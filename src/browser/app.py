"""Local HTTP surface for browser play.

A front end onto the same session as the terminal app (``terminal.cli`` and
``terminal.tui``). The app imports only ``arena`` and ``stl.engine``, the
imports that ``docs/PROJECTS.toml`` allows it. It shares the
rules text and the prepared art with the terminal through
``arena.presentation``, and it never imports the terminal renderer. The STL
engine remains the only referee.

The server holds exactly one live session, because it is a local single-player
surface, but that session sits inside one repeated-opponent series exactly as
``terminal play --games N`` does. ``series.py`` builds the games and keeps the
record of the finished ones; this module sequences requests onto it under one
lock and serializes state.

Every mutating endpoint carries the sequence number the client last saw, so a
double-submitted action or a stale second tab is rejected rather than silently
replayed.
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Callable

from fastapi import FastAPI, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from arena.presentation.rules_text import rules_body
from arena.presentation.scene_art import PANEL_ROOT, SceneArt
from arena.presentation.sprites import encode_png
from arena.session import Phase, PlaySession, SessionPhaseError
from browser.schema import (
    ActionRequest,
    NewSessionRequest,
    SequencedRequest,
    Snapshot,
    snapshot_from_session,
)
from browser.series import Series, SeriesConfig, SessionConfig

HalFactory = Callable[[], object]

# Manga panels the client may request by name, under PANEL_ROOT. The rules
# spread opens the game; it is the page the chapter states its rules on.
PANELS = {"stl_rules": ("stl_rules.png", "image/png")}

# The built browser client. ``npm --prefix src/browser/webclient run build``
# writes it; when it exists the Python server serves it alone, with no Vite
# process, and when it does not the root route explains how to get it.
DEFAULT_WEBCLIENT_DIST = Path(__file__).resolve().parent / "webclient" / "dist"

_NO_CLIENT_PAGE = """<!doctype html>
<meta charset="utf-8">
<title>Surpassing The Leader</title>
<body style="background:#000;color:#f2f2f2;font:14px/1.6 ui-monospace,monospace;padding:32px">
<h1 style="font-size:15px;letter-spacing:.22em">SURPASSING THE LEADER</h1>
<p>The API is running, but the browser client has not been built.</p>
<p>Either build it once and reload this page:</p>
<pre>npm --prefix src/browser/webclient install
npm --prefix src/browser/webclient run build</pre>
<p>or run the development client, which proxies to this server:</p>
<pre>npm --prefix src/browser/webclient run dev</pre>
</body>
"""


def create_app(
    *,
    hal_factory: HalFactory,
    config: SessionConfig | None = None,
    series: SeriesConfig | None = None,
    art_loader: Callable[[], object] | None = None,
    webclient_dist: Path | None = DEFAULT_WEBCLIENT_DIST,
    sequence_start: int = 0,
) -> FastAPI:
    """Build the app. ``hal_factory`` is called once, never inside a request.

    ``sequence_start`` numbers the first session. The hosted server opens a
    fresh app after a restart and continues the player's sequence there, so a
    request from before the restart stays stale.

    Provider construction can memory-map a multi-gigabyte artifact, and the
    ``abstract`` provider can even build a tablebase from scratch. Neither
    belongs on a request path, so both happen here at startup.
    """

    app = FastAPI(title="Surpassing The Leader")
    # The Vite dev server is a different origin; the API is bound to loopback.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    played = Series(hal_factory(), config, series, sequence_start=sequence_start)
    # The hosted adapter checkpoints provider memory at game boundaries.
    app.state.hal_provider = played.provider
    lock = threading.Lock()
    art_cache: dict[str, object] = {}

    def _require(sequence: int, session: PlaySession) -> None:
        if sequence != session.sequence:
            raise HTTPException(
                status_code=409,
                detail=f"stale sequence {sequence}; session is at {session.sequence}",
            )

    @app.get("/api/rules")
    def rules() -> dict[str, object]:
        return {
            "human_name": played.session.human_display_name,
            "hal_label": played.series_config.hal_label,
            "pure_dth": played.config.pure_dth,
            "lines": list(rules_body()),
        }

    @app.get("/api/session", response_model=Snapshot)
    def read_session() -> Snapshot:
        with lock:
            return snapshot_from_session(played.session)

    @app.post("/api/session", response_model=Snapshot)
    def new_session(request: NewSessionRequest) -> Snapshot:
        with lock:
            previous = played.session
            _require(request.sequence, previous)
            if previous.phase not in (Phase.RULES, Phase.GAME_OVER):
                raise HTTPException(
                    status_code=409,
                    detail=f"cannot replace an active session in phase {previous.phase.value}",
                )
            session = played.next_game(
                human_name=request.human_name,
                start_clock=request.start_clock,
                max_half_rounds=request.max_half_rounds,
                seed=request.seed,
            )
            return snapshot_from_session(session)

    @app.post("/api/session/restart", response_model=Snapshot)
    def restart(request: SequencedRequest) -> Snapshot:
        with lock:
            _require(request.sequence, played.session)
            return snapshot_from_session(played.restart())

    @app.post("/api/session/begin", response_model=Snapshot)
    def begin(request: SequencedRequest) -> Snapshot:
        with lock:
            session = played.session
            _require(request.sequence, session)
            try:
                session.begin()
            except SessionPhaseError as error:
                raise HTTPException(status_code=409, detail=str(error)) from error
            if session.phase is Phase.GAME_OVER:
                played.record_finished_game()
            return snapshot_from_session(session)

    @app.post("/api/session/action", response_model=Snapshot)
    def act(request: ActionRequest) -> Snapshot:
        with lock:
            session = played.session
            _require(request.sequence, session)
            if session.phase is not Phase.AWAITING_ACTION:
                raise HTTPException(status_code=409, detail=f"phase is {session.phase.value}")
            if request.second not in session.legal_actions():
                raise HTTPException(
                    status_code=422,
                    detail=f"{request.second} is not a legal second this turn",
                )
            try:
                session.submit(request.second)
            except SessionPhaseError as error:
                raise HTTPException(status_code=409, detail=str(error)) from error
            return snapshot_from_session(session)

    @app.post("/api/session/ack", response_model=Snapshot)
    def acknowledge(request: SequencedRequest) -> Snapshot:
        with lock:
            session = played.session
            _require(request.sequence, session)
            try:
                session.acknowledge()
            except SessionPhaseError as error:
                raise HTTPException(status_code=409, detail=str(error)) from error
            if session.phase is Phase.GAME_OVER:
                played.record_finished_game()
            return snapshot_from_session(session)

    @app.get("/api/transcript")
    def read_transcript() -> dict[str, object]:
        """The public series so far; ``Series.public_transcript`` states its scope."""

        with lock:
            return played.public_transcript()

    @app.get("/art/panel/{name}")
    def panel(name: str) -> Response:
        """A manga panel from ``PANEL_ROOT``, by bare name; only known names."""

        entry = PANELS.get(name)
        if entry is None:
            raise HTTPException(status_code=404, detail=f"no panel {name}")
        filename, media_type = entry
        path = PANEL_ROOT / filename
        if not path.is_file():
            raise HTTPException(status_code=404, detail=f"panel {name} is not on disk")
        return Response(
            content=path.read_bytes(),
            media_type=media_type,
            headers={"Cache-Control": "no-cache"},
        )

    @app.get("/art/{character}/{pose}/{index}.png")
    def frame(character: str, pose: str, index: int) -> Response:
        if "art" not in art_cache:
            art_cache["art"] = (art_loader or SceneArt.load)()
        sprite = art_cache["art"].frame(character, pose, index)
        if sprite is None:
            raise HTTPException(status_code=404, detail=f"no art for {character}/{pose}")
        return Response(
            content=encode_png(sprite),
            media_type="image/png",
            headers={"Cache-Control": "no-cache"},
        )

    if webclient_dist is not None and (webclient_dist / "index.html").is_file():

        @app.middleware("http")
        async def _revalidate_page(request, call_next):
            # Vite names its bundles by content hash, so they may be cached
            # freely; the page that references them must be re-checked on
            # every load or a rebuild stays invisible until a hard refresh.
            response = await call_next(request)
            if request.url.path in ("/", "/index.html"):
                response.headers["Cache-Control"] = "no-cache"
            return response

        # Registered last so the API and art routes above take precedence.
        app.mount("/", StaticFiles(directory=webclient_dist, html=True), name="webclient")
    else:

        @app.get("/", response_class=HTMLResponse, include_in_schema=False)
        def no_client() -> str:
            return _NO_CLIENT_PAGE

    return app
