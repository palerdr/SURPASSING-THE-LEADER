"""Local HTTP surface for browser play.

A third front end onto the same session, alongside ``arena.cli`` and
``arena.tui``. It imports only ``stl`` and ``arena``, so it introduces no new
peer project and no new import edge that ``AGENTS.md`` forbids. It shares the
rules text and the prepared art with the terminal through
``arena.presentation``, and it never imports the terminal renderer. The STL
engine remains the only referee; this module sequences requests and serializes
state.

The server holds exactly one live session, because it is a local single-player
surface, but that session sits inside one repeated-opponent series exactly as
``arena play --games N`` does: Hal is built once and keeps its opponent model
across games, each finished game is appended to a public transcript in the
CLI's ``arena-public-play-session-v1`` shape, and each new game receives a
fresh canonical referee whose seed is the base seed plus the game index.

Every mutating endpoint carries the sequence number the client last saw, so a
double-submitted action or a stale second tab is rejected rather than silently
replayed.
"""

from __future__ import annotations

import json
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from fastapi import FastAPI, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from arena.presentation.rules_text import rules_body
from arena.presentation.scene_art import PANEL_ROOT, SceneArt
from arena.presentation.sprites import encode_png
from arena.session import (
    CANONICAL_HAL_NAME,
    CANONICAL_HUMAN_NAME,
    Phase,
    PlaySession,
    SessionPhaseError,
    validate_human_display_name,
)
from arena.variants import PureDTHGame
from arena.web.schema import (
    ActionRequest,
    NewSessionRequest,
    SequencedRequest,
    Snapshot,
    snapshot_from_session,
)
from stl.engine.game import (
    OPENING_START_CLOCK,
    PHYSICALITY_BAKU,
    PHYSICALITY_HAL,
    Game,
    Player,
    Referee,
)

HalFactory = Callable[[], object]

# Manga panels the client may request by name, under PANEL_ROOT. The rules
# spread opens the game; it is the page the chapter states its rules on.
PANELS = {"stl_rules": ("stl_rules.png", "image/png")}

TRANSCRIPT_SCHEMA = "arena-public-play-session-v1"

# The built browser client. ``npm --prefix src/arena/webclient run build``
# writes it; when it exists the Python server serves it alone, with no Vite
# process, and when it does not the root route explains how to get it.
DEFAULT_WEBCLIENT_DIST = Path(__file__).resolve().parents[1] / "webclient" / "dist"

_NO_CLIENT_PAGE = """<!doctype html>
<meta charset="utf-8">
<title>Surpassing The Leader</title>
<body style="background:#000;color:#f2f2f2;font:14px/1.6 ui-monospace,monospace;padding:32px">
<h1 style="font-size:15px;letter-spacing:.22em">SURPASSING THE LEADER</h1>
<p>The API is running, but the browser client has not been built.</p>
<p>Either build it once and reload this page:</p>
<pre>npm --prefix src/arena/webclient install
npm --prefix src/arena/webclient run build</pre>
<p>or run the development client, which proxies to this server:</p>
<pre>npm --prefix src/arena/webclient run dev</pre>
</body>
"""


@dataclass
class SessionConfig:
    """Per-game settings. ``pure_dth`` is fixed for the whole server."""

    human_name: str = CANONICAL_HUMAN_NAME
    seed: int | None = None
    start_clock: int = OPENING_START_CLOCK
    max_half_rounds: int | None = None
    pure_dth: bool = False

    def __post_init__(self) -> None:
        self.human_name = validate_human_display_name(self.human_name)
        if self.seed is not None and (
            isinstance(self.seed, bool) or not isinstance(self.seed, int)
        ):
            raise TypeError("seed must be an integer or None")
        if isinstance(self.start_clock, bool) or not isinstance(self.start_clock, int):
            raise TypeError("start_clock must be an integer")
        if self.start_clock < 0:
            raise ValueError("start_clock must be nonnegative")
        if self.max_half_rounds is not None:
            if isinstance(self.max_half_rounds, bool) or not isinstance(
                self.max_half_rounds, int
            ):
                raise TypeError("max_half_rounds must be an integer or None")
            if self.max_half_rounds <= 0:
                raise ValueError("max_half_rounds must be positive")
        if not isinstance(self.pure_dth, bool):
            raise TypeError("pure_dth must be a boolean")


@dataclass
class SeriesConfig:
    """Session-wide presentation and record-keeping, as ``arena play`` takes it."""

    hal_agent: str = "dth"
    public_hal_label: str | None = None
    conceal_hal_details: bool = False
    transcript_path: Path | None = None

    @property
    def hal_label(self) -> str:
        return self.public_hal_label or self.hal_agent


def _new_session(
    hal_agent: object,
    config: SessionConfig,
    *,
    game_index: int = 0,
    game_seed: int | None = None,
    sequence_start: int = 0,
    reset_provider: bool = True,
) -> PlaySession:
    import random

    from arena.contracts import reset_provider_game

    hal = Player(name=CANONICAL_HAL_NAME, physicality=PHYSICALITY_HAL)
    human = Player(name=CANONICAL_HUMAN_NAME, physicality=PHYSICALITY_BAKU)
    game_type: type[Game] = Game
    if config.pure_dth:
        game_type = PureDTHGame
    game = game_type(
        player1=hal,
        player2=human,
        referee=Referee(),
        rng=random.Random(game_seed),
    )
    game.game_clock = config.start_clock
    if reset_provider:
        reset_provider_game(getattr(hal_agent, "provider", None))
    return PlaySession(
        game=game,
        hal_agent=hal_agent,
        hal=hal,
        human=human,
        human_display_name=config.human_name,
        game_index=game_index,
        game_seed=game_seed,
        start_clock=config.start_clock,
        max_half_rounds=config.max_half_rounds,
        sequence_start=sequence_start,
        pure_dth=config.pure_dth,
    )


def _game_seed(base_seed: int | None, game_index: int) -> int | None:
    """The CLI's per-game seed rule: base seed plus game index."""

    return None if base_seed is None else base_seed + game_index


def write_play_transcript(destination: str | Path, transcript: dict[str, object]) -> Path:
    """Atomically write a public transcript, exactly as the CLI does."""

    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(transcript, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)
    return path


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

    base_config = config or SessionConfig()
    series_config = series or SeriesConfig()
    hal_agent = hal_factory()
    # The hosted adapter checkpoints provider memory at game boundaries.
    app.state.hal_provider = getattr(hal_agent, "provider", None)
    series_seed = base_config.seed
    lock = threading.Lock()
    state: dict[str, object] = {
        "session": _new_session(
            hal_agent,
            base_config,
            game_seed=_game_seed(base_config.seed, 0),
            sequence_start=sequence_start,
        ),
        "game_index": 0,
        "recorded": False,
    }
    transcript: dict[str, object] = {
        "schema_version": TRANSCRIPT_SCHEMA,
        "hal_agent": series_config.hal_agent,
        "public_hal_label": series_config.public_hal_label,
        "human_name": base_config.human_name,
        "base_seed": base_config.seed,
        "start_clock": base_config.start_clock,
        "start_clock_sequence": None,
        # A browser series is open-ended; the CLI records its --games count here.
        "requested_games": None,
        "pure_dth": base_config.pure_dth,
        "games": [],
    }
    tally = {"human_wins": 0, "hal_wins": 0, "no_winner": 0, "stopped": 0}
    art_cache: dict[str, object] = {}

    def _session() -> PlaySession:
        session = state["session"]
        assert isinstance(session, PlaySession)
        return session

    def _game_index() -> int:
        index = state["game_index"]
        assert isinstance(index, int)
        return index

    def _require(sequence: int, session: PlaySession) -> None:
        if sequence != session.sequence:
            raise HTTPException(
                status_code=409,
                detail=f"stale sequence {sequence}; session is at {session.sequence}",
            )

    def _hal_details() -> dict[str, object]:
        details: dict[str, object] = {}
        match_summary = getattr(getattr(hal_agent, "provider", None), "match_summary", None)
        if callable(match_summary):
            details["hal_summary"] = match_summary()
        diagnostics = getattr(
            getattr(hal_agent, "provider", None), "experiment_diagnostics", None
        )
        if callable(diagnostics):
            details["hal_diagnostics"] = diagnostics()
        return details

    def _record_finished_game(session: PlaySession) -> None:
        """Append a terminal game to the series once, then persist the file."""

        if state["recorded"]:
            return
        record = session.finish()
        games = transcript["games"]
        assert isinstance(games, list)
        games.append(record)
        if session.stopped:
            tally["stopped"] += 1
        elif session.game.winner is None:
            tally["no_winner"] += 1
        elif session.game.winner is session.human:
            tally["human_wins"] += 1
        else:
            tally["hal_wins"] += 1
        state["recorded"] = True
        if series_config.transcript_path is not None:
            write_play_transcript(
                series_config.transcript_path, {**transcript, **_hal_details()}
            )

    @app.get("/api/rules")
    def rules() -> dict[str, object]:
        return {
            "human_name": _session().human_display_name,
            "hal_label": series_config.hal_label,
            "pure_dth": base_config.pure_dth,
            "lines": list(rules_body()),
        }

    @app.get("/api/session", response_model=Snapshot)
    def read_session() -> Snapshot:
        with lock:
            return snapshot_from_session(_session())

    @app.post("/api/session", response_model=Snapshot)
    def new_session(request: NewSessionRequest) -> Snapshot:
        with lock:
            previous = _session()
            _require(request.sequence, previous)
            if previous.phase not in (Phase.RULES, Phase.GAME_OVER):
                raise HTTPException(
                    status_code=409,
                    detail=f"cannot replace an active session in phase {previous.phase.value}",
                )
            finished = previous.phase is Phase.GAME_OVER
            if finished:
                _record_finished_game(previous)
                state["game_index"] = _game_index() + 1
            game_index = _game_index()
            config = SessionConfig(
                human_name=(
                    request.human_name
                    if request.human_name is not None
                    else previous.human_display_name
                ),
                seed=series_seed,
                start_clock=(
                    request.start_clock
                    if request.start_clock is not None
                    else base_config.start_clock
                ),
                max_half_rounds=(
                    request.max_half_rounds
                    if request.max_half_rounds is not None
                    else base_config.max_half_rounds
                ),
                pure_dth=base_config.pure_dth,
            )
            game_seed = (
                request.seed
                if request.seed is not None
                else _game_seed(series_seed, game_index)
            )
            state["session"] = _new_session(
                hal_agent,
                config,
                game_index=game_index,
                game_seed=game_seed,
                sequence_start=previous.sequence + 1,
                reset_provider=finished,
            )
            state["recorded"] = False
            transcript["human_name"] = config.human_name
            return snapshot_from_session(_session())

    @app.post("/api/session/restart", response_model=Snapshot)
    def restart(request: SequencedRequest) -> Snapshot:
        nonlocal series_seed
        with lock:
            previous = _session()
            _require(request.sequence, previous)
            sequence = previous.sequence + 1
            series_seed = _game_seed(base_config.seed, sequence)
            state["session"] = _new_session(
                hal_agent, base_config, game_seed=series_seed, sequence_start=sequence
            )
            state["game_index"] = 0
            state["recorded"] = False
            transcript["games"] = []
            transcript["human_name"] = base_config.human_name
            transcript["base_seed"] = series_seed
            tally.update({key: 0 for key in tally})
            return snapshot_from_session(_session())

    @app.post("/api/session/begin", response_model=Snapshot)
    def begin(request: SequencedRequest) -> Snapshot:
        with lock:
            session = _session()
            _require(request.sequence, session)
            try:
                session.begin()
            except SessionPhaseError as error:
                raise HTTPException(status_code=409, detail=str(error)) from error
            if session.phase is Phase.GAME_OVER:
                _record_finished_game(session)
            return snapshot_from_session(session)

    @app.post("/api/session/action", response_model=Snapshot)
    def act(request: ActionRequest) -> Snapshot:
        with lock:
            session = _session()
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
            session = _session()
            _require(request.sequence, session)
            try:
                session.acknowledge()
            except SessionPhaseError as error:
                raise HTTPException(status_code=409, detail=str(error)) from error
            if session.phase is Phase.GAME_OVER:
                _record_finished_game(session)
            return snapshot_from_session(session)

    @app.get("/api/transcript")
    def read_transcript() -> dict[str, object]:
        """The public series so far: finished games plus the live game's reveals.

        Everything here has already been revealed on an outcome screen. The
        live game's history contains only resolved half-rounds, so a request
        made while the human is deciding cannot see an unrevealed second.
        """

        with lock:
            session = _session()
            payload: dict[str, object] = {
                **transcript,
                "tally": dict(tally),
                "current_game": {
                    "game_index": session.game_index,
                    "seed": session.game_seed,
                    "start_clock": session.start_clock,
                    "phase": session.phase.value,
                    "half_rounds": session.half_rounds,
                    "public_history": list(session.public_history),
                },
            }
            if not series_config.conceal_hal_details:
                payload.update(_hal_details())
            return payload

    @app.get("/art/panel/{name}")
    def panel(name: str) -> Response:
        """A manga panel from ``art/panels``, by bare name; only known names."""

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
