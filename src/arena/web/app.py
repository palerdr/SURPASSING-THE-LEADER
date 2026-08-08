"""Local HTTP surface for browser play.

A third front end onto the same session, alongside ``arena.cli`` and
``arena.tui``. It imports only ``stl`` and ``arena``, so it introduces no new
peer project and no new import edge that ``AGENTS.md`` forbids. The STL engine
remains the only referee; this module sequences requests and serializes state.

The server holds exactly one session, because it is a local single-player
surface. Every mutating endpoint carries the sequence number the client last
saw, so a double-submitted action or a stale second tab is rejected rather than
silently replayed.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Callable

from fastapi import FastAPI, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware

from arena.session import Phase, PlaySession, SessionPhaseError
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


@dataclass
class SessionConfig:
    human_name: str = "Baku"
    seed: int | None = None
    start_clock: int = OPENING_START_CLOCK
    max_half_rounds: int | None = None


def _new_session(hal_agent: object, config: SessionConfig) -> PlaySession:
    import random

    from arena.contracts import reset_provider_game

    hal = Player(name="Hal", physicality=PHYSICALITY_HAL)
    human = Player(name=config.human_name, physicality=PHYSICALITY_BAKU)
    game = Game(
        player1=hal,
        player2=human,
        referee=Referee(),
        rng=random.Random(config.seed),
    )
    game.game_clock = config.start_clock
    reset_provider_game(getattr(hal_agent, "provider", None))
    return PlaySession(
        game=game,
        hal_agent=hal_agent,
        hal=hal,
        human=human,
        game_seed=config.seed,
        start_clock=config.start_clock,
        max_half_rounds=config.max_half_rounds,
    )


def create_app(
    *,
    hal_factory: HalFactory,
    config: SessionConfig | None = None,
    art_loader: Callable[[], object] | None = None,
) -> FastAPI:
    """Build the app. ``hal_factory`` is called once, never inside a request.

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
    hal_agent = hal_factory()
    lock = threading.Lock()
    state: dict[str, object] = {"session": _new_session(hal_agent, base_config)}
    art_cache: dict[str, object] = {}

    def _session() -> PlaySession:
        session = state["session"]
        assert isinstance(session, PlaySession)
        return session

    def _require(sequence: int, session: PlaySession) -> None:
        if sequence != session.sequence:
            raise HTTPException(
                status_code=409,
                detail=f"stale sequence {sequence}; session is at {session.sequence}",
            )

    @app.get("/api/rules")
    def rules() -> dict[str, object]:
        from arena.tui import rules_body

        return {"human_name": _session().human.name, "lines": list(rules_body())}

    @app.get("/api/session", response_model=Snapshot)
    def read_session() -> Snapshot:
        with lock:
            return snapshot_from_session(_session())

    @app.post("/api/session", response_model=Snapshot)
    def new_session(request: NewSessionRequest) -> Snapshot:
        with lock:
            config = SessionConfig(
                human_name=request.human_name or base_config.human_name,
                seed=request.seed if request.seed is not None else base_config.seed,
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
            )
            state["session"] = _new_session(hal_agent, config)
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
                session.finish()
            return snapshot_from_session(session)

    @app.get("/art/{character}/{pose}/{index}.png")
    def frame(character: str, pose: str, index: int) -> Response:
        from arena.sprites import encode_png
        from arena.tui import SceneArt

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

    return app
