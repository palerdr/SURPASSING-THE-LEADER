"""Wire types for the browser front end.

This module owns the single seat-scoped serializer. Everything the browser is
allowed to know passes through :func:`snapshot_from_session` and nothing else,
so the hidden-information rule has exactly one place to be enforced and exactly
one place to be tested.

The rule: an unrevealed action never leaves the process. Hal's second does not
exist during :attr:`~arena.session.Phase.AWAITING_ACTION` (see
:meth:`arena.session.PlaySession.submit`), and ``last_outcome`` is populated
only once the half-round has resolved.
"""

from __future__ import annotations

from pydantic import BaseModel

from arena.session import Phase, PlaySession
from stl.engine.game import CYLINDER_MAX, LS_WINDOW_END, LS_WINDOW_START, TOTAL_TTD_MAX

# Phases in which the reveal is public and may be serialized.
_REVEALED = (Phase.AWAITING_ACK, Phase.GAME_OVER)


class PlayerView(BaseModel):
    name: str
    cylinder_seconds: float
    ttd_seconds: float
    deaths: int
    is_human: bool


class OutcomeView(BaseModel):
    """The reveal. Only ever built from a resolved half-round record."""

    dropper: str
    checker: str
    drop_time: int
    check_time: int
    result: str
    st_gained: float
    death_duration: float
    survived: bool | None
    survival_probability: float | None
    game_over: bool
    winner_name: str | None


class Snapshot(BaseModel):
    sequence: int
    phase: str
    human_name: str
    clock_display: str
    clock_seconds: float
    round: int
    half: int
    turn_duration: int
    leap_window: bool
    dropper_name: str
    checker_name: str
    human_role: str
    legal_seconds: list[int]
    players: list[PlayerView]
    cylinder_max: float
    ttd_max: float
    half_rounds: int
    last_outcome: OutcomeView | None
    winner_name: str | None
    stopped: bool


class ActionRequest(BaseModel):
    sequence: int
    second: int


class SequencedRequest(BaseModel):
    sequence: int


class NewSessionRequest(BaseModel):
    human_name: str | None = None
    seed: int | None = None
    start_clock: int | None = None
    max_half_rounds: int | None = None


def _outcome_view(session: PlaySession) -> OutcomeView | None:
    record = session.last_record
    if record is None or session.phase not in _REVEALED:
        return None
    return OutcomeView(
        dropper=record.dropper,
        checker=record.checker,
        drop_time=int(record.drop_time),
        check_time=int(record.check_time),
        result=record.result.value,
        st_gained=float(record.st_gained),
        death_duration=float(record.death_duration),
        survived=record.survived,
        survival_probability=record.survival_probability,
        game_over=bool(session.game.game_over),
        winner_name=session.winner_name,
    )


def snapshot_from_session(session: PlaySession) -> Snapshot:
    """Serialize everything the browser may see, and nothing more."""

    game = session.game
    dropper, checker = session.roles()
    turn_duration = session.turn_duration()
    # Legality is the engine's call and is only meaningful while the human is
    # on the clock; sending it otherwise would invite the client to act early.
    legal = (
        list(session.legal_actions()) if session.phase is Phase.AWAITING_ACTION else []
    )
    return Snapshot(
        sequence=session.sequence,
        phase=session.phase.value,
        human_name=session.human.name,
        clock_display=game.format_game_clock(),
        clock_seconds=float(game.game_clock),
        round=int(game.round_num + 1),
        half=int(game.current_half),
        turn_duration=int(turn_duration),
        leap_window=LS_WINDOW_START <= game.game_clock <= LS_WINDOW_END,
        dropper_name=dropper.name,
        checker_name=checker.name,
        human_role=session.human_role(),
        legal_seconds=legal,
        players=[
            PlayerView(
                name=player.name,
                cylinder_seconds=float(player.cylinder),
                ttd_seconds=float(player.ttd),
                deaths=int(player.deaths),
                is_human=player is session.human,
            )
            for player in (game.player1, game.player2)
        ],
        cylinder_max=float(CYLINDER_MAX),
        ttd_max=float(TOTAL_TTD_MAX),
        half_rounds=session.half_rounds,
        last_outcome=_outcome_view(session),
        winner_name=session.winner_name,
        stopped=session.stopped,
    )
