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

from typing import Annotated, Literal

from pydantic import BaseModel, Field, field_validator

from arena.session import Phase, PlaySession, validate_human_display_name
from stl.engine.game import CYLINDER_MAX, TOTAL_TTD_MAX, HalfRoundResult

# Phases in which the reveal is public and may be serialized.
_REVEALED = (Phase.AWAITING_ACK, Phase.GAME_OVER)


class PlayerView(BaseModel):
    """One display row with server-owned seat, character, and current role."""

    name: str
    character: Literal["hal", "baku"]
    role: Literal["dropper", "checker"]
    cylinder_seconds: float
    ttd_seconds: float
    deaths: int
    is_human: bool


class OutcomeView(BaseModel):
    """A reveal with distinct engine-game-over and session-ending signals."""

    dropper: str
    checker: str
    drop_time: int
    check_time: int
    result: HalfRoundResult
    st_gained: float
    death_duration: float
    survived: bool | None
    survival_probability: float | None
    game_over: bool
    session_ending: bool
    winner_name: str | None


class Snapshot(BaseModel):
    sequence: int
    phase: Phase
    human_name: str
    clock_display: str
    clock_seconds: float
    round: int
    half: int
    turn_duration: int
    leap_window: bool
    dropper_name: str
    checker_name: str
    human_role: Literal["dropper", "checker"]
    legal_seconds: list[int]
    players: list[PlayerView]
    cylinder_max: float
    ttd_max: float
    half_rounds: int
    last_outcome: OutcomeView | None
    winner_name: str | None
    winner_is_human: bool | None
    stopped: bool


class ActionRequest(BaseModel):
    sequence: Annotated[int, Field(strict=True, ge=0)]
    second: Annotated[int, Field(strict=True, ge=1, le=61)]


class SequencedRequest(BaseModel):
    sequence: Annotated[int, Field(strict=True, ge=0)]


class NewSessionRequest(BaseModel):
    sequence: Annotated[int, Field(strict=True, ge=0)]
    human_name: str | None = None
    seed: Annotated[int, Field(strict=True)] | None = None
    start_clock: Annotated[int, Field(strict=True, ge=0)] | None = None
    max_half_rounds: Annotated[int, Field(strict=True, ge=1)] | None = None

    @field_validator("human_name")
    @classmethod
    def _valid_human_name(cls, value: str | None) -> str | None:
        return None if value is None else validate_human_display_name(value)


def _outcome_view(session: PlaySession) -> OutcomeView | None:
    record = session.last_record
    if record is None or session.phase not in _REVEALED:
        return None
    return OutcomeView(
        dropper=session.display_canonical_name(record.dropper),
        checker=session.display_canonical_name(record.checker),
        drop_time=int(record.drop_time),
        check_time=int(record.check_time),
        result=record.result,
        st_gained=float(record.st_gained),
        death_duration=float(record.death_duration),
        survived=record.survived,
        survival_probability=record.survival_probability,
        game_over=bool(session.game.game_over),
        session_ending=session.terminal,
        winner_name=session.display_canonical_name(session.winner_name),
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
        phase=session.phase,
        human_name=session.human_display_name,
        clock_display=game.format_game_clock(),
        clock_seconds=float(game.game_clock),
        round=int(game.round_num + 1),
        half=int(game.current_half),
        turn_duration=int(turn_duration),
        leap_window=game.is_leap_second_turn(),
        dropper_name=session.display_name(dropper),
        checker_name=session.display_name(checker),
        human_role=session.human_role(),
        legal_seconds=legal,
        players=[
            PlayerView(
                name=session.display_name(player),
                character="baku" if player is session.human else "hal",
                role="dropper" if player is dropper else "checker",
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
        winner_name=session.display_canonical_name(session.winner_name),
        winner_is_human=(
            None if game.winner is None else game.winner is session.human
        ),
        stopped=session.stopped,
    )
