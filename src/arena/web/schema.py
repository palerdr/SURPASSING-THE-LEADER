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

import unicodedata
from typing import Annotated, Literal

from pydantic import BaseModel, Field, field_validator

from arena.session import (
    CANONICAL_HAL_NAME,
    Phase,
    PlaySession,
    validate_human_display_name,
)
from arena.web.names import is_offensive
from stl.engine.game import (
    CYLINDER_MAX,
    TOTAL_TTD_MAX,
    TURN_DURATION_LEAP,
    HalfRoundResult,
)

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
    """A reveal with distinct engine-game-over and session-ending signals.

    ``round`` and ``half`` are the resolved half-round's own, not the next
    half's: the engine has already advanced by the time the reveal is shown.
    """

    round: int
    half: int
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
    game_index: int
    pure_dth: bool
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


LEADERBOARD_NAME_LENGTH = 16
LEADERBOARD_SIZE = 10
# Control, format (bidi and zero-width), surrogate, private-use, unassigned,
# and line or paragraph separators.
_HIDDEN_CATEGORIES = {"Cc", "Cf", "Cs", "Co", "Cn", "Zl", "Zp"}
# Blank glyphs that Unicode files as letters or symbols: the Hangul fillers
# and the empty braille cell.
_BLANK_GLYPHS = {"\u115f", "\u1160", "\u3164", "\uffa0", "\u2800"}


class LeaderboardEntry(BaseModel):
    """One standing: a player's latest game, which that player won."""

    rank: int
    name: str
    score: float
    half_rounds: int
    is_you: bool


class Leaderboard(BaseModel):
    """The top standings plus the requesting player's own.

    Player identifiers and seeds stay in the ledger. ``your_score`` is set when
    the player's latest game was a win; ``your_rank`` also needs a posted name.
    """

    entries: list[LeaderboardEntry]
    your_rank: int | None
    your_name: str | None
    your_score: float | None


def top_standings(payload: dict[str, object]) -> Leaderboard:
    """The board a browser may see: the ledger's answer, cut to the top ten."""

    board = Leaderboard.model_validate(payload)
    board.entries = board.entries[:LEADERBOARD_SIZE]
    return board


class PlayerNameRequest(BaseModel):
    name: str

    @field_validator("name")
    @classmethod
    def _valid_name(cls, value: str) -> str:
        # Every player reads this name, so it must show as the text it holds:
        # no direction overrides, no invisible characters, no stacked marks.
        name = validate_human_display_name(unicodedata.normalize("NFC", value))
        if len(name) > LEADERBOARD_NAME_LENGTH:
            raise ValueError(
                f"a leaderboard name has at most {LEADERBOARD_NAME_LENGTH} characters"
            )
        categories = [unicodedata.category(character) for character in name]
        if any(category in _HIDDEN_CATEGORIES for category in categories) or any(
            character in _BLANK_GLYPHS for character in name
        ):
            raise ValueError("a leaderboard name must use visible characters")
        # Fullwidth and styled letters fold to plain ones, so they cannot spell Hal.
        if unicodedata.normalize("NFKC", name).casefold() == CANONICAL_HAL_NAME.casefold():
            raise ValueError(f"{CANONICAL_HAL_NAME!r} is reserved for the opponent")
        if not any(category[0] in "LN" for category in categories):
            raise ValueError("a leaderboard name needs a letter or a digit")
        marks = 0
        for category in categories:
            marks = marks + 1 if category[0] == "M" else 0
            if marks > 2:
                raise ValueError("a leaderboard name has too many combining marks")
        if is_offensive(name):
            raise ValueError("choose another leaderboard name")
        return name


def _outcome_view(session: PlaySession) -> OutcomeView | None:
    record = session.last_record
    if record is None or session.phase not in _REVEALED:
        return None
    return OutcomeView(
        round=int(record.round_num + 1),
        half=int(record.half),
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
        game_index=session.game_index,
        pure_dth=session.pure_dth,
        human_name=session.human_display_name,
        clock_display=game.format_game_clock(),
        clock_seconds=float(game.game_clock),
        round=int(game.round_num + 1),
        half=int(game.current_half),
        turn_duration=int(turn_duration),
        # The leap turn is in effect only when the engine lengthened the
        # turn; a pure-DTH game keeps 60 seconds however the clock reads.
        leap_window=turn_duration == TURN_DURATION_LEAP,
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
