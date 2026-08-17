"""Headless, resumable play session for one canonical STL game.

The terminal loop in :mod:`arena.cli` used to drive a game by *pulling* the
human's action through a blocking ``input()``. That works for a terminal and
nowhere else: a browser, a socket, or any client-driven front end needs the
server to suspend at "waiting for the human" and resume when a request arrives.

:class:`PlaySession` is that suspend point, expressed as a phase machine. It
owns no rendering and no I/O. The STL engine remains the only referee; this
class only sequences the calls the CLI loop already made, in the same order.

Exactly one of the Dropper and the Checker is the human, so a half-round has
two suspend points: the human's action, and the acknowledgement of the reveal.

Hidden information is preserved structurally. Hal's action is computed inside
:meth:`PlaySession.submit`, *after* the human's second has been accepted, so it
does not exist anywhere in the process while the human is deciding. No snapshot
taken during :attr:`Phase.AWAITING_ACTION` can leak it, because there is nothing
to leak. This is behaviourally identical to the old loop: ``decision_from_game``
reads only pre-decision state, which the human's choice does not mutate, and
``PolicyDrivenAgent`` still draws from its generator exactly once per half-round.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from arena.agent import public_state_from_game
from arena.contracts import (
    PublicDecisionState,
    PublicGameOutcome,
    PublicHalfRound,
    end_provider_game,
    observe_provider,
)
from stl.engine.actions import legal_seconds, validate_action
from stl.engine.game import Game, HalfRoundRecord, Player


CANONICAL_HAL_NAME = "Hal"
CANONICAL_HUMAN_NAME = "Baku"
MAX_DISPLAY_NAME_LENGTH = 64


class Phase(str, Enum):
    """Where a session is waiting."""

    RULES = "rules"
    AWAITING_ACTION = "awaiting_action"
    AWAITING_ACK = "awaiting_ack"
    GAME_OVER = "game_over"


class SessionPhaseError(RuntimeError):
    """A transition was requested from a phase that does not allow it."""


def validate_human_display_name(name: str) -> str:
    """Validate a presentation label without changing rule-bearing identity."""

    if not isinstance(name, str):
        raise TypeError("human display name must be a string")
    normalized = name.strip()
    if not normalized:
        raise ValueError("human display name must not be empty")
    if len(normalized) > MAX_DISPLAY_NAME_LENGTH:
        raise ValueError(
            f"human display name must be at most {MAX_DISPLAY_NAME_LENGTH} characters"
        )
    if normalized.casefold() == CANONICAL_HAL_NAME.casefold():
        raise ValueError(f"{CANONICAL_HAL_NAME!r} is reserved for the opponent")
    if any(ord(character) < 32 or ord(character) == 127 for character in normalized):
        raise ValueError("human display name must not contain control characters")
    return normalized


def public_player_state(player: Player) -> dict[str, float | int | str]:
    """One player's public load state, as recorded in a play transcript."""

    return {
        "name": player.name,
        "cylinder_seconds": float(player.cylinder),
        "ttd_seconds": float(player.ttd),
        "deaths": int(player.deaths),
    }


@dataclass(slots=True)
class PlaySession:
    """One canonical game, advanced a half-round at a time by a caller.

    ``hal`` and ``human`` must be the two players of ``game``; which of them
    drops in a given half is the engine's decision, not this class's.
    """

    game: Game
    hal_agent: object
    hal: Player
    human: Player
    human_display_name: str = CANONICAL_HUMAN_NAME
    game_index: int = 0
    game_seed: int | None = None
    start_clock: int = 0
    max_half_rounds: int | None = None
    sequence_start: int = 0

    phase: Phase = field(default=Phase.RULES, init=False)
    sequence: int = field(default=0, init=False)
    half_rounds: int = field(default=0, init=False)
    last_record: HalfRoundRecord | None = field(default=None, init=False)
    public_history: list[dict[str, object]] = field(default_factory=list, init=False)
    _finished: bool = field(default=False, init=False)

    def __post_init__(self) -> None:
        players = (self.game.player1, self.game.player2)
        if self.hal is self.human or not any(player is self.hal for player in players):
            raise ValueError("hal and human must be distinct players owned by the game")
        if not any(player is self.human for player in players):
            raise ValueError("hal and human must be distinct players owned by the game")
        if self.hal.name != CANONICAL_HAL_NAME or self.human.name != CANONICAL_HUMAN_NAME:
            raise ValueError(
                "Arena engine players must retain the canonical Hal and Baku identities"
            )
        self.human_display_name = validate_human_display_name(self.human_display_name)
        if isinstance(self.sequence_start, bool) or not isinstance(self.sequence_start, int):
            raise TypeError("sequence_start must be an integer")
        if self.sequence_start < 0:
            raise ValueError("sequence_start must be nonnegative")
        self.sequence = self.sequence_start

    def display_name(self, player: Player) -> str:
        """Return the presentation label for a game-owned player."""

        if player is self.human:
            return self.human_display_name
        if player is self.hal:
            return self.hal.name
        raise ValueError("player is not owned by this session")

    def display_canonical_name(self, name: str | None) -> str | None:
        """Map a canonical engine name to its public presentation label."""

        if name is None:
            return None
        return self.human_display_name if name == self.human.name else name

    # ------------------------------------------------------------------
    # Pure queries
    # ------------------------------------------------------------------
    def roles(self) -> tuple[Player, Player]:
        """The (dropper, checker) pair for the current half."""

        return self.game.get_roles_for_half(self.game.current_half)

    def turn_duration(self) -> int:
        return self.game.get_turn_duration()

    def human_role(self) -> str:
        dropper, _ = self.roles()
        return "dropper" if dropper is self.human else "checker"

    def hal_role(self) -> str:
        return "checker" if self.human_role() == "dropper" else "dropper"

    def legal_actions(self) -> tuple[int, ...]:
        """The human's legal seconds. Only the engine decides legality."""

        return legal_seconds(self.human.name, self.human_role(), self.turn_duration())

    def pre_decision_state(self) -> PublicDecisionState:
        return public_state_from_game(self.game, turn_duration=self.turn_duration())

    @property
    def stopped(self) -> bool:
        """True when the session ended on the half-round cap, not a win."""

        return (
            self.phase is Phase.GAME_OVER
            and not self.game.game_over
            and self._cap_reached()
        )

    @property
    def winner_name(self) -> str | None:
        return self.game.winner.name if self.game.winner is not None else None

    def _cap_reached(self) -> bool:
        return self.max_half_rounds is not None and self.half_rounds >= self.max_half_rounds

    def _terminal(self) -> bool:
        return self.game.game_over or self._cap_reached()

    @property
    def terminal(self) -> bool:
        """Whether the next acknowledgement ends this session."""

        return self._terminal()

    # ------------------------------------------------------------------
    # Transitions
    # ------------------------------------------------------------------
    def begin(self) -> None:
        """Leave the rules screen and start play."""

        if self.phase is not Phase.RULES:
            raise SessionPhaseError(f"begin() requires {Phase.RULES}, in {self.phase}")
        self.phase = Phase.GAME_OVER if self._terminal() else Phase.AWAITING_ACTION
        self.sequence += 1

    def submit(self, second: int) -> HalfRoundRecord:
        """Accept the human's second, resolve the half-round, return the record.

        Hal's action is chosen here rather than before, so that it cannot be
        observed while the human is still deciding.
        """

        if self.phase is not Phase.AWAITING_ACTION:
            raise SessionPhaseError(
                f"submit() requires {Phase.AWAITING_ACTION}, in {self.phase}"
            )

        turn_duration = self.turn_duration()
        human_role = self.human_role()
        lifecycle_state = public_state_from_game(self.game, turn_duration=turn_duration)
        public_state = {
            "clock_seconds": float(self.game.game_clock),
            "clock_display": self.game.format_game_clock(),
            "round": int(self.game.round_num + 1),
            "half": int(self.game.current_half),
            "turn_duration": int(turn_duration),
            "players": [
                {
                    **public_player_state(player),
                    "name": self.display_name(player),
                }
                for player in (self.game.player1, self.game.player2)
            ],
        }

        # The human's action is validated before Hal is consulted, so a rejected
        # second costs nothing and reveals nothing.
        if isinstance(second, bool) or not isinstance(second, int):
            raise ValueError(
                f"Illegal action second={second!r}; seconds must be integers"
            )
        validate_action(
            second,
            actor=self.human.name,
            role=human_role,
            turn_duration=turn_duration,
        )
        hal_second = self.hal_agent.choose_action(self.game, self.hal_role(), turn_duration)
        if isinstance(hal_second, bool) or not isinstance(hal_second, int):
            raise ValueError(
                f"Illegal Hal action second={hal_second!r}; seconds must be integers"
            )
        validate_action(
            hal_second,
            actor=self.hal.name,
            role=self.hal_role(),
            turn_duration=turn_duration,
        )

        if human_role == "dropper":
            drop, check = int(second), int(hal_second)
        else:
            drop, check = int(hal_second), int(second)

        record = self.game.play_half_round(drop, check)
        observe_provider(
            getattr(self.hal_agent, "provider", None),
            PublicHalfRound(
                game_index=self.game_index,
                half_round_index=self.half_rounds,
                pre_decision_state=lifecycle_state,
                dropper_name=record.dropper,
                checker_name=record.checker,
                drop_time=int(record.drop_time),
                check_time=int(record.check_time),
                outcome=record.result.value,
                game_over=bool(self.game.game_over),
                winner_name=self.winner_name,
            ),
        )
        self.public_history.append(
            {
                "public_state_before": public_state,
                "dropper": self.display_canonical_name(record.dropper),
                "checker": self.display_canonical_name(record.checker),
                "drop_second": int(record.drop_time),
                "check_second": int(record.check_time),
                "result": record.result.value,
                "squandered_seconds": float(record.st_gained),
                "death_duration_seconds": float(record.death_duration),
                "survived": record.survived,
                "survival_probability": record.survival_probability,
            }
        )
        self.half_rounds += 1
        self.last_record = record
        self.phase = Phase.AWAITING_ACK
        self.sequence += 1
        return record

    def acknowledge(self) -> None:
        """Dismiss the reveal and either continue or end the game."""

        if self.phase is not Phase.AWAITING_ACK:
            raise SessionPhaseError(
                f"acknowledge() requires {Phase.AWAITING_ACK}, in {self.phase}"
            )
        self.phase = Phase.GAME_OVER if self._terminal() else Phase.AWAITING_ACTION
        self.sequence += 1

    def finish(self) -> dict[str, object]:
        """Notify the provider once and return this game's public transcript."""

        if self.phase is not Phase.GAME_OVER:
            raise SessionPhaseError(
                f"finish() requires {Phase.GAME_OVER}, in {self.phase}"
            )
        if not self._finished:
            end_provider_game(
                getattr(self.hal_agent, "provider", None),
                PublicGameOutcome(
                    game_index=self.game_index,
                    winner_name=self.winner_name,
                    half_rounds=self.half_rounds,
                ),
            )
            self._finished = True
        return {
            "game_index": self.game_index,
            "seed": self.game_seed,
            "start_clock": self.start_clock,
            "winner": self.display_canonical_name(self.winner_name),
            "stopped": self.stopped,
            "half_rounds": self.half_rounds,
            "public_history": self.public_history,
        }
