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


class Phase(str, Enum):
    """Where a session is waiting."""

    RULES = "rules"
    AWAITING_ACTION = "awaiting_action"
    AWAITING_ACK = "awaiting_ack"
    GAME_OVER = "game_over"


class SessionPhaseError(RuntimeError):
    """A transition was requested from a phase that does not allow it."""


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
    game_index: int = 0
    game_seed: int | None = None
    start_clock: int = 0
    max_half_rounds: int | None = None

    phase: Phase = field(default=Phase.RULES, init=False)
    sequence: int = field(default=0, init=False)
    half_rounds: int = field(default=0, init=False)
    last_record: HalfRoundRecord | None = field(default=None, init=False)
    public_history: list[dict[str, object]] = field(default_factory=list, init=False)
    _finished: bool = field(default=False, init=False)

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

        return not self.game.game_over

    @property
    def winner_name(self) -> str | None:
        return self.game.winner.name if self.game.winner is not None else None

    def _cap_reached(self) -> bool:
        return self.max_half_rounds is not None and self.half_rounds >= self.max_half_rounds

    def _terminal(self) -> bool:
        return self.game.game_over or self._cap_reached()

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
                public_player_state(self.game.player1),
                public_player_state(self.game.player2),
            ],
        }

        # The human's action is validated before Hal is consulted, so a rejected
        # second costs nothing and reveals nothing.
        validate_action(
            int(second),
            actor=self.human.name,
            role=human_role,
            turn_duration=turn_duration,
        )
        hal_second = self.hal_agent.choose_action(self.game, self.hal_role(), turn_duration)
        validate_action(
            int(hal_second),
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
                "dropper": record.dropper,
                "checker": record.checker,
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
            "winner": self.winner_name,
            "stopped": self.stopped,
            "half_rounds": self.half_rounds,
            "public_history": self.public_history,
        }
