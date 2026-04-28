"""Exact-second transition expansion for rigorous CFR/search."""

from __future__ import annotations

from dataclasses import dataclass

from src.Constants import TURN_DURATION_NORMAL
from src.Game import Game, HalfRoundRecord

from environment.legal_actions import legal_max_second

from .exact_state import ExactPublicState, exact_public_state
from .utility import terminal_value


@dataclass(frozen=True)
class ExactSearchConfig:
    """Knowledge/legality switches for exact action expansion."""

    hal_leap_deduced: bool = False
    hal_memory_impaired: bool = False
    perspective_name: str = "Hal"


@dataclass(frozen=True)
class ExactJointAction:
    drop_time: int
    check_time: int


@dataclass(frozen=True)
class ExactTransition:
    action: ExactJointAction
    probability: float
    state: ExactPublicState
    terminal_value: float | None
    record: HalfRoundRecord


class ExactGameSnapshot:
    """Copy/restore mutable Game state for exact tree traversal."""

    __slots__ = (
        "p1_cylinder", "p1_ttd", "p1_deaths", "p1_alive", "p1_dh_len",
        "p2_cylinder", "p2_ttd", "p2_deaths", "p2_alive", "p2_dh_len",
        "cprs", "clock", "current_half", "round_num", "game_over",
        "winner", "loser", "hist_len",
    )

    def __init__(self, game: Game):
        self.p1_cylinder = game.player1.cylinder
        self.p1_ttd = game.player1.ttd
        self.p1_deaths = game.player1.deaths
        self.p1_alive = game.player1.alive
        self.p1_dh_len = len(game.player1.death_history)
        self.p2_cylinder = game.player2.cylinder
        self.p2_ttd = game.player2.ttd
        self.p2_deaths = game.player2.deaths
        self.p2_alive = game.player2.alive
        self.p2_dh_len = len(game.player2.death_history)
        self.cprs = game.referee.cprs_performed
        self.clock = game.game_clock
        self.current_half = game.current_half
        self.round_num = game.round_num
        self.game_over = game.game_over
        self.winner = game.winner
        self.loser = game.loser
        self.hist_len = len(game.history)

    def restore(self, game: Game) -> None:
        game.player1.cylinder = self.p1_cylinder
        game.player1.ttd = self.p1_ttd
        game.player1.deaths = self.p1_deaths
        game.player1.alive = self.p1_alive
        del game.player1.death_history[self.p1_dh_len:]
        game.player2.cylinder = self.p2_cylinder
        game.player2.ttd = self.p2_ttd
        game.player2.deaths = self.p2_deaths
        game.player2.alive = self.p2_alive
        del game.player2.death_history[self.p2_dh_len:]
        game.referee.cprs_performed = self.cprs
        game.game_clock = self.clock
        game.current_half = self.current_half
        game.round_num = self.round_num
        game.game_over = self.game_over
        game.winner = self.winner
        game.loser = self.loser
        del game.history[self.hist_len:]


def legal_seconds_for_current_role(game: Game, actor_name: str, role: str, config: ExactSearchConfig) -> range:
    turn_duration = game.get_turn_duration()
    max_second = legal_max_second(
        actor_name,
        role,
        turn_duration,
        hal_leap_deduced=config.hal_leap_deduced,
        hal_memory_impaired=config.hal_memory_impaired,
    )
    if role == "checker":
        max_second = min(max_second, max(turn_duration, TURN_DURATION_NORMAL))
    return range(1, max_second + 1)


def enumerate_joint_actions(game: Game, config: ExactSearchConfig | None = None) -> list[ExactJointAction]:
    config = config or ExactSearchConfig()
    dropper, checker = game.get_roles_for_half(game.current_half)
    drop_seconds = legal_seconds_for_current_role(game, dropper.name, "dropper", config)
    check_seconds = legal_seconds_for_current_role(game, checker.name, "checker", config)
    return [ExactJointAction(d, c) for d in drop_seconds for c in check_seconds]


def expand_joint_action(
    game: Game,
    action: ExactJointAction,
    config: ExactSearchConfig | None = None,
) -> tuple[ExactTransition, ...]:
    """Expand a joint action into deterministic/no-death or survival chance branches.

    The input game is restored to its original state before this function returns.
    """
    config = config or ExactSearchConfig()
    snap = ExactGameSnapshot(game)
    probe = game.resolve_half_round(action.drop_time, action.check_time, survived_outcome=None)
    death_occurred = probe.survived is not None
    survival_probability = probe.survival_probability
    snap.restore(game)

    if not death_occurred:
        record = game.resolve_half_round(action.drop_time, action.check_time, survived_outcome=None)
        transition = ExactTransition(
            action=action,
            probability=1.0,
            state=exact_public_state(game),
            terminal_value=terminal_value(game, perspective_name=config.perspective_name),
            record=record,
        )
        snap.restore(game)
        return (transition,)

    assert survival_probability is not None
    branches: list[ExactTransition] = []
    for survived, probability in ((True, survival_probability), (False, 1.0 - survival_probability)):
        if probability <= 0.0:
            continue
        record = game.resolve_half_round(action.drop_time, action.check_time, survived_outcome=survived)
        branches.append(
            ExactTransition(
                action=action,
                probability=probability,
                state=exact_public_state(game),
                terminal_value=terminal_value(game, perspective_name=config.perspective_name),
                record=record,
            )
        )
        snap.restore(game)
    return tuple(branches)

