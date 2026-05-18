"""Small exact tactical positions for CFR/search audits.

These are tablebase-style fixtures, not training reward shortcuts. They exist
to provide known exact targets for diagnostics, MCTS, MCCFR, and value nets.

Scenarios with a non-None ``expected_value`` are pinned tablebase entries:
the value is determinable by construction (forced terminal sequences) and
must not drift. Scenarios with ``expected_value`` left as None are paired
or action-comparison fixtures whose tests assert relational invariants
(e.g. monotonicity, action dominance).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from src.Constants import PHYSICALITY_BAKU, PHYSICALITY_HAL
from src.Game import Game
from src.Player import Player
from src.Referee import Referee

from .exact import ExactSearchConfig


@dataclass(frozen=True)
class TacticalScenario:
    name: str
    game: Game
    config: ExactSearchConfig
    half_round_horizon: int
    expected_note: str
    expected_value: float | None = None
    tags: tuple[str, ...] = field(default_factory=tuple)
    holdout: bool = False


def _base_game(*, clock: float = 720.0, current_half: int = 1) -> Game:
    hal = Player(name="Hal", physicality=PHYSICALITY_HAL)
    baku = Player(name="Baku", physicality=PHYSICALITY_BAKU)
    game = Game(player1=hal, player2=baku, referee=Referee(), first_dropper=hal)
    game.seed(0)
    game.game_clock = clock
    game.current_half = current_half
    return game


def forced_baku_overflow_death() -> TacticalScenario:
    """Hal drops, Baku checks with cylinder already at 299.

    Every legal exact action either succeeds with at least ST=1 or fails and
    injects the full cylinder. Both paths are terminal for Baku because
    death_duration reaches 300 seconds.
    """
    game = _base_game()
    game.player2.cylinder = 299.0
    return TacticalScenario(
        name="forced_baku_overflow_death",
        game=game,
        config=ExactSearchConfig(),
        half_round_horizon=1,
        expected_note="All joint actions are terminal Hal wins.",
        expected_value=1.0,
        tags=("near_overflow", "forced_terminal", "hal_win"),
    )


def forced_hal_overflow_death() -> TacticalScenario:
    """Baku drops on half 2, Hal checks with cylinder at 299.

    Symmetric of ``forced_baku_overflow_death``: every joint action is
    terminal Baku-win because Hal's death_duration hits 300 regardless of
    success/fail.
    """
    game = _base_game(current_half=2)
    game.player1.cylinder = 299.0
    return TacticalScenario(
        name="forced_hal_overflow_death",
        game=game,
        config=ExactSearchConfig(),
        half_round_horizon=1,
        expected_note="All joint actions are terminal Baku wins.",
        expected_value=-1.0,
        tags=("near_overflow", "forced_terminal", "baku_win"),
    )


def safe_budget_pressure_at_cylinder_241() -> TacticalScenario:
    """Baku checker at cyl=241: drop=1 + check=60 forces terminal overflow.

    Paired with ``safe_budget_pressure_at_cylinder_240``. At cyl=241 a single
    cell of the joint matrix (drop=1, check=60) is terminal Hal-win because
    ST=59 pushes cylinder to exactly 300; failed checks are also terminal.
    Hal's mixed-strategy value is strictly positive.
    """
    game = _base_game()
    game.player2.cylinder = 241.0
    return TacticalScenario(
        name="safe_budget_pressure_at_cylinder_241",
        game=game,
        config=ExactSearchConfig(),
        half_round_horizon=1,
        expected_note="Hal can force a terminal cell at (drop=1, check=60); strictly positive equilibrium value.",
        expected_value=None,
        tags=("safe_budget", "threshold_pair", "checker_pressure"),
    )


def safe_budget_pressure_at_cylinder_240() -> TacticalScenario:
    """Baku checker at cyl=240: check=60 always succeeds without overflow.

    Paired with ``safe_budget_pressure_at_cylinder_241``. With cyl=240 the
    maximum reachable post-success cylinder is 240+59=299 < 300, so check=60
    is a guaranteed-survive strategy and Hal cannot force a terminal cell.
    """
    game = _base_game()
    game.player2.cylinder = 240.0
    return TacticalScenario(
        name="safe_budget_pressure_at_cylinder_240",
        game=game,
        config=ExactSearchConfig(),
        half_round_horizon=1,
        expected_note="check=60 is a safe pure strategy for Baku; equilibrium value is 0 unresolved.",
        expected_value=None,
        tags=("safe_budget", "threshold_pair", "checker_pressure"),
    )


def cpr_degradation_fresh_referee() -> TacticalScenario:
    """Baku checker, cylinder=180, fresh referee (cprs_performed=0).

    Paired with ``cpr_degradation_fatigued_referee``. Used with a forced-fail
    joint action (drop late, check early) so the survival branch fires; the
    paired scenarios assert that fatigue lowers Baku's survival probability
    and therefore raises Hal's value on the fail branch.
    """
    game = _base_game()
    game.player2.cylinder = 180.0
    return TacticalScenario(
        name="cpr_degradation_fresh_referee",
        game=game,
        config=ExactSearchConfig(),
        half_round_horizon=1,
        expected_note="Reference state for the CPR-fatigue pair; cprs_performed=0.",
        expected_value=None,
        tags=("cpr_fatigue", "monotonic_pair", "survival_branch"),
    )


def cpr_degradation_fatigued_referee() -> TacticalScenario:
    """Baku checker, cylinder=180, fatigued referee (cprs_performed=10).

    Paired with ``cpr_degradation_fresh_referee``. cprs_performed=10 sits at
    the REFEREE_FLOOR-bound regime; Baku's survival probability on a forced
    failed check is strictly lower than the fresh-referee pair, so the
    branch-weighted value tilts further toward Hal.
    """
    game = _base_game()
    game.player2.cylinder = 180.0
    game.referee.cprs_performed = 10
    return TacticalScenario(
        name="cpr_degradation_fatigued_referee",
        game=game,
        config=ExactSearchConfig(),
        half_round_horizon=1,
        expected_note="Fatigue counterpart; cprs_performed=10 sits at REFEREE_FLOOR.",
        expected_value=None,
        tags=("cpr_fatigue", "monotonic_pair", "survival_branch"),
    )


def baku_dropper_leap_window_alignment() -> TacticalScenario:
    """Baku is dropper and Hal is checker at the leap-window half-round."""
    game = _base_game(clock=3540.0, current_half=2)
    return TacticalScenario(
        name="baku_dropper_leap_window_alignment",
        game=game,
        config=ExactSearchConfig(),
        half_round_horizon=2,
        expected_note="Baku can legally drop at 61 while Hal checker remains capped at 60.",
        expected_value=None,
        tags=("forced_leap", "role_alignment", "baku_dropper_61", "holdout"),
        holdout=True,
    )


def hal_dropper_leap_window_asymmetry() -> TacticalScenario:
    """Hal is dropper at the leap-window half-round; Hal remains capped at 60."""
    game = _base_game(clock=3540.0, current_half=1)
    return TacticalScenario(
        name="hal_dropper_leap_window_asymmetry",
        game=game,
        config=ExactSearchConfig(),
        half_round_horizon=2,
        expected_note="Wrong-role leap alignment: Hal dropper cannot use second 61.",
        expected_value=None,
        tags=("forced_leap", "role_alignment", "hal_dropper", "holdout"),
        holdout=True,
    )


def near_overflow_marginal_baku_294() -> TacticalScenario:
    """Baku checker at cylinder=294: one-second spreads become decisive."""
    game = _base_game()
    game.player2.cylinder = 294.0
    return TacticalScenario(
        name="near_overflow_marginal_baku_294",
        game=game,
        config=ExactSearchConfig(),
        half_round_horizon=2,
        expected_note="Marginal near-overflow bucket for audit diagnostics.",
        expected_value=None,
        tags=("near_overflow", "marginal", "holdout"),
        holdout=True,
    )


def death_trade_double_pressure() -> TacticalScenario:
    """Both players carry high cylinder and prior-death pressure."""
    game = _base_game(clock=2580.0, current_half=1)
    game.player1.cylinder = 220.0
    game.player2.cylinder = 235.0
    game.player1.ttd = 180.0
    game.player2.ttd = 240.0
    game.player1.deaths = 1
    game.player2.deaths = 1
    game.referee.cprs_performed = 2
    return TacticalScenario(
        name="death_trade_double_pressure",
        game=game,
        config=ExactSearchConfig(),
        half_round_horizon=3,
        expected_note="Death-trade pressure state: both sides near costly failure branches.",
        expected_value=None,
        tags=("death_trade", "cpr_fatigue", "holdout"),
        holdout=True,
    )


def role_alignment_active_lsr_runway() -> TacticalScenario:
    """Active LSR label with a short pre-leap runway."""
    game = _base_game(clock=3420.0, current_half=2)
    return TacticalScenario(
        name="role_alignment_active_lsr_runway",
        game=game,
        config=ExactSearchConfig(),
        half_round_horizon=3,
        expected_note="Variation-2 role-alignment audit state before the leap window.",
        expected_value=None,
        tags=("role_alignment", "active_lsr", "holdout"),
        holdout=True,
    )


def role_alignment_variation4_post_engineering() -> TacticalScenario:
    """Variation-4 comparison state for post-engineering parity checks."""
    game = _base_game(clock=3300.0, current_half=2)
    return TacticalScenario(
        name="role_alignment_variation4_post_engineering",
        game=game,
        config=ExactSearchConfig(),
        half_round_horizon=3,
        expected_note="Variation-4 role-alignment comparison bucket.",
        expected_value=None,
        tags=("role_alignment", "variation4", "holdout"),
        holdout=True,
    )
