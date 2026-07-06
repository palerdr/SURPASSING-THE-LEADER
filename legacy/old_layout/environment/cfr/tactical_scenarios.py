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

from src.Constants import CYLINDER_MAX, FAILED_CHECK_PENALTY, PHYSICALITY_BAKU, PHYSICALITY_HAL
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


# ── Phase F: pinned-tablebase expansion ───────────────────────────────────
# All scenarios below are forced-terminal extensions of the
# ``forced_*_overflow_death`` template: cylinder=299 on the checker side
# means every joint action drives the LP to a Hal-win (half=1, baku
# checker) or Baku-win (half=2, hal checker) within one half-round.
# Each variant changes one or more axes (clock, ttd, deaths, fatigue,
# leap-window proximity) so feature vectors are distinct from the original
# two pins while preserving the all-terminal structure that makes
# ``unresolved_probability == 0`` provable.


# Category 1: Forced overflow at varied clock positions ───────────────────


def forced_baku_overflow_mid_clock() -> TacticalScenario:
    """Forced Baku overflow at mid-game clock — value-invariant w.r.t. clock."""
    game = _base_game(clock=1800.0, current_half=1)
    game.player2.cylinder = 299.0
    return TacticalScenario(
        name="forced_baku_overflow_mid_clock",
        game=game,
        config=ExactSearchConfig(),
        half_round_horizon=1,
        expected_note="Same all-terminal structure as the opening pin; clock=1800 differentiates features.",
        expected_value=1.0,
        tags=("near_overflow", "forced_terminal", "hal_win", "clock_variant"),
    )


def forced_hal_overflow_mid_clock() -> TacticalScenario:
    """Forced Hal overflow at mid-game clock."""
    game = _base_game(clock=1800.0, current_half=2)
    game.player1.cylinder = 299.0
    return TacticalScenario(
        name="forced_hal_overflow_mid_clock",
        game=game,
        config=ExactSearchConfig(),
        half_round_horizon=1,
        expected_note="Symmetric counterpart at clock=1800.",
        expected_value=-1.0,
        tags=("near_overflow", "forced_terminal", "baku_win", "clock_variant"),
    )


def forced_baku_overflow_pre_leap() -> TacticalScenario:
    """Forced Baku overflow just before the leap-second window."""
    game = _base_game(clock=3450.0, current_half=1)
    game.player2.cylinder = 299.0
    return TacticalScenario(
        name="forced_baku_overflow_pre_leap",
        game=game,
        config=ExactSearchConfig(),
        half_round_horizon=1,
        expected_note="Pre-leap clock pin (3450, rounds_until_leap_window flag active).",
        expected_value=1.0,
        tags=("near_overflow", "forced_terminal", "hal_win", "pre_leap"),
    )


def forced_hal_overflow_pre_leap() -> TacticalScenario:
    """Forced Hal overflow just before the leap-second window."""
    game = _base_game(clock=3450.0, current_half=2)
    game.player1.cylinder = 299.0
    return TacticalScenario(
        name="forced_hal_overflow_pre_leap",
        game=game,
        config=ExactSearchConfig(),
        half_round_horizon=1,
        expected_note="Pre-leap clock pin, Baku-win counterpart.",
        expected_value=-1.0,
        tags=("near_overflow", "forced_terminal", "baku_win", "pre_leap"),
    )


# Category 2: Leap-window and post-leap pins ──────────────────────────────


def forced_baku_overflow_leap_window_open() -> TacticalScenario:
    """Forced Baku overflow at the leap-window opening (clock=3540)."""
    game = _base_game(clock=3540.0, current_half=1)
    game.player2.cylinder = 299.0
    return TacticalScenario(
        name="forced_baku_overflow_leap_window_open",
        game=game,
        config=ExactSearchConfig(),
        half_round_horizon=1,
        expected_note="Leap-window open (clock=3540); is_leap_second_turn flag active.",
        expected_value=1.0,
        tags=("near_overflow", "forced_terminal", "hal_win", "leap_window"),
    )


def forced_hal_overflow_leap_window_open() -> TacticalScenario:
    """Forced Hal overflow at the leap-window opening, half=2."""
    game = _base_game(clock=3540.0, current_half=2)
    game.player1.cylinder = 299.0
    return TacticalScenario(
        name="forced_hal_overflow_leap_window_open",
        game=game,
        config=ExactSearchConfig(),
        half_round_horizon=1,
        expected_note="Leap-window open, Baku-win counterpart.",
        expected_value=-1.0,
        tags=("near_overflow", "forced_terminal", "baku_win", "leap_window"),
    )


def forced_baku_overflow_leap_window_late() -> TacticalScenario:
    """Forced Baku overflow inside the leap-second window (clock=3580)."""
    game = _base_game(clock=3580.0, current_half=1)
    game.player2.cylinder = 299.0
    return TacticalScenario(
        name="forced_baku_overflow_leap_window_late",
        game=game,
        config=ExactSearchConfig(),
        half_round_horizon=1,
        expected_note="Deep inside leap window (clock=3580).",
        expected_value=1.0,
        tags=("near_overflow", "forced_terminal", "hal_win", "leap_window"),
    )


def forced_baku_overflow_post_leap() -> TacticalScenario:
    """Forced Baku overflow just after the leap-second window."""
    game = _base_game(clock=3600.0, current_half=1)
    game.player2.cylinder = 299.0
    return TacticalScenario(
        name="forced_baku_overflow_post_leap",
        game=game,
        config=ExactSearchConfig(),
        half_round_horizon=1,
        expected_note="Post-leap clock; leap flags inactive.",
        expected_value=1.0,
        tags=("near_overflow", "forced_terminal", "hal_win", "post_leap"),
    )


# Category 3: Fatigue and TTD pressure ────────────────────────────────────


def forced_baku_overflow_fatigued_referee() -> TacticalScenario:
    """Forced Baku overflow with a fatigued referee (cprs_performed=10)."""
    game = _base_game(clock=720.0, current_half=1)
    game.player2.cylinder = 299.0
    game.referee.cprs_performed = 10
    return TacticalScenario(
        name="forced_baku_overflow_fatigued_referee",
        game=game,
        config=ExactSearchConfig(),
        half_round_horizon=1,
        expected_note="cprs=10 raises the cpr feature; all-terminal structure unchanged.",
        expected_value=1.0,
        tags=("near_overflow", "forced_terminal", "hal_win", "cpr_fatigue"),
    )


def forced_hal_overflow_fatigued_referee() -> TacticalScenario:
    """Forced Hal overflow with a fatigued referee (cprs_performed=10)."""
    game = _base_game(clock=720.0, current_half=2)
    game.player1.cylinder = 299.0
    game.referee.cprs_performed = 10
    return TacticalScenario(
        name="forced_hal_overflow_fatigued_referee",
        game=game,
        config=ExactSearchConfig(),
        half_round_horizon=1,
        expected_note="Baku-win counterpart at cprs=10.",
        expected_value=-1.0,
        tags=("near_overflow", "forced_terminal", "baku_win", "cpr_fatigue"),
    )


def forced_baku_overflow_high_ttd() -> TacticalScenario:
    """Forced Baku overflow with elevated baku_ttd=240."""
    game = _base_game(clock=720.0, current_half=1)
    game.player2.cylinder = 299.0
    game.player2.ttd = 240.0
    return TacticalScenario(
        name="forced_baku_overflow_high_ttd",
        game=game,
        config=ExactSearchConfig(),
        half_round_horizon=1,
        expected_note="baku_ttd=240 raises the cardiac modifier on the death branch; outcome still pinned.",
        expected_value=1.0,
        tags=("near_overflow", "forced_terminal", "hal_win", "ttd_pressure"),
    )


# Category 4: Asymmetric death-count pins ─────────────────────────────────


def forced_baku_overflow_with_baku_deaths() -> TacticalScenario:
    """Forced Baku overflow when Baku already carries a prior death."""
    game = _base_game(clock=720.0, current_half=1)
    game.player2.cylinder = 299.0
    game.player2.deaths = 1
    return TacticalScenario(
        name="forced_baku_overflow_with_baku_deaths",
        game=game,
        config=ExactSearchConfig(),
        half_round_horizon=1,
        expected_note="Asymmetric: baku_deaths=1; feature dim 5 (baku_deaths) is 0.25 here.",
        expected_value=1.0,
        tags=("near_overflow", "forced_terminal", "hal_win", "asymmetric_deaths"),
    )


def forced_hal_overflow_with_hal_deaths() -> TacticalScenario:
    """Forced Hal overflow when Hal already carries a prior death."""
    game = _base_game(clock=720.0, current_half=2)
    game.player1.cylinder = 299.0
    game.player1.deaths = 1
    return TacticalScenario(
        name="forced_hal_overflow_with_hal_deaths",
        game=game,
        config=ExactSearchConfig(),
        half_round_horizon=1,
        expected_note="Asymmetric: hal_deaths=1; feature dim 4 is 0.25.",
        expected_value=-1.0,
        tags=("near_overflow", "forced_terminal", "baku_win", "asymmetric_deaths"),
    )


def forced_baku_overflow_with_hal_deaths() -> TacticalScenario:
    """Forced Baku overflow when Hal (not Baku) carries a prior death.

    Verifies that opponent's prior deaths do not invert the equilibrium —
    Baku is the one about to die, regardless of Hal's death history.
    """
    game = _base_game(clock=720.0, current_half=1)
    game.player2.cylinder = 299.0
    game.player1.deaths = 1
    return TacticalScenario(
        name="forced_baku_overflow_with_hal_deaths",
        game=game,
        config=ExactSearchConfig(),
        half_round_horizon=1,
        expected_note="Hal-deaths set, Baku still about-to-overflow; pin tests that opponent deaths don't flip the outcome.",
        expected_value=1.0,
        tags=("near_overflow", "forced_terminal", "hal_win", "asymmetric_deaths"),
    )


def forced_hal_overflow_with_baku_deaths() -> TacticalScenario:
    """Forced Hal overflow when Baku (not Hal) carries a prior death."""
    game = _base_game(clock=720.0, current_half=2)
    game.player1.cylinder = 299.0
    game.player2.deaths = 1
    return TacticalScenario(
        name="forced_hal_overflow_with_baku_deaths",
        game=game,
        config=ExactSearchConfig(),
        half_round_horizon=1,
        expected_note="Baku-deaths set, Hal still about-to-overflow; symmetric counterpart.",
        expected_value=-1.0,
        tags=("near_overflow", "forced_terminal", "baku_win", "asymmetric_deaths"),
    )


# Category 5: Both-players-near-overflow pins ─────────────────────────────


def both_overflow_baku_dies_first() -> TacticalScenario:
    """Both players at cylinder=299, half=1 — Baku (checker) dies first."""
    game = _base_game(clock=720.0, current_half=1)
    game.player1.cylinder = 299.0
    game.player2.cylinder = 299.0
    return TacticalScenario(
        name="both_overflow_baku_dies_first",
        game=game,
        config=ExactSearchConfig(),
        half_round_horizon=1,
        expected_note="Symmetric cylinders, role determines outcome: half=1 → Baku checker → Baku dies first.",
        expected_value=1.0,
        tags=("near_overflow", "forced_terminal", "hal_win", "double_overflow"),
    )


def both_overflow_hal_dies_first() -> TacticalScenario:
    """Both players at cylinder=299, half=2 — Hal (checker) dies first."""
    game = _base_game(clock=720.0, current_half=2)
    game.player1.cylinder = 299.0
    game.player2.cylinder = 299.0
    return TacticalScenario(
        name="both_overflow_hal_dies_first",
        game=game,
        config=ExactSearchConfig(),
        half_round_horizon=1,
        expected_note="Symmetric cylinders, half=2 → Hal checker → Hal dies first.",
        expected_value=-1.0,
        tags=("near_overflow", "forced_terminal", "baku_win", "double_overflow"),
    )


# ── Phase F-2: interior-valued pins (survivable forced fail) ───────────────
# The Phase-8/F pins above are ALL ±1.0 forced-overflow terminals, so the
# pinned ruler only constrains the BOUNDARY of [-1, 1]. The pins below constrain
# the INTERIOR.
#
# Construction (leap-window forced fail on Hal):
#   In the leap-second window Baku (dropper) may legally drop at 61 while Hal
#   (checker) is capped at 60 (environment/legal_actions.py: Hal may never check
#   61). So Baku drop=61 forces Hal to fail on EVERY check — a single death of
#   duration hal_cylinder + FAILED_CHECK_PENALTY. Chosen < CYLINDER_MAX so the
#   referee roll is survivable (0 < p < 1):
#       die  (prob 1-p): Hal permanently dies                      → Baku win (-1)
#       live (prob p)  : Hal revived → next round Baku is the checker at cyl=299,
#                        a forced overflow → Baku permanent death → Hal win (+1)
#   Both branches terminate within 2 half-rounds (drop=61 is strictly dominant
#   for Baku, since any drop ≤ 60 lets Hal check safely and reach the +1
#   continuation), so unresolved_probability == 0 and the exact value is 2p-1.
#
# The expected_value is derived from the engine's OWN survival formula
# (Referee.compute_survival_probability) — never read back from
# solve_exact_finite_horizon — so verify_pinned_value stays a genuine
# cross-check of the solver rather than a tautology.
#
# This forced-fail lever is leap-window- and Hal-checker-only: outside the leap
# window no player can force the opponent to fail, and Hal can never drop 61, so
# there is deliberately no Baku-checker mirror. That asymmetry IS the structural
# point of the leap second.

_INTERIOR_FAIL_HAL_CYLINDER: float = 120.0


def _survivable_fail_value(hal_cylinder: float, cprs_performed: int) -> float:
    """Exact Hal-perspective value (2p-1) of a leap-window forced-fail-on-Hal pin.

    Derived independently of the solver: a forced fail injects
    ``hal_cylinder + FAILED_CHECK_PENALTY`` (< CYLINDER_MAX by construction),
    survived with probability ``p`` from the engine's survival curve; the two
    terminal branches are +1 (survive → Baku forced overflow next round) and -1
    (die). Hence 2p-1.
    """
    death_duration = hal_cylinder + FAILED_CHECK_PENALTY
    if death_duration >= CYLINDER_MAX:
        raise ValueError(
            f"hal_cylinder={hal_cylinder} gives death_duration {death_duration} >= "
            f"{CYLINDER_MAX}: the forced fail would be unsurvivable (p=0) and the pin "
            "would collapse to the ±1 boundary, defeating its interior purpose."
        )
    referee = Referee(cprs_performed=cprs_performed)
    dying_hal = Player(name="Hal", physicality=PHYSICALITY_HAL)
    p = referee.compute_survival_probability(dying_hal, death_duration=death_duration)
    return 2.0 * p - 1.0


def _interior_fail_game(hal_cylinder: float, cprs_performed: int) -> Game:
    """Leap-window half-2 state: Baku dropper (may use 61), Hal checker (capped 60)."""
    game = _base_game(clock=3540.0, current_half=2)
    game.player1.cylinder = hal_cylinder   # Hal is the checker this half-round
    game.player2.cylinder = 299.0          # Baku is the checker NEXT round → forced overflow
    game.referee.cprs_performed = cprs_performed
    return game


def forced_hal_fail_survivable_fresh() -> TacticalScenario:
    """Interior pin: survivable forced fail on Hal, fresh referee → value 2p-1 > 0."""
    game = _interior_fail_game(_INTERIOR_FAIL_HAL_CYLINDER, cprs_performed=0)
    return TacticalScenario(
        name="forced_hal_fail_survivable_fresh",
        game=game,
        config=ExactSearchConfig(),
        half_round_horizon=2,
        expected_note=(
            "Leap-window Baku drop=61 forces Hal to fail every check (death_duration "
            "180, survivable). Survive → Baku@299 overflow next round (Hal win); die → "
            "Baku win. Exact interior value 2p-1 at fresh referee (p=0.784 → +0.568)."
        ),
        expected_value=_survivable_fail_value(_INTERIOR_FAIL_HAL_CYLINDER, cprs_performed=0),
        tags=("leap_window", "forced_fail", "interior_value", "survivable_death",
              "hal_checker", "cpr_fatigue_pair"),
    )


def forced_hal_fail_survivable_fatigued() -> TacticalScenario:
    """Interior pin: survivable forced fail on Hal, fatigued referee (cprs=10).

    Monotone counterpart to ``forced_hal_fail_survivable_fresh``: more referee
    fatigue lowers Hal's revival probability on the forced fail, lowering the
    interior value (+0.568 fresh → -0.3728 fatigued).
    """
    game = _interior_fail_game(_INTERIOR_FAIL_HAL_CYLINDER, cprs_performed=10)
    return TacticalScenario(
        name="forced_hal_fail_survivable_fatigued",
        game=game,
        config=ExactSearchConfig(),
        half_round_horizon=2,
        expected_note=(
            "Fatigued counterpart (cprs=10 → referee floor 0.4): p=0.3136 → value "
            "-0.3728. Pairs with the fresh pin for the monotone fatigue invariant."
        ),
        expected_value=_survivable_fail_value(_INTERIOR_FAIL_HAL_CYLINDER, cprs_performed=10),
        tags=("leap_window", "forced_fail", "interior_value", "survivable_death",
              "hal_checker", "cpr_fatigue_pair"),
    )


def forced_hal_fail_survivable_deep() -> TacticalScenario:
    """Interior pin on the death-duration axis: Hal cyl=180 → death_duration 240.

    p=0.488 → value -0.024, a near-balanced anchor at the center of [-1, 1] — the
    region the all-±1 Phase-F ruler never constrained.
    """
    game = _interior_fail_game(180.0, cprs_performed=0)
    return TacticalScenario(
        name="forced_hal_fail_survivable_deep",
        game=game,
        config=ExactSearchConfig(),
        half_round_horizon=2,
        expected_note=(
            "Death-duration axis: hal_cylinder=180 → death_duration 240 → p=0.488 → "
            "value -0.024. Near-zero interior anchor at the center of the value range."
        ),
        expected_value=_survivable_fail_value(180.0, cprs_performed=0),
        tags=("leap_window", "forced_fail", "interior_value", "survivable_death",
              "hal_checker", "death_duration_axis"),
    )
