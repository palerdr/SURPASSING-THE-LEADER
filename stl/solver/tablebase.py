"""Pinned scenarios, tactical fixtures, and Tier A tablebase primitives."""


# ============================================================================
# Tactical scenarios
# ============================================================================

"""Small exact tactical positions for CFR/search audits.

These are tablebase-style fixtures, not training reward shortcuts. They exist
to provide known exact targets for diagnostics, MCTS, MCCFR, and value nets.

Scenarios with a non-None ``expected_value`` are pinned tablebase entries:
the value is determinable by construction (forced terminal sequences) and
must not drift. Scenarios with ``expected_value`` left as None are paired
or action-comparison fixtures whose tests assert relational invariants
(e.g. monotonicity, action dominance).
"""


from dataclasses import dataclass, field

from stl.engine.game import CYLINDER_MAX, FAILED_CHECK_PENALTY, PHYSICALITY_BAKU, PHYSICALITY_HAL
from stl.engine.game import Game
from stl.engine.game import Player
from stl.engine.game import Referee

from stl.solver.exact import ExactSearchConfig


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
    """Hal drops, Baku checks with cylinder already at 300.

    Every legal exact action either succeeds into an already-full cylinder or
    fails and injects the full cylinder. Both paths are terminal for Baku because
    death_duration reaches 300 seconds.
    """
    game = _base_game()
    game.player2.cylinder = 300.0
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
    """Baku drops on half 2, Hal checks with cylinder at 300.

    Symmetric of ``forced_baku_overflow_death``: every joint action is
    terminal Baku-win because Hal's death_duration hits 300 regardless of
    success/fail.
    """
    game = _base_game(current_half=2)
    game.player1.cylinder = 300.0
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
# ``forced_*_overflow_death`` template: cylinder=300 on the checker side
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
    game.player2.cylinder = 300.0
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
    game.player1.cylinder = 300.0
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
    game.player2.cylinder = 300.0
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
    game.player1.cylinder = 300.0
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
    game.player2.cylinder = 300.0
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
    game.player1.cylinder = 300.0
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
    game.player2.cylinder = 300.0
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
    game.player2.cylinder = 300.0
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
    game.player2.cylinder = 300.0
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
    game.player1.cylinder = 300.0
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
    game.player2.cylinder = 300.0
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
    game.player2.cylinder = 300.0
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
    game.player1.cylinder = 300.0
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
    game.player2.cylinder = 300.0
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
    game.player1.cylinder = 300.0
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
    """Both players at cylinder=300, half=1 — Baku (checker) dies first."""
    game = _base_game(clock=720.0, current_half=1)
    game.player1.cylinder = 300.0
    game.player2.cylinder = 300.0
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
    """Both players at cylinder=300, half=2 — Hal (checker) dies first."""
    game = _base_game(clock=720.0, current_half=2)
    game.player1.cylinder = 300.0
    game.player2.cylinder = 300.0
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
#       live (prob p)  : Hal revived -> next round Baku is the checker at cyl=300,
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
    game.player2.cylinder = 300.0          # Baku is the checker NEXT round -> forced overflow
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
            "180, survivable). Survive -> Baku@300 overflow next round (Hal win); die -> "
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


# ============================================================================
# Pinned scenario registry
# ============================================================================

"""Lazy registry of exact tablebase targets.

Each entry in ``REGISTRY`` is a factory that produces a fresh
``TacticalScenario``. Call ``solve_target(name)`` to materialise a scenario
and run ``solve_exact_finite_horizon`` against it. Scenarios with a non-None
``expected_value`` are pinned: the solver value is exact by construction
(forced terminal sequences) and any drift is a regression. Relational
scenarios (``expected_value`` None) ship in pairs; their tests assert
monotonicity, action dominance, or threshold inequalities rather than a
single fixed number.

This module is part of the rigorous CFR core: no shaping, no value-net
frontier, no curriculum or stage labels. Targets exist purely as supervised
labels for downstream MCCFR / MCTS / value-net training.
"""


import math
from collections.abc import Callable

from stl.solver.exact import ExactSolveResult, solve_exact_finite_horizon


ScenarioFactory = Callable[[], TacticalScenario]


REGISTRY: dict[str, ScenarioFactory] = {
    factory.__name__: factory
    for factory in (
        # Original pinned (Phase 8)
        forced_baku_overflow_death,
        forced_hal_overflow_death,
        # Original relational (Phase 8)
        safe_budget_pressure_at_cylinder_241,
        safe_budget_pressure_at_cylinder_240,
        cpr_degradation_fresh_referee,
        cpr_degradation_fatigued_referee,
        # Original holdout diagnostics (Phase 8)
        baku_dropper_leap_window_alignment,
        hal_dropper_leap_window_asymmetry,
        near_overflow_marginal_baku_294,
        death_trade_double_pressure,
        role_alignment_active_lsr_runway,
        role_alignment_variation4_post_engineering,
        # Phase F: pinned-tablebase expansion (17 new) ───────────────────
        # Clock variants
        forced_baku_overflow_mid_clock,
        forced_hal_overflow_mid_clock,
        forced_baku_overflow_pre_leap,
        forced_hal_overflow_pre_leap,
        # Leap-window variants
        forced_baku_overflow_leap_window_open,
        forced_hal_overflow_leap_window_open,
        forced_baku_overflow_leap_window_late,
        forced_baku_overflow_post_leap,
        # Fatigue / TTD pressure
        forced_baku_overflow_fatigued_referee,
        forced_hal_overflow_fatigued_referee,
        forced_baku_overflow_high_ttd,
        # Asymmetric death pins
        forced_baku_overflow_with_baku_deaths,
        forced_hal_overflow_with_hal_deaths,
        forced_baku_overflow_with_hal_deaths,
        forced_hal_overflow_with_baku_deaths,
        # Double-overflow
        both_overflow_baku_dies_first,
        both_overflow_hal_dies_first,
        # Phase F-2: interior-valued pins (survivable leap-window forced fail) ─
        forced_hal_fail_survivable_fresh,
        forced_hal_fail_survivable_fatigued,
        forced_hal_fail_survivable_deep,
    )
}


def scenario_names() -> tuple[str, ...]:
    return tuple(REGISTRY.keys())


def get_scenario(name: str) -> TacticalScenario:
    if name not in REGISTRY:
        raise KeyError(f"unknown tablebase scenario: {name}")
    return REGISTRY[name]()


def materialize_all() -> tuple[TacticalScenario, ...]:
    return tuple(factory() for factory in REGISTRY.values())


def pinned_scenarios(*, include_holdout: bool = False) -> tuple[TacticalScenario, ...]:
    return tuple(
        s
        for s in materialize_all()
        if s.expected_value is not None and (include_holdout or not s.holdout)
    )


def scenarios_by_tag(tag: str) -> tuple[TacticalScenario, ...]:
    return tuple(s for s in materialize_all() if tag in s.tags)


def solve_target(name: str) -> ExactSolveResult:
    """Run the exact finite-horizon solver against a registered scenario."""
    scenario = get_scenario(name)
    return solve_exact_finite_horizon(
        scenario.game,
        scenario.half_round_horizon,
        scenario.config,
    )


def verify_pinned_value(name: str, *, abs_tol: float = 1e-9) -> ExactSolveResult:
    """Solve a pinned scenario and raise if the value drifted from its pin."""
    scenario = get_scenario(name)
    if scenario.expected_value is None:
        raise ValueError(f"scenario {name!r} has no pinned expected_value")
    result = solve_target(name)
    if not math.isclose(result.value_for_hal, scenario.expected_value, abs_tol=abs_tol):
        raise AssertionError(
            f"tablebase value drift for {name!r}: "
            f"got {result.value_for_hal!r}, expected {scenario.expected_value!r}"
        )
    return result


# ============================================================================
# Analytic backward map
# ============================================================================

"""Analytic stage-transition map for backward induction (tablebase pilot).

At a half-round decision state, the 60x60 (61x60 in the leap turn) joint
action space collapses to at most 61 distinct outcomes:

- success cells (check >= drop) depend ONLY on ST = check - drop:
  the checker's cylinder grows by ST; at >= CYLINDER_MAX the injection is
  immediate and always fatal (death_duration = 300 => survival probability
  is exactly 0 in src/Referee.py).
- every fail cell (check < drop) produces the SAME death event:
  duration min(cylinder + 60, CYLINDER_MAX), survival probability taken
  from the engine's own referee (invariant G4: no reimplemented chance).

``analytic_stage_outcomes`` computes that outcome set by arithmetic plus
one referee call; ``verify_stage_outcomes_against_engine`` proves it
matches live ``Game.resolve_half_round`` probes outcome class by outcome
class. Backward-induction solvers assemble per-state payoff matrices from
these outcomes and child-value lookups in O(60) instead of ~9,000 engine
calls per node — the cost inversion that makes a full-game sweep feasible.

Everything here is EXACT: no action coarsening, terminal-only utilities,
chance probabilities from the engine referee. Interval-valued sweeps
(bounded unknowns at unsolved frontiers) live outside this namespace.
"""


from dataclasses import dataclass

import numpy as np

from stl.engine.game import (
    CYLINDER_MAX,
    DEATH_PROCEDURE_OVERHEAD,
    FAILED_CHECK_PENALTY,
    TURN_DURATION_NORMAL,
    WITHIN_ROUND_OVERHEAD,
)
from stl.engine.game import Game

from stl.solver.exact import ExactGameSnapshot, exact_public_state

MAX_ST = TURN_DURATION_NORMAL - 1  # check <= 60, drop >= 1 => successful ST in 0..59
ST_COUNT = MAX_ST + 1


@dataclass(frozen=True)
class StageSuccess:
    st: int
    checker_cylinder_after: float
    overflow: bool  # True => immediate injection, p = 0, dropper wins


@dataclass(frozen=True)
class StageFail:
    death_duration: float
    survival_probability: float  # engine referee value; 0.0 when duration hits the cap
    # Survive branch deltas: checker cylinder -> 0, ttd += duration,
    # deaths += 1, referee cprs += 1. Fatal branch: dropper wins.


@dataclass(frozen=True)
class StageOutcomes:
    dropper_name: str
    checker_name: str
    checker_is_hal: bool
    turn_duration: int
    drop_seconds: tuple[int, ...]
    check_seconds: tuple[int, ...]
    successes: tuple[StageSuccess, ...]  # indexed by st
    fail: StageFail


def analytic_stage_outcomes(game: Game) -> StageOutcomes:
    """The stage's distinct outcomes, by arithmetic + one referee call."""
    from stl.engine.actions import legal_max_second

    dropper, checker = game.get_roles_for_half(game.current_half)
    turn_duration = game.get_turn_duration()
    drop_max = legal_max_second(dropper.name, "dropper", turn_duration)
    check_max = legal_max_second(checker.name, "checker", turn_duration)

    successes = []
    for st in range(0, MAX_ST + 1):
        cyl_after = checker.cylinder + st
        successes.append(
            StageSuccess(
                st=st,
                checker_cylinder_after=cyl_after,
                overflow=cyl_after >= CYLINDER_MAX,
            )
        )

    fail_duration = min(checker.cylinder + FAILED_CHECK_PENALTY, CYLINDER_MAX)
    p_survive = game.referee.compute_survival_probability(
        checker, death_duration=fail_duration
    )

    return StageOutcomes(
        dropper_name=dropper.name,
        checker_name=checker.name,
        checker_is_hal=checker.name.lower() == "hal",
        turn_duration=turn_duration,
        drop_seconds=tuple(range(1, drop_max + 1)),
        check_seconds=tuple(range(1, check_max + 1)),
        successes=tuple(successes),
        fail=StageFail(
            death_duration=float(fail_duration),
            survival_probability=float(p_survive),
        ),
    )


# ── Engine equivalence ─────────────────────────────────────────────────────


def _expected_clock_after(game: Game, *, death_duration: float | None) -> float:
    """Mirror the engine's clock bookkeeping for one resolved half-round."""
    clock = game.game_clock + game.get_turn_duration()
    if death_duration is not None:
        clock += death_duration + DEATH_PROCEDURE_OVERHEAD
    if game.current_half == 1:
        return clock + WITHIN_ROUND_OVERHEAD
    # Engine snap: next wall-clock minute (pre-leap multiples of 60; the
    # 8:59 minute holds 61 seconds, so post-leap minutes sit at 3601+60n).
    gc = int(clock)
    if gc < 3600:
        snapped = ((gc // 60) + 1) * 60
        return float(3601 if snapped == 3600 else snapped)
    if gc <= 3600:
        return 3601.0
    return float(3601 + (((gc - 3601) // 60) + 1) * 60)


def verify_stage_outcomes_against_engine(game: Game) -> None:
    """Assert the analytic map reproduces live engine probes exactly.

    Probes one representative joint action per outcome class (every ST,
    the tie cell, the fail cell with both forced outcomes) and compares
    full public states. Raises AssertionError on any mismatch.
    """
    outcomes = analytic_stage_outcomes(game)
    snap = ExactGameSnapshot(game)
    base = exact_public_state(game)
    checker_name = outcomes.checker_name

    def probe(drop: int, check: int, survived: bool | None):
        record = game.resolve_half_round(drop, check, survived_outcome=survived)
        state = exact_public_state(game)
        snap.restore(game)
        return record, state

    def check_field(state, name, expected, ctx):
        actual = getattr(state, name)
        assert actual == expected, (
            f"{ctx}: engine {name}={actual!r} != analytic {expected!r} "
            f"(base state {base!r})"
        )

    checker_is_p1 = base.p1_name == checker_name
    cyl_field = "p1_cylinder" if checker_is_p1 else "p2_cylinder"
    ttd_field = "p1_ttd" if checker_is_p1 else "p2_ttd"
    deaths_field = "p1_deaths" if checker_is_p1 else "p2_deaths"

    # Success classes: one probe per ST, plus an extra diagonal cell for ST=0.
    for success in outcomes.successes:
        st = success.st
        cells = [(1, 1 + st)] if st > 0 else [(1, 1), (2, 2)]
        for drop, check in cells:
            if check > max(outcomes.check_seconds):
                continue
            if success.overflow:
                record, state = probe(drop, check, False)
                assert record.death_duration == float(CYLINDER_MAX), (
                    f"overflow ST={st}: engine duration {record.death_duration}"
                )
                assert record.survival_probability == 0.0, (
                    f"overflow ST={st}: engine p={record.survival_probability} != 0"
                )
                check_field(state, "game_over", True, f"overflow ST={st}")
                check_field(state, "winner_name", outcomes.dropper_name, f"overflow ST={st}")
            else:
                record, state = probe(drop, check, None)
                assert record.st_gained == st, (
                    f"cell ({drop},{check}): engine ST {record.st_gained} != {st}"
                )
                check_field(state, cyl_field, success.checker_cylinder_after, f"success ST={st}")
                check_field(state, "game_over", False, f"success ST={st}")
                expected_clock = _expected_clock_after(game, death_duration=None)
                check_field(state, "game_clock", expected_clock, f"success ST={st}")

    # Fail class: one cell, both chance branches.
    fail = outcomes.fail
    drop, check = 2, 1
    record, state = probe(drop, check, True if fail.survival_probability > 0 else False)
    assert record.death_duration == fail.death_duration, (
        f"fail: engine duration {record.death_duration} != {fail.death_duration}"
    )
    assert record.survival_probability == fail.survival_probability, (
        f"fail: engine p={record.survival_probability} != {fail.survival_probability}"
    )
    if fail.survival_probability > 0:
        check_field(state, cyl_field, 0.0, "fail+survive")
        check_field(state, ttd_field, getattr(base, ttd_field) + fail.death_duration, "fail+survive")
        check_field(state, deaths_field, getattr(base, deaths_field) + 1, "fail+survive")
        check_field(state, "referee_cprs", base.referee_cprs + 1, "fail+survive")
        expected_clock = _expected_clock_after(game, death_duration=fail.death_duration)
        check_field(state, "game_clock", expected_clock, "fail+survive")
        # Fatal branch too.
        _, dead_state = probe(drop, check, False)
        check_field(dead_state, "game_over", True, "fail+die")
        check_field(dead_state, "winner_name", outcomes.dropper_name, "fail+die")
    else:
        check_field(state, "game_over", True, "fail (always fatal)")
        check_field(state, "winner_name", outcomes.dropper_name, "fail (always fatal)")


# ── Payoff-matrix assembly ─────────────────────────────────────────────────

_ST_INDEX_CACHE: dict = {}


def _st_index_matrix(n_drop: int, n_check: int) -> tuple[np.ndarray, np.ndarray]:
    """(fail_mask, st_index) for a drop x check grid of 1-based seconds."""
    key = (n_drop, n_check)
    if key not in _ST_INDEX_CACHE:
        drops = np.arange(1, n_drop + 1)[:, None]
        checks = np.arange(1, n_check + 1)[None, :]
        fail_mask = checks < drops
        st = checks - drops  # only meaningful where not fail
        _ST_INDEX_CACHE[key] = (fail_mask, st)
    return _ST_INDEX_CACHE[key]


def assemble_payoff_matrix(
    n_drop: int,
    n_check: int,
    success_values: np.ndarray,
    fail_value: float,
) -> np.ndarray:
    """Payoff matrix over 1-based (drop, check) grids from outcome values.

    ``success_values``: length-ST_COUNT array, success_values[st] = value
    after a successful check with that ST (already +-1 for overflow
    terminals). ``fail_value``: the chance-weighted value of the shared
    fail cell (p * V_survive + (1-p) * terminal).
    """
    fail_mask, st_idx = _st_index_matrix(n_drop, n_check)
    matrix = np.where(fail_mask, fail_value, success_values[st_idx])
    return matrix.astype(np.float64)


def stage_matrix_from_values(
    outcomes: StageOutcomes,
    success_values: np.ndarray,
    fail_value: float,
) -> np.ndarray:
    """Hal-perspective payoff matrix for a state's StageOutcomes."""
    return assemble_payoff_matrix(
        len(outcomes.drop_seconds),
        len(outcomes.check_seconds),
        success_values,
        fail_value,
    )


# ============================================================================
# Epoch sweep
# ============================================================================

"""Interval-valued backward induction over one (ttd_hal, ttd_baku, cprs) epoch.

State convention (matches the Tier-0 pilot): V[bit, cyl_hal, cyl_baku],
bit 0 = Hal drops / Baku checks, bit 1 = Baku drops / Hal checks; values
are Hal-perspective. Within an epoch only cylinders move (success: the
checker's cylinder grows by ST; cylinders are the only resetting state),
so a single sweep by descending cylinder sum is exact — the post-leap
quotient is a DAG.

Transitions out of the epoch happen only on a SURVIVED death (checker's
ttd grows, cprs + 1): their values are supplied by ``survive_value`` —
either a solved deeper epoch's table or a certified bracket. The minimax
value of a matrix is monotone in its entries, so sweeping once with all
lower edges and once with all upper edges yields certified [lo, hi]
brackets at every state.

Death chance probabilities come from the engine referee per (player,
cylinder) — see ``survival_table`` — never reimplemented (G4).
"""


import time
from dataclasses import dataclass
from typing import Callable

import numpy as np

from stl.solver.exact import solve_minimax
from stl.engine.game import (
    CYLINDER_MAX,
    FAILED_CHECK_PENALTY,
    PHYSICALITY_BAKU,
    PHYSICALITY_HAL,
)
from stl.engine.game import Player
from stl.engine.game import Referee

CYL = int(CYLINDER_MAX)

# survive_value(checker_is_hal, death_duration, next_bit, other_cylinder)
#   -> (lo, hi) Hal-perspective value bracket of the post-revival state
#      (checker cylinder reset to 0, checker ttd += duration, cprs + 1).
SurviveValueFn = Callable[[bool, int, int, int], tuple[float, float]]


@dataclass(frozen=True)
class EpochSpec:
    ttd_hal: float
    ttd_baku: float
    cprs: int

    @property
    def deaths_total_hint(self) -> int:
        """Engine invariant: cprs == total deaths."""
        return self.cprs


def survival_table(*, name: str, ttd: float, cprs: int) -> np.ndarray:
    """p_survive[cylinder] for this player's failed check, via the engine
    referee (G4). Index = checker cylinder BEFORE the +60 penalty."""
    physicality = PHYSICALITY_HAL if name.lower() == "hal" else PHYSICALITY_BAKU
    probe = Player(name=name, physicality=physicality)
    probe.ttd = float(ttd)
    referee = Referee()
    referee.cprs_performed = int(cprs)

    table = np.zeros(CYL, dtype=np.float64)
    for cyl in range(CYL):
        duration = min(cyl + FAILED_CHECK_PENALTY, CYLINDER_MAX)
        table[cyl] = referee.compute_survival_probability(
            probe, death_duration=duration
        )
    return table


def bracket_survive_value(lo: float = -1.0, hi: float = 1.0) -> SurviveValueFn:
    """Frontier bracket for unsolved deeper epochs."""

    def fn(checker_is_hal: bool, duration: int, next_bit: int, other_cyl: int):
        return (lo, hi)

    return fn


def solve_epoch(
    spec: EpochSpec,
    survive_value: SurviveValueFn,
    *,
    survival_overrides: tuple[np.ndarray, np.ndarray] | None = None,
    min_cyl: int = 0,
    progress: Callable[[str], None] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Solve one epoch; returns (V_lo, V_hi), each [2, CYL, CYL] float64.

    ``min_cyl`` restricts the sweep to states with BOTH cylinders >=
    min_cyl — a self-contained region (cylinders only grow), used by
    tests. Entries outside the region are NaN.

    ``survival_overrides`` replaces the engine-derived (hal, baku)
    survival tables — used by tests to force the all-deaths-fatal
    degenerate case, which must reproduce the Tier-0 limit game exactly.
    """
    if survival_overrides is not None:
        p_hal, p_baku = survival_overrides
    else:
        p_hal = survival_table(name="Hal", ttd=spec.ttd_hal, cprs=spec.cprs)
        p_baku = survival_table(name="Baku", ttd=spec.ttd_baku, cprs=spec.cprs)

    V_lo = np.full((2, CYL, CYL), np.nan, dtype=np.float64)
    V_hi = np.full((2, CYL, CYL), np.nan, dtype=np.float64)
    start = time.perf_counter()
    states = 0

    for cyl_sum in range(2 * (CYL - 1), 2 * min_cyl - 1, -1):
        ch_lo = max(min_cyl, cyl_sum - (CYL - 1))
        ch_hi = min(CYL - 1, cyl_sum - min_cyl)
        for ch in range(ch_lo, ch_hi + 1):
            cb = cyl_sum - ch
            succ_by_bit: dict[int, tuple[np.ndarray, np.ndarray]] = {}
            fail_by_bit: dict[int, tuple[float, float]] = {}

            for bit in (0, 1):
                checker_is_hal = bit == 1
                checker_cyl = ch if checker_is_hal else cb
                other_cyl = cb if checker_is_hal else ch
                terminal = -1.0 if checker_is_hal else 1.0  # checker dies
                next_bit = 1 - bit

                # Success outcome values per ST. ST=0 is a same-cylinder
                # role swap and is filled by the local fixed-point solve below.
                succ_lo = np.full(ST_COUNT, terminal)
                succ_hi = np.full(ST_COUNT, terminal)
                room = (CYL - 1) - checker_cyl
                if room >= 1:
                    take = min(MAX_ST, room)
                    if checker_is_hal:
                        succ_lo[1 : take + 1] = V_lo[next_bit, ch + 1 : ch + take + 1, cb]
                        succ_hi[1 : take + 1] = V_hi[next_bit, ch + 1 : ch + take + 1, cb]
                    else:
                        succ_lo[1 : take + 1] = V_lo[next_bit, ch, cb + 1 : cb + take + 1]
                        succ_hi[1 : take + 1] = V_hi[next_bit, ch, cb + 1 : cb + take + 1]

                # Shared fail outcome.
                p = (p_hal if checker_is_hal else p_baku)[checker_cyl]
                if p <= 0.0:
                    fail_lo = fail_hi = terminal
                else:
                    duration = min(checker_cyl + FAILED_CHECK_PENALTY, CYLINDER_MAX)
                    s_lo, s_hi = survive_value(
                        checker_is_hal, int(duration), next_bit, int(other_cyl)
                    )
                    fail_lo = p * s_lo + (1.0 - p) * terminal
                    fail_hi = p * s_hi + (1.0 - p) * terminal

                succ_by_bit[bit] = (succ_lo, succ_hi)
                fail_by_bit[bit] = (fail_lo, fail_hi)

            def solve_local(upper: bool) -> np.ndarray:
                values = np.zeros(2, dtype=np.float64)
                idx = 1 if upper else 0
                for _ in range(200):
                    updated = np.empty(2, dtype=np.float64)
                    for bit in (0, 1):
                        succ = succ_by_bit[bit][idx].copy()
                        succ[0] = values[1 - bit]
                        fail = fail_by_bit[bit][idx]
                        matrix = assemble_payoff_matrix(60, 60, succ, fail)
                        if bit == 0:
                            _, value = solve_minimax(matrix)       # Hal (dropper) maximizes
                        else:
                            _, neg = solve_minimax(-matrix)        # Baku (dropper) minimizes
                            value = -neg
                        updated[bit] = value
                    if float(np.max(np.abs(updated - values))) <= 1e-10:
                        return updated
                    values = updated
                return values

            V_lo[:, ch, cb] = solve_local(upper=False)
            V_hi[:, ch, cb] = solve_local(upper=True)
            states += 2

        if progress is not None and cyl_sum % 100 == 0:
            elapsed = time.perf_counter() - start
            progress(
                f"cyl_sum={cyl_sum} states={states} elapsed={elapsed:.0f}s "
                f"({1000 * elapsed / max(states, 1):.2f} ms/state)"
            )

    return V_lo, V_hi


# ============================================================================
# Tier A runtime lookup
# ============================================================================

"""Runtime access to the generated Tier A interval tablebase.

Tier A is an interval-valued post-leap quotient over states with <= 1
total death. It is deliberately outside ``environment.cfr``: these values
are certified brackets useful for search frontiers, not part of the
terminal-only exact oracle.
"""


import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

from stl.solver.search import LeafEvaluation, normalize_leaf_evaluation
from stl.solver.exact import terminal_value
from stl.engine.actions import ACTION_SIZE, legal_max_second
from stl.engine.game import LS_WINDOW_END
from stl.engine.game import Game


DEFAULT_TIER_A_DIR = Path(__file__).resolve().parents[2] / "checkpoints" / "tablebase" / "tier_a"


@dataclass(frozen=True)
class TierAInterval:
    lo: float
    hi: float
    source: str
    bit: int
    hal_cylinder: int
    baku_cylinder: int

    @property
    def width(self) -> float:
        return self.hi - self.lo

    @property
    def midpoint(self) -> float:
        return 0.5 * (self.lo + self.hi)


@dataclass(frozen=True)
class TierALookupResult:
    interval: TierAInterval | None
    miss_reason: str | None = None

    @property
    def hit(self) -> bool:
        return self.interval is not None


def _player_by_name(game: Game, name: str):
    for player in (game.player1, game.player2):
        if player.name.lower() == name.lower():
            return player
    return None


def _int_index(value: float, *, name: str) -> tuple[int | None, str | None]:
    idx = int(round(float(value)))
    if abs(float(value) - idx) > 1e-9:
        return None, f"{name}_non_integer"
    if not (0 <= idx < 300):
        return None, f"{name}_out_of_range"
    return idx, None


def _bit_for_current_roles(game: Game) -> tuple[int | None, str | None]:
    if game.game_over:
        return None, "terminal"
    dropper, checker = game.get_roles_for_half(game.current_half)
    if dropper.name.lower() == "hal" and checker.name.lower() == "baku":
        return 0, None
    if dropper.name.lower() == "baku" and checker.name.lower() == "hal":
        return 1, None
    return None, "unsupported_roles"


class TierALookup:
    """Lazy lookup for ``checkpoints/tablebase/tier_a`` interval artifacts."""

    def __init__(self, root: str | os.PathLike[str] = DEFAULT_TIER_A_DIR, *, verify: bool = False) -> None:
        self.root = Path(root)
        self._cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        if verify:
            self.verify_manifest()

    def verify_manifest(self) -> dict[str, str]:
        manifest_path = self.root / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(f"Tier A manifest not found: {manifest_path}")
        with manifest_path.open() as fh:
            manifest: dict[str, str] = json.load(fh)
        for name, expected in manifest.items():
            path = self.root / name
            if not path.exists():
                raise FileNotFoundError(f"Tier A artifact listed in manifest is missing: {path}")
            with path.open("rb") as fh:
                actual = hashlib.sha256(fh.read()).hexdigest()
            if actual != expected:
                raise ValueError(f"Tier A artifact hash mismatch for {name}: {actual} != {expected}")
        return manifest

    def _load(self, name: str) -> tuple[np.ndarray, np.ndarray]:
        if name not in self._cache:
            path = self.root / name
            if not path.exists():
                raise FileNotFoundError(f"Tier A artifact not found: {path}")
            with np.load(path) as data:
                self._cache[name] = (
                    data["lo"].astype(np.float64),
                    data["hi"].astype(np.float64),
                )
        return self._cache[name]

    def _artifact_for_game(self, game: Game) -> tuple[str | None, str | None]:
        if game.game_clock <= float(LS_WINDOW_END):
            return None, "not_post_leap"
        if game.get_turn_duration() != 60:
            return None, "non_normal_turn"
        if game.referee.cprs_performed != game.player1.deaths + game.player2.deaths:
            return None, "cprs_death_mismatch"

        hal = _player_by_name(game, "Hal")
        baku = _player_by_name(game, "Baku")
        if hal is None or baku is None:
            return None, "missing_hal_or_baku"

        total_deaths = int(hal.deaths + baku.deaths)
        if total_deaths == 0:
            if game.referee.cprs_performed != 0 or hal.ttd != 0.0 or baku.ttd != 0.0:
                return None, "unsupported_d0_epoch"
            return "d0.npz", None

        if total_deaths == 1 and game.referee.cprs_performed == 1:
            if hal.deaths == 1 and baku.deaths == 0 and baku.ttd == 0.0:
                ttd, reason = _int_index(hal.ttd, name="hal_ttd")
                if reason is not None:
                    return None, reason
                if not (60 <= ttd <= 299):
                    return None, "hal_ttd_out_of_tier_a_range"
                return f"d1_hal_{ttd}.npz", None
            if baku.deaths == 1 and hal.deaths == 0 and hal.ttd == 0.0:
                ttd, reason = _int_index(baku.ttd, name="baku_ttd")
                if reason is not None:
                    return None, reason
                if not (60 <= ttd <= 299):
                    return None, "baku_ttd_out_of_tier_a_range"
                return f"d1_baku_{ttd}.npz", None
            return None, "unsupported_d1_epoch"

        return None, "too_many_deaths"

    def lookup(self, game: Game) -> TierALookupResult:
        terminal = terminal_value(game, perspective_name="Hal")
        if terminal is not None:
            return TierALookupResult(
                TierAInterval(float(terminal), float(terminal), "terminal", -1, -1, -1)
            )

        bit, reason = _bit_for_current_roles(game)
        if reason is not None:
            return TierALookupResult(None, reason)

        hal = _player_by_name(game, "Hal")
        baku = _player_by_name(game, "Baku")
        if hal is None or baku is None:
            return TierALookupResult(None, "missing_hal_or_baku")
        ch, reason = _int_index(hal.cylinder, name="hal_cylinder")
        if reason is not None:
            return TierALookupResult(None, reason)
        cb, reason = _int_index(baku.cylinder, name="baku_cylinder")
        if reason is not None:
            return TierALookupResult(None, reason)

        artifact, reason = self._artifact_for_game(game)
        if reason is not None:
            return TierALookupResult(None, reason)
        if not (self.root / artifact).exists():
            return TierALookupResult(None, "artifact_missing")

        lo, hi = self._load(artifact)
        low = float(lo[bit, ch, cb])
        high = float(hi[bit, ch, cb])
        if not np.isfinite(low) or not np.isfinite(high):
            return TierALookupResult(None, "non_finite_interval")
        if low > high + 1e-7:
            return TierALookupResult(None, "unordered_interval")
        return TierALookupResult(TierAInterval(low, high, artifact, bit, ch, cb))


def uniform_policy_for_game(game: Game) -> tuple[np.ndarray, np.ndarray]:
    if game.game_over:
        return np.zeros(ACTION_SIZE, dtype=np.float64), np.zeros(ACTION_SIZE, dtype=np.float64)
    dropper, checker = game.get_roles_for_half(game.current_half)
    turn_duration = game.get_turn_duration()
    drop_max = legal_max_second(dropper.name, "dropper", turn_duration)
    check_max = legal_max_second(checker.name, "checker", turn_duration)
    drop = np.zeros(ACTION_SIZE, dtype=np.float64)
    check = np.zeros(ACTION_SIZE, dtype=np.float64)
    if drop_max > 0:
        drop[1 : drop_max + 1] = 1.0 / drop_max
    if check_max > 0:
        check[1 : check_max + 1] = 1.0 / check_max
    return drop, check


class TierAEvaluator:
    """Leaf evaluator wrapper that short-circuits low-width Tier A hits."""

    def __init__(
        self,
        fallback: Callable[[Game], LeafEvaluation | float],
        *,
        lookup: TierALookup | None = None,
        max_width: float = 0.0,
        use_midpoint_for_wide: bool = False,
        preserve_fallback_policy: bool = True,
    ) -> None:
        self.fallback = fallback
        self.lookup = lookup or TierALookup()
        self.max_width = float(max_width)
        self.use_midpoint_for_wide = bool(use_midpoint_for_wide)
        self.preserve_fallback_policy = bool(preserve_fallback_policy)
        self.hits = 0
        self.wide_hits = 0
        self.misses: dict[str, int] = {}

    def __call__(self, game: Game) -> LeafEvaluation:
        result = self.lookup.lookup(game)
        if result.interval is not None:
            interval = result.interval
            if interval.width <= self.max_width or self.use_midpoint_for_wide:
                self.hits += 1
                if self.preserve_fallback_policy and not game.game_over:
                    _, drop, check = normalize_leaf_evaluation(self.fallback(game), game)
                else:
                    drop, check = uniform_policy_for_game(game)
                return float(interval.midpoint), drop, check
            self.wide_hits += 1
            self.misses["wide_interval"] = self.misses.get("wide_interval", 0) + 1
        else:
            reason = result.miss_reason or "unknown"
            self.misses[reason] = self.misses.get(reason, 0) + 1
        return normalize_leaf_evaluation(self.fallback(game), game)


def frontier_interval_fn(
    lookup: TierALookup | None = None,
    *,
    max_width: float | None = None,
) -> Callable[[Game], tuple[float, float] | None]:
    table = lookup or TierALookup()

    def fn(game: Game) -> tuple[float, float] | None:
        result = table.lookup(game)
        if result.interval is None:
            return None
        if max_width is not None and result.interval.width > max_width:
            return None
        return result.interval.lo, result.interval.hi

    return fn


# ============================================================================
# Epoch tablebase exports
# ============================================================================

"""Certified tablebase construction (plan Phase 3).

Backward induction over the post-leap clock-free quotient, one epoch
(ttd_hal, ttd_baku, cprs) at a time. Where children leave the solved
region (a survived death entering an unsolved epoch), their values are
BRACKETED and the sweep propagates certified [lo, hi] intervals — the
declared approximation layer, which is why this package lives OUTSIDE
the exact-only ``environment/cfr`` namespace. The per-state transition
structure itself comes from the engine-equivalence-verified analytic map
(``environment/cfr/backward.py``); survival probabilities come from the
engine referee (invariant G4).
"""

import sys

tier_a = sys.modules[__name__]

__all__ = [
    "EpochSpec",
    "solve_epoch",
    "survival_table",
    "TierAEvaluator",
    "TierAInterval",
    "TierALookup",
    "TierALookupResult",
    "frontier_interval_fn",
    "tier_a",
]
