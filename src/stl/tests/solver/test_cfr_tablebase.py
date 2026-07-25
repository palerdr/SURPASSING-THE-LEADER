"""Tests for the rigorous CFR tablebase registry."""

import os
import sys

import pytest

sys.path.insert(0, os.getcwd())

from stl.solver.exact import evaluate_joint_action
from stl.solver.exact import ExactJointAction
from stl.solver.tablebase import (
    REGISTRY,
    get_scenario,
    materialize_all,
    pinned_scenarios,
    scenario_names,
    scenarios_by_tag,
    solve_target,
    verify_pinned_value,
)


EXPECTED_NAMES = (
    # Original Phase 8 scenarios
    "forced_baku_overflow_death",
    "forced_hal_overflow_death",
    "safe_budget_pressure_at_cylinder_240",
    "safe_budget_pressure_at_cylinder_239",
    "cpr_degradation_fresh_referee",
    "cpr_degradation_fatigued_referee",
    "baku_dropper_leap_window_alignment",
    "hal_dropper_leap_window_asymmetry",
    "near_overflow_marginal_baku_294",
    "death_trade_double_pressure",
    "role_alignment_active_lsr_runway",
    "role_alignment_variation4_post_engineering",
    # Phase F pinned-tablebase expansion (17 scenarios)
    "forced_baku_overflow_mid_clock",
    "forced_hal_overflow_mid_clock",
    "forced_baku_overflow_pre_leap",
    "forced_hal_overflow_pre_leap",
    "forced_baku_overflow_leap_window_open",
    "forced_hal_overflow_leap_window_open",
    "forced_baku_overflow_leap_window_late",
    "forced_baku_overflow_post_leap",
    "forced_baku_overflow_fatigued_referee",
    "forced_hal_overflow_fatigued_referee",
    "forced_baku_overflow_high_ttd",
    "forced_baku_overflow_with_baku_deaths",
    "forced_hal_overflow_with_hal_deaths",
    "forced_baku_overflow_with_hal_deaths",
    "forced_hal_overflow_with_baku_deaths",
    "both_overflow_baku_dies_first",
    "both_overflow_hal_dies_first",
    # Phase F-2 interior-valued pins (survivable leap-window forced fail)
    "forced_hal_fail_survivable_fresh",
    "forced_hal_fail_survivable_fatigued",
    "forced_hal_fail_survivable_deep",
)

INTERIOR_PIN_NAMES = (
    "forced_hal_fail_survivable_fresh",
    "forced_hal_fail_survivable_fatigued",
    "forced_hal_fail_survivable_deep",
)

PINNED_NAMES = tuple(
    name for name, factory in REGISTRY.items() if factory().expected_value is not None
)


def test_registry_exposes_all_named_scenarios():
    assert set(scenario_names()) == set(EXPECTED_NAMES)
    assert set(REGISTRY.keys()) == set(EXPECTED_NAMES)


def test_unknown_scenario_name_raises_key_error():
    with pytest.raises(KeyError):
        get_scenario("not_a_real_scenario")


def test_materialize_all_returns_one_scenario_per_factory():
    materialized = materialize_all()
    assert len(materialized) == len(REGISTRY)
    assert {s.name for s in materialized} == set(EXPECTED_NAMES)


def test_pinned_scenarios_include_phase_f_expansion():
    """Phase F brought REGISTRY pinned count from 2 → 19. Two original
    overflow pins plus 17 forced-terminal extensions across clock, leap-
    window, fatigue, ttd, asymmetric-deaths, and double-overflow axes.
    """
    pinned = pinned_scenarios()
    pinned_names = {s.name for s in pinned}
    # The two originals
    assert "forced_baku_overflow_death" in pinned_names
    assert "forced_hal_overflow_death" in pinned_names
    # Phase F expansion present and accounted for
    assert len(pinned_names) >= 19, (
        f"expected >=19 pinned scenarios after Phase F, got {len(pinned_names)}: {pinned_names}"
    )


def test_scenarios_by_tag_filters_correctly():
    safe_budget = scenarios_by_tag("safe_budget")
    assert {s.name for s in safe_budget} == {
        "safe_budget_pressure_at_cylinder_240",
        "safe_budget_pressure_at_cylinder_239",
    }
    # near_overflow is widely used across Phase F pins; just assert
    # the original three remain present rather than enumerating all.
    near_overflow = {s.name for s in scenarios_by_tag("near_overflow")}
    assert "forced_baku_overflow_death" in near_overflow
    assert "forced_hal_overflow_death" in near_overflow
    assert "near_overflow_marginal_baku_294" in near_overflow


def test_pinned_baku_overflow_value_is_plus_one():
    result = verify_pinned_value("forced_baku_overflow_death")
    assert result.value_for_hal == pytest.approx(1.0)
    assert result.breakdown.hal_win_probability == pytest.approx(1.0)
    assert result.unresolved_probability == pytest.approx(0.0)


def test_pinned_hal_overflow_value_is_minus_one():
    result = verify_pinned_value("forced_hal_overflow_death")
    assert result.value_for_hal == pytest.approx(-1.0)
    assert result.breakdown.baku_win_probability == pytest.approx(1.0)
    assert result.unresolved_probability == pytest.approx(0.0)


def test_verify_pinned_value_rejects_unpinned_scenarios():
    with pytest.raises(ValueError):
        verify_pinned_value("safe_budget_pressure_at_cylinder_240")


def test_safe_budget_threshold_pair_distinguishes_cylinder_value():
    # cyl=240 lets Hal force a terminal cell at (drop=1, check=60);
    # cyl=239's max-success cylinder is 299 < 300, so check=60 is a safe pure
    # strategy and Hal cannot force any terminal cell at horizon=1.
    pressure_240 = solve_target("safe_budget_pressure_at_cylinder_240")
    pressure_239 = solve_target("safe_budget_pressure_at_cylinder_239")

    assert pressure_240.value_for_hal > pressure_239.value_for_hal
    assert pressure_239.value_for_hal == pytest.approx(0.0)


def test_late_cpr_degradation_pair_increases_hal_value_under_forced_fail():
    fresh = get_scenario("cpr_degradation_fresh_referee")
    fatigued = get_scenario("cpr_degradation_fatigued_referee")
    forced_fail = ExactJointAction(drop_time=60, check_time=1)

    fresh_branch = evaluate_joint_action(fresh.game, forced_fail, half_round_horizon=1)
    fatigued_branch = evaluate_joint_action(fatigued.game, forced_fail, half_round_horizon=1)

    assert fresh_branch.unresolved_probability > 0.0
    assert fatigued_branch.unresolved_probability > 0.0
    assert fatigued_branch.hal_win_probability > fresh_branch.hal_win_probability
    assert fatigued_branch.value > fresh_branch.value


# ── Phase F acceptance: every pinned scenario satisfies strict criteria ───


@pytest.mark.parametrize("name", PINNED_NAMES)
def test_every_pinned_scenario_resolves_to_expected_value(name):
    """Strict acceptance gate for every pinned tablebase entry:

      1. ``solve_exact_finite_horizon`` returns ``unresolved_probability == 0``
         (chance tree fully resolves at the scenario's stated horizon).
      2. ``value_for_hal`` matches ``expected_value`` within 1e-6.

    Any scenario added to REGISTRY with a non-None ``expected_value`` must
    satisfy both; the parameterized form fires once per scenario so the
    failing one is named in the assertion output.
    """
    scenario = REGISTRY[name]()
    result = solve_target(name)
    assert result.unresolved_probability == pytest.approx(0.0, abs=1e-6), (
        f"{name} has unresolved_probability {result.unresolved_probability:.6f} "
        f"(expected 0). Scenario is not all-terminal at horizon "
        f"{scenario.half_round_horizon}; drop the pin or deepen the horizon."
    )
    assert result.value_for_hal == pytest.approx(scenario.expected_value, abs=1e-6), (
        f"{name} value drifted: got {result.value_for_hal:.6f}, "
        f"expected {scenario.expected_value:.6f}"
    )


def test_pinned_scenarios_have_distinct_feature_vectors():
    """No two pinned scenarios may produce identical ValueNet feature
    vectors. If they do, the corpus's tablebase records can't be
    distinguished by the net, so any net hedging on tablebase predictions
    is masked into a single MSE statistic rather than per-scenario drift.

    Uses the same feature-hash logic as ``corpus_diagnostics`` so a
    failure here predicts a Phase F → Phase G collision before training.
    """
    from stl.learning.model import extract_features
    from stl.learning.corpus_diagnostics import _hash_features

    pinned = pinned_scenarios()
    hashes: dict[str, list[str]] = {}
    for scenario in pinned:
        h = _hash_features(extract_features(scenario.game))
        hashes.setdefault(h, []).append(scenario.name)

    collisions = {h: names for h, names in hashes.items() if len(names) > 1}
    assert not collisions, (
        f"Pinned scenarios collide on feature vectors (the net can't tell them "
        f"apart): {collisions}"
    )


# ── Phase F-2 acceptance: interior-valued pins ────────────────────────────


@pytest.mark.parametrize("name", INTERIOR_PIN_NAMES)
def test_interior_pins_are_genuinely_interior(name):
    """Phase F-2 pins exist specifically to constrain the INTERIOR of [-1, 1].

    Every Phase-8/F pin is a ±1 forced-overflow boundary terminal, so the pinned
    ruler never constrained interior values. A pin here whose value sat on the
    boundary would be mislabeled — the whole construction is a *survivable*
    (0 < p < 1) forced fail. Asserts: all-terminal (unresolved 0), genuinely
    interior (|value| < 1), and the engine-derived pin matches the solver to 1e-9.
    """
    result = solve_target(name)
    assert result.unresolved_probability == pytest.approx(0.0, abs=1e-9), (
        f"{name} unresolved_probability {result.unresolved_probability:.6f} != 0; "
        "not all-terminal at horizon 2."
    )
    assert abs(result.value_for_hal) < 1.0 - 1e-6, (
        f"{name} value {result.value_for_hal:.6f} is on the ±1 boundary, not interior."
    )
    # Cross-checks the engine-derived expected_value against the solver at 1e-9.
    verify_pinned_value(name)


def test_interior_fail_pins_monotone_in_referee_fatigue():
    """Relational invariant backing the interior-pin family: more CPR fatigue
    lowers Hal's revival probability on the forced fail, so the fresh pin's
    interior value must strictly exceed the fatigued pin's. The fresh state is
    Hal-favored (p > 0.5 → value > 0); at the referee floor the fatigued state
    is Baku-favored (p < 0.5 → value < 0), so the pair also straddles zero.
    """
    fresh = solve_target("forced_hal_fail_survivable_fresh")
    fatigued = solve_target("forced_hal_fail_survivable_fatigued")

    assert fresh.value_for_hal > fatigued.value_for_hal
    assert fresh.value_for_hal > 0.0 > fatigued.value_for_hal


def test_interior_pins_break_the_all_boundary_ruler():
    """The defining Phase F-2 outcome: the pinned ruler now contains at least one
    genuinely interior anchor. Before F-2 every pinned value was exactly ±1, so a
    value net could score the tablebase source perfectly while learning nothing
    about the interior of [-1, 1]. This guards that the ruler keeps interior
    coverage."""
    interior = [
        s for s in pinned_scenarios()
        if abs(s.expected_value) < 1.0 - 1e-6
    ]
    assert interior, "no interior pinned anchors present; the ruler is all-boundary again"
    assert {s.name for s in interior} >= set(INTERIOR_PIN_NAMES)

