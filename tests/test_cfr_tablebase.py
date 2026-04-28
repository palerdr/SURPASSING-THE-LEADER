"""Tests for the rigorous CFR tablebase registry."""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from environment.cfr.exact_solver import evaluate_joint_action
from environment.cfr.exact_transition import ExactJointAction
from environment.cfr.tablebase import (
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
    "forced_baku_overflow_death",
    "forced_hal_overflow_death",
    "leap_second_check_61_probe",
    "safe_budget_pressure_at_cylinder_241",
    "safe_budget_pressure_at_cylinder_240",
    "cpr_degradation_fresh_referee",
    "cpr_degradation_fatigued_referee",
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


def test_pinned_scenarios_are_forced_overflow_pair():
    pinned = pinned_scenarios()
    assert {s.name for s in pinned} == {
        "forced_baku_overflow_death",
        "forced_hal_overflow_death",
    }


def test_scenarios_by_tag_filters_correctly():
    near_overflow = scenarios_by_tag("near_overflow")
    assert {s.name for s in near_overflow} == {
        "forced_baku_overflow_death",
        "forced_hal_overflow_death",
    }
    safe_budget = scenarios_by_tag("safe_budget")
    assert {s.name for s in safe_budget} == {
        "safe_budget_pressure_at_cylinder_241",
        "safe_budget_pressure_at_cylinder_240",
    }


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
        verify_pinned_value("leap_second_check_61_probe")


def test_safe_budget_threshold_pair_distinguishes_cylinder_value():
    # cyl=241 lets Hal force a single terminal cell at (drop=1, check=60);
    # cyl=240's max-success cylinder is 299 < 300, so check=60 is a safe pure
    # strategy and Hal cannot force any terminal cell at horizon=1.
    pressure_241 = solve_target("safe_budget_pressure_at_cylinder_241")
    pressure_240 = solve_target("safe_budget_pressure_at_cylinder_240")

    assert pressure_241.value_for_hal > pressure_240.value_for_hal
    assert pressure_240.value_for_hal == pytest.approx(0.0)


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


def test_leap_second_probe_check_61_dominates_check_60_against_drop_61():
    scenario = get_scenario("leap_second_check_61_probe")

    check_60 = evaluate_joint_action(
        scenario.game,
        ExactJointAction(drop_time=61, check_time=60),
        half_round_horizon=1,
        config=scenario.config,
    )
    check_61 = evaluate_joint_action(
        scenario.game,
        ExactJointAction(drop_time=61, check_time=61),
        half_round_horizon=1,
        config=scenario.config,
    )

    assert check_61.value > check_60.value
    assert check_60.baku_win_probability > 0.0
    assert check_61.unresolved_probability == pytest.approx(1.0)
