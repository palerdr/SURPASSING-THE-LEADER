"""Tests for selective exact-second search."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.getcwd())

from stl.solver.search import generate_candidates
from stl.solver.exact import solve_exact_finite_horizon
from stl.solver.search import (
    audit_against_full_width,
    selective_solve,
)
from stl.solver.tablebase import get_scenario
from stl.engine.actions import ACTION_SIZE


def _audit(name: str):
    scenario = get_scenario(name)
    return audit_against_full_width(
        scenario.game,
        scenario.half_round_horizon,
        scenario.config,
    )


def test_selective_solve_pins_forced_baku_overflow_to_plus_one():
    scenario = get_scenario("forced_baku_overflow_death")
    result = selective_solve(scenario.game, scenario.half_round_horizon, scenario.config)

    assert result.value_for_hal == pytest.approx(1.0)
    assert result.unresolved_probability == pytest.approx(0.0)
    assert result.dropper_strategy.sum() == pytest.approx(1.0)
    assert result.checker_strategy.sum() == pytest.approx(1.0)


def test_selective_solve_with_evaluator_returns_value_at_horizon_0():
    scenario = get_scenario("safe_budget_pressure_at_cylinder_240")

    class FixedEvaluator:
        def __call__(self, game):
            del game
            return 0.7, np.zeros(ACTION_SIZE), np.zeros(ACTION_SIZE)

    result = selective_solve(scenario.game, 0, scenario.config, evaluator=FixedEvaluator())

    assert result.value_for_hal == pytest.approx(0.7)
    assert result.unresolved_probability == pytest.approx(0.0)


def test_selective_solve_without_evaluator_preserves_cutoff_unresolved():
    scenario = get_scenario("safe_budget_pressure_at_cylinder_240")
    result = selective_solve(scenario.game, 0, scenario.config)

    assert result.value_for_hal == pytest.approx(0.0)
    assert result.unresolved_probability == pytest.approx(1.0)


def test_selective_solve_pins_forced_hal_overflow_to_minus_one():
    scenario = get_scenario("forced_hal_overflow_death")
    result = selective_solve(scenario.game, scenario.half_round_horizon, scenario.config)

    assert result.value_for_hal == pytest.approx(-1.0)


def test_selective_audit_zero_gap_on_pinned_overflow_pair():
    for name in ("forced_baku_overflow_death", "forced_hal_overflow_death"):
        audit = _audit(name)
        assert audit.value_gap == pytest.approx(0.0, abs=1e-9), name
        assert audit.candidate_joint_count <= audit.full_width_joint_count


def test_selective_audit_zero_gap_on_safe_budget_pair():
    for name in ("safe_budget_pressure_at_cylinder_241", "safe_budget_pressure_at_cylinder_240"):
        audit = _audit(name)
        assert audit.value_gap == pytest.approx(0.0, abs=1e-9), name


def test_selective_audit_zero_gap_on_cpr_pair():
    for name in ("cpr_degradation_fresh_referee", "cpr_degradation_fatigued_referee"):
        audit = _audit(name)
        assert audit.value_gap == pytest.approx(0.0, abs=1e-9), name


def test_selective_solve_uses_far_fewer_actions_than_full_width():
    scenario = get_scenario("forced_baku_overflow_death")
    audit = audit_against_full_width(
        scenario.game,
        scenario.half_round_horizon,
        scenario.config,
    )
    assert audit.candidate_joint_count * 4 <= audit.full_width_joint_count


def test_selective_solve_accepts_explicit_candidates_override():
    scenario = get_scenario("forced_baku_overflow_death")
    candidates = generate_candidates(scenario.game, scenario.config)
    result = selective_solve(
        scenario.game,
        scenario.half_round_horizon,
        scenario.config,
        candidates=candidates,
    )
    assert result.value_for_hal == pytest.approx(1.0)
    assert result.drop_seconds == candidates.drop_seconds
    assert result.check_seconds == candidates.check_seconds


def test_selective_solve_at_terminal_state_returns_pinned_value_without_candidates():
    scenario = get_scenario("forced_baku_overflow_death")
    # Run forward once so the game ends, then re-solve.
    full = solve_exact_finite_horizon(scenario.game, scenario.half_round_horizon, scenario.config)
    assert full.value_for_hal == pytest.approx(1.0)


def test_selective_solve_distribution_sums_to_one():
    for name in (
        "forced_baku_overflow_death",
        "forced_hal_overflow_death",
        "safe_budget_pressure_at_cylinder_241",
        "safe_budget_pressure_at_cylinder_240",
    ):
        scenario = get_scenario(name)
        result = selective_solve(scenario.game, scenario.half_round_horizon, scenario.config)
        assert result.dropper_strategy.sum() == pytest.approx(1.0), name
        assert result.checker_strategy.sum() == pytest.approx(1.0), name
