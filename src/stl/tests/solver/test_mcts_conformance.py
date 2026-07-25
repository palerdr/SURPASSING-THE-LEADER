from __future__ import annotations

import json

import numpy as np
import pytest

from stl.solver.mcts_conformance import (
    MCTSConformanceGateThresholds,
    certify_strict_unique_pure_saddle,
    conformance_report_digest,
    evaluate_conformance_gate,
    lift_policy_to_full_actions,
    run_mcts_conformance,
    unique_equilibrium_policy_tv,
    write_conformance_report,
)
from stl.solver.tablebase import (
    forced_baku_overflow_death,
    forced_hal_overflow_death,
)


@pytest.fixture(scope="module")
def smoke_report():
    return run_mcts_conformance(
        (forced_hal_overflow_death(), forced_baku_overflow_death()),
        budgets=(4, 8),
        seeds=(0, 1),
        action_mode="candidate",
    )


def test_bounded_smoke_report_has_frozen_order_and_required_metrics(smoke_report):
    assert smoke_report.scenario_names == (
        "forced_baku_overflow_death",
        "forced_hal_overflow_death",
    )
    assert smoke_report.budgets == (4, 8)
    assert smoke_report.seeds == (0, 1)
    assert len(smoke_report.records) == 8
    assert [
        (record.scenario_name, record.budget, record.seed)
        for record in smoke_report.records
    ] == sorted(
        (record.scenario_name, record.budget, record.seed)
        for record in smoke_report.records
    )

    for record in smoke_report.records:
        assert record.action_mode == "candidate"
        assert record.exact_dropper_actions == 60
        assert record.exact_checker_actions == 60
        # Near-overflow candidate states intentionally use the full exact grid;
        # the omitted-action audit found a profitable cyclic response chain.
        assert record.search_dropper_actions == record.exact_dropper_actions
        assert record.search_checker_actions == record.exact_checker_actions
        assert record.absolute_value_error == pytest.approx(0.0)
        assert record.full_width_saddle_gap == pytest.approx(0.0)
        assert record.dropper_normalized_entropy >= 0.0
        assert record.checker_normalized_entropy >= 0.0
        assert 0 < record.root_unique_cells_visited <= record.root_visits
        # Constant payoff matrices are maximally degenerate, not unique.
        assert record.unique_equilibrium_certified is False
        assert record.dropper_policy_tv is None
        assert record.checker_policy_tv is None


def test_smoke_gate_can_use_explicit_small_budgets(smoke_report):
    gate = evaluate_conformance_gate(
        smoke_report,
        MCTSConformanceGateThresholds(
            evaluation_budget=8,
            comparison_budget=4,
            median_absolute_value_error=1e-12,
            p95_absolute_value_error=1e-12,
            maximum_saddle_gap=1e-12,
            maximum_root_value_std=1e-12,
            maximum_fixture_median_worsening=1e-12,
        ),
    )
    assert gate.passed
    assert gate.failures == ()
    assert gate.median_absolute_value_error == pytest.approx(0.0)
    assert gate.maximum_saddle_gap == pytest.approx(0.0)

    missing_frozen_budgets = evaluate_conformance_gate(smoke_report)
    assert not missing_frozen_budgets.passed
    assert any("1024" in failure for failure in missing_frozen_budgets.failures)
    assert any("256" in failure for failure in missing_frozen_budgets.failures)


def test_report_and_writer_are_deterministic(tmp_path):
    first = run_mcts_conformance(
        (forced_baku_overflow_death(),),
        budgets=(4,),
        seeds=(7,),
    )
    second = run_mcts_conformance(
        (forced_baku_overflow_death(),),
        budgets=(4,),
        seeds=(7,),
    )
    assert first == second
    assert conformance_report_digest(first) == conformance_report_digest(second)

    first_path = tmp_path / "first.json"
    second_path = tmp_path / "second.json"
    first_digest = write_conformance_report(first, first_path)
    second_digest = write_conformance_report(second, second_path)
    assert first_digest == second_digest
    assert first_path.read_bytes() == second_path.read_bytes()
    assert (
        json.loads(first_path.read_text(encoding="utf-8"))["report_sha256"]
        == first_digest
    )


def test_candidate_policy_is_lifted_without_reordering_or_hidden_mass():
    lifted = lift_policy_to_full_actions(
        search_actions=(1, 60),
        search_policy=np.array([0.25, 0.75]),
        full_actions=(1, 2, 3, 60),
    )
    np.testing.assert_allclose(lifted, np.array([0.25, 0.0, 0.0, 0.75]))
    with pytest.raises(ValueError, match="absent"):
        lift_policy_to_full_actions((61,), (1.0,), (1, 2, 3, 60))


def test_policy_tv_requires_strict_unique_pure_saddle_certificate():
    # Hal is the row/dropper maximizer.  (row=0, column=0) is strict in both
    # directions: 0 > -1 down its column and 0 < 2 across its row.
    strict = np.array([[0.0, 2.0], [-1.0, 1.0]])
    assert certify_strict_unique_pure_saddle(strict, hal_is_dropper=True) == (0, 0)
    tv = unique_equilibrium_policy_tv(
        strict,
        np.array([0.75, 0.25]),
        np.array([0.5, 0.5]),
        hal_is_dropper=True,
    )
    assert tv == pytest.approx((0.25, 0.5))

    degenerate = np.ones((2, 2), dtype=np.float64)
    assert certify_strict_unique_pure_saddle(degenerate, hal_is_dropper=True) is None
    assert (
        unique_equilibrium_policy_tv(
            degenerate,
            np.array([0.5, 0.5]),
            np.array([0.5, 0.5]),
            hal_is_dropper=True,
        )
        is None
    )


def test_full_width_smoke_reports_literal_action_mode():
    report = run_mcts_conformance(
        (forced_baku_overflow_death(),),
        budgets=(2,),
        seeds=(0,),
        action_mode="full_width",
    )
    record = report.records[0]
    assert record.action_mode == "full_width"
    assert record.search_dropper_actions == record.exact_dropper_actions == 60
    assert record.search_checker_actions == record.exact_checker_actions == 60


def test_conformance_can_inject_a_candidate_leaf_evaluator():
    calls = 0

    class Evaluator:
        def __call__(self, _game):
            nonlocal calls
            calls += 1
            return 0.25

    report = run_mcts_conformance(
        (forced_baku_overflow_death(),),
        budgets=(2,),
        seeds=(0,),
        evaluator_factory=Evaluator,
    )
    assert len(report.records) == 1
    assert calls > 0
