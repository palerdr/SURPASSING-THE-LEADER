from pathlib import Path

import numpy as np
import pytest

from stl.learning.bellman import (
    BellmanRootSpec,
    build_bellman_bundle,
    bundle_training_records,
    candidate_action_representability,
    evaluate_bellman_gate,
    load_bellman_bundle,
    save_bellman_bundle,
    select_causal_roots,
)
from stl.learning.replay import validate_record
from stl.solver.mcts_conformance import BellmanLookupEvaluator
from stl.solver.tablebase import get_scenario


@pytest.fixture(scope="module")
def fatal_bundle():
    scenario = get_scenario("forced_baku_overflow_death")
    return build_bellman_bundle(
        [BellmanRootSpec(scenario.name, scenario.game, (("fixture", "fatal"),))]
    )


def test_balanced_selector_is_deterministic_and_covers_all_factor_levels():
    first = select_causal_roots(count=32, salt="test")
    repeated = select_causal_roots(count=32, salt="test")
    assert [row.name for row in first] == [row.name for row in repeated]
    strata = {pair for row in first for pair in row.strata}
    assert {value for name, value in strata if name == "checker"} == {"hal", "baku"}
    assert len({value for name, value in strata if name == "checker_cylinder"}) == 8


def test_fatal_fixture_closure_contains_every_cell_and_exact_backup(fatal_bundle):
    assert len(fatal_bundle.roots) == 1
    assert len(fatal_bundle.successors) == 61
    assert len(fatal_bundle.branches) == 3600
    root = fatal_bundle.roots[0]
    np.testing.assert_allclose(root.q3_for_hal, 1.0)
    assert root.value_h3 == pytest.approx(1.0)


def test_bellman_records_carry_separate_root_horizons(fatal_bundle):
    records, certificates = bundle_training_records(
        fatal_bundle, source_artifact="unit-test-bellman"
    )
    root_rows = [row for row in records if row.source.startswith("bellman_root")]
    assert {(row.source, row.value_horizon_half_rounds) for row in root_rows} == {
        ("bellman_root_h2", 2),
        ("bellman_root_h3", 3),
    }
    assert root_rows[0].features.tobytes() == root_rows[1].features.tobytes()
    assert sum(row.dropper_dist.sum() > 0 for row in root_rows) == 1
    assert len(certificates) == 1
    for row in records:
        validate_record(row)


def test_bellman_bundle_round_trips_without_pickle(tmp_path: Path, fatal_bundle):
    path = tmp_path / "closure.npz"
    save_bellman_bundle(fatal_bundle, path)
    loaded = load_bellman_bundle(path)
    assert loaded.plan_digest == fatal_bundle.plan_digest
    assert len(loaded.branches) == len(fatal_bundle.branches)
    np.testing.assert_array_equal(
        loaded.roots[0].q3_for_hal, fatal_bundle.roots[0].q3_for_hal
    )


def test_exact_lookup_passes_bellman_gate_and_candidate_preflight(fatal_bundle):
    evaluator = BellmanLookupEvaluator(fatal_bundle)

    def predict(game, *, horizon):
        return evaluator(game, value_horizon=horizon)

    report = evaluate_bellman_gate(fatal_bundle, predict)
    assert report["passed"]
    assert candidate_action_representability(fatal_bundle)["passed"]
