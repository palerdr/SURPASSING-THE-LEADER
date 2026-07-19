import numpy as np

from stl.learning.bellman import BellmanRootSpec, build_bellman_bundle
from stl.solver.mcts_conformance import run_bounded_mcts_conformance
from stl.solver.search import MCTSConfig, mcts_search
from stl.solver.tablebase import get_scenario


class _RecordingEvaluator:
    supports_horizon_context = True

    def __init__(self):
        self.horizons = []

    def __call__(self, game, *, value_horizon=None):
        from stl.solver.search import uniform_policy_for_current_roles

        self.horizons.append(value_horizon)
        drop, check = uniform_policy_for_current_roles(game)
        return 0.0, drop, check


def test_depth_one_search_queries_v3_root_and_v2_children():
    scenario = get_scenario("safe_budget_pressure_at_cylinder_239")
    evaluator = _RecordingEvaluator()
    mcts_search(
        scenario.game,
        MCTSConfig(
            iterations=4,
            exploration_c=1.0,
            max_depth=1,
            root_value_horizon=3,
        ),
        evaluator,
        np.random.default_rng(0),
        scenario.config,
    )
    assert 3 in evaluator.horizons
    assert 2 in evaluator.horizons


def test_bounded_conformance_uses_v3_oracle_with_v2_lookup():
    scenario = get_scenario("forced_baku_overflow_death")
    bundle = build_bellman_bundle(
        [BellmanRootSpec(scenario.name, scenario.game, ())]
    )
    report = run_bounded_mcts_conformance(
        bundle, budgets=(4,), seeds=(0,)
    )
    assert report.schema_version == "stl.mcts_bounded_conformance.v1"
    assert report.records[0].exact_value_for_hal == 1.0
