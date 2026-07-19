import numpy as np
import pytest
import dth.mcts as pure_mcts

from dth.mcts import (
    AnchoredLeafEvaluator,
    Evaluation,
    ExactLeafEvaluator,
    ExactTargetStore,
    ExactValueEvaluator,
    MCTSConfig,
    expand,
    make_node,
    mcts_search,
    mcts_search_ladder,
    payoff_from_exact_targets,
    refresh_policies,
    simulate,
    summarize_audit,
)
from dth.solver import payoff, solve, solve_matrix


class _ConstantEvaluator:
    def __call__(self, state, horizon) -> Evaluation:
        del state, horizon
        policy = np.full(60, 1.0 / 60)
        return Evaluation(0.25, policy, policy)


def test_exact_evaluator_nodes_remain_solved_leaves() -> None:
    state = (0, 0, 0, 0)
    node = make_node(state, 1)
    table = {(state, 1): node}
    evaluator = ExactValueEvaluator()
    config = MCTSConfig(simulations=0)
    action_rng = np.random.default_rng(1)
    chance_rng = np.random.default_rng(2)

    first = simulate(node, table, evaluator, config, action_rng, chance_rng)
    second = simulate(node, table, evaluator, config, action_rng, chance_rng)

    assert second == pytest.approx(first)
    assert node.exact_value == pytest.approx(first)
    assert node.visits == 0


def test_anchored_evaluator_excludes_root_and_certifies_child() -> None:
    root = (0, 0, 0, 0)
    child = (1, 0, 0, 0)
    evaluator = AnchoredLeafEvaluator(
        _ConstantEvaluator(),
        ExactTargetStore({(root, 2): 0.75, (child, 1): -0.5}),
        root,
        2,
    )

    root_evaluation = evaluator(root, 2)
    child_evaluation = evaluator(child, 1)

    assert root_evaluation.value == pytest.approx(0.25)
    assert not root_evaluation.exact
    assert child_evaluation.value == pytest.approx(-0.5)
    assert child_evaluation.exact


def test_anchored_evaluator_can_limit_exact_horizons() -> None:
    root = (0, 0, 0, 0)
    child = (1, 0, 0, 0)
    evaluator = AnchoredLeafEvaluator(
        _ConstantEvaluator(),
        ExactTargetStore({(child, 1): -0.5, (child, 2): -0.75}),
        root,
        3,
        frozenset({1}),
    )

    assert evaluator(child, 1).exact
    assert not evaluator(child, 2).exact


def test_exact_target_store_loads_state_horizon_values(tmp_path) -> None:
    path = tmp_path / "exact_targets.npz"
    np.savez_compressed(
        path,
        states=np.asarray([[1, 2, 3, 4]], dtype=np.int16),
        horizons=np.asarray([2], dtype=np.uint8),
        values=np.asarray([-0.25], dtype=np.float32),
    )

    store = ExactTargetStore.load(path)

    assert store.values[((1, 2, 3, 4), 2)] == pytest.approx(-0.25)


def test_exact_target_store_reconstructs_horizon_one_payoff() -> None:
    state = (0, 0, 0, 0)

    reconstructed = payoff_from_exact_targets(
        state,
        1,
        ExactTargetStore({}),
    )

    assert reconstructed == pytest.approx(payoff(state, 1))


def test_horizon_one_full_width_root_recovers_exact_value() -> None:
    state = (240, 0, 0, 0)
    evaluator = ExactLeafEvaluator(state, 1)
    result = mcts_search(
        state,
        1,
        evaluator,
        MCTSConfig(simulations=0, root_warmup_cells=3600),
        np.random.default_rng(1),
        np.random.default_rng(2),
    )

    assert result.value == pytest.approx(0.5)
    assert result.unique_cells == 3600
    assert result.warmup_visits == 3600


def test_full_width_expected_chance_backup_is_seed_invariant() -> None:
    state = (0, 0, 0, 0)
    evaluator = ExactLeafEvaluator(state, 1)
    config = MCTSConfig(simulations=0, root_warmup_cells=3600)

    first = mcts_search(
        state,
        1,
        evaluator,
        config,
        np.random.default_rng(1),
        np.random.default_rng(2),
    )
    second = mcts_search(
        state,
        1,
        evaluator,
        config,
        np.random.default_rng(3),
        np.random.default_rng(4),
    )

    assert first.value == pytest.approx(solve(state, 1).value)
    assert second.value == pytest.approx(first.value)
    assert second.mean_q_drop_policy == pytest.approx(first.mean_q_drop_policy)
    assert second.mean_q_check_policy == pytest.approx(first.mean_q_check_policy)


def test_max_depth_keeps_network_frontier_fixed() -> None:
    state = (0, 0, 0, 0)
    result = mcts_search(
        state,
        2,
        _ConstantEvaluator(),
        MCTSConfig(simulations=8, max_depth=1),
        np.random.default_rng(1),
        np.random.default_rng(2),
    )

    assert result.root_visits == 8


def test_budget_ladder_matches_one_shared_search_prefix() -> None:
    state = (0, 0, 0, 0)
    config = MCTSConfig(simulations=8, max_depth=1)
    ladder = mcts_search_ladder(
        state,
        2,
        _ConstantEvaluator(),
        config,
        np.random.default_rng(1),
        np.random.default_rng(2),
        budgets=[0, 4, 8],
    )
    standalone = mcts_search(
        state,
        2,
        _ConstantEvaluator(),
        config,
        np.random.default_rng(1),
        np.random.default_rng(2),
    )

    assert ladder[0].search_visits == 0
    assert ladder[4].search_visits == 4
    assert ladder[8].joint_visits == pytest.approx(standalone.joint_visits)
    assert ladder[8].drop_policy == pytest.approx(standalone.drop_policy)
    assert ladder[8].check_policy == pytest.approx(standalone.check_policy)


def test_internal_backup_returns_matrix_value_not_selected_outcome() -> None:
    state = (0, 0, 0, 0)
    node = make_node(state, 1)
    matrix = payoff(state, 1)
    node.expanded = True
    node.prior_value = 0.0
    node.joint_visits.fill(1)
    node.joint_value_sum[:] = matrix
    node.visits = 3600
    config = MCTSConfig(simulations=0)
    refresh_policies(node, config)
    exact_value = solve_matrix(matrix)[0]

    returned = simulate(
        node,
        {(state, 1): node},
        _ConstantEvaluator(),
        config,
        np.random.default_rng(7),
        np.random.default_rng(8),
    )

    assert returned == pytest.approx(exact_value)


def test_exploration_lp_failure_falls_back_to_mean_policy(monkeypatch) -> None:
    state = (0, 0, 0, 0)
    node = make_node(state, 1)
    node.expanded = True
    node.joint_visits.fill(1)
    node.joint_value_sum[:] = payoff(state, 1)
    node.visits = 3600
    real_solve = pure_mcts.solve_matrix
    calls = 0

    def fail_exploration(matrix):
        nonlocal calls
        calls += 1
        if calls > 1:
            raise RuntimeError("LP saddle gap too large: test")
        return real_solve(matrix)

    monkeypatch.setattr(pure_mcts, "solve_matrix", fail_exploration)
    refresh_policies(node, MCTSConfig(simulations=0, exploration_scale=0.0))

    assert node.selection_drop == pytest.approx(node.mean_q_drop)
    assert node.selection_check == pytest.approx(node.mean_q_check)


def test_internal_warmup_is_scoped_below_the_root() -> None:
    result = mcts_search(
        (0, 0, 0, 0),
        2,
        _ConstantEvaluator(),
        MCTSConfig(
            simulations=0,
            root_warmup_cells=1,
            internal_warmup_cells=2,
            max_depth=2,
        ),
        np.random.default_rng(1),
        np.random.default_rng(2),
    )

    assert result.root_visits == 1
    assert result.warmup_visits == 1


def test_internal_warmup_can_be_limited_by_remaining_horizon() -> None:
    state = (0, 0, 0, 0)
    evaluator = _ConstantEvaluator()
    node = make_node(state, 2)
    table = {(state, 2): node}
    config = MCTSConfig(
        simulations=0,
        internal_warmup_cells=2,
        internal_warmup_horizons=frozenset({2}),
        max_depth=2,
    )
    expand(node, evaluator, config)

    simulate(
        node,
        table,
        evaluator,
        config,
        np.random.default_rng(1),
        np.random.default_rng(2),
        forced_joint_action=(0, 0),
    )

    child_nodes = [child for key, child in table.items() if key != (state, 2)]
    assert len(child_nodes) == 1
    assert child_nodes[0].remaining_horizon == 1
    assert child_nodes[0].visits == 0


def test_audit_summary_tracks_mean_q_gap_and_coverage() -> None:
    record = {
        "evaluator": "exact_leaf",
        "budget": 0,
        "warmup_cells": 3600,
        "state": [240, 0, 0, 0],
        "horizon": 1,
        "mcts_value": 0.5,
        "value_error": 0.0,
        "saddle_gap": 0.02,
        "mean_q_saddle_gap": 0.01,
        "coverage_fraction": 1.0,
        "unique_cells": 3600,
    }

    summary = summarize_audit([record])["exact_leaf"]["0"]

    assert summary["median_mean_q_saddle_gap"] == pytest.approx(0.01)
    assert summary["mean_coverage_fraction"] == pytest.approx(1.0)
