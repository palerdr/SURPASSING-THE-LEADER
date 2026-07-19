import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from dth.mcts import Evaluation
from dth.generate_dataset import TARGET_SCHEMA
from dth.network import DTHNetworkConfig, DTHPolicyValueNet
from dth.self_play import SelfPlayConfig, generate_self_play, write_self_play
from dth.solver import CHECKER_ACTIONS, DROPPER_ACTIONS, payoff, solve, solve_matrix
from dth.train import (
    DecisionRoot,
    ExactTargets,
    SelfPlayTargets,
    TargetRows,
    _batch_loss,
    decision_checkpoint_better,
    decision_guard_passes,
    decision_training_objective,
    evaluate_decision_roots,
    exact_frontier_indices,
    exact_frontier_replay_indices,
    grouped_state_split,
    load_self_play_targets,
    prepare_decision_loss_roots,
    soft_cross_entropy,
    train_exact,
)


class _UniformEvaluator:
    def __call__(self, state, horizon) -> Evaluation:
        del state, horizon
        policy = np.full(60, 1.0 / 60)
        return Evaluation(0.0, policy, policy)


def test_grouped_split_prevents_physical_state_leakage():
    states = np.asarray(
        [
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [1, 0, 0, 0],
            [2, 0, 0, 0],
            [3, 0, 0, 0],
        ],
        dtype=np.int16,
    )
    train, validation = grouped_state_split(
        states, validation_fraction=0.25, seed=4
    )
    train_states = {tuple(row) for row in states[train]}
    validation_states = {tuple(row) for row in states[validation]}
    assert train_states.isdisjoint(validation_states)
    assert sorted(np.concatenate((train, validation)).tolist()) == list(range(5))


def test_soft_cross_entropy_prefers_matching_logits():
    target = torch.tensor([[1.0, 0.0]])
    matching = soft_cross_entropy(torch.tensor([[5.0, -5.0]]), target)
    reversed_logits = soft_cross_entropy(torch.tensor([[-5.0, 5.0]]), target)
    assert matching.item() < reversed_logits.item()


def test_self_play_targets_remain_a_separate_audited_source(tmp_path):
    config = SelfPlayConfig(
        checkpoint=str(tmp_path / "missing.pt"),
        output=str(tmp_path / "replay"),
        games=1,
        max_half_rounds=1,
        simulations=0,
        root_warmup_cells=0,
        starts=[{"state": [240, 0, 0, 0], "horizon": 1}],
    )
    replay = generate_self_play(config, evaluator=_UniformEvaluator())
    npz_path, _ = write_self_play(replay, config.output)

    targets, summary = load_self_play_targets(
        npz_path,
        max_saddle_gap=2.0,
        min_coverage_fraction=0.0,
    )

    assert len(targets) == 1
    assert summary["source_type"] == "mcts_self_play"
    assert summary["accepted_rows"] == 1
    assert targets.value_weights.tolist() == [
        0.0 if replay.arrays["truncated"][0] else 1.0
    ]


def _policy_targets(rows: int) -> np.ndarray:
    return np.full((rows, 60), 1.0 / 60.0, dtype=np.float32)


def test_both_h5_orientations_have_122_unique_h4_frontier_rows():
    roots = [
        {"state": [239, 0, 0, 240], "horizon": 5},
        {"state": [0, 240, 239, 0], "horizon": 5},
    ]
    from dth.generate_dataset import live_successors

    children = sorted(
        live_successors((239, 0, 0, 240))
        | live_successors((0, 240, 239, 0))
    )
    rows = len(children)
    targets = ExactTargets(
        states=np.asarray(children, dtype=np.int16),
        horizons=np.full(rows, 4, dtype=np.uint8),
        values=np.zeros(rows, dtype=np.float32),
        drop_policies=_policy_targets(rows),
        check_policies=_policy_targets(rows),
        value_weights=np.ones(rows, dtype=np.float32),
        dataset_version="test",
        schema_version="test",
    )

    indices = exact_frontier_indices(targets, roots)

    assert rows == 122
    assert len(indices) == 122


def test_dual_frontier_groups_preserve_independent_repeat_counts():
    from dth.generate_dataset import live_successors

    h5_roots = [
        {"state": [239, 0, 0, 240], "horizon": 5},
        {"state": [0, 240, 239, 0], "horizon": 5},
    ]
    h4_roots = [{"state": [239, 0, 0, 240], "horizon": 4}]
    h4_children = sorted(
        live_successors((239, 0, 0, 240))
        | live_successors((0, 240, 239, 0))
    )
    h3_children = sorted(live_successors((239, 0, 0, 240)))
    states = h4_children + h3_children
    horizons = [4] * len(h4_children) + [3] * len(h3_children)
    rows = len(states)
    targets = ExactTargets(
        states=np.asarray(states, dtype=np.int16),
        horizons=np.asarray(horizons, dtype=np.uint8),
        values=np.zeros(rows, dtype=np.float32),
        drop_policies=_policy_targets(rows),
        check_policies=_policy_targets(rows),
        value_weights=np.ones(rows, dtype=np.float32),
        dataset_version="test",
        schema_version="test",
    )

    indices, summary = exact_frontier_replay_indices(
        targets,
        {
            "groups": [
                {"name": "h5", "roots": h5_roots, "repeats": 32},
                {"name": "h4", "roots": h4_roots, "repeats": 64},
            ]
        },
    )

    assert len(h4_children) == 122
    assert len(h3_children) == 61
    assert len(indices) == 7_808
    assert summary["rows"] == 183
    assert [group["training_rows"] for group in summary["groups"]] == [
        3_904,
        3_904,
    ]


def test_truncated_rows_keep_policy_loss_but_mask_value_loss():
    targets = SelfPlayTargets(
        states=np.zeros((1, 4), dtype=np.int16),
        horizons=np.ones(1, dtype=np.uint8),
        values=np.zeros(1, dtype=np.float32),
        drop_policies=_policy_targets(1),
        check_policies=_policy_targets(1),
        value_weights=np.zeros(1, dtype=np.float32),
    )
    rows = TargetRows(targets, np.asarray([0]), horizon_scale=3.0)
    loader = DataLoader(rows, batch_size=1)
    model = DTHPolicyValueNet(DTHNetworkConfig(hidden_width=4, hidden_layers=1))
    for parameter in model.parameters():
        torch.nn.init.zeros_(parameter)

    value_only = _batch_loss(
        model,
        next(iter(loader)),
        policy_weight=0.0,
        device=torch.device("cpu"),
    )
    policy_only = _batch_loss(
        model,
        next(iter(loader)),
        policy_weight=1.0,
        device=torch.device("cpu"),
    )

    assert torch.isfinite(value_only)
    assert value_only.item() == 0.0
    assert policy_only.item() > 0.0


def test_terminal_rows_keep_full_value_loss():
    targets = SelfPlayTargets(
        states=np.zeros((1, 4), dtype=np.int16),
        horizons=np.ones(1, dtype=np.uint8),
        values=np.ones(1, dtype=np.float32),
        drop_policies=_policy_targets(1),
        check_policies=_policy_targets(1),
        value_weights=np.ones(1, dtype=np.float32),
    )
    rows = TargetRows(targets, np.asarray([0]), horizon_scale=3.0)
    model = DTHPolicyValueNet(DTHNetworkConfig(hidden_width=4, hidden_layers=1))
    for parameter in model.parameters():
        torch.nn.init.zeros_(parameter)

    loss = _batch_loss(
        model,
        next(iter(DataLoader(rows, batch_size=1))),
        policy_weight=0.0,
        device=torch.device("cpu"),
    )

    assert loss.item() == 1.0


def test_decision_selection_is_gap_first_and_can_retain_epoch_zero():
    baseline = {"max_saddle_gap": 0.2, "max_value_error": 0.1}
    incumbent = baseline
    better_value_only = {"max_saddle_gap": 0.2, "max_value_error": 0.01}
    better_gap = {"max_saddle_gap": 0.19, "max_value_error": 0.2}

    assert not decision_checkpoint_better(
        better_value_only,
        incumbent,
        baseline,
        candidate_validation_loss=0.01,
        incumbent_validation_loss=0.1,
        minimum_gap_improvement=1e-6,
        tie_tolerance=1e-6,
    )
    assert decision_checkpoint_better(
        better_gap,
        incumbent,
        baseline,
        candidate_validation_loss=1.0,
        incumbent_validation_loss=0.1,
        minimum_gap_improvement=1e-6,
        tie_tolerance=1e-6,
    )


def test_decision_guard_rejects_h4_regression():
    baseline = {"max_saddle_gap": 0.13, "max_value_error": 0.02}
    regressed = {"max_saddle_gap": 0.14, "max_value_error": 0.02}
    assert not decision_guard_passes(regressed, baseline, tolerance=1e-6)


def test_h5_decision_audit_matches_an_exact_zero_child_matrix():
    state = (239, 0, 0, 240)
    from dth.train import approximate_payoff_from_network

    model = DTHPolicyValueNet(DTHNetworkConfig(hidden_width=4, hidden_layers=1))
    for parameter in model.parameters():
        torch.nn.init.zeros_(parameter)
    matrix = approximate_payoff_from_network(
        model,
        state,
        5,
        device=torch.device("cpu"),
    )
    value, _, _ = solve_matrix(matrix)
    root = DecisionRoot(state, 5, value, matrix)

    result = evaluate_decision_roots(model, [root], device=torch.device("cpu"))

    assert result["max_saddle_gap"] <= 1e-6
    assert result["max_value_error"] <= 1e-6


def test_decision_loss_is_zero_for_an_exact_saddle_and_positive_when_exploitable():
    model = DTHPolicyValueNet(DTHNetworkConfig(hidden_width=4, hidden_layers=1))
    for parameter in model.parameters():
        torch.nn.init.zeros_(parameter)
    exact_matrix = np.zeros((60, 60), dtype=np.float64)
    exact_root = DecisionRoot((0, 0, 0, 0), 1, 0.0, exact_matrix)
    prepared = prepare_decision_loss_roots(
        model, [exact_root], device=torch.device("cpu")
    )

    zero_loss, zero_metrics = decision_training_objective(
        model,
        prepared,
        saddle_gap_weight=1.0,
        matrix_weight=0.0,
        matrix_top_k=16,
    )
    assert zero_loss.item() == 0.0
    assert zero_metrics["mean_saddle_gap"].item() == 0.0

    exploitable_matrix = np.zeros((60, 60), dtype=np.float64)
    exploitable_matrix[0, :] = 1.0
    exploitable_root = DecisionRoot(
        (0, 0, 0, 0), 1, 1.0, exploitable_matrix
    )
    prepared = prepare_decision_loss_roots(
        model, [exploitable_root], device=torch.device("cpu")
    )
    _, metrics = decision_training_objective(
        model,
        prepared,
        saddle_gap_weight=1.0,
        matrix_weight=0.0,
        matrix_top_k=16,
    )
    assert metrics["mean_saddle_gap"].item() > 0.9


def test_decision_matrix_loss_backpropagates_through_child_values():
    state = (239, 0, 0, 240)
    exact_matrix = solve(state, 2)
    root = DecisionRoot(state, 2, exact_matrix.value, payoff(state, 2))
    model = DTHPolicyValueNet(DTHNetworkConfig(hidden_width=4, hidden_layers=1))
    prepared = prepare_decision_loss_roots(model, [root], device=torch.device("cpu"))

    loss, metrics = decision_training_objective(
        model,
        prepared,
        saddle_gap_weight=0.0,
        matrix_weight=1.0,
        matrix_top_k=32,
    )
    loss.backward()

    assert torch.isfinite(loss)
    assert metrics["mean_matrix_top_k_error"].item() > 0.0
    assert model.value_head.weight.grad is not None
    assert torch.isfinite(model.value_head.weight.grad).all()


def test_decision_loss_uses_the_worst_root_instead_of_the_mean():
    model = DTHPolicyValueNet(DTHNetworkConfig(hidden_width=4, hidden_layers=1))
    for parameter in model.parameters():
        torch.nn.init.zeros_(parameter)
    hard_matrix = np.zeros((60, 60), dtype=np.float64)
    hard_matrix[0, :] = 1.0
    easier_matrix = 0.5 * hard_matrix
    roots = [
        DecisionRoot((0, 0, 0, 0), 1, 1.0, hard_matrix),
        DecisionRoot((1, 0, 0, 0), 1, 0.5, easier_matrix),
    ]
    prepared = prepare_decision_loss_roots(model, roots, device=torch.device("cpu"))

    loss, metrics = decision_training_objective(
        model,
        prepared,
        saddle_gap_weight=1.0,
        matrix_weight=0.0,
        matrix_top_k=16,
    )

    assert loss.item() == pytest.approx(metrics["worst_saddle_gap"].item())
    assert loss.item() > metrics["mean_saddle_gap"].item()


def test_root_value_preservation_loss_backpropagates_through_child_values():
    state = (239, 0, 0, 240)
    exact = solve(state, 2)
    root = DecisionRoot(
        state,
        2,
        exact.value,
        payoff(state, 2),
        value_preservation_weight=1.0,
    )
    model = DTHPolicyValueNet(DTHNetworkConfig(hidden_width=4, hidden_layers=1))
    prepared = prepare_decision_loss_roots(model, [root], device=torch.device("cpu"))

    loss, metrics = decision_training_objective(
        model,
        prepared,
        saddle_gap_weight=0.0,
        matrix_weight=0.0,
        matrix_top_k=16,
    )
    loss.backward()

    assert loss.item() == pytest.approx(metrics["worst_root_value_error"].item())
    assert loss.item() > 0.0
    assert model.value_head.weight.grad is not None
    assert torch.isfinite(model.value_head.weight.grad).all()


def test_decision_training_retains_epoch_zero_without_gap_improvement(tmp_path):
    states = [(240, 0, 0, 0), (0, 0, 0, 0)]
    solutions = [solve(state, 1) for state in states]
    dataset = tmp_path / "exact.npz"
    np.savez_compressed(
        dataset,
        states=np.asarray(states, dtype=np.int16),
        horizons=np.ones(2, dtype=np.uint8),
        values=np.asarray([item.value for item in solutions], dtype=np.float32),
        drop_policies=np.asarray(
            [item.drop_policy for item in solutions], dtype=np.float32
        ),
        check_policies=np.asarray(
            [item.check_policy for item in solutions], dtype=np.float32
        ),
        saddle_gaps=np.asarray(
            [item.saddle_gap for item in solutions], dtype=np.float32
        ),
        drop_actions=np.asarray(DROPPER_ACTIONS, dtype=np.int16),
        check_actions=np.asarray(CHECKER_ACTIONS, dtype=np.int16),
        dataset_version=np.asarray("decision-test"),
        schema_version=np.asarray(TARGET_SCHEMA),
    )
    report = train_exact(
        {
            "dataset": str(dataset),
            "output_dir": str(tmp_path / "checkpoint"),
            "seed": 4,
            "device": "cpu",
            "model": {
                "hidden_width": 4,
                "hidden_layers": 1,
                "horizon_scale": 3.0,
            },
            "training": {
                "epochs": 1,
                "batch_size": 2,
                "learning_rate": 0.0,
                "weight_decay": 0.0,
                "validation_fraction": 0.5,
                "policy_weight": 0.1,
                "selection_metric": "decision",
                "patience": 0,
                "minimum_delta": 1e-6,
                "log_every": 1,
            },
            "decision_selection": {
                "exact_targets": str(dataset),
                "roots": [{"state": list(states[0]), "horizon": 1}],
                "guard_roots": [{"state": list(states[1]), "horizon": 1}],
                "guard_tolerance": 1e-6,
                "minimum_gap_improvement": 1e-6,
                "tie_tolerance": 1e-6,
            },
            "audit": {"max_rows_per_horizon": 1},
        }
    )

    assert report["best_epoch"] == 0
    assert not report["decision_selection"]["improved_from_epoch_zero"]


def test_decision_training_can_retain_different_width_baseline(tmp_path):
    states = [(240, 0, 0, 0), (0, 0, 0, 0)]
    solutions = [solve(state, 1) for state in states]
    dataset = tmp_path / "exact.npz"
    np.savez_compressed(
        dataset,
        states=np.asarray(states, dtype=np.int16),
        horizons=np.ones(2, dtype=np.uint8),
        values=np.asarray([item.value for item in solutions], dtype=np.float32),
        drop_policies=np.asarray(
            [item.drop_policy for item in solutions], dtype=np.float32
        ),
        check_policies=np.asarray(
            [item.check_policy for item in solutions], dtype=np.float32
        ),
        saddle_gaps=np.asarray(
            [item.saddle_gap for item in solutions], dtype=np.float32
        ),
        drop_actions=np.asarray(DROPPER_ACTIONS, dtype=np.int16),
        check_actions=np.asarray(CHECKER_ACTIONS, dtype=np.int16),
        dataset_version=np.asarray("external-baseline-test"),
        schema_version=np.asarray(TARGET_SCHEMA),
    )
    baseline_config = DTHNetworkConfig(hidden_width=4, hidden_layers=1)
    baseline_model = DTHPolicyValueNet(baseline_config)
    baseline_checkpoint = tmp_path / "baseline.pt"
    torch.save(
        {
            "state_dict": baseline_model.state_dict(),
            "model_config": baseline_config.to_dict(),
        },
        baseline_checkpoint,
    )

    report = train_exact(
        {
            "dataset": str(dataset),
            "output_dir": str(tmp_path / "checkpoint"),
            "seed": 4,
            "device": "cpu",
            "model": {
                "hidden_width": 8,
                "hidden_layers": 1,
                "horizon_scale": 3.0,
            },
            "training": {
                "epochs": 1,
                "batch_size": 2,
                "learning_rate": 0.0,
                "weight_decay": 0.0,
                "validation_fraction": 0.5,
                "policy_weight": 0.1,
                "selection_metric": "decision",
                "patience": 0,
                "minimum_delta": 1e-6,
                "log_every": 1,
            },
            "decision_selection": {
                "exact_targets": str(dataset),
                "baseline_checkpoint": str(baseline_checkpoint),
                "roots": [{"state": list(states[0]), "horizon": 1}],
                "guard_roots": [{"state": list(states[1]), "horizon": 1}],
                "guard_tolerance": 1e-6,
                "minimum_gap_improvement": 2.0,
                "tie_tolerance": 1e-6,
            },
            "audit": {"max_rows_per_horizon": 1},
        }
    )

    selected = torch.load(report["checkpoint"], weights_only=False)
    assert report["best_epoch"] == 0
    assert report["model"]["hidden_width"] == 8
    assert report["selected_model"]["hidden_width"] == 4
    assert selected["model_config"]["hidden_width"] == 4
    assert report["decision_selection"]["baseline_checkpoint"] == str(
        baseline_checkpoint
    )
