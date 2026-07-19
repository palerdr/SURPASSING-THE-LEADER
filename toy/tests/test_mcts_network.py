import numpy as np
import torch

from toy.mcts import (
    ToyMCTSConfig,
    conformance_report,
    make_exact_evaluator,
    make_tablebase_evaluator,
    mcts_search,
    run_conformance,
)
from toy.network import ToyPolicyValueNet, feature_dim, make_network_evaluator
from toy.rules import Bucket12Fixed50Rules
from toy.tablebase import build_tablebase


def test_mcts_result_has_normalized_role_policies_and_root_visits():
    rules = Bucket12Fixed50Rules(max_half_rounds=2)
    result = mcts_search(
        rules.initial_state(),
        2,
        make_exact_evaluator(rules),
        ToyMCTSConfig(rules=rules, iterations=16, max_depth=2),
        np.random.default_rng(1),
        np.random.default_rng(2),
    )
    assert result.root_visits == 16
    assert result.root_unique_cells_visited >= 1
    assert np.isclose(result.improved_dropper_policy.sum(), 1.0)
    assert np.isclose(result.improved_checker_policy.sum(), 1.0)
    assert result.improved_dropper_policy.shape == (12,)
    assert result.improved_checker_policy.shape == (12,)


def test_mcts_linear_policy_average_is_finite_at_large_budget():
    rules = Bucket12Fixed50Rules(max_half_rounds=1)
    result = mcts_search(
        rules.initial_state(),
        1,
        make_exact_evaluator(rules),
        ToyMCTSConfig(rules=rules, iterations=1024, max_depth=1),
        np.random.default_rng(7),
        np.random.default_rng(8),
    )
    assert np.all(np.isfinite(result.improved_dropper_policy))
    assert np.all(np.isfinite(result.improved_checker_policy))
    assert np.isclose(result.improved_dropper_policy.sum(), 1.0)


def test_mcts_conformance_report_is_deterministic():
    rules = Bucket12Fixed50Rules(max_half_rounds=2)
    records = run_conformance(
        rules,
        (rules.initial_state(),),
        horizons=(1,),
        budgets=(8,),
        seeds=(0, 1),
    )
    report_a = conformance_report(records)
    report_b = conformance_report(records)
    assert report_a == report_b
    assert len(report_a["records"]) == 2
    assert all(report_a["summary"]["gates"].values())


def test_tablebase_leaf_evaluator_round_trip():
    rules = Bucket12Fixed50Rules(max_half_rounds=1)
    tablebase = build_tablebase(rules, max_horizon=0)
    evaluator = make_tablebase_evaluator(tablebase, rules)
    value, drop, check = evaluator(rules.initial_state(), 0, rules)
    assert value == 0.0
    assert drop.shape == (12,)
    assert check.shape == (12,)


def test_network_shapes_and_legal_policy_mask():
    rules = Bucket12Fixed50Rules()
    model = ToyPolicyValueNet(feature_dim(rules), rules.action_size)
    features = torch.zeros((3, feature_dim(rules)), dtype=torch.float32)
    values, drop_logits, check_logits = model(features)
    assert values.shape == (3, 1)
    assert drop_logits.shape == (3, 12)
    assert check_logits.shape == (3, 12)
    evaluator = make_network_evaluator(model, rules)
    value, drop, check = evaluator(rules.initial_state(), 8, rules)
    assert -1.0 <= value <= 1.0
    assert np.isclose(drop.sum(), 1.0)
    assert np.isclose(check.sum(), 1.0)
    assert np.all(drop >= 0.0)
    assert np.all(check >= 0.0)
