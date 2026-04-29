"""Tests for MCTS Step 3: exploration bonus and selection."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from environment.cfr.mcts import (
    MCTSNode,
    _exploration_augmented_matrix,
    _select_joint_action,
    make_node,
)
from environment.cfr.tactical_scenarios import (
    forced_baku_overflow_death,
    forced_hal_overflow_death,
)


def test_exploration_bonus_at_fresh_node_is_zero_so_q_explore_equals_q():
    # N_node = 0 → sqrt(N_node) = 0 → U is the zero matrix → Q_explore == Q.
    scenario = forced_baku_overflow_death()
    node = make_node(scenario.game, scenario.config)

    q_explore = _exploration_augmented_matrix(node, exploration_c=1.0)

    assert q_explore.shape == node.Q.shape
    np.testing.assert_array_equal(q_explore, node.Q)


def test_exploration_bonus_shape_matches_node_matrix_shape():
    scenario = forced_baku_overflow_death()
    node = make_node(scenario.game, scenario.config)
    D = len(node.drop_seconds)
    C = len(node.check_seconds)

    # Manually mature the node so the bonus is non-trivial.
    node.N_node = 100

    q_explore = _exploration_augmented_matrix(node, exploration_c=1.0)
    assert q_explore.shape == (D, C)


def test_exploration_bonus_shrinks_for_well_visited_cells():
    # With N_node=100 and N_cell[0,0]=99 (heavily visited) vs N_cell[0,1]=0
    # (untouched), cell (0,0)'s bonus is √100 / (1 + 99) = 0.1×prior×c, while
    # cell (0,1)'s is √100 / (1 + 0) = 10×prior×c. Ratio 100×.
    scenario = forced_baku_overflow_death()
    node = make_node(scenario.game, scenario.config)
    assert len(node.check_seconds) >= 2  # need at least two columns for the test

    node.N_node = 100
    node.N_cell[0, 0] = 99
    node.N_cell[0, 1] = 0

    q_explore = _exploration_augmented_matrix(node, exploration_c=1.0)

    assert q_explore[0, 1] > q_explore[0, 0]
    # The 100x ratio is deterministic from the formula.
    ratio = q_explore[0, 1] / q_explore[0, 0]
    assert ratio == pytest.approx(100.0)


def test_exploration_constant_scales_bonus_linearly():
    # U is linear in c, so doubling c doubles U (with Q=0 it doubles Q_explore).
    scenario = forced_baku_overflow_death()
    node = make_node(scenario.game, scenario.config)
    node.N_node = 50

    q_c1 = _exploration_augmented_matrix(node, exploration_c=1.0)
    q_c2 = _exploration_augmented_matrix(node, exploration_c=2.0)

    np.testing.assert_allclose(q_c2, 2.0 * q_c1)


def test_select_joint_action_returns_indices_within_matrix_bounds():
    scenario = forced_baku_overflow_death()
    node = make_node(scenario.game, scenario.config)
    D = len(node.drop_seconds)
    C = len(node.check_seconds)
    rng = np.random.default_rng(42)

    for _ in range(50):
        d_idx, c_idx = _select_joint_action(node, exploration_c=1.0, rng=rng)
        assert isinstance(d_idx, int)
        assert isinstance(c_idx, int)
        assert 0 <= d_idx < D
        assert 0 <= c_idx < C


def test_select_joint_action_is_deterministic_under_same_seed():
    scenario = forced_baku_overflow_death()
    node = make_node(scenario.game, scenario.config)

    rng_a = np.random.default_rng(123)
    rng_b = np.random.default_rng(123)

    samples_a = [_select_joint_action(node, exploration_c=1.0, rng=rng_a) for _ in range(20)]
    samples_b = [_select_joint_action(node, exploration_c=1.0, rng=rng_b) for _ in range(20)]

    assert samples_a == samples_b


def test_select_joint_action_avoids_dropper_rows_with_strongly_negative_q():
    # Plant Q[0, :] = -1 (every checker response is a loss for the dropper)
    # and zero elsewhere. With N_node large and a small exploration constant,
    # the Q signal dominates the U bonus and the dropper-as-Hal should put
    # little probability on row 0.
    scenario = forced_baku_overflow_death()
    node = make_node(scenario.game, scenario.config)
    assert node.hal_is_dropper is True

    D = len(node.drop_seconds)
    node.Q[0, :] = -1.0
    node.N_node = 10_000

    rng = np.random.default_rng(0)
    counts = np.zeros(D, dtype=np.int64)
    n_samples = 500
    for _ in range(n_samples):
        d_idx, _ = _select_joint_action(node, exploration_c=0.01, rng=rng)
        counts[d_idx] += 1

    # Row 0 is dominated; equilibrium puts ~zero probability on it.
    assert counts[0] < n_samples * 0.05


def test_select_joint_action_works_when_hal_is_checker():
    # forced_hal_overflow_death sets up Hal as the checker (current_half=2,
    # first_dropper=hal). This exercises the `else` branch in selection where
    # the matrix is negated/transposed differently.
    scenario = forced_hal_overflow_death()
    node = make_node(scenario.game, scenario.config)
    assert node.hal_is_dropper is False
    D = len(node.drop_seconds)
    C = len(node.check_seconds)
    rng = np.random.default_rng(0)

    for _ in range(50):
        d_idx, c_idx = _select_joint_action(node, exploration_c=1.0, rng=rng)
        assert 0 <= d_idx < D
        assert 0 <= c_idx < C


def test_select_joint_action_strategies_at_fresh_node_are_valid_distributions():
    # Solve_minimax on the all-zero exploration matrix should return strategies
    # that are proper probability distributions; rng.choice will validate this
    # (it raises if probabilities don't sum to 1.0). We run a single selection
    # and rely on rng.choice's internal validation.
    scenario = forced_baku_overflow_death()
    node = make_node(scenario.game, scenario.config)
    rng = np.random.default_rng(99)

    # If strategies didn't sum to 1.0, rng.choice would raise ValueError.
    d_idx, c_idx = _select_joint_action(node, exploration_c=1.0, rng=rng)
    assert d_idx >= 0 and c_idx >= 0
