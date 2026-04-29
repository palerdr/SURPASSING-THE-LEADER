"""Tests for MCTS Step 5: _backup propagates leaf value through the path."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from environment.cfr.mcts import _backup, make_node
from environment.cfr.tactical_scenarios import forced_baku_overflow_death


def _fresh_node():
    scenario = forced_baku_overflow_death()
    return make_node(scenario.game, scenario.config)


def test_backup_single_step_updates_one_cell_only():
    node = _fresh_node()
    D = len(node.drop_seconds)
    C = len(node.check_seconds)

    _backup([(node, 0, 0)], value=1.0)

    assert node.N_node == 1
    assert node.N_cell[0, 0] == 1
    assert node.N_cell.sum() == 1
    assert node.Q[0, 0] == 1.0
    # Every other cell must remain at zero.
    other_cells_visits = node.N_cell.copy()
    other_cells_visits[0, 0] = 0
    assert other_cells_visits.sum() == 0
    other_cells_q = node.Q.copy()
    other_cells_q[0, 0] = 0
    assert other_cells_q.sum() == 0


def test_backup_running_mean_averages_two_visits_to_same_cell():
    node = _fresh_node()

    _backup([(node, 0, 0)], value=1.0)
    _backup([(node, 0, 0)], value=-1.0)

    assert node.N_cell[0, 0] == 2
    assert node.Q[0, 0] == pytest.approx(0.0)


def test_backup_running_mean_handles_three_values_correctly():
    # Mean of (1.0, -1.0, 0.5) = 0.5 / 3 ≈ 0.1667
    node = _fresh_node()
    for v in (1.0, -1.0, 0.5):
        _backup([(node, 0, 0)], value=v)

    assert node.N_cell[0, 0] == 3
    assert node.Q[0, 0] == pytest.approx((1.0 - 1.0 + 0.5) / 3)


def test_backup_preserves_n_node_equals_n_cell_sum_invariant():
    # Visit several different cells and assert the invariant holds at every
    # step. This is the test that catches `=+` typos and any future change
    # that increments only one counter.
    node = _fresh_node()
    cells = [(0, 0), (0, 1), (1, 0), (0, 0), (1, 1), (0, 0)]

    for d_idx, c_idx in cells:
        _backup([(node, d_idx, c_idx)], value=0.5)
        assert node.N_node == node.N_cell.sum()

    assert node.N_node == len(cells)


def test_backup_multi_node_path_updates_every_node_once():
    # Build two distinct nodes, run one backup with a path through both.
    # Each node should be visited exactly once, not twice.
    scenario_a = forced_baku_overflow_death()
    node_a = make_node(scenario_a.game, scenario_a.config)
    scenario_b = forced_baku_overflow_death()
    node_b = make_node(scenario_b.game, scenario_b.config)

    _backup([(node_a, 0, 0), (node_b, 1, 1)], value=0.5)

    assert node_a.N_node == 1
    assert node_a.N_cell[0, 0] == 1
    assert node_a.Q[0, 0] == pytest.approx(0.5)

    assert node_b.N_node == 1
    assert node_b.N_cell[1, 1] == 1
    assert node_b.Q[1, 1] == pytest.approx(0.5)


def test_backup_with_empty_path_is_a_noop():
    node = _fresh_node()
    pre_q = node.Q.copy()
    pre_n = node.N_node

    _backup([], value=0.5)

    assert node.N_node == pre_n
    np.testing.assert_array_equal(node.Q, pre_q)


def test_backup_negative_leaf_value_propagates_correctly():
    # Hal-perspective values can be -1.0; the running-mean math should not
    # depend on sign.
    node = _fresh_node()
    _backup([(node, 0, 0)], value=-1.0)
    assert node.Q[0, 0] == -1.0
    assert node.N_cell[0, 0] == 1
