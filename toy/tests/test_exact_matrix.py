import numpy as np

from toy.exact import solve_all_states, solve_exact
from toy.matrix import saddle_gap, solve_matrix
from toy.rules import Bucket12Fixed50Rules
from toy.state import ToyState
from toy.tablebase import build_tablebase


def test_matrix_lp_normalizes_and_has_small_saddle_gap():
    payoff = np.asarray([[1.0, -1.0], [-1.0, 1.0]])
    equilibrium = solve_matrix(payoff, row_is_hal=True)
    assert np.allclose(equilibrium.row_strategy.sum(), 1.0)
    assert np.allclose(equilibrium.column_strategy.sum(), 1.0)
    assert np.all(equilibrium.row_strategy >= 0.0)
    assert np.all(equilibrium.column_strategy >= 0.0)
    assert abs(equilibrium.value_for_hal) < 1e-8
    assert equilibrium.saddle_gap <= 2e-7
    expected, _row_gain, _column_gain, gap = saddle_gap(
        payoff,
        equilibrium.row_strategy,
        equilibrium.column_strategy,
        row_is_hal=True,
    )
    assert abs(expected) < 1e-8
    assert gap <= 2e-7


def test_exact_horizon_zero_and_cell_branch_value():
    rules = Bucket12Fixed50Rules()
    zero = solve_exact(ToyState(), 0, rules)
    assert zero.value_for_hal == 0.0
    assert zero.truncated_probability == 1.0
    assert zero.transitions == ()

    result = solve_exact(ToyState(baku_load=59), 1, rules)
    assert len(result.transitions) == 144
    overflow_cell = result.payoff_for_hal[0, 1]
    assert np.isclose(overflow_cell, 0.5)
    assert result.saddle_gap <= 2e-7


def test_v0_state_enumeration_and_zero_horizon_tablebase():
    rules = Bucket12Fixed50Rules()
    states = tuple(rules.enumerate_states())
    assert len(states) == 60 * 60 * 2
    assert states[0] == ToyState(role_phase=0)
    assert states[1] == ToyState(role_phase=1)
    assert states[-1] == ToyState(hal_load=59, baku_load=59, role_phase=1)
    rows = solve_all_states(rules, max_horizon=0)
    assert len(rows) == 7200
    assert all(result.value_for_hal == 0.0 for _state, _horizon, result in rows)
    tablebase = build_tablebase(rules, max_horizon=0)
    assert tablebase["arrays"]["states"].shape == (7200, 3)
    assert tablebase["arrays"]["policy_active"].sum() == 0
