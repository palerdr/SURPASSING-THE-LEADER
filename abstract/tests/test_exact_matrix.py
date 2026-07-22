import numpy as np

from abstract.exact import enumerate_reachable_states, solve_exact
from abstract.matrix import saddle_gap, solve_matrix
from abstract.rules import Bucket6TTDCurve95Rules
from abstract.state import AbstractState


def test_matrix_lp_normalizes_and_certifies_matching_pennies() -> None:
    payoff = np.asarray([[1.0, -1.0], [-1.0, 1.0]])
    equilibrium = solve_matrix(payoff)
    assert np.allclose(equilibrium.row_strategy.sum(), 1.0)
    assert np.allclose(equilibrium.column_strategy.sum(), 1.0)
    assert abs(equilibrium.value) < 1e-8
    expected, _row_gain, _column_gain, gap = saddle_gap(
        payoff,
        equilibrium.row_strategy,
        equilibrium.column_strategy,
    )
    assert abs(expected) < 1e-8
    assert gap <= 2e-7


def test_exact_solver_uses_role_swap_and_terminal_values_without_a_horizon() -> None:
    rules = Bucket6TTDCurve95Rules()
    terminal_state = AbstractState(checker_load=29, checker_ttd=0, dropper_load=0, dropper_ttd=0)
    result = solve_exact(terminal_state, rules)
    assert result.value_for_dropper == 1.0
    assert result.dropper_win_probability == 1.0
    assert result.checker_win_probability == 0.0
    assert len(result.transitions) == 36
    assert result.saddle_gap <= 2e-7


def test_reachable_closure_is_exhaustive_from_the_initial_state() -> None:
    rules = Bucket6TTDCurve95Rules()
    states = enumerate_reachable_states(rules)
    assert len(states) == 576_270
    assert states[0] == AbstractState()
    assert states[-1].potential == 118
    assert all(state.potential <= 118 for state in states)
