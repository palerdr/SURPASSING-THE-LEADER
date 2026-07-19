import numpy as np
import pytest

from stl.solver.exact import (
    CFRPlusConfig,
    compute_payoff_matrix,
    solve_cfr_plus,
    solve_cfr_plus_rust,
    solve_minimax,
)

rs = pytest.importorskip("stl_solver_rs")


def _rust_solve(payoff: np.ndarray, *, iterations: int = 2000, average_delay: int = 100):
    matrix = np.ascontiguousarray(payoff, dtype=np.float64)
    return rs.solve_cfr_plus_rs(
        matrix,
        iterations=iterations,
        average_delay=average_delay,
        linear_weighting=True,
    )


def test_rust_cfr_plus_solves_matching_pennies():
    payoff = np.array([[1.0, -1.0], [-1.0, 1.0]], dtype=np.float64)

    strategy, value = _rust_solve(payoff, iterations=500, average_delay=50)

    np.testing.assert_allclose(strategy, [0.5, 0.5], atol=1e-3)
    assert value == pytest.approx(0.0, abs=1e-3)


def test_rust_cfr_plus_matches_python_on_exact_second_matrix_60():
    payoff = compute_payoff_matrix(0.0, turn_duration=60) / 60.0

    py_strategy, py_value = solve_cfr_plus(
        payoff,
        CFRPlusConfig(iterations=2000, average_delay=100),
    )
    rs_strategy, rs_value = _rust_solve(payoff, iterations=2000, average_delay=100)

    assert rs_strategy.shape == py_strategy.shape
    assert rs_strategy.sum() == pytest.approx(1.0)
    assert np.all(rs_strategy >= 0.0)
    np.testing.assert_allclose(rs_strategy, py_strategy, atol=1e-12)
    assert rs_value == pytest.approx(py_value, abs=1e-12)


def test_exact_wrapper_matches_direct_rust_binding():
    payoff = compute_payoff_matrix(0.0, turn_duration=60) / 60.0
    config = CFRPlusConfig(iterations=2000, average_delay=100)

    wrapper_strategy, wrapper_value = solve_cfr_plus_rust(payoff.T, config)
    direct_strategy, direct_value = _rust_solve(payoff.T, iterations=2000, average_delay=100)

    np.testing.assert_allclose(wrapper_strategy, direct_strategy, atol=1e-12)
    assert wrapper_value == pytest.approx(direct_value, abs=1e-12)


def test_rust_cfr_plus_tracks_lp_on_exact_second_matrix_61():
    payoff = compute_payoff_matrix(0.0, turn_duration=61) / 60.0
    _lp_strategy, lp_value = solve_minimax(payoff)

    rs_strategy, rs_value = _rust_solve(payoff, iterations=2000, average_delay=100)

    assert rs_strategy.shape == (61,)
    assert rs_strategy.sum() == pytest.approx(1.0)
    assert np.all(rs_strategy >= 0.0)
    assert rs_value == pytest.approx(lp_value, abs=5e-4)


def test_rust_cfr_plus_rejects_zero_iterations():
    payoff = np.eye(2, dtype=np.float64)

    with pytest.raises(ValueError, match="iterations"):
        rs.solve_cfr_plus_rs(payoff, iterations=0)
