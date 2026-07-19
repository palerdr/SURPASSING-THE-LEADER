import numpy as np
import pytest

from dth.cfr import solve_matrix_cfr_plus
from dth.solver import solve_matrix


def test_cfr_plus_converges_on_matching_pennies():
    matrix = np.asarray([[1.0, -1.0], [-1.0, 1.0]])

    result = solve_matrix_cfr_plus(matrix, iterations=20_000)

    assert result.value == pytest.approx(0.0, abs=2e-3)
    assert result.drop_policy == pytest.approx([0.5, 0.5], abs=2e-3)
    assert result.check_policy == pytest.approx([0.5, 0.5], abs=2e-3)
    assert result.saddle_gap <= 5e-3


def test_cfr_plus_matches_lp_on_a_rectangular_matrix():
    matrix = np.asarray(
        [
            [1.0, -1.0, 0.5],
            [-0.5, 1.0, -1.0],
        ]
    )
    exact_value, _, _ = solve_matrix(matrix)

    result = solve_matrix_cfr_plus(matrix, iterations=50_000)

    assert result.value == pytest.approx(exact_value, abs=5e-3)
    assert result.saddle_gap <= 1e-2
    assert result.drop_policy.sum() == pytest.approx(1.0)
    assert result.check_policy.sum() == pytest.approx(1.0)


def test_cfr_plus_is_deterministic_and_can_stop_on_a_gap_certificate():
    matrix = np.asarray([[1.0, 1.0], [0.0, 0.0]])

    first = solve_matrix_cfr_plus(
        matrix,
        iterations=10_000,
        gap_tolerance=1e-4,
        check_every=100,
    )
    second = solve_matrix_cfr_plus(
        matrix,
        iterations=10_000,
        gap_tolerance=1e-4,
        check_every=100,
    )

    assert first.iterations < 10_000
    assert first.saddle_gap <= 1e-4
    assert np.array_equal(first.drop_policy, second.drop_policy)
    assert np.array_equal(first.check_policy, second.check_policy)


@pytest.mark.parametrize(
    "matrix",
    [np.asarray([]), np.asarray([[np.nan]]), np.asarray([1.0, 2.0])],
)
def test_cfr_plus_rejects_invalid_matrices(matrix):
    with pytest.raises(ValueError):
        solve_matrix_cfr_plus(matrix)
