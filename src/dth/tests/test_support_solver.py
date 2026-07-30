"""Parity and certification tests for the support-restricted solver."""

import numpy as np
import pytest

from dth.solver import (
    SADDLE_GAP_TOLERANCE,
    reconstruct_transition_class_matrix,
    solve_matrix,
)
from dth.support_solver import (
    solve_certified_matrix_fast,
    solve_matrix_by_support,
    solve_matrix_single_lp,
    solve_pure_saddle_point,
)


def saddle_gap(matrix, drop, check):
    return float(np.max(matrix @ check) - np.min(matrix.T @ drop))


def test_matching_pennies_is_solved_to_the_uniform_value():
    matrix = np.array([[1.0, -1.0], [-1.0, 1.0]])
    solution = solve_matrix_by_support(matrix)
    assert solution.saddle_gap <= SADDLE_GAP_TOLERANCE
    assert solution.value == pytest.approx(0.0, abs=1e-9)
    assert solution.drop_policy == pytest.approx([0.5, 0.5], abs=1e-9)


def test_saddle_point_game_needs_one_iteration():
    # Row 1 dominates and column 0 dominates, so the seed is already optimal.
    matrix = np.array([[0.0, 1.0], [2.0, 3.0]])
    solution = solve_matrix_by_support(matrix)
    assert solution.value == pytest.approx(2.0, abs=1e-9)
    assert solution.iterations == 1
    assert solution.drop_support == 1
    assert solution.check_support == 1


def test_asymmetric_random_games_match_the_highs_oracle():
    rng = np.random.default_rng(11)
    for _ in range(40):
        matrix = rng.uniform(-1.0, 1.0, size=(rng.integers(2, 25), rng.integers(2, 25)))
        expected, _, _ = solve_matrix(matrix)
        solution = solve_matrix_by_support(matrix)
        assert solution.saddle_gap <= SADDLE_GAP_TOLERANCE
        assert solution.value == pytest.approx(expected, abs=1e-6)


def test_degenerate_games_with_many_tied_actions_still_certify():
    # Half the actions exactly tied is the shape that stalls the learned
    # evaluator at the h4 promotion anchor; the solver must still close.
    rng = np.random.default_rng(5)
    for _ in range(20):
        matrix = rng.uniform(-1.0, 1.0, size=(12, 12))
        matrix[:, :6] = matrix[:, :1]
        matrix[:6, :] = matrix[:1, :]
        expected, _, _ = solve_matrix(matrix)
        solution = solve_matrix_by_support(matrix)
        assert solution.saddle_gap <= SADDLE_GAP_TOLERANCE
        assert solution.value == pytest.approx(expected, abs=1e-6)


def test_transition_class_matrices_match_the_oracle():
    rng = np.random.default_rng(3)
    for _ in range(10):
        successful = rng.uniform(-1.0, 1.0, size=60)
        failed = float(rng.uniform(-1.0, 1.0))
        built = reconstruct_transition_class_matrix(successful, failed)
        expected, _, _ = solve_matrix(built)
        solution = solve_matrix_by_support(built)
        assert solution.saddle_gap <= SADDLE_GAP_TOLERANCE
        assert solution.value == pytest.approx(expected, abs=1e-6)


def test_certified_wrapper_falls_back_to_highs_on_failure(monkeypatch):
    import dth.support_solver as module

    def fail(*args, **kwargs):
        raise RuntimeError("forced")

    monkeypatch.setattr(module, "solve_pure_saddle_point", fail)
    monkeypatch.setattr(module, "solve_matrix_single_lp", fail)
    matrix = np.array([[1.0, -1.0], [-1.0, 1.0]])
    value, drop, check, backend = solve_certified_matrix_fast(matrix)
    assert backend == "highs"
    assert value == pytest.approx(0.0, abs=1e-9)
    assert saddle_gap(matrix, drop, check) <= SADDLE_GAP_TOLERANCE


def test_non_finite_matrices_are_rejected():
    with pytest.raises(ValueError):
        solve_matrix_by_support(np.array([[np.nan, 0.0], [0.0, 1.0]]))


def test_pure_saddle_point_path_matches_the_oracle():
    matrix = np.array([[0.0, 1.0], [2.0, 3.0]])
    value, drop, check = solve_pure_saddle_point(matrix)
    assert value == pytest.approx(2.0, abs=1e-12)
    assert saddle_gap(matrix, drop, check) <= SADDLE_GAP_TOLERANCE


def test_pure_saddle_point_path_refuses_games_that_need_mixing():
    with pytest.raises(RuntimeError):
        solve_pure_saddle_point(np.array([[1.0, -1.0], [-1.0, 1.0]]))


def test_single_lp_dual_extraction_matches_the_oracle():
    rng = np.random.default_rng(17)
    for _ in range(40):
        matrix = rng.uniform(-1.0, 1.0, size=(rng.integers(2, 30), rng.integers(2, 30)))
        expected, _, _ = solve_matrix(matrix)
        value, drop, check = solve_matrix_single_lp(matrix)
        assert saddle_gap(matrix, drop, check) <= SADDLE_GAP_TOLERANCE
        assert value == pytest.approx(expected, abs=1e-6)


def test_single_lp_on_transition_class_matrices():
    rng = np.random.default_rng(23)
    for _ in range(10):
        built = reconstruct_transition_class_matrix(
            rng.uniform(-1.0, 1.0, size=60), float(rng.uniform(-1.0, 1.0))
        )
        expected, _, _ = solve_matrix(built)
        value, drop, check = solve_matrix_single_lp(built)
        assert saddle_gap(built, drop, check) <= SADDLE_GAP_TOLERANCE
        assert value == pytest.approx(expected, abs=1e-6)


def test_fast_ladder_agrees_with_the_oracle_and_labels_its_path():
    rng = np.random.default_rng(29)
    seen = set()
    for _ in range(60):
        matrix = rng.uniform(-1.0, 1.0, size=(8, 8))
        expected, _, _ = solve_matrix(matrix)
        value, drop, check, backend = solve_certified_matrix_fast(matrix)
        seen.add(backend)
        assert saddle_gap(matrix, drop, check) <= SADDLE_GAP_TOLERANCE
        assert value == pytest.approx(expected, abs=1e-6)
    assert seen <= {"pure-saddle-point", "single-lp-dual", "highs"}
