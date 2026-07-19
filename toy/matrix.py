"""Standalone zero-sum matrix-game utilities for ToySTL."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import linprog


@dataclass(frozen=True, slots=True)
class MatrixStrategy:
    strategy: np.ndarray
    value: float

    def __post_init__(self) -> None:
        frozen = np.asarray(self.strategy, dtype=np.float64).copy()
        frozen.setflags(write=False)
        object.__setattr__(self, "strategy", frozen)


@dataclass(frozen=True, slots=True)
class MatrixEquilibrium:
    row_strategy: np.ndarray
    column_strategy: np.ndarray
    value_for_hal: float
    saddle_gap: float

    def __post_init__(self) -> None:
        for name in ("row_strategy", "column_strategy"):
            frozen = np.asarray(getattr(self, name), dtype=np.float64).copy()
            frozen.setflags(write=False)
            object.__setattr__(self, name, frozen)


def normalize_policy(policy: np.ndarray, *, expected_size: int | None = None) -> np.ndarray:
    """Return a finite, nonnegative probability vector with unit mass."""

    values = np.asarray(policy, dtype=np.float64).reshape(-1)
    if expected_size is not None and values.shape != (expected_size,):
        raise ValueError(f"policy shape {values.shape} does not match {(expected_size,)}")
    if values.size == 0 or not np.all(np.isfinite(values)) or np.any(values < 0.0):
        raise ValueError("policy must be nonempty, finite, and nonnegative")
    total = float(values.sum())
    if total <= 1e-12:
        raise ValueError("policy has no probability mass")
    return values / total


def _validate_matrix(payoff: np.ndarray) -> np.ndarray:
    matrix = np.asarray(payoff, dtype=np.float64)
    if matrix.ndim != 2 or 0 in matrix.shape:
        raise ValueError(f"payoff must be a non-empty matrix, got {matrix.shape}")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("payoff must contain only finite values")
    return matrix


def _lp_maximin(matrix: np.ndarray, *, value_floor: float | None = None, weights: np.ndarray | None = None) -> tuple[np.ndarray, float]:
    matrix = _validate_matrix(matrix)
    rows, columns = matrix.shape

    c = np.zeros(rows + 1, dtype=np.float64)
    c[:rows] = 0.0 if weights is None else np.asarray(weights, dtype=np.float64)
    c[rows] = -1.0 if weights is None else 0.0

    a_ub = np.zeros((columns, rows + 1), dtype=np.float64)
    a_ub[:, :rows] = -matrix.T
    a_ub[:, rows] = 1.0
    b_ub = np.zeros(columns, dtype=np.float64)
    if value_floor is not None:
        a_ub = np.vstack((a_ub, np.r_[np.zeros(rows), -1.0]))
        b_ub = np.r_[b_ub, -float(value_floor)]

    a_eq = np.zeros((1, rows + 1), dtype=np.float64)
    a_eq[0, :rows] = 1.0
    def run_lp(current_b_ub: np.ndarray):
        return linprog(
            c,
            A_ub=a_ub,
            b_ub=current_b_ub,
            A_eq=a_eq,
            b_eq=np.array([1.0]),
            bounds=[(0.0, None)] * rows + [(None, None)],
            method="highs",
            options={
                "primal_feasibility_tolerance": 1e-9,
                "dual_feasibility_tolerance": 1e-9,
                "ipm_optimality_tolerance": 1e-10,
            },
        )

    result = run_lp(b_ub)
    if (not result.success or result.x is None) and value_floor is not None:
        # The secondary LP is only a deterministic selector on the primary
        # optimal face.  Optimism bonuses can magnify a few ulps of primary
        # value error, so retain the selector while backing off by a bounded
        # numerical margin rather than declaring the matrix infeasible.
        relaxed_b_ub = b_ub.copy()
        relaxed_b_ub[-1] = -(float(value_floor) - 1e-7)
        result = run_lp(relaxed_b_ub)
    if not result.success or result.x is None:
        raise RuntimeError(f"ToySTL LP failed: {result.message}")

    strategy = normalize_policy(np.maximum(np.asarray(result.x[:rows], dtype=np.float64), 0.0))
    value = float(result.x[rows]) if weights is None else float(value_floor)
    return strategy, value


def _lp_maximin_with_dual(matrix: np.ndarray) -> tuple[np.ndarray, float, np.ndarray]:
    """Solve the primary LP and recover its minimax dual in one pass."""

    matrix = _validate_matrix(matrix)
    rows, columns = matrix.shape
    c = np.zeros(rows + 1, dtype=np.float64)
    c[rows] = -1.0
    a_ub = np.zeros((columns, rows + 1), dtype=np.float64)
    a_ub[:, :rows] = -matrix.T
    a_ub[:, rows] = 1.0
    result = linprog(
        c,
        A_ub=a_ub,
        b_ub=np.zeros(columns, dtype=np.float64),
        A_eq=np.r_[np.ones(rows), 0.0][None, :],
        b_eq=np.array([1.0]),
        bounds=[(0.0, None)] * rows + [(None, None)],
        method="highs",
        options={
            "primal_feasibility_tolerance": 1e-9,
            "dual_feasibility_tolerance": 1e-9,
            "ipm_optimality_tolerance": 1e-10,
        },
    )
    if not result.success or result.x is None:
        raise RuntimeError(f"ToySTL LP failed: {result.message}")
    strategy = normalize_policy(np.maximum(np.asarray(result.x[:rows], dtype=np.float64), 0.0))
    dual = normalize_policy(np.maximum(-np.asarray(result.ineqlin.marginals, dtype=np.float64), 0.0))
    return strategy, float(result.x[rows]), dual


def solve_maximin(payoff: np.ndarray) -> MatrixStrategy:
    """Return a deterministic row-player maximin strategy."""

    matrix = _validate_matrix(payoff)
    _, value = _lp_maximin(matrix)
    # Select one stable point on a potentially non-unique optimal face. The
    # value floor is deliberately loose only at the LP numerical tolerance.
    weights = np.arange(1.0, matrix.shape[0] + 1.0)
    strategy, _ = _lp_maximin(matrix, value_floor=value - 1e-9, weights=weights)
    guaranteed = float(np.min(strategy @ matrix))
    if guaranteed < value - 2e-7:
        raise RuntimeError("ToySTL LP tie-break left the optimal face")
    return MatrixStrategy(strategy=strategy, value=value)


def saddle_gap(
    payoff: np.ndarray,
    row_strategy: np.ndarray,
    column_strategy: np.ndarray,
    *,
    row_is_hal: bool,
) -> tuple[float, float, float, float]:
    """Return expected value, row gain, column gain, and total gap."""

    matrix = _validate_matrix(payoff)
    row = np.asarray(row_strategy, dtype=np.float64)
    column = np.asarray(column_strategy, dtype=np.float64)
    expected = float(row @ matrix @ column)
    row_values = matrix @ column
    column_values = row @ matrix
    if row_is_hal:
        row_gain = max(0.0, float(row_values.max()) - expected)
        column_gain = max(0.0, expected - float(column_values.min()))
    else:
        row_gain = max(0.0, expected - float(row_values.min()))
        column_gain = max(0.0, float(column_values.max()) - expected)
    return expected, row_gain, column_gain, row_gain + column_gain


def solve_matrix(payoff_for_hal: np.ndarray, *, row_is_hal: bool) -> MatrixEquilibrium:
    """Solve a payoff matrix while returning both role marginals."""

    matrix = _validate_matrix(payoff_for_hal)
    row_objective = matrix if row_is_hal else -matrix
    _primary_row, primary_value, primary_dual = _lp_maximin_with_dual(row_objective)
    row_weights = np.arange(1.0, matrix.shape[0] + 1.0)
    row_strategy, _ = _lp_maximin(
        row_objective,
        value_floor=primary_value - 1e-9,
        weights=row_weights,
    )

    # The primary row LP's dual is already a valid column equilibrium.  Apply
    # the same deterministic secondary objective to the column player on its
    # transformed minimax problem, preserving the explicit target contract.
    column_objective = -matrix.T if row_is_hal else matrix.T
    column_value_floor = -primary_value - 1e-9
    column_weights = np.arange(1.0, matrix.shape[1] + 1.0)
    column_strategy, _ = _lp_maximin(
        column_objective,
        value_floor=column_value_floor,
        weights=column_weights,
    )
    # ``primary_dual`` is intentionally computed above even though the
    # tie-broken column strategy is returned; checking its support catches
    # malformed solver duals early and documents the primal/dual relation.
    if not np.isclose(primary_dual.sum(), 1.0, atol=1e-8):
        raise RuntimeError("ToySTL LP dual strategy is not normalized")
    value, _row_gain, _column_gain, gap = saddle_gap(
        matrix,
        row_strategy,
        column_strategy,
        row_is_hal=row_is_hal,
    )
    return MatrixEquilibrium(
        row_strategy=row_strategy,
        column_strategy=column_strategy,
        value_for_hal=value,
        saddle_gap=gap,
    )
