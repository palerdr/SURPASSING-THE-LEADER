"""Exact zero-sum matrix-game LP utilities for the abstract solver."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize._highspy._core import HighsModelStatus, _Highs, kHighsInf


@dataclass(frozen=True, slots=True)
class MatrixEquilibrium:
    row_strategy: np.ndarray
    column_strategy: np.ndarray
    value: float
    saddle_gap: float

    def __post_init__(self) -> None:
        for name in ("row_strategy", "column_strategy"):
            frozen = np.asarray(getattr(self, name), dtype=np.float64).copy()
            frozen.setflags(write=False)
            object.__setattr__(self, name, frozen)


class _PersistentHighsMatrixLP:
    """One reusable HiGHS model for a fixed dense matrix shape.

    The tablebase needs hundreds of thousands of tiny LPs. Reusing the fixed
    row-player model changes only its payoff coefficients, remains exact, and
    avoids rebuilding HiGHS options for each state.
    """

    def __init__(self, rows: int, columns: int) -> None:
        self.rows = rows
        self.columns = columns
        self.highs = _Highs()
        self.highs.setOptionValue("output_flag", False)
        self.highs.setOptionValue("primal_feasibility_tolerance", 1e-9)
        self.highs.setOptionValue("dual_feasibility_tolerance", 1e-9)
        self.highs.setOptionValue("ipm_optimality_tolerance", 1e-10)
        variable_count = rows + 1  # mixed row policy and guaranteed value
        self.highs.addCols(
            variable_count,
            np.r_[np.zeros(rows), -1.0],
            np.r_[np.zeros(rows), -kHighsInf],
            np.full(variable_count, kHighsInf),
            0,
            np.zeros(variable_count + 1, dtype=np.int32),
            np.array([], dtype=np.int32),
            np.array([], dtype=np.float64),
        )
        starts = [0]
        indices: list[int] = []
        values: list[float] = []
        for _column in range(columns):
            indices.extend(range(rows))
            values.extend([0.0] * rows)
            indices.append(rows)
            values.append(1.0)
            starts.append(len(indices))
        indices.extend(range(rows))
        values.extend([1.0] * rows)
        starts.append(len(indices))
        self.highs.addRows(
            columns + 1,
            np.r_[-np.full(columns, kHighsInf), 1.0],
            np.r_[np.zeros(columns), 1.0],
            len(indices),
            np.asarray(starts, dtype=np.int32),
            np.asarray(indices, dtype=np.int32),
            np.asarray(values, dtype=np.float64),
        )

    def solve(self, matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        for column in range(self.columns):
            for row in range(self.rows):
                self.highs.changeCoeff(column, row, -float(matrix[row, column]))
        status = self.highs.run()
        if str(status) != "HighsStatus.kOk" or self.highs.getModelStatus() != HighsModelStatus.kOptimal:
            raise RuntimeError(f"Abstract LP failed: {self.highs.modelStatusToString(self.highs.getModelStatus())}")
        solution = self.highs.getSolution()
        row = normalize_policy(np.maximum(np.asarray(solution.col_value[: self.rows], dtype=np.float64), 0.0))
        column = normalize_policy(np.maximum(-np.asarray(solution.row_dual[: self.columns], dtype=np.float64), 0.0))
        return row, column


_PERSISTENT_SOLVERS: dict[tuple[int, int], _PersistentHighsMatrixLP] = {}


def normalize_policy(policy: np.ndarray, *, expected_size: int | None = None) -> np.ndarray:
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


def saddle_gap(
    payoff: np.ndarray,
    row_strategy: np.ndarray,
    column_strategy: np.ndarray,
) -> tuple[float, float, float, float]:
    """Expected row payoff plus both players' unilateral best-response gains."""

    matrix = _validate_matrix(payoff)
    row = normalize_policy(row_strategy, expected_size=matrix.shape[0])
    column = normalize_policy(column_strategy, expected_size=matrix.shape[1])
    expected = float(row @ matrix @ column)
    row_gain = max(0.0, float((matrix @ column).max()) - expected)
    column_gain = max(0.0, expected - float((row @ matrix).min()))
    return expected, row_gain, column_gain, row_gain + column_gain


def solve_matrix(payoff: np.ndarray) -> MatrixEquilibrium:
    """Solve the finite simultaneous zero-sum matrix for its row maximizer."""

    matrix = _validate_matrix(payoff)
    # Most late-game abstract matrices have a pure saddle.  Detecting it is
    # exact and avoids invoking a general LP for a certificate that is already
    # visible from row minima and column maxima.
    row_minima = matrix.min(axis=1)
    column_maxima = matrix.max(axis=0)
    lower = float(row_minima.max())
    upper = float(column_maxima.min())
    if np.isclose(lower, upper, atol=1e-12, rtol=0.0):
        for row_index in np.flatnonzero(np.isclose(row_minima, lower, atol=1e-12, rtol=0.0)):
            for column_index in np.flatnonzero(np.isclose(column_maxima, upper, atol=1e-12, rtol=0.0)):
                if np.isclose(matrix[row_index, column_index], lower, atol=1e-12, rtol=0.0):
                    row_strategy = np.zeros(matrix.shape[0], dtype=np.float64)
                    column_strategy = np.zeros(matrix.shape[1], dtype=np.float64)
                    row_strategy[row_index] = 1.0
                    column_strategy[column_index] = 1.0
                    value, _row_gain, _column_gain, gap = saddle_gap(
                        matrix,
                        row_strategy,
                        column_strategy,
                    )
                    if gap > 2e-7:
                        continue
                    return MatrixEquilibrium(
                        row_strategy=row_strategy,
                        column_strategy=column_strategy,
                        value=value,
                        saddle_gap=gap,
                    )
    key = matrix.shape
    solver = _PERSISTENT_SOLVERS.get(key)
    if solver is None:
        solver = _PersistentHighsMatrixLP(*key)
        _PERSISTENT_SOLVERS[key] = solver
    row_strategy, column_strategy = solver.solve(matrix)
    value, _row_gain, _column_gain, gap = saddle_gap(matrix, row_strategy, column_strategy)
    if gap > 2e-7:
        raise RuntimeError(f"Abstract LP saddle gap too large: {gap}")
    return MatrixEquilibrium(
        row_strategy=row_strategy,
        column_strategy=column_strategy,
        value=value,
        saddle_gap=gap,
    )
