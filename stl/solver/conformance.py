"""Small, deterministic conformance reports for Python solver boundaries."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from stl.solver.exact import CFRPlusConfig, solve_cfr_plus, solve_minimax


@dataclass(frozen=True)
class MatrixParityRecord:
    rows: int
    columns: int
    lp_value: float
    cfr_value: float
    value_error: float
    saddle_gap: float


def matrix_saddle_gap(
    payoff: np.ndarray,
    row_strategy: np.ndarray,
    column_strategy: np.ndarray,
) -> float:
    """Return the local saddle gap for a row-max/column-min zero-sum game."""

    matrix = np.asarray(payoff, dtype=np.float64)
    row = np.asarray(row_strategy, dtype=np.float64)
    column = np.asarray(column_strategy, dtype=np.float64)
    if matrix.shape != (len(row), len(column)):
        raise ValueError("strategy lengths do not match payoff shape")
    expected = float(row @ matrix @ column)
    row_gain = max(0.0, float(np.max(matrix @ column)) - expected)
    column_gain = max(0.0, expected - float(np.min(row @ matrix)))
    return row_gain + column_gain


def audit_cfr_plus_matrix(
    payoff: np.ndarray,
    config: CFRPlusConfig | None = None,
) -> MatrixParityRecord:
    """Compare bounded Python CFR+ with the LP oracle on one matrix."""

    matrix = np.asarray(payoff, dtype=np.float64)
    if matrix.ndim != 2 or 0 in matrix.shape:
        raise ValueError("payoff must be a non-empty 2D matrix")
    row_lp, lp_value = solve_minimax(matrix)
    del row_lp
    row_cfr, _ = solve_cfr_plus(matrix, config)
    column_cfr, _ = solve_cfr_plus((-matrix).T, config)
    cfr_value = float(row_cfr @ matrix @ column_cfr)
    return MatrixParityRecord(
        rows=matrix.shape[0],
        columns=matrix.shape[1],
        lp_value=float(lp_value),
        cfr_value=cfr_value,
        value_error=abs(cfr_value - float(lp_value)),
        saddle_gap=matrix_saddle_gap(matrix, row_cfr, column_cfr),
    )
