"""Zero-sum matrix-game solvers owned by the CFR package."""

from __future__ import annotations

import numpy as np
from scipy.optimize import linprog


def solve_minimax(payoff: np.ndarray) -> tuple[np.ndarray, float]:
    """Solve the row player's maximin strategy for a zero-sum payoff matrix."""
    m, n = payoff.shape
    if m == 1:
        return np.array([1.0]), float(payoff[0].min())
    if n == 1:
        best = int(np.argmax(payoff[:, 0]))
        strategy = np.zeros(m)
        strategy[best] = 1.0
        return strategy, float(payoff[best, 0])

    c = np.zeros(m + 1)
    c[m] = -1.0

    a_ub = np.zeros((n, m + 1))
    a_ub[:, :m] = -payoff.T
    a_ub[:, m] = 1.0
    b_ub = np.zeros(n)

    a_eq = np.zeros((1, m + 1))
    a_eq[0, :m] = 1.0
    b_eq = np.array([1.0])

    result = linprog(
        c,
        A_ub=a_ub,
        b_ub=b_ub,
        A_eq=a_eq,
        b_eq=b_eq,
        bounds=[(0, None)] * m + [(None, None)],
        method="highs",
    )

    if not result.success or result.x is None:
        uniform = np.ones(m) / m
        return uniform, float(np.min(uniform @ payoff))

    strategy = np.maximum(result.x[:m], 0.0)
    total = strategy.sum()
    if total > 1e-9:
        strategy /= total
    else:
        strategy = np.ones(m) / m
    return strategy, float(result.x[m])

