from __future__ import annotations

import numpy as np
from scipy.optimize import linprog

from .types import BeliefState


def solve_minimax(payoff: np.ndarray) -> tuple[np.ndarray, float]:
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

    A_ub = np.zeros((n, m + 1))
    A_ub[:, :m] = -payoff.T
    A_ub[:, m] = 1.0
    b_ub = np.zeros(n)

    A_eq = np.zeros((1, m + 1))
    A_eq[0, :m] = 1.0
    b_eq = np.array([1.0])

    bounds = [(0, None)] * m + [(None, None)]

    result = linprog(
        c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs"
    )

    if not result.success or result.x is None:
        uniform = np.ones(m) / m
        worst_case = float(np.min(uniform @ payoff))
        return uniform, worst_case

    strategy = np.maximum(result.x[:m], 0.0)
    s = strategy.sum()
    if s > 1e-9:
        strategy = strategy / s
    else:
        strategy = np.ones(m) / m
    value = float(result.x[m])
    return strategy, value


def best_response(belief: BeliefState, payoff: np.ndarray, hal_is_dropper: bool = True) -> tuple[np.ndarray, float]:
    probs = belief.baku_check_probs if hal_is_dropper else belief.baku_drop_probs
    baku_strategy = np.array(probs)
    ev_per_hal_action = payoff @ baku_strategy
    best_idx = int(np.argmax(ev_per_hal_action))
    strategy = np.zeros(payoff.shape[0])
    strategy[best_idx] = 1.0
    return strategy, float(ev_per_hal_action[best_idx])
