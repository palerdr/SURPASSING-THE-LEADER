from __future__ import annotations
import numpy as np
from scipy.optimize import linprog
from .types import BeliefState


def solve_minimax(payoff: np.ndarray) -> tuple[np.ndarray, float]:
    m, n = payoff.shape  # m = Hal's buckets, n = Baku's buckets

    # Cost: minimize -v. v is the last variable (index m).
    c = np.zeros(m + 1)
    c[m] = -1.0

    # Inequality constraints: v - Σ x[i]*A[i][j] ≤ 0, one per Baku bucket j
    # Row j: [-A[0][j], -A[1][j], ..., -A[m-1][j], 1] ≤ 0
    A_ub = np.zeros((n, m + 1))
    A_ub[:, :m] = -payoff.T  # -A transposed: each row j gets -A[i][j] for all i
    A_ub[:, m] = 1.0  # +v
    b_ub = np.zeros(n)

    # Equality: x[0] + ... + x[m-1] = 1
    A_eq = np.zeros((1, m + 1))
    A_eq[0, :m] = 1.0
    b_eq = np.array([1.0])

    # Bounds: x[i] ≥ 0, v is unbounded
    bounds = [(0, None)] * m + [(None, None)]

    result = linprog(
        c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs"
    )

    strategy = result.x[:m]
    value = result.x[m]
    return strategy, value


def best_response(belief: BeliefState, payoff: np.ndarray) -> tuple[np.ndarray, float]:
    baku_strategy = np.array(belief.baku_predicted_bucket_probs)
    ev_per_hal_action = payoff @ baku_strategy
    best_idx = int(np.argmax(ev_per_hal_action))
    strategy = np.zeros(payoff.shape[0])
    strategy[best_idx] = 1.0
    return strategy, float(ev_per_hal_action[best_idx])
