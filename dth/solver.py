from typing import Tuple
from enum import IntEnum
import numpy as np
from functools import cache
from scipy.optimize import linprog
from dataclasses import dataclass

DROPPER_ACTIONS = list(range(1, 61))
CHECKER_ACTIONS = list(range(1, 61))
ST_SUPPORT = list(range(300))


def successful_check(c,d): return d <= c
def st(c,d): return c - d + 1
def failed_check_dose(st): return st + 60
def overflow(st): return st >= 300
def survive_injection(st, ttd):
    dose = failed_check_dose(st)
    return dose < 300 and dose + ttd <= 300
def revival_model(st, ttd):
    if not survive_injection(st, ttd):
        return 0.0
    q = st + 60
    st_term = 1.0 - (q / 300.0) ** 3
    ttd_term = 2.0 ** (-ttd / 240.0)
    return st_term * ttd_term

type NTState = Tuple[int, int, int, int]
class TState(IntEnum):
    W = 1
    L = 0

type State = NTState | TState
type Branch = tuple[float, State]
type Distribution = tuple[Branch, ...]
def reward(x: State):
    if x == 1:
        return 1
    elif x == 0:
        return -1
    else:
        return 0
    
def transition(x: NTState, d: int, c: int) -> Distribution:
    """From a state and given actions, give the distribution of child states"""
    sc, tc, sd, td = x
    # Check succeeds. Reaching the cylinder cap dumps a fatal 300-second dose.
    if successful_check(c,d):
        next_st = sc + st(c,d)
        if overflow(next_st):
            return (
                (1.0, TState.W),
            )
        return (
            (1.0, (sd, td, next_st, tc)),
            )
    #check fails
    else:
        p = revival_model(sc, tc)
        #die off injection
        if p == 0.0:
            return (
                (1.0, TState.W),
            )
        #get revived off injection
        else:
            return (
                    (p, (sd, td, 0, tc + sc + 60)),
                    (1.0 - p, TState.W),
                    )

@dataclass(frozen=True)
class Solution:
    value: float
    drop_policy: tuple[float, ...] | None
    check_policy: tuple[float, ...] | None
    saddle_gap: float

# E[Terminal or -V(s', h-1)]
def action_value(
        x: NTState,
        d: int,
        c: int,
        horizon: int,
) -> float:
    total = 0.0
    for probability, child in transition(x, d, c):
        if isinstance(child, TState):
            child_value = reward(child)
        else:
            child_value = -value(child, horizon - 1)
        total += probability * child_value
    return total

def payoff(x: NTState, horizon: int) -> np.ndarray:
    """Builds the payoff matrix for the current state
    with the continuation values for the given horizon"""
    matrix = np.empty(
        (len(DROPPER_ACTIONS), len(CHECKER_ACTIONS)),
        dtype=np.float64
    )
    
    for i, d in enumerate(DROPPER_ACTIONS):
        for j, c in enumerate(CHECKER_ACTIONS):
            matrix[i, j] = action_value(x, d, c, horizon)
    
    return matrix


@cache
def solve(x: NTState, horizon: int) -> Solution:
    """Solves a state for a given horizon"""
    if horizon == 0:
        return Solution(
            value=0.0,
            drop_policy=None,
            check_policy=None,
            saddle_gap=0.0,
        )

    matrix = payoff(x, horizon)
    matrix_value, p, q = solve_matrix(matrix)

    lower = np.min(matrix.T @ p)
    upper = np.max(matrix @ q)

    return Solution(
        value=matrix_value,
        drop_policy=tuple(p),
        check_policy=tuple(q),
        saddle_gap=upper - lower,
    )


def value(x: NTState, horizon: int) -> float:
    return solve(x, horizon).value


def solve_matrix(matrix: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    """solve the minimax linear program for the value and d,c strategies"""
    rows, cols = matrix.shape
    # Dropper: maximize v subject to M.T @ p >= v.
    drop_result = linprog(
        c=np.concatenate([np.zeros(rows), [-1.0]]),
        A_ub=np.hstack([
            -matrix.T,
            np.ones((cols, 1)),
        ]),
        b_ub=np.zeros(cols),
        A_eq=np.hstack([
            np.ones((1, rows)),
            np.zeros((1, 1)),
        ]),
        b_eq=np.array([1.0]),
        bounds=[(0.0, None)] * rows + [(None, None)],
        method="highs",
    )

    if not drop_result.success:
        raise RuntimeError(
            f"Dropper LP failed: {drop_result.message}"
        )
    # Checker: minimize w subject to M @ q <= w.
    check_result = linprog(
        c=np.concatenate([np.zeros(cols), [1.0]]),
        A_ub=np.hstack([
            matrix,
            -np.ones((rows, 1)),
        ]),
        b_ub=np.zeros(rows),
        A_eq=np.hstack([
            np.ones((1, cols)),
            np.zeros((1, 1)),
        ]),
        b_eq=np.array([1.0]),
        bounds=[(0.0, None)] * cols + [(None, None)],
        method="highs",
    )

    if not check_result.success:
        raise RuntimeError(
            f"Checker LP failed: {check_result.message}"
        )
    #normalize the policies
    drop_policy = np.clip(drop_result.x[:-1], 0.0, None)
    check_policy = np.clip(check_result.x[:-1], 0.0, None)
    drop_policy /= drop_policy.sum()
    check_policy /= check_policy.sum()
    
    lower_bound = np.min(matrix.T @ drop_policy)
    upper_bound = np.max(matrix @ check_policy)
    saddle_gap = upper_bound - lower_bound

    if saddle_gap > 1e-6:
        raise RuntimeError(
            f"LP saddle gap too large: {saddle_gap}"
        )
    matrix_value = (lower_bound + upper_bound) / 2.0

    return matrix_value, drop_policy, check_policy

