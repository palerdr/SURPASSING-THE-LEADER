"""Approximate zero-sum matrix solving with deterministic CFR+.

This module is deliberately separate from :mod:`dth.solver`.  The LP in that
module remains the only solver used to create certified exactly solved targets.  CFR+
is useful for measuring how an iterative regret solver behaves on the local
simultaneous-action matrices used by search.
"""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CFRPlusSolution:
    value: float
    drop_policy: np.ndarray
    check_policy: np.ndarray
    lower_bound: float
    upper_bound: float
    saddle_gap: float
    iterations: int


def _regret_matching_plus(regrets: np.ndarray) -> np.ndarray:
    positive = np.maximum(regrets, 0.0)
    total = float(positive.sum())
    if total <= 0.0:
        return np.full(regrets.shape, 1.0 / regrets.size, dtype=np.float64)
    return positive / total


def solve_matrix_cfr_plus(
    matrix: np.ndarray,
    *,
    iterations: int = 10_000,
    averaging_delay: int = 0,
    gap_tolerance: float | None = None,
    check_every: int = 100,
) -> CFRPlusSolution:
    """Approximately solve a finite zero-sum matrix game with CFR+.

    Rows are Dropper actions and maximize payoff; columns are Checker actions
    and minimize it.  Regrets are clipped at zero after every update and the
    returned strategies use linear iteration weighting after ``averaging_delay``.
    A requested tolerance is an early-stop condition, not an exactness claim.
    """

    payoffs = np.asarray(matrix, dtype=np.float64)
    if payoffs.ndim != 2 or payoffs.shape[0] == 0 or payoffs.shape[1] == 0:
        raise ValueError("matrix must be a non-empty two-dimensional array")
    if not np.isfinite(payoffs).all():
        raise ValueError("matrix entries must be finite")
    if iterations <= 0:
        raise ValueError("iterations must be positive")
    if averaging_delay < 0:
        raise ValueError("averaging delay must be nonnegative")
    if gap_tolerance is not None and gap_tolerance < 0.0:
        raise ValueError("gap tolerance must be nonnegative")
    if check_every <= 0:
        raise ValueError("check interval must be positive")

    rows, columns = payoffs.shape
    drop_regrets = np.zeros(rows, dtype=np.float64)
    check_regrets = np.zeros(columns, dtype=np.float64)
    drop_sum = np.zeros(rows, dtype=np.float64)
    check_sum = np.zeros(columns, dtype=np.float64)
    completed = 0

    for iteration in range(1, iterations + 1):
        drop_policy = _regret_matching_plus(drop_regrets)
        check_policy = _regret_matching_plus(check_regrets)
        expected = float(drop_policy @ payoffs @ check_policy)

        drop_regrets = np.maximum(drop_regrets + payoffs @ check_policy - expected, 0.0)
        check_regrets = np.maximum(check_regrets + expected - drop_policy @ payoffs, 0.0)

        weight = max(0, iteration - averaging_delay)
        if weight:
            drop_sum += float(weight) * drop_policy
            check_sum += float(weight) * check_policy
        completed = iteration

        if (
            gap_tolerance is not None
            and iteration % check_every == 0
            and drop_sum.sum() > 0.0
            and check_sum.sum() > 0.0
        ):
            average_drop = drop_sum / drop_sum.sum()
            average_check = check_sum / check_sum.sum()
            lower = float(np.min(payoffs.T @ average_drop))
            upper = float(np.max(payoffs @ average_check))
            if upper - lower <= gap_tolerance:
                break

    if drop_sum.sum() == 0.0 or check_sum.sum() == 0.0:
        # The averaging delay may cover the entire requested run.
        average_drop = _regret_matching_plus(drop_regrets)
        average_check = _regret_matching_plus(check_regrets)
    else:
        average_drop = drop_sum / drop_sum.sum()
        average_check = check_sum / check_sum.sum()

    lower = float(np.min(payoffs.T @ average_drop))
    upper = float(np.max(payoffs @ average_check))
    gap = max(0.0, upper - lower)
    return CFRPlusSolution(
        value=(lower + upper) / 2.0,
        drop_policy=average_drop,
        check_policy=average_check,
        lower_bound=lower,
        upper_bound=upper,
        saddle_gap=gap,
        iterations=completed,
    )
