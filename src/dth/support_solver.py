"""Cheap certified paths to a matrix-game equilibrium, ahead of the oracle.

Every path here returns an answer carrying the *same* certificate
:func:`solver.solve_matrix` produces: the saddle gap of the returned pair
measured against the full payoff matrix, checked against
``SADDLE_GAP_TOLERANCE``.  Nothing is accepted on a weaker test, and every
path fails closed so the caller falls back to HiGHS.

Measured on 600 DTH h2/h3 matrices (2026-07-30, one core):

===============================  ==============  ============================
path                             throughput      applies to
===============================  ==============  ============================
``solve_matrix`` (two HiGHS LPs)   172 solves/s   everything, the oracle
pure saddle point                  ~free          43.7% of matrices
single LP + dual extraction        349 solves/s   everything
double oracle                      3.7 solves/s   correct but far slower here
===============================  ==============  ============================

Two things drive that table.  First, a linprog call costs about 1.3 ms of
fixed marshalling before any simplex work happens -- a 2x2 program costs
1.31 ms and a 60x60 costs 4.36 ms -- so the oracle spends roughly half its
time paying that toll twice.  Extracting the checker's strategy from the
dropper program's duals removes one of the two calls outright.  Second, a
large minority of DTH matrices have a pure saddle point, which two reductions
find for free.

The double oracle is retained and tested but is *not* in the default ladder,
because it is measured slower on this game and the reason is worth recording:
DTH matrices are massively degenerate (at the h4 promotion anchor 33 of 60
checker replies and 31 of 60 dropper actions are tied at the value), so a
small final support does not imply a small sufficient subgame.  The median
run grows the subgame to the full 60x60 before it can certify.  Small support
is not the same property as fast double-oracle convergence, and on this game
only the first one holds.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import linprog

from dth.solver import SADDLE_GAP_TOLERANCE, solve_matrix

__all__ = [
    "SupportSolution",
    "certify",
    "solve_pure_saddle_point",
    "solve_matrix_single_lp",
    "solve_matrix_by_support",
    "solve_certified_matrix_fast",
]


@dataclass(frozen=True)
class SupportSolution:
    """One certified equilibrium plus the work it took to find it."""

    value: float
    drop_policy: np.ndarray
    check_policy: np.ndarray
    saddle_gap: float
    iterations: int
    drop_support: int
    check_support: int
    subgame_rows: int
    subgame_cols: int


def certify(
    matrix: np.ndarray,
    drop_policy: np.ndarray,
    check_policy: np.ndarray,
    *,
    tolerance: float = SADDLE_GAP_TOLERANCE,
) -> tuple[float, float]:
    """Return ``(value, saddle_gap)`` or raise if the pair misses tolerance."""

    lower = float(np.min(matrix.T @ drop_policy))
    upper = float(np.max(matrix @ check_policy))
    gap = max(0.0, upper - lower)
    if gap > tolerance:
        raise RuntimeError(f"saddle gap too large: {gap}")
    return (lower + upper) / 2.0, gap


def solve_pure_saddle_point(
    matrix: np.ndarray,
    *,
    tolerance: float = SADDLE_GAP_TOLERANCE,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Solve games whose maximin and minimax pure strategies already meet.

    Two reductions and two indexing operations, no program of any kind.  Raises
    when the game genuinely needs mixing.
    """

    values = np.asarray(matrix, dtype=np.float64)
    row_mins = values.min(axis=1)
    col_maxes = values.max(axis=0)
    row = int(np.argmax(row_mins))
    col = int(np.argmin(col_maxes))
    if col_maxes[col] - row_mins[row] > tolerance:
        raise RuntimeError("game has no pure saddle point")
    drop = np.zeros(values.shape[0])
    drop[row] = 1.0
    check = np.zeros(values.shape[1])
    check[col] = 1.0
    value, _ = certify(values, drop, check, tolerance=tolerance)
    return value, drop, check


def solve_matrix_single_lp(
    matrix: np.ndarray,
    *,
    tolerance: float = SADDLE_GAP_TOLERANCE,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Solve both sides from one program, taking the checker off its duals.

    The dropper program's inequality multipliers are exactly the checker's
    optimal mixture, so the second HiGHS call the oracle makes is redundant.
    The pair is still certified against the full matrix before it is returned,
    so a degenerate or unusable dual set fails closed instead of being trusted.
    """

    values = np.asarray(matrix, dtype=np.float64)
    rows, cols = values.shape
    result = linprog(
        c=np.concatenate([np.zeros(rows), [-1.0]]),
        A_ub=np.hstack([-values.T, np.ones((cols, 1))]),
        b_ub=np.zeros(cols),
        A_eq=np.hstack([np.ones((1, rows)), np.zeros((1, 1))]),
        b_eq=np.array([1.0]),
        bounds=[(0.0, None)] * rows + [(None, None)],
        method="highs",
    )
    if not result.success:
        raise RuntimeError(f"dropper LP failed: {result.message}")
    drop = np.clip(result.x[:-1], 0.0, None)
    total = drop.sum()
    if total <= 0.0:
        raise RuntimeError("dropper LP returned an empty mixture")
    drop /= total

    marginals = getattr(result, "ineqlin", None)
    if marginals is None:
        raise RuntimeError("solver returned no inequality duals")
    check = np.clip(-np.asarray(marginals.marginals, dtype=np.float64), 0.0, None)
    dual_total = check.sum()
    if dual_total <= 0.0:
        raise RuntimeError("dual mixture is degenerate")
    check /= dual_total

    value, _ = certify(values, drop, check, tolerance=tolerance)
    return value, drop, check


def _seed_indices(matrix: np.ndarray) -> tuple[int, int]:
    """Maximin row and minimax column: the security-level pure strategies."""

    return int(np.argmax(matrix.min(axis=1))), int(np.argmin(matrix.max(axis=0)))


def _solve_subgame(
    submatrix: np.ndarray,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Solve a restricted game, closed form when it is 1xN or Nx1."""

    rows, cols = submatrix.shape
    if rows == 1:
        column = int(np.argmin(submatrix[0]))
        check = np.zeros(cols)
        check[column] = 1.0
        return float(submatrix[0, column]), np.ones(1), check
    if cols == 1:
        row = int(np.argmax(submatrix[:, 0]))
        drop = np.zeros(rows)
        drop[row] = 1.0
        return float(submatrix[row, 0]), drop, np.ones(1)
    return solve_matrix(submatrix)


def solve_matrix_by_support(
    matrix: np.ndarray,
    *,
    tolerance: float = SADDLE_GAP_TOLERANCE,
    max_iterations: int | None = None,
) -> SupportSolution:
    """Grow a restricted subgame one best response at a time (double oracle).

    Correct but, on DTH's degenerate matrices, slower than solving the whole
    program -- see the module docstring.  Kept because the negative result is
    worth preserving and because it is the right algorithm on games whose
    equilibria are genuinely low-dimensional.

    Raises ``RuntimeError`` if it cannot certify within tolerance, so callers
    fall back rather than accept a weaker answer.
    """

    values = np.asarray(matrix, dtype=np.float64)
    if values.ndim != 2 or values.size == 0:
        raise ValueError("support solving needs a non-empty two-dimensional matrix")
    if not np.all(np.isfinite(values)):
        raise ValueError("support solving needs a finite matrix")

    rows, cols = values.shape
    seed_row, seed_col = _seed_indices(values)
    row_set = [seed_row]
    col_set = [seed_col]
    limit = rows + cols if max_iterations is None else int(max_iterations)

    for iteration in range(1, limit + 1):
        row_index = np.asarray(row_set, dtype=np.intp)
        col_index = np.asarray(col_set, dtype=np.intp)
        try:
            subgame_value, sub_drop, sub_check = _solve_subgame(
                values[np.ix_(row_index, col_index)]
            )
        except RuntimeError as error:
            raise RuntimeError(f"restricted subgame failed: {error}") from error

        drop = np.zeros(rows)
        drop[row_index] = sub_drop
        check = np.zeros(cols)
        check[col_index] = sub_check

        # Best responses in the FULL matrix; this is what certifies the answer.
        row_payoffs = values @ check
        col_payoffs = values.T @ drop
        upper = float(np.max(row_payoffs))
        lower = float(np.min(col_payoffs))
        gap = max(0.0, upper - lower)
        if gap <= tolerance:
            return SupportSolution(
                value=(lower + upper) / 2.0,
                drop_policy=drop,
                check_policy=check,
                saddle_gap=gap,
                iterations=iteration,
                drop_support=int(np.count_nonzero(drop > 0.0)),
                check_support=int(np.count_nonzero(check > 0.0)),
                subgame_rows=len(row_set),
                subgame_cols=len(col_set),
            )

        # Add the most profitable deviation for whichever side has one, judged
        # against the *restricted* game's value -- the quantity the lifted pair
        # actually guarantees.  Exact ties, which are abundant in DTH, must not
        # count as deviations or the subgame grows without improving.
        grew = False
        best_row = int(np.argmax(row_payoffs))
        if (
            row_payoffs[best_row] > subgame_value + tolerance / 2.0
            and best_row not in row_set
        ):
            row_set.append(best_row)
            grew = True
        best_col = int(np.argmin(col_payoffs))
        if (
            col_payoffs[best_col] < subgame_value - tolerance / 2.0
            and best_col not in col_set
        ):
            col_set.append(best_col)
            grew = True
        if not grew:
            raise RuntimeError(
                f"support solving stalled at saddle gap {gap} with no new deviation"
            )

    raise RuntimeError(f"support solving did not converge within {limit} iterations")


def solve_certified_matrix_fast(
    matrix: np.ndarray,
) -> tuple[float, np.ndarray, np.ndarray, str]:
    """Cheapest certified path first, HiGHS as the retained oracle.

    Mirrors :func:`solver.solve_certified_matrix`'s signature, and never
    returns an answer that solver's own tolerance would reject.
    """

    values = np.asarray(matrix, dtype=np.float64)
    try:
        value_, drop, check = solve_pure_saddle_point(values)
    except (ValueError, RuntimeError):
        pass
    else:
        return value_, drop, check, "pure-saddle-point"
    try:
        value_, drop, check = solve_matrix_single_lp(values)
    except (ValueError, RuntimeError, AttributeError, np.linalg.LinAlgError):
        pass
    else:
        return value_, drop, check, "single-lp-dual"
    value_, drop, check = solve_matrix(values)
    return value_, drop, check, "highs"
