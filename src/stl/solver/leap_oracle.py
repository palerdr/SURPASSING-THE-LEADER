"""Scalar behavioral authority for the certified leap stage."""
from dataclasses import dataclass
import math

import numpy as np
from scipy.optimize import linprog

GAP = 1e-6
CERT_RADIUS = 1e-10
_fma = getattr(math, "fma", lambda a, b, c: a*b+c)


@dataclass(frozen=True)
class StageResult:
    value: float
    kind: int
    lower: float
    upper: float
    support: tuple[int, int] | None = None

    @property
    def gap(self):
        return self.upper - self.lower


def stage_matrix(s, f, window):
    matrix = np.full((60+int(window), 60), f, dtype=np.float64)
    for row in range(60):
        matrix[row, row:] = s[:60-row]
    return matrix


def policy_masks(p, q):
    return tuple(int.from_bytes(np.packbits(np.asarray(x) > 0, bitorder='little').tobytes(), 'little')
                 for x in (p, q))


def solve_lp(s, f, window, *, support=False):
    matrix = stage_matrix(s, f, window)
    rows = len(matrix)
    failure = 'HiGHS failed the 1e-6 saddle gate'
    # We retain the successful path of each prior retry. A common offset
    # changes the LP coordinates; we certify against the original matrix.
    attempts = [('highs', None), ('highs-ipm', None),
                ('highs-ipm', f), ('highs-ipm', float(matrix[0, 0]))]
    for method, offset in attempts:
        lp_matrix = matrix if offset is None else matrix-offset
        result = linprog(np.r_[np.zeros(rows), -1.],
                         A_ub=np.c_[-lp_matrix.T, np.ones(60)], b_ub=np.zeros(60),
                         A_eq=[np.r_[np.ones(rows), 0.]], b_eq=[1.],
                         bounds=[(0, None)]*rows+[(None, None)], method=method,
                         options={'primal_feasibility_tolerance': 1e-9,
                                  'dual_feasibility_tolerance': 1e-9})
        if not result.success:
            failure = f'HiGHS failed: {result.message}'
            continue
        p = np.maximum(result.x[:-1], 0); q = np.maximum(-result.ineqlin.marginals, 0)
        p /= p.sum(); q /= q.sum()
        lower = float((p @ matrix).min()); upper = float((matrix @ q).max())
        if not math.isfinite(upper-lower) or upper-lower > GAP:
            failure = 'HiGHS failed the 1e-6 saddle gate'
            continue
        return StageResult((lower+upper)/2, 3, lower, upper, policy_masks(p, q) if support else None)
    raise RuntimeError(failure)


def solve_stage(s, f, window, *, allow_lp=True):
    s = np.asarray(s, dtype=np.float64)
    if s.shape != (60,) or not np.isfinite(s).all() or not math.isfinite(f):
        raise ValueError('stage needs 60 finite successes and a finite failure')
    f = float(f); s = list(map(float, s))
    lo, hi = min(s), max(s)
    lower, upper = max(lo, min(f, s[0])), min(hi, max(f, s[0]))
    if window:
        lower, upper = max(lower, f), max(upper, f)
    kind = 0
    if upper-lower <= GAP:
        value = (lower+upper)/2
        if window and f >= (max(lo, min(f, s[0]))+min(hi, max(f, s[0])))/2:
            kind = 2
        return StageResult(value, kind, lower, upper)
    d = s[0]-f
    if abs(d) >= 1e-12:
        b = [0.] + [(s[k-1]-s[k])/d for k in range(1, 60)]
        r = [1.]
        for k in range(1, 60):
            acc = 0.
            try:
                for j in range(k):
                    acc = _fma(b[k-j], r[j], acc)
            except (OverflowError, ValueError):
                acc = math.nan
            r.append(acc)
            if not math.isfinite(acc):
                break
        q = [max(x, 0.) if math.isfinite(x) else math.nan for x in reversed(r)]
        total = sum(q)
        if math.isfinite(total) and total > 0:
            if min(r) >= 0 and lo >= -2 and hi <= 2 and abs(f) <= 2 and abs(d) >= GAP:
                value = f+d/total
                lower, upper = value-CERT_RADIUS, value+CERT_RADIUS
            else:
                q = np.array(q)/total
                payoffs = stage_matrix(s, f, False) @ q
                lower, upper = float(payoffs.min()), float(payoffs.max())
                value = (lower+upper)/2
            if window:
                lower, upper = max(lower, f), max(upper, f)
                kind = 2 if f >= value else 1
                value = max(value, f)
            else:
                kind = 1
            if math.isfinite(upper-lower) and upper-lower <= GAP:
                return StageResult(value, kind, lower, upper)
    if allow_lp:
        return solve_lp(s, f, window)
    return StageResult(math.nan, 255, -math.inf, math.inf)
