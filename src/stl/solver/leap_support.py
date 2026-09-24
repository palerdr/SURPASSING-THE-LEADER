"""We certify reduced support guesses before sending a stage to HiGHS."""
from dataclasses import dataclass
import math

import numpy as np

from stl.solver.leap_oracle import GAP, StageResult, stage_matrix

ALL_ACTIONS = (1 << 60)-1
EDGE_LIMIT = 64


@dataclass(frozen=True)
class SupportResult:
    stage: StageResult
    p: np.ndarray
    q: np.ndarray
    p_mask: int
    q_mask: int
    attempts: int


def reverse_support(mask):
    return int(f'{mask:060b}'[::-1], 2)


def kink_index(s):
    """We return the first index k of the most negative step s[k]-s[k-1], or None.

    A nondecreasing s with f > s[0] gives nonnegative recurrence weights, so
    a residue stage needs at least one negative step.
    """
    s = np.asarray(s, np.float64)
    steps = s[1:]-s[:-1]
    k = int(np.argmin(steps))
    return k+1 if steps[k] < 0 else None


def holes(mask):
    """We list the maximal runs of absent actions as inclusive (start, end) pairs."""
    runs = []; i = 0
    while i < 60:
        if mask >> i & 1:
            i += 1
            continue
        start = i
        while i+1 < 60 and not mask >> (i+1) & 1:
            i += 1
        runs.append((start, i)); i += 1
    return runs


def move_holes(mask, near, old, delta):
    """We move each hole with the nearer of the anchors near and 59-old.

    A hole at the first anchor moves by delta and a hole at the second by
    -delta. A hole that starts at action 0 keeps its place. We clip moved
    holes to actions 0..59.
    """
    far = 59-old; out = ALL_ACTIONS
    for start, end in holes(mask):
        shift = 0 if start == 0 else (delta if abs(end-near) <= abs(end-far) else -delta)
        for x in range(max(start+shift, 0), min(end+shift, 59)+1):
            out &= ~(1 << x)
    return out


def kink_seed(p_mask, q_mask, old, new):
    """We move a certified support from kink old to kink new.

    Dropper holes use the anchors old and 59-old; Checker holes use old-1 and
    59-old. We return the masks unchanged when either kink is absent or the
    two kinks agree.
    """
    if old is None or new is None or old == new:
        return p_mask, q_mask
    delta = new-old
    return move_holes(p_mask, old, old, delta), move_holes(q_mask, old-1, old, delta)


def recurrence(s, f):
    d = s[0]-f
    if abs(d) < 1e-12:
        return None
    b = np.r_[0., (s[:-1]-s[1:])/d]
    r = np.zeros(60); r[0] = 1.
    for k in range(1, 60):
        acc = 0.
        for j in range(k):
            acc = math.fma(float(b[k-j]), float(r[j]), acc)
        r[k] = acc
    return r if np.isfinite(r).all() else None


def reduced_weights(r, p_mask, q_mask):
    """We solve the hole constraints in g, then reconstruct the Checker mix."""
    if not p_mask or not q_mask:
        return None
    rows = [i for i in range(60) if p_mask >> i & 1]
    columns = np.array([bool(q_mask >> i & 1) for i in range(60)])
    g = np.array([i for i in range(59) if not (p_mask >> i & 1 and p_mask >> (i+1) & 1)]+[59])
    equations = []
    for j in np.flatnonzero(~columns):
        equations.append(np.array([r[t-j] if t >= j else 0. for t in g]))
    for a, c in zip(rows, rows[1:]):
        if c > a+1:
            equations.append(((g >= a) & (g < c)).astype(float))
    equations.append(np.cumsum(r)[g])
    if len(equations) != len(g):
        return None
    a = np.array(equations); rhs = np.zeros(len(g)); rhs[-1] = 1.
    try:
        weights = np.linalg.solve(a, rhs)
    except np.linalg.LinAlgError:
        return None
    q = np.zeros(60)
    for j in np.flatnonzero(columns):
        q[j] = sum(r[t-j]*x for t, x in zip(g, weights) if t >= j)
    if not np.isfinite(q).all() or np.min(q) < 0 or q.sum() <= 0:
        return None
    return q/q.sum()


def shifted_isolated_holes(mask, mode):
    result = mask
    for i in range(60):
        if mask >> i & 1:
            continue
        if (i > 0 and not (mask >> (i-1) & 1)) or (i < 59 and not (mask >> (i+1) & 1)):
            continue
        direction = (-1 if i < 30 else 1) if mode == 0 else ((1 if i < 30 else -1) if mode == 1 else (-1 if mode == 2 else 1))
        target = i+direction
        if 0 <= target < 60 and mask >> target & 1:
            result = (result | (1 << i)) & ~(1 << target)
    return result


def edge_candidates(p_mask, q_mask):
    """We bound edge probes so an unsuccessful guess still reaches HiGHS."""
    seen = {(p_mask, q_mask)}; count = 0
    # We move isolated holes together when two edges travel in opposite
    # directions, then try individual paired support boundaries.
    for mode in range(4):
        pair = (shifted_isolated_holes(p_mask, mode), shifted_isolated_holes(q_mask, mode))
        if pair not in seen:
            seen.add(pair); yield pair; count += 1
    for shift in range(59):
        if ((p_mask >> shift) ^ (p_mask >> (shift+1))) & 1 and ((q_mask >> (58-shift)) ^ (q_mask >> (59-shift))) & 1:
            pair = (p_mask ^ (3 << shift), q_mask ^ (3 << (58-shift)))
            if pair not in seen:
                seen.add(pair); yield pair; count += 1
    def neighbors(mask):
        output = [mask]
        i = 0
        while i < 60:
            if mask >> i & 1:
                i += 1
                continue
            start = i
            while i+1 < 60 and not (mask >> (i+1) & 1):
                i += 1
            end = i
            if start > 0:
                output.append(mask ^ (1 << (start-1)) ^ (1 << end))
            if end < 59:
                output.append(mask ^ (1 << start) ^ (1 << (end+1)))
            i += 1
        for i in range(60):
            bit = (mask >> i) & 1
            boundary = (i > 0 and ((mask >> (i-1)) & 1) != bit) or (i < 59 and ((mask >> (i+1)) & 1) != bit)
            if boundary:
                output.append(mask ^ (1 << i))
            if i < 59 and ((mask >> (i+1)) & 1) != bit:
                output.append(mask ^ (3 << i))
        return output
    for pp in neighbors(p_mask):
        for qq in neighbors(q_mask):
            pair = (pp, qq)
            if pp.bit_count() == qq.bit_count() and pair not in seen:
                if count >= EDGE_LIMIT:
                    return
                seen.add(pair); yield pair; count += 1


def solve_supported(s, f, window, p_mask, q_mask, *, edges=True, max_attempts=None):
    """We accept a guessed support only after a complete saddle certificate.

    You can cap the number of supports we try, counting the guess as the
    first, with max_attempts.
    """
    s = np.asarray(s, np.float64); p_mask = int(p_mask); q_mask = int(q_mask)
    if s.shape != (60,) or not np.isfinite(s).all() or not math.isfinite(f):
        raise ValueError('stage needs 60 finite successes and a finite failure')
    if min(p_mask, q_mask) < 0 or (p_mask | q_mask) & ~ALL_ACTIONS:
        raise ValueError('support bits must name actions 0 through 59')
    if not p_mask or not q_mask:
        return None
    r = recurrence(s, f)
    if r is None:
        return None
    candidates = [(p_mask, q_mask)]
    if edges:
        candidates.extend(edge_candidates(p_mask, q_mask))
    matrix = stage_matrix(s, f, False)
    for attempt, (pp, qq) in enumerate(candidates, 1):
        if max_attempts is not None and attempt > max_attempts:
            return None
        q = reduced_weights(r, pp, qq)
        if q is None:
            continue
        reverse_p = reduced_weights(r, reverse_support(qq), reverse_support(pp))
        if reverse_p is None:
            continue
        p = reverse_p[::-1].copy()
        lower = float((p @ matrix).min()); upper = float((matrix @ q).max())
        square_value = (lower+upper)/2
        kind = 4 if attempt == 1 else 5
        if window:
            lower, upper = max(lower, f), max(upper, f)
            if f >= square_value:
                kind = 2
            if f >= float((p @ matrix).min()):
                p = np.r_[np.zeros(60), 1.]
            else:
                p = np.r_[p, 0.]
        if math.isfinite(upper-lower) and -1e-12 <= upper-lower <= GAP:
            return SupportResult(StageResult(max(square_value, f) if window else square_value, kind, lower, upper), p, q, pp, qq, attempt)
    return None
