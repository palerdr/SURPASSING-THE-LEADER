"""Drop the Handkerchief tablebase: one certified game value per state class, from a single backward
pass over the potential layers phi = 1200..0 (architecture.md)."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import numba as nb
import numpy as np
from numba import njit, prange
from scipy.optimize import linprog

# ---- rules and sizes ---------------------------------------------------------------------------
MAX_ST = 299                  # squandered time 0..299
MAX_TTD = 300                 # accrued toxin time 0..300
CAPACITY = 300                # a successful check that grows ST to >= CAPACITY wins for the mover
DOSE = 60                     # toxin time taken on a failed check
LAGS = 60                     # actions 1..60; a successful check elapses lag = c - d + 1 seconds
N_ALIVE = 16711               # alive profiles: ids 0..16710 in the normative order
N_DEAD = 300                  # dead sentinels: id N_ALIVE + st
N = N_ALIVE + N_DEAD          # 17011 profiles
CLASSES = N * N               # 289,374,121 state classes; class (pc, pd) is V[pc, pd]
WIN = N                       # child column meaning "the mover wins": V[:, WIN] == -1.0, so -V reads +1
DEAD_PHI = 301                # potential offset (and stored TTD) of a dead profile
MAX_PHI = MAX_ST + DEAD_PHI   # profile potential 0..600
MAX_LAYER = 2 * MAX_PHI       # class potential 0..1200
MAX_SADDLE_GAP = 1e-6         # the certificate gate
PURE, SUPPORT, LP = 0, 1, 2   # solver rungs, stored in K
UNSOLVED = 255                # K sentinel; V uses NaN
RECHECK_SAMPLES = 1200        # classes re-derived independently in finalize
ARTIFACTS = Path(__file__).resolve().parent / "artifacts"   # V.npy and K.npy land here (gitignored)

ANCHORS = [  # ((s_c, t_c, s_d, t_d), certified value), architecture.md section 6
    ((0, 0, 0, 0), 0.08985007280951046),
    ((240, 0, 240, 0), 0.3372132166291093),
    ((10, 60, 200, 0), -0.7944428916469297),
    ((150, 90, 30, 120), 0.7244093036356785),
    ((250, 300, 40, 0), 0.9981152817381969),
    ((100, 140, 100, 140), 0.1877386378276193),
]


@dataclass
class ProfileTable:
    alive_id: np.ndarray          # (MAX_ST + 1, MAX_TTD + 1) int32; -1 where not alive / off-domain
    st: np.ndarray                # (N,) int32
    ttd: np.ndarray               # (N,) int32; DEAD_PHI for dead profiles
    alive: np.ndarray             # (N,) bool
    phi: np.ndarray               # (N,) int32 = st + ttd
    rev: np.ndarray               # (N,) float64 revival probability (0 for dead)
    fail_child: np.ndarray        # (N,) int32; failed-check child id, or WIN
    success_children: np.ndarray  # (N, LAGS) int32; [p, lag - 1] = child id, or WIN
    bucket: list[np.ndarray]      # MAX_PHI + 1 int32 arrays; bucket[v] = ascending ids with phi == v


@dataclass
class SolveResult:
    maximin: float      # lower bound on the value: the Checker's best reply to the Dropper's mix
    minimax: float      # upper bound on the value: the Dropper's best reply to the Checker's mix
    kind: int           # PURE / SUPPORT / LP
    drop: np.ndarray | None = None    # the certifying Dropper mix (rows), when the rung produced one
    check: np.ndarray | None = None   # the certifying Checker mix (columns)
    saddle_gap: float = field(init=False)
    certified: bool = field(init=False)
    value: float | None = field(init=False)   # the certificate midpoint; None when uncertified

    def __post_init__(self):
        self.maximin = float(self.maximin)
        self.minimax = float(self.minimax)
        self.saddle_gap = self.minimax - self.maximin
        self.certified = bool(np.isfinite(self.saddle_gap) and self.saddle_gap <= MAX_SADDLE_GAP)
        self.value = 0.5 * (self.maximin + self.minimax) if self.certified else None


# ---- profile tables ----------------------------------------------------------------------------
def survives_injection(s: int, t: int) -> bool:
    dose = s + DOSE
    return dose <= MAX_ST and dose + t <= MAX_TTD


def revival_probability(s: int, t: int) -> float:
    if not survives_injection(s, t):
        return 0.0
    return 0.95 * (1 - (s / 240.0)) * (0.75 ** (t / 60.0))


def build_table() -> ProfileTable:
    """Profiles in the normative order, every transition table, and the phi buckets (architecture.md 4.1)."""
    alive_id = np.full((MAX_ST + 1, MAX_TTD + 1), -1, dtype=np.int32)
    st = np.empty(N, dtype=np.int32)
    ttd = np.empty(N, dtype=np.int32)

    next_id = 0
    for t in [0] + list(range(60, MAX_TTD + 1)):
        for s in range(MAX_ST + 1):
            if survives_injection(s, t):
                alive_id[s, t] = next_id
                st[next_id] = s
                ttd[next_id] = t
                next_id += 1
    assert next_id == N_ALIVE, "Wrong number of Alive Profiles indexed into profile table"

    st[N_ALIVE:] = np.arange(N_DEAD, dtype=np.int32)
    ttd[N_ALIVE:] = DEAD_PHI

    alive = np.arange(N) < N_ALIVE
    phi = st + ttd

    succ = np.empty((N, LAGS), dtype=np.int32)
    fail = np.empty(N, dtype=np.int32)
    rev = np.zeros(N, dtype=np.float64)

    for p in range(N):
        s, t = int(st[p]), int(ttd[p])
        is_alive = p < N_ALIVE
        for lag in range(1, LAGS + 1):
            grown = s + lag
            if grown >= CAPACITY:
                succ[p, lag - 1] = WIN
            elif is_alive and survives_injection(grown, t):
                succ[p, lag - 1] = alive_id[grown, t]
            else:
                succ[p, lag - 1] = N_ALIVE + grown
        if is_alive:
            rev[p] = revival_probability(s, t)
            new_ttd = s + t + DOSE
            fail[p] = alive_id[0, new_ttd] if survives_injection(0, new_ttd) else N_ALIVE
        else:
            fail[p] = WIN

    bucket = [np.flatnonzero(phi == v).astype(np.int32) for v in range(MAX_PHI + 1)]

    has_child = succ != WIN
    assert (phi[succ[has_child]] > np.broadcast_to(phi[:, None], succ.shape)[has_child]).all(), \
        "phi must strictly increase on successful checks"
    has_child = fail != WIN
    assert (phi[fail[has_child]] > phi[has_child]).all(), "phi must strictly increase on failed checks"

    return ProfileTable(
        alive_id=alive_id, st=st, ttd=ttd, alive=alive, phi=phi, rev=rev,
        fail_child=fail, success_children=succ, bucket=bucket,
    )


def profile(s: int, t: int, table: ProfileTable) -> int:
    """Profile id of (ST, TTD); dead profiles collapse to the sentinel for their ST."""
    if not survives_injection(s, t):
        return N_ALIVE + s
    pid = int(table.alive_id[s, t])
    if pid == -1:
        raise IndexError("alive TTD in 1..59 is off-domain")
    return pid


def encode_state(sc: int, tc: int, sd: int, td: int, table: ProfileTable) -> int:
    """Flat class index pc * N + pd of the state (s_c, t_c, s_d, t_d): Checker first, as everywhere else."""
    return profile(sc, tc, table) * N + profile(sd, td, table)


def decode_class(c: int) -> tuple[int, int]:
    return divmod(int(c), N)


# ---- values and layers -------------------------------------------------------------------------
def values_array() -> np.ndarray:
    """V[pc, pd] = value; NaN until solved; the extra column WIN holds -1.0."""
    V = np.full((N, N + 1), np.nan, dtype=np.float64)
    V[:, WIN] = -1.0
    return V


def kinds_array() -> np.ndarray:
    return np.full((N, N), UNSOLVED, dtype=np.uint8)


def class_values(pc: int, pd: int, V: np.ndarray, table: ProfileTable) -> tuple[np.ndarray, float]:
    """The 60 successful-check values and the failed-check value of one class (architecture.md 4.3)."""
    s = np.empty(LAGS, dtype=np.float64)
    for lag in range(1, LAGS + 1):
        child = int(table.success_children[pc, lag - 1])
        s[lag - 1] = 1.0 if child == WIN else -V[pd, child]
    child = int(table.fail_child[pc])
    if child == WIN:
        f = 1.0
    else:
        p = float(table.rev[pc])
        f = p * (-V[pd, child]) + (1.0 - p)
    if not (np.isfinite(s).all() and np.isfinite(f)):
        raise RuntimeError(f"unsolved child read for class ({pc}, {pd}): schedule bug")
    return s, float(f)


def layer_pairs(P: int, table: ProfileTable) -> tuple[np.ndarray, np.ndarray]:
    """All classes (pc, pd) with phi[pc] + phi[pd] == P, pd-major so neighbours read the same row of V."""
    pcs, pds = [], []
    for a in range(max(0, P - MAX_PHI), min(MAX_PHI, P) + 1):
        bc, bd = table.bucket[a], table.bucket[P - a]
        if bc.size and bd.size:
            pds.append(np.repeat(bd, bc.size))
            pcs.append(np.tile(bc, bd.size))
    if not pcs:
        return np.empty(0, dtype=np.int32), np.empty(0, dtype=np.int32)
    return np.concatenate(pcs), np.concatenate(pds)


# ---- solver ladder, scalar (the oracle: tests and the finalize recheck) --------------------------
def _M_dot(s: np.ndarray, f: float, q: np.ndarray) -> np.ndarray:
    # (M q)[i] = f * sum_{j<i} q[j] + sum_{k>=i} s[k-i] q[k]
    n = len(q)
    Mq = np.zeros(n)
    Mq[1:] = f * np.cumsum(q)[:-1]
    for j in range(n):
        Mq[:n - j] += s[j] * q[j:]
    return Mq


def full_matrix(s: np.ndarray, f: float) -> np.ndarray:
    # M[d, c] = s[c - d] when c >= d (successful check with lag c - d + 1), else f (failed check)
    n = len(s)
    g = np.arange(n)[None, :] - np.arange(n)[:, None]
    return np.where(g >= 0, s[g.clip(min=0)], f)


def try_rung1(s: np.ndarray, f: float) -> SolveResult:
    """Pure saddle point. Constant: ~120 comparisons (min and max of the 60 success values); row d of M is
    (d-1) copies of f then s[0:n-d+1], so the row/column scans collapse to min/max of s. The maximin row is
    row 1 (minimum s) or row 60 (min(f, s[0])); the minimax column is column 60 (maximum s) or column 1
    (max(f, s[0])). Those pure actions are the certifying pair."""
    n = len(s)
    s0, lo, hi = s[0], s.min(), s.max()
    drop = np.zeros(n)
    check = np.zeros(n)
    drop[0 if lo >= min(f, s0) else n - 1] = 1.0
    check[n - 1 if hi <= max(f, s0) else 0] = 1.0
    return SolveResult(maximin=max(lo, min(f, s0)), minimax=min(hi, max(f, s0)), kind=PURE, drop=drop, check=check)


def try_rung2(s: np.ndarray, f: float) -> SolveResult | None:
    """Full-support equalizer. Constant: ~3.6k multiply-adds (a 59-step recurrence and one Toeplitz matvec,
    1,770 each) plus 60 divisions.

    Consecutive rows of M differ by d0 = s[0] - f on the diagonal and dS = diff(s) above it, so
    (M q)[i] == (M q)[i+1] for all i becomes r[k] = -sum_{m<k} dS[m] r[k-1-m] / d0 with r[0] = 1, and
    q = clip(reverse(r), 0) / sum (Checker). M is persymmetric, M[d, c] == M[n-1-c, n-1-d], so for the
    mirror p = reverse(q) (Dropper) (p M)[c] == (M q)[n-1-c] whatever q is: the max and min of M q are the
    two-sided certificate bounds, clipped or not.
    """
    n = len(s)
    d0 = s[0] - f
    dS = np.diff(s)
    if abs(d0) < 1e-12:
        return None
    r = np.zeros(n)
    r[0] = 1.0
    with np.errstate(over="ignore", invalid="ignore"):
        for k in range(1, n):
            r[k] = -np.dot(dS[:k], r[k - 1::-1]) / d0
    if not np.all(np.isfinite(r)):
        return None
    q = np.clip(r[::-1], 0.0, None)
    if q.sum() <= 0:
        return None
    q /= q.sum()
    Mq = _M_dot(s, f, q)
    return SolveResult(maximin=Mq.min(), minimax=Mq.max(), kind=SUPPORT, drop=q[::-1].copy(), check=q)


def try_rung3(s: np.ndarray, f: float) -> SolveResult:
    """LP residue. Constant but ~1000x heavier than rung 2: one 60-variable LP on a dense 60x61 tableau
    (~n^2 work per simplex pivot, ~n pivots), about 1 ms per call including scipy/HiGHS overhead."""
    s, f = np.asarray(s, dtype=float), float(f)
    n = len(s)
    M = full_matrix(s, f)
    # maximize v  s.t.  M^T p >= v * 1,  sum(p) = 1,  p >= 0   (the Dropper's mix p; q from the duals)
    c = np.zeros(n + 1)
    c[n] = -1.0
    A_ub = np.hstack([-M.T, np.ones((n, 1))])
    A_eq = np.zeros((1, n + 1))
    A_eq[0, :n] = 1.0
    bnds = [(0, None)] * n + [(None, None)]

    best = None
    for method in ("highs-ds", "highs-ipm"):
        res = linprog(c, A_ub=A_ub, b_ub=np.zeros(n), A_eq=A_eq, b_eq=[1.0], bounds=bnds, method=method)
        if not res.success:
            continue
        p = np.clip(res.x[:n], 0, None)
        p /= p.sum()
        q = np.clip(-res.ineqlin.marginals, 0, None)
        if q.sum() <= 0:
            continue
        q /= q.sum()
        out = SolveResult(maximin=(p @ M).min(), minimax=(M @ q).max(), kind=LP, drop=p, check=q)
        if out.certified:
            return out
        if best is None or out.saddle_gap < best.saddle_gap:
            best = out
    return best if best is not None else SolveResult(maximin=np.nan, minimax=np.nan, kind=LP)


def solve_class(s: np.ndarray, f: float) -> SolveResult:
    """The ladder for one class (architecture.md 4.7)."""
    res = try_rung1(s, f)
    if res.certified:
        return res
    res = try_rung2(s, f)
    if res is not None and res.certified:
        return res
    res = try_rung3(s, f)
    if not res.certified:
        raise RuntimeError(f"uncertified class (gap {res.saddle_gap}) - aborting build")
    return res


# ---- solver ladder, numba kernel (rungs 1-2 fused per class; the LP stays in scipy) -------------
@njit(cache=True)
def _ladder(s, f, r, q):
    """Rungs 1-2 on one class (~120 comparisons, then ~3.6k multiply-adds) in L1-resident scratch.
    Returns (value, kind); kind == LP means 'not certified here, hand it to the LP'."""
    n = s.shape[0]
    lo = s[0]
    hi = s[0]
    for k in range(1, n):
        if s[k] < lo:
            lo = s[k]
        if s[k] > hi:
            hi = s[k]
    s0 = s[0]
    maximin = max(lo, min(f, s0))
    minimax = min(hi, max(f, s0))
    if minimax - maximin <= MAX_SADDLE_GAP:
        return 0.5 * (maximin + minimax), PURE

    d0 = s0 - f
    if abs(d0) < 1e-12:
        return 0.0, LP
    # r[k] = -sum_{m<k} (s[m+1] - s[m]) r[k-1-m] / d0
    r[0] = 1.0
    for k in range(1, n):
        acc = 0.0
        for m in range(k):
            acc += (s[m + 1] - s[m]) * r[k - 1 - m]
        r[k] = -acc / d0
    # q = clip(reverse(r), 0) / sum
    qsum = 0.0
    for k in range(n):
        rk = r[n - 1 - k]
        if not np.isfinite(rk):
            return 0.0, LP
        qk = rk if rk > 0.0 else 0.0
        q[k] = qk
        qsum += qk
    if qsum <= 0.0:
        return 0.0, LP
    for k in range(n):
        q[k] /= qsum
    # (M q)[i] = f * sum_{j<i} q[j] + sum_{k>=i} s[k-i] q[k]; track only its min and max
    cum = 0.0
    mn = np.inf
    mx = -np.inf
    for i in range(n):
        acc = f * cum
        for k in range(i, n):
            acc += s[k - i] * q[k]
        if acc < mn:
            mn = acc
        if acc > mx:
            mx = acc
        cum += q[i]
    if mx - mn <= MAX_SADDLE_GAP:
        return 0.5 * (mn + mx), SUPPORT
    return 0.0, LP


@njit(parallel=True, cache=True)
def _solve_layer_kernel(pcs, pds, V, K, succ, fail, rev, scratch, need_lp):
    """One layer in parallel: gather the 61 children of each class, run the fused ladder, write V and K.
    Every read is a finished layer and every write is this class's own cell, so there are no races.
    prange splits the layer statically across threads (numba's workqueue layer; tbb has no macOS-arm64 wheel)."""
    for i in prange(pcs.shape[0]):
        tid = nb.get_thread_id()
        s = scratch[tid, 0]
        r = scratch[tid, 1]
        q = scratch[tid, 2]
        pc = pcs[i]
        pd = pds[i]
        for k in range(LAGS):
            s[k] = -V[pd, succ[pc, k]]
        p = rev[pc]
        f = p * (-V[pd, fail[pc]]) + (1.0 - p)
        value, kind = _ladder(s, f, r, q)
        if kind == LP:
            need_lp[i] = True
        else:
            need_lp[i] = False
            V[pc, pd] = value
            K[pc, pd] = kind


def solve_layer(P: int, V: np.ndarray, K: np.ndarray, table: ProfileTable, scratch: np.ndarray) -> tuple[int, np.ndarray]:
    """Solve layer P: the kernel for rungs 1-2, then the LP worklist. Returns (classes, counts by rung)."""
    pcs, pds = layer_pairs(P, table)
    if pcs.size == 0:
        return 0, np.zeros(3, dtype=np.int64)
    need_lp = np.empty(pcs.size, dtype=np.bool_)
    _solve_layer_kernel(pcs, pds, V, K, table.success_children, table.fail_child, table.rev, scratch, need_lp)
    for i in np.flatnonzero(need_lp):
        pc, pd = int(pcs[i]), int(pds[i])
        res = try_rung3(*class_values(pc, pd, V, table))
        if not res.certified:
            raise RuntimeError(f"uncertified class ({pc}, {pd}) (gap {res.saddle_gap}) - aborting build")
        V[pc, pd] = res.value
        K[pc, pd] = LP
    return pcs.size, np.bincount(K[pcs, pds], minlength=3).astype(np.int64)


# ---- sweep -------------------------------------------------------------------------------------
def recheck(c: int, V: np.ndarray, table: ProfileTable) -> SolveResult:
    """Re-derive one class from its stored children with the scalar ladder (architecture.md 4.10)."""
    pc, pd = decode_class(c)
    s, f = class_values(pc, pd, V, table)
    return solve_class(s, f)


def finalize(V: np.ndarray, K: np.ndarray, table: ProfileTable, log=print) -> None:
    """Every class solved and in range, every rung recorded, and 1,200 strided classes re-derived (4.9)."""
    core = V[:, :N]
    assert not np.isnan(core).any(), "unsolved classes remain"
    assert (np.abs(core) <= 1 + 1e-9).all(), "a value left [-1, 1]"
    assert (K <= LP).all(), "a class has no rung recorded"
    worst = 0.0
    for c in np.linspace(0, CLASSES - 1, RECHECK_SAMPLES).astype(np.int64):
        pc, pd = decode_class(c)
        err = abs(recheck(c, V, table).value - V[pc, pd])
        worst = max(worst, err)
        assert err <= MAX_SADDLE_GAP, f"recheck mismatch at class {c}: {err}"
    log(f"finalize: {RECHECK_SAMPLES} rechecks ok, worst |diff| = {worst:.3e}")


def verify_anchors(V: np.ndarray, table: ProfileTable) -> list[tuple[tuple, float, float, bool]]:
    """(state, expected, got, ok) for the six reference values of architecture.md section 6."""
    out = []
    for (sc, tc, sd, td), expected in ANCHORS:
        c = encode_state(sc, tc, sd, td, table)
        pc, pd = decode_class(c)
        got = float(V[pc, pd])
        out.append(((sc, tc, sd, td), expected, got, abs(got - expected) <= MAX_SADDLE_GAP))
    return out


def build(table: ProfileTable, stop_at: int = 0, log=print) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The whole sweep (architecture.md 4.8): layers MAX_LAYER down to stop_at, N^2 = 289,374,121 classes
    times the constant per-class ladder. Returns V, K, counts by rung."""
    V, K = values_array(), kinds_array()
    scratch = np.empty((nb.get_num_threads(), 3, LAGS), dtype=np.float64)
    counts = np.zeros(3, dtype=np.int64)
    t0 = time.perf_counter()
    for P in range(MAX_LAYER, stop_at - 1, -1):
        n, c = solve_layer(P, V, K, table, scratch)
        counts += c
        if n and P % 25 == 0:
            log(f"layer {P:4d}: {n:8d} classes | {counts.sum():10d} solved | {time.perf_counter() - t0:7.1f}s")
    return V, K, counts


def main() -> None:
    table = build_table()
    V, K, counts = build(table)
    finalize(V, K, table)
    for state, expected, got, ok in verify_anchors(V, table):
        print(f"anchor {state}: expected {expected:.16f} got {got:.16f} {'ok' if ok else 'MISMATCH'}")
    print(f"pure={counts[0]} support={counts[1]} lp={counts[2]} total={counts.sum()}")
    ARTIFACTS.mkdir(exist_ok=True)
    np.save(ARTIFACTS / "V.npy", V)
    np.save(ARTIFACTS / "K.npy", K)


if __name__ == "__main__":
    main()
