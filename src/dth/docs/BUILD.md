# Building the complete DTH tablebase from scratch in Python

This is a self-contained recipe for computing the exact solution of pure Drop
the Handkerchief: one certified game value for every one of its 289,374,121
state classes. It assumes nothing but Python and
[`uv`](https://docs.astral.sh/uv/): no part of this repository — code,
configs, or artifacts — is needed. The rules are restated here in full; their
canonical source is this repository (`docs/REVIVAL_MODEL.md` freezes the
revival surface, and `src/dth/` remains the behavioral authority), but you do
not have to read it. If you can code and follow instructions, you can build
this.

The complete program is about 300 lines and appears below in five blocks.
Concatenate them, in order, into one file `build_tablebase.py`. Every block
is the exact code that was validated end to end against the repository's
independently implemented certified artifact.

## What you will build

A directory `tablebase/` containing:

- `value.npy` — 289,374,121 float64 values (2.16 GiB), one per state class,
  each the exact game value from the current Dropper's perspective, in
  `[-1, +1]`, certified to a saddle gap of at most `1e-6`.
- `kind.npy` — one uint8 per class (276 MiB): which solver produced the
  value (0 pure, 1 support-certified, 2 LP).
- `progress.json` — the resume checkpoint.

Optimal strategies are not stored, and do not need to be: any class's
equilibrium pair is recomputable in about a millisecond from the stored child
values (the `recheck` function below does exactly that).

## Setup

```bash
uv init dth-tablebase
cd dth-tablebase
uv add numpy scipy
```

That is the entire environment (validated with numpy 2.5.2 and scipy 1.18.0;
any recent versions work — the LP solver, HiGHS, ships inside SciPy). Create
`build_tablebase.py` from the blocks in the next sections, then skip to
**Run it**.

## The game

A live state is `x = (s_c, t_c, s_d, t_d)`: squandered time (ST, `0..299`)
and accrued toxin time (TTD, `0..300`) for the current Checker and current
Dropper. Each turn both players simultaneously pick a literal second in
`1..60` — Dropper `d`, Checker `c`.

- **Successful check** (`d <= c`): the Checker's ST grows by the inclusive
  elapsed time `c - d + 1`. If it reaches 300, the current Dropper wins.
  Otherwise the roles swap and play continues.
- **Failed check** (`d > c`): the Checker takes the dose `q = s_c + 60`.
  Survival is possible exactly when `q < 300` and `q + t_c <= 300`
  (equivalently `s_c <= 239` and `s_c + t_c <= 240`); when it is, the
  revival probability is the frozen surface
  `0.95 * (1 - s_c/240) * 0.75^(t_c/60)`. Death is a win for the Dropper. A
  revived Checker returns at ST 0 with TTD grown to `t_c + s_c + 60`, and
  the roles swap.

Payoffs are zero-sum, `+1` for the winner and `-1` for the loser, so a value
is an expected payoff in `[-1, +1]` for the current Dropper. Because every
transition swaps the roles, a child's value is negated wherever a parent
reads it.

## Why this is buildable: four ideas

1. **A per-player quotient.** TTD is read in exactly one place — the revival
   probability — and once a player's `(ST, TTD)` profile cannot survive a
   failed check, that is permanent, so all dead TTDs collapse to one
   sentinel per ST. TTD is also transition-closed over `{0} | [60, 300]`
   (it starts at 0 and revival always adds at least 60). That leaves 16,711
   alive profiles plus 300 dead sentinels: 17,011 per-player profiles and
   `17,011^2 = 289,374,121` two-player classes, 18.2x fewer than the ~5.27
   billion raw reachable states. A class's index,
   `checker_profile * 17_011 + dropper_profile`, is its address in
   `value.npy`.

2. **A potential that every move strictly increases.** Let `rho = TTD` while
   alive and `301` once dead, and `phi(x) = s_c + s_d + rho_c + rho_d`.
   Success raises the Checker's ST (or kills a profile, jumping `rho` past
   any live TTD); failure adds exactly 60 to a surviving Checker's
   `s + rho`. No transition stays inside a `phi` layer, so solving layers in
   descending order `phi = 1200 .. 0` is backward induction that always
   reads finished children — and the graph never has to be materialized: a
   layer is a union of bucket rectangles.

3. **61 numbers per class, not 3,600.** The full 60x60 payoff matrix of a
   class has only 61 distinct entries: `M[d, c] = success[c - d]` when
   `c >= d`, else `failed`. So the sweep gathers 61 continuation values per
   class and reconstructs `M` only for the classes that need it.

4. **A certified solver ladder.** Each class first gets an O(60) pure-saddle
   test; the rest get a full-support equalizer solve; a small residue goes
   to linear programming. Whatever rung answers, the answer is accepted only
   if its saddle gap against the full matrix,
   `max_d (Mq)_d - min_c (pM)_c`, is at most `1e-6`, and the stored value is
   the midpoint of that certificate. Failing every rung aborts the build —
   there is no weaker acceptance path.

## The code

### Block 1 — rules

```python
"""Build the complete pure-DTH tablebase: 289,374,121 exact class values."""

import json
import os
import sys
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from scipy.optimize import linprog

GAP = 1e-6                      # frozen certification gate on the saddle gap
OUT = Path("tablebase")         # artifact directory
N_ALIVE = 16_711                # alive (ST, TTD) profiles
N = 17_011                      # profiles: alive + 300 dead sentinels
CLASSES = N * N                 # 289,374,121 two-player classes
ALIVE_TTDS = (0, *range(60, 301))


# ---------------------------------------------------------------- rules

def survives(s, t):
    """Can a player at ST s, TTD t survive a failed check (dose s + 60)?"""
    return s + 60 < 300 and s + 60 + t <= 300


def revival(s, t):
    """Frozen revival surface; zero exactly when survival is impossible."""
    return 0.95 * (1.0 - s / 240.0) * 0.75 ** (t / 60.0) if survives(s, t) else 0.0
```

### Block 2 — the quotient and its rule tables

Profiles are enumerated in a fixed order (alive profiles by TTD ascending,
ST ascending inside each TTD, then the 300 dead sentinels by ST); that order
defines every class index, so treat it as part of the format. For each
profile the table precomputes its 60 success children (`-1` when the grown ST
reaches 300: the mover wins), its failure child (`-1` for a dead Checker: the
Dropper wins), and its revival probability — after this, the sweep never
evaluates a rule again, it only gathers.

```python
# ------------------------------------------------------- profile tables

def build_tables():
    alive_id = np.full((300, 301), -1, dtype=np.int64)
    st, ttd = [], []
    for t in ALIVE_TTDS:                      # enumeration order is normative
        for s in range(300):
            if survives(s, t):
                alive_id[s, t] = len(st)
                st.append(s)
                ttd.append(t)
    assert len(st) == N_ALIVE
    for s in range(300):                      # dead sentinels, by ST
        st.append(s)
        ttd.append(-1)
    st = np.array(st, dtype=np.int64)
    ttd = np.array(ttd, dtype=np.int64)
    phi = np.where(ttd >= 0, st + ttd, st + 301)

    succ = np.full((N, 60), -1, dtype=np.int64)   # -1: overflow, mover wins
    fail = np.full(N, -1, dtype=np.int64)         # -1: dead checker, Dropper wins
    rev = np.zeros(N, dtype=np.float64)
    for p in range(N):
        s, t = int(st[p]), int(ttd[p])
        for lag in range(1, 61):
            grown = s + lag
            if grown >= 300:
                continue
            child = alive_id[grown, t] if t >= 0 else -1
            succ[p, lag - 1] = child if child >= 0 else N_ALIVE + grown
        if t >= 0:
            rev[p] = revival(s, t)
            child = alive_id[0, t + s + 60]
            fail[p] = child if child >= 0 else N_ALIVE
    buckets = [np.flatnonzero(phi == v).astype(np.int64) for v in range(601)]
    return SimpleNamespace(alive_id=alive_id, st=st, ttd=ttd, phi=phi,
                           succ=succ, fail=fail, rev=rev, buckets=buckets)


def profile(T, s, t):
    if not survives(s, t):
        return N_ALIVE + s                    # dead: TTD is discarded exactly
    pid = int(T.alive_id[s, t])
    if pid < 0:
        raise ValueError(f"alive TTD {t} in 1..59 is outside the domain")
    return pid


def encode(T, state):
    sc, tc, sd, td = state
    return profile(T, sc, tc) * N + profile(T, sd, td)
```

### Block 3 — solving one class

`pure_scan` exploits the matrix structure: row `d` is `d` copies of `failed`
followed by `success[0 .. 59-d]`, so all 60 row minima come from one reversed
prefix-minimum (and column maxima from one prefix-maximum). When maximin and
minimax agree within the gate, the class is solved for the cost of two scans.
Otherwise `support_solve` solves the two full-support equalizer systems —
`M q = v` with `q` on the simplex makes the Dropper indifferent, and
`M^T p = v` the Checker — and certifies the pair against `M`. The LP residue
formulates the standard minimax programs for both players and is certified
the same way; an interior-point retry covers the handful of matrices HiGHS's
dual simplex mishandles, and the gate is never loosened.

```python
# ------------------------------------------------------- class solvers

_D = np.arange(60)[:, None]                   # Dropper second minus one
_C = np.arange(60)[None, :]                   # Checker second minus one
_ON = _C >= _D                                # successful-check cells
_LAG = np.maximum(_C - _D, 0)                 # index into the success values


def full_matrix(success, failed):
    """The literal 60x60 Dropper-payoff matrix from 61 continuation values."""
    return np.where(_ON, success[_LAG], failed)


def pure_scan(success, failed):
    """Vectorized maximin/minimax over a block, O(60) per class.

    Row d of the implied matrix is d copies of `failed` then
    success[0 .. 59-d], so row minima are a reversed prefix-min of the
    success values (with `failed` folded in for d > 0) and column maxima
    are a prefix-max (with `failed` folded in for c < 59).
    """
    pmin = np.minimum.accumulate(success, axis=1)
    row_min = pmin[:, ::-1].copy()
    row_min[:, 1:] = np.minimum(row_min[:, 1:], failed[:, None])
    maximin = row_min.max(axis=1)
    pmax = np.maximum.accumulate(success, axis=1)
    col_max = pmax.copy()
    col_max[:, :-1] = np.maximum(col_max[:, :-1], failed[:, None])
    minimax = col_max.min(axis=1)
    return maximin, minimax


def support_solve(M):
    """Full-support equalizer solve; certified or None."""
    A = np.zeros((61, 61))
    A[:60, 60] = -1.0
    A[60, :60] = 1.0
    b = np.zeros(61)
    b[60] = 1.0
    try:
        A[:60, :60] = M
        check = np.linalg.solve(A, b)[:60]    # Checker mix equalizing rows
        A[:60, :60] = M.T
        drop = np.linalg.solve(A, b)[:60]     # Dropper mix equalizing columns
    except np.linalg.LinAlgError:
        return None
    if min(check.min(), drop.min()) < -1e-12:
        return None
    check = np.clip(check, 0.0, None)
    drop = np.clip(drop, 0.0, None)
    check /= check.sum()
    drop /= drop.sum()
    upper = float((M @ check).max())
    lower = float((drop @ M).min())
    if upper - lower > GAP:
        return None
    return (lower + upper) / 2.0


def _two_lps(M, method):
    ones = np.ones((60, 1))
    simplex = np.r_[np.ones(60), 0.0][None, :]
    bounds = [(0.0, None)] * 60 + [(None, None)]
    r_drop = linprog(np.r_[np.zeros(60), -1.0], A_ub=np.hstack([-M.T, ones]),
                     b_ub=np.zeros(60), A_eq=simplex, b_eq=[1.0],
                     bounds=bounds, method=method)
    r_check = linprog(np.r_[np.zeros(60), 1.0], A_ub=np.hstack([M, -ones]),
                      b_ub=np.zeros(60), A_eq=simplex, b_eq=[1.0],
                      bounds=bounds, method=method)
    if not (r_drop.success and r_check.success):
        raise RuntimeError(f"HiGHS ({method}) failed on a class matrix")
    drop = np.clip(r_drop.x[:60], 0.0, None)
    check = np.clip(r_check.x[:60], 0.0, None)
    drop /= drop.sum()
    check /= check.sum()
    upper = float((M @ check).max())
    lower = float((drop @ M).min())
    if upper - lower > GAP:
        raise RuntimeError(f"LP pair ({method}) missed the certificate gate")
    return (lower + upper) / 2.0


def lp_solve(M):
    """LP residue: dual simplex, then an interior-point retry. Never a
    weaker gate — a matrix failing both aborts the build."""
    try:
        return _two_lps(M, "highs")
    except RuntimeError:
        return _two_lps(M, "highs-ipm")
```

### Block 4 — the sweep

The classes of potential `P` are exactly the rectangles
`bucket(a) x bucket(P - a)` over per-profile potentials. Per rectangle, the
child values of a whole block of classes are gathered with three NumPy
indexing expressions: a success child's value is `1.0` on overflow, else the
negated stored value of class `dropper * N + succ[checker][lag]`; the failure
value is `1.0` for a dead Checker, else
`revival * (-V(child)) + (1 - revival)`. Reading NaN — an unsolved child —
means the schedule is broken, and the build aborts.

Each layer is committed before the next begins: values and kinds are written
and flushed, then `progress.json` is atomically replaced. Killing the process
at any moment therefore loses at most the layer in flight, which re-runs
idempotently on the next start.

```python
# ------------------------------------------------------------ the sweep

def solve_layer(P, T, value):
    """Solve every class of potential P, reading only deeper (solved) values."""
    ids, vals, kinds = [], [], []
    mixed_ids, mixed_vals, mixed_kinds = [], [], []
    for a in range(max(0, P - 600), min(600, P) + 1):
        checkers = T.buckets[a]
        droppers = T.buckets[P - a]
        if not len(checkers) or not len(droppers):
            continue
        succ = T.succ[checkers]               # (W, 60)
        fail = T.fail[checkers]               # (W,)
        rev = T.rev[checkers]                 # (W,)
        width = len(checkers)
        block = max(1, 32_768 // width)
        for start in range(0, len(droppers), block):
            dr = droppers[start:start + block]
            child = dr[:, None, None] * N + succ[None, :, :]
            S = np.where(succ[None, :, :] < 0, 1.0, -value[np.maximum(child, 0)])
            fchild = dr[:, None] * N + fail[None, :]
            F = np.where(fail[None, :] < 0, 1.0,
                         rev[None, :] * (-value[np.maximum(fchild, 0)])
                         + (1.0 - rev[None, :]))
            S = S.reshape(-1, 60)
            F = F.reshape(-1)
            if not (np.isfinite(S).all() and np.isfinite(F).all()):
                raise RuntimeError("read an unsolved child value: schedule bug")
            cls = (checkers[None, :] * N + dr[:, None]).reshape(-1)
            maximin, minimax = pure_scan(S, F)
            pure = minimax - maximin <= GAP
            ids.append(cls[pure])
            vals.append((maximin[pure] + minimax[pure]) / 2.0)
            kinds.append(np.zeros(int(pure.sum()), dtype=np.uint8))
            for j in np.flatnonzero(~pure):
                M = full_matrix(S[j], F[j])
                v = support_solve(M)
                kind = 1
                if v is None:
                    v = lp_solve(M)
                    kind = 2
                mixed_ids.append(int(cls[j]))
                mixed_vals.append(v)
                mixed_kinds.append(kind)
    ids.append(np.array(mixed_ids, dtype=np.int64))
    vals.append(np.array(mixed_vals, dtype=np.float64))
    kinds.append(np.array(mixed_kinds, dtype=np.uint8))
    ids = np.concatenate(ids)
    vals = np.concatenate(vals)
    kinds = np.concatenate(kinds)
    counters = {"pure": int((kinds == 0).sum()), "support": int((kinds == 1).sum()),
                "lp": int((kinds == 2).sum())}
    return ids, vals, kinds, counters


def save_progress(progress):
    with tempfile.NamedTemporaryFile("w", dir=OUT, suffix=".json",
                                     delete=False) as handle:
        handle.write(json.dumps(progress))
    os.replace(handle.name, OUT / "progress.json")


def sweep(max_layers=None):
    T = build_tables()
    OUT.mkdir(exist_ok=True)
    if (OUT / "progress.json").exists():
        progress = json.loads((OUT / "progress.json").read_text())
        value = np.load(OUT / "value.npy", mmap_mode="r+")
        kind = np.load(OUT / "kind.npy", mmap_mode="r+")
    else:
        value = np.lib.format.open_memmap(OUT / "value.npy", mode="w+",
                                          dtype=np.float64, shape=(CLASSES,))
        for i in range(0, CLASSES, 16_000_000):
            value[i:i + 16_000_000] = np.nan
        kind = np.lib.format.open_memmap(OUT / "kind.npy", mode="w+",
                                         dtype=np.uint8, shape=(CLASSES,))
        progress = {"next": 1200, "pure": 0, "support": 0, "lp": 0}
        save_progress(progress)
    done = 0
    while progress["next"] >= 0 and (max_layers is None or done < max_layers):
        P = progress["next"]
        started = time.perf_counter()
        ids, vals, kinds, counters = solve_layer(P, T, value)
        value[ids] = vals                     # commit, then checkpoint
        kind[ids] = kinds
        value.flush()
        kind.flush()
        for key in counters:
            progress[key] += counters[key]
        progress["next"] = P - 1
        save_progress(progress)
        done += 1
        solved = progress["pure"] + progress["support"] + progress["lp"]
        print(f"phi={P:4d} {len(ids):9,} classes in "
              f"{time.perf_counter() - started:8.2f}s | "
              f"{100.0 * solved / CLASSES:6.2f}% done", flush=True)
    if progress["next"] < 0:
        finalize(T, value, kind, progress)


def finalize(T, value, kind, progress):
    for i in range(0, CLASSES, 16_000_000):
        chunk = np.asarray(value[i:i + 16_000_000])
        if not np.isfinite(chunk).all() or np.abs(chunk).max() > 1.0 + 1e-9:
            raise RuntimeError("finalize found an unsolved or out-of-range value")
        if np.asarray(kind[i:i + 16_000_000]).max() > 2:
            raise RuntimeError("finalize found an unknown solver kind")
    worst = 0.0
    for cid in range(0, CLASSES, CLASSES // 1_200):   # deterministic re-check
        worst = max(worst, abs(recheck(T, cid, value) - float(value[cid])))
    if worst > GAP:
        raise RuntimeError(f"recheck disagreed with a stored value by {worst}")
    progress["complete"] = True
    save_progress(progress)
    print(f"complete: {progress['pure']:,} pure / {progress['support']:,} "
          f"support / {progress['lp']:,} lp; recheck worst gap {worst:.3e}")
```

### Block 5 — verification

`recheck` is the audit primitive: rebuild one class's 61 continuation values
from the stored children and solve it again, independently of whatever the
sweep stored. The anchors are certified reference values; the second one is
special because it is independently derivable — in the dead-vs-dead band
every failed check is a certain Dropper win, so that whole region can be
re-solved with none of this machinery.

```python
# ----------------------------------------------------------- verification

def recheck(T, cid, value):
    """Re-derive one class from its stored children and re-solve it."""
    checker, dropper = divmod(int(cid), N)
    succ = T.succ[checker]
    S = np.where(succ < 0, 1.0, -value[np.maximum(dropper * N + succ, 0)])
    f = int(T.fail[checker])
    F = 1.0 if f < 0 else float(T.rev[checker] * (-value[dropper * N + f])
                                + (1.0 - T.rev[checker]))
    maximin, minimax = pure_scan(S[None, :], np.array([F]))
    if minimax[0] - maximin[0] <= GAP:
        return float(maximin[0] + minimax[0]) / 2.0
    M = full_matrix(S, F)
    v = support_solve(M)
    return lp_solve(M) if v is None else v


ANCHORS = {
    (0, 0, 0, 0): 0.08985007280951046,        # the root of the whole game
    (240, 0, 240, 0): 0.3372132166291093,     # independent dead-band reference
    (10, 60, 200, 0): -0.7944428916469297,
    (150, 90, 30, 120): 0.7244093036356785,
    (250, 300, 40, 0): 0.9981152817381969,
    (100, 140, 100, 140): 0.1877386378276193,
}


def verify():
    T = build_tables()
    progress = json.loads((OUT / "progress.json").read_text())
    if not progress.get("complete"):
        raise SystemExit("the sweep has not finished")
    value = np.load(OUT / "value.npy", mmap_mode="r")
    for state, expected in ANCHORS.items():
        stored = float(value[encode(T, state)])
        print(f"V{state} = {stored!r} (expected {expected!r}, "
              f"difference {abs(stored - expected):.3e})")
        if abs(stored - expected) > GAP:
            raise SystemExit("anchor mismatch: the build is wrong")
    print("all anchors match")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "verify":
        verify()
    else:
        sweep(int(sys.argv[1]) if len(sys.argv) > 1 else None)
```

## Run it

```bash
uv run python build_tablebase.py
```

That builds to completion, or resumes if a checkpoint exists. To bound one
session, pass a layer count; to check a finished artifact, pass `verify`:

```bash
uv run python build_tablebase.py 200
uv run python build_tablebase.py verify
```

The first start spends about a minute filling the 2.16 GiB value array with
NaN, then prints one line per committed layer:

```text
phi=1101       100 classes in     0.04s |   0.00% done
```

What to expect, all measured on one ordinary desktop core with this exact
code:

- The bulk of the game (`phi <= 600`, 283.3M classes, largest layer
  1,678,715 classes at `phi = 374`) runs at roughly 9,500 classes/s.
- The mixed band (`phi 601..840`, 6.0M classes) is LP-heavier and runs at
  1,200–3,500 classes/s; the dead band above it is negligible.
- Total: **about nine hours single-core**, roughly 2.5 GiB of disk, and a
  few GiB of RAM.
- Rough routing mix at the end: ~334.6k pure, ~288.85M support-certified,
  ~191k LP. A few hundred borderline classes may route differently from
  build to build; their values are certified either way.

Interrupt freely; re-running the same command continues from the last
committed layer.

## How you know it is right

- **Nothing is accepted uncertified.** Every stored value passed the `1e-6`
  full-matrix saddle-gap gate when it was produced, whichever rung produced
  it, and an unsolvable matrix aborts the build instead of storing anything.
- **Finalize re-audits.** It scans all 289,374,121 values for NaN and range
  violations, then independently re-derives 1,200 evenly strided classes
  from their stored children and requires agreement within the gate.
- **The anchors pin the result.** `verify` checks six certified reference
  values, including the root `V(0,0,0,0) = 0.08985007280951046` and the
  independently derivable dead-band value
  `V(240,0,240,0) = 0.3372132166291093`.
- **This exact code was cross-validated.** Layer-by-layer comparison against
  this repository's independently implemented, Rust-parity-pinned certified
  artifact — sampled layers spanning every region of the game, 5.9M classes
  in the largest four alone — agreed to at most `5.3e-7` (only on
  LP-routed classes; elsewhere agreement is at machine precision), the
  finalize recheck's worst gap on the complete value set was `2.0e-10`, and
  all six anchors matched.

## Relation to this repository

`src/dth/` builds the same values via `python -m dth complete` (see
[`WORKFLOWS.md`](WORKFLOWS.md)) with two extras this recipe deliberately
omits: a warm-started small-support rung, and arithmetic pinned
operation-by-operation so a Rust kernel can reproduce the artifact byte for
byte (`DTH_COMPLETE_PARITY.md`). Those constraints are what make the
repository's pure-Python backend roughly 190x slower than this recipe; free
of them, plain LAPACK and vectorized NumPy do the same certified work in
hours. The repository remains the authority on the rules; this document is a
faithful, independent restatement of them.
