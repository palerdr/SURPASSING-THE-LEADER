# Drop the Handkerchief: the complete tablebase

A language-neutral recipe for computing the exact solution of pure Drop the
Handkerchief: one certified game value for each of its 289,374,121 state
classes, from a single backward pass. Everything is stated as pseudocode. You
need a language with 64-bit floats, a linear-programming solver for the
residue of classes that the cheap rungs cannot certify, and about 3 GiB of
memory. `main.py` in this directory implements exactly this document (its
docstrings cite the section numbers below), and the paper carries the proofs.

## 1. What you are building

| Output | Contents |
| --- | --- |
| `V` | 289,374,121 float64 values (2.16 GiB), one per state class: the exact game value from the current Dropper's perspective, in [−1, +1], certified to a saddle gap of at most 1e-6 |
| `K` | one byte per class: which solver rung produced the value (0 pure, 1 equalizer, 2 LP) |

Optimal strategies are not stored and do not need to be: any class's
equilibrium pair is recomputable in microseconds from the stored child values
(Section 4.10).

## 2. The game

A live state is `(s_c, t_c, s_d, t_d)`: squandered time (ST, 0..299) and
accrued toxin time (TTD, 0..300) for the current Checker and the current
Dropper. Each turn, both players simultaneously pick a literal second in
1..60.

```text
function SURVIVES(s, t):                 # can this profile survive a failed check?
    dose ← s + 60
    return dose < 300 and dose + t ≤ 300

function REVIVAL(s, t):                  # the frozen revival surface
    if not SURVIVES(s, t): return 0
    return 0.95 · (1 − s/240) · 0.75^(t/60)
```

One simultaneous turn (Dropper picks `d`, Checker picks `c`, both in 1..60):

```text
if d ≤ c:                                # successful check
    s'_c ← s_c + (c − d + 1)             # inclusive elapsed time
    if s'_c ≥ 300: DROPPER WINS
    else: next state ← (s_d, t_d, s'_c, t_c)          # roles swap
else:                                    # failed check: the Checker takes dose s_c + 60
    with probability REVIVAL(s_c, t_c):
        next state ← (s_d, t_d, 0, t_c + s_c + 60)     # revived at ST 0, roles swap
    otherwise: DROPPER WINS
```

Payoffs are zero-sum: +1 to the winner, −1 to the loser. `V(state)` is the
current Dropper's expected payoff. Because every transition swaps the roles, a
child's value is negated wherever a parent reads it.

## 3. Why this is buildable: four ideas

1. **A per-player quotient.** TTD is read in exactly one place, the revival
   probability, and once a profile fails `SURVIVES` that is permanent, so all
   dead TTDs collapse to one sentinel per ST. TTD is also transition-closed
   over `{0} ∪ [60, 300]`. Result: 16,711 alive profiles + 300 dead sentinels
   = 17,011 profiles, and 17,011² = 289,374,121 classes, 18.2× fewer than the
   5.27 billion raw states.
2. **A potential every move strictly increases.** With `rho = TTD` while
   alive and 301 once dead, define `phi = s_c + s_d + rho_c + rho_d`. No
   transition stays inside a `phi` layer, so solving layers in descending
   order `phi = 1200..0` always reads finished children. The graph is never
   materialized.
3. **61 numbers per class, not 3,600.** The 60×60 payoff matrix has only 61
   distinct entries: `M[d][c] = success[c − d + 1]` when `d ≤ c`, else
   `failed`. It is Toeplitz above the diagonal, constant below it, and
   persymmetric: `M[d][c] = M[61 − c][61 − d]`.
4. **A certified solver ladder.** An O(60) pure test, then a 59-step
   recurrence that yields both equilibrium strategies at once, then linear
   programming. Every answer must pass the same test, a saddle gap of at most
   1e-6 against the full matrix, and the stored value is always the
   certificate midpoint. Failing every rung aborts the build.

## 4. The components, in pseudocode

### 4.1 The quotient and its rule tables

Profile ids 0..16,710 are the alive profiles in a fixed, normative order; ids
16,711..17,010 are the dead sentinels by ST. All rules are evaluated once,
here; the sweep only gathers.

```text
procedure BUILD_TABLES():
    # alive profiles, in the normative order:
    # TTD ascending over {0} then 60..300, ST ascending inside each TTD
    next_id ← 0
    for t in (0, 60, 61, ..., 300):
        for s in 0..299:
            if SURVIVES(s, t):
                alive_id[s][t] ← next_id
                ST[next_id] ← s;  TTD[next_id] ← t
                next_id ← next_id + 1
    assert next_id = 16711
    for s in 0..299:                     # dead sentinels, by ST
        ST[16711 + s] ← s;  TTD[16711 + s] ← DEAD

    for each profile p in 0..17010:
        phi[p] ← ST[p] + TTD[p] if p is alive, else ST[p] + 301
        for lag in 1..60:                # successful-check children
            grown ← ST[p] + lag
            if grown ≥ 300:  succ[p][lag] ← WIN        # capacity reached: the mover wins
            else if p is dead or not SURVIVES(grown, TTD[p]):
                succ[p][lag] ← dead sentinel at ST grown
            else:  succ[p][lag] ← alive_id[grown][TTD[p]]
        if p is alive:                   # failed-check child
            rev[p] ← REVIVAL(ST[p], TTD[p])
            t' ← TTD[p] + ST[p] + 60
            fail[p] ← alive_id[0][t'] if SURVIVES(0, t'), else dead sentinel at ST 0
        else:
            rev[p] ← 0
            fail[p] ← WIN                # a dead Checker loses every failed check

    bucket[v] ← all profiles p with phi[p] = v,  for v in 0..600
    assert phi strictly increases along every succ and fail edge
```

### 4.2 Class indexing

```text
function PROFILE(s, t):
    if not SURVIVES(s, t):  return 16711 + s        # dead: TTD is discarded exactly
    if alive_id[s][t] is undefined:  error          # alive TTD in 1..59 is off-domain
    return alive_id[s][t]

function ENCODE(s_c, t_c, s_d, t_d):
    return PROFILE(s_c, t_c) · 17011 + PROFILE(s_d, t_d)
```

### 4.3 Gathering one class's 61 continuation values

```text
function CLASS_VALUES(pc, pd, V):        # pc: Checker profile, pd: Dropper profile
    for lag in 1..60:
        if succ[pc][lag] = WIN:  success[lag] ← +1
        else:  success[lag] ← −V[pd · 17011 + succ[pc][lag]]      # roles swap: negate
    if fail[pc] = WIN:  failed ← +1
    else:  failed ← rev[pc] · (−V[pd · 17011 + fail[pc]]) + (1 − rev[pc])
    abort the build if any value read above is UNSOLVED           # schedule bug
    return (success, failed)
```

The full matrix, only when a rung needs it:
`M[d][c] ← success[c − d + 1] if d ≤ c, else failed` for `d, c` in 1..60.

### 4.4 Rung 1: pure saddle point, O(60)

Row `d` of `M` is `(d − 1)` copies of `failed` followed by
`success[1..61−d]`; column `c` is `success[1..c]` plus `failed` when
`c < 60`. Every row minimum and column maximum is therefore one of four
numbers, and the best pure actions are known in advance.

```text
function PURE_TEST(success, failed):
    lo ← min(success);  hi ← max(success)
    maximin ← max(lo, min(failed, success[1]))          # row 1, or row 60
    minimax ← min(hi, max(failed, success[1]))          # column 60, or column 1
    dropper ← row 1 if lo ≥ min(failed, success[1]) else row 60         # a pure action
    checker ← column 60 if hi ≤ max(failed, success[1]) else column 1   # a pure action
    return (maximin, minimax, dropper, checker)
```

The test is exact: min and max select entries and add no rounding.

### 4.5 Rung 2: the equalizer recurrence

Consecutive rows of `M` differ by `d0 = success[1] − failed` on the diagonal
and by `Δ_m = success[m + 1] − success[m]` above it. A Checker mix that makes
the Dropper indifferent between consecutive seconds therefore satisfies a
59-step recurrence. Because `M` is persymmetric, the reversed vector is the
Dropper's mix, and one matrix-vector product certifies both sides.

```text
function EQUALIZER(success, failed):
    d0 ← success[1] − failed
    if |d0| < 1e-12:  return FAIL
    r[1] ← 1
    for k in 2..60:  r[k] ← −( Σ_{m=1}^{k−1} Δ_m · r[k−m] ) / d0
    if any r[k] is not finite:  return FAIL
    q[c] ← max(r[61 − c], 0)  for c in 1..60           # Checker mix: r reversed, negatives clipped
    if Σ q = 0:  return FAIL
    q ← q / Σ q
    p[d] ← q[61 − d]  for d in 1..60                   # Dropper mix: the mirror image
    y ← M q            # y[d] = failed · Σ_{c<d} q[c] + Σ_{c≥d} success[c − d + 1] · q[c]
    return ( lower ← min(y),  upper ← max(y),  p, q )   # (p M)[c] = y[61 − c] by persymmetry
```

The clip cannot invalidate the certificate: `lower ≤ value ≤ upper` holds for
every mixed pair, and the mirror identity holds for every reversed pair. Cost:
1,770 multiply-adds for `r` and 1,770 for `y`.

### 4.6 Rung 3: the LP residue

```text
function LP_SOLVE(M):
    for method in (dual simplex, interior point):      # retries change the solver, never the gate
        p ← argmax v  subject to  Mᵀ p ≥ v · 1,  Σ p = 1,  p ≥ 0
        q ← the dual multipliers of the 60 row constraints, clipped to ≥ 0, renormalized
        lower ← min(p M);  upper ← max(M q)
        if upper − lower ≤ 1e-6:  return (lower, upper, p, q)
    abort the build                                    # nothing is ever stored uncertified
```

### 4.7 The ladder for one class

```text
function SOLVE_CLASS(success, failed):
    (lower, upper, p, q) ← PURE_TEST(success, failed)
    if upper − lower ≤ 1e-6:  return ( (lower + upper)/2, PURE, p, q )
    result ← EQUALIZER(success, failed)
    if result ≠ FAIL and result.upper − result.lower ≤ 1e-6:
        return ( (result.lower + result.upper)/2, EQUALIZER, result.p, result.q )
    (lower, upper, p, q) ← LP_SOLVE(full matrix from success, failed)
    return ( (lower + upper)/2, LP, p, q )
```

### 4.8 Layers and the sweep

The classes of potential `P` are exactly the rectangles
`bucket[a] × bucket[P − a]`. `V` lives in memory (2.2 GiB), every entry
initialised to UNSOLVED; `K` likewise. Classes inside one layer are
independent: every read is a finished layer and every write is the class's
own cell, so a layer can be solved in parallel with no synchronisation.

```text
procedure SOLVE_LAYER(P):
    classes ← all (pc, pd) with pc in bucket[a], pd in bucket[P − a], for every valid a,
              ordered so that neighbours share pd (they read the same row of V)
    for each (pc, pd) in classes, in parallel:
        (success, failed) ← CLASS_VALUES(pc, pd, V)
        run PURE_TEST, then EQUALIZER, in a per-thread scratch of 3 × 60 floats
        if either certifies:  V[pc · 17011 + pd] ← midpoint;  K[...] ← rung
        else:  add (pc, pd) to the LP worklist
    for each (pc, pd) in the LP worklist:                   # serial; empty on the real table
        V[...] ← LP_SOLVE(full matrix of CLASS_VALUES(pc, pd, V));  K[...] ← LP

procedure SWEEP():
    for P in 1200 down to 0:  SOLVE_LAYER(P)
    FINALIZE()
    write V and K to disk
```

The whole sweep takes under a minute (Section 5), so there is no checkpoint:
an interrupted build is simply rerun.

### 4.9 Finalize

```text
procedure FINALIZE():
    for every class c:
        assert V[c] is solved,  |V[c]| ≤ 1 + 1e-9,  K[c] in {PURE, EQUALIZER, LP}
    for 1,200 evenly strided classes c:
        assert | RECHECK(c).value − V[c] | ≤ 1e-6         # independent re-derivation
```

### 4.10 Recheck: the audit primitive and the policy oracle

```text
function RECHECK(c):
    (pc, pd) ← (c div 17011, c mod 17011)
    (success, failed) ← CLASS_VALUES(pc, pd, V)      # rebuilt from the stored children
    return SOLVE_CLASS(success, failed)              # solved again, independently: value, rung, p, q
```

`RECHECK` is deliberately a second, scalar implementation of the ladder, not a
call into the sweep's kernel; the finalize audit compares the two. Its `p`
and `q` are the optimal mixed strategies, so the table never needs to store
them.

## 5. Cost, measured

| Setting | Rate | Whole table |
| --- | --- | --- |
| fifteen threads (Apple M5 Pro laptop: five performance and ten efficiency cores) | 5.9 M classes/s | 48.9 s |
| one thread | 0.49 M classes/s | about ten minutes |

Memory: about 2.5 GiB (`V` 2.16 GiB, `K` 0.27 GiB, tables and scratch). Disk:
the same, written once at the end.

Routing on the complete table: 334,177 classes pure, 289,039,944 equalizer,
0 LP. Clipping the equalizer's negative entries, instead of rejecting the
candidate, is what empties the LP residue; a few hundred borderline classes
may route differently between builds, and their values are certified either
way.

## 6. Verification anchors

After `FINALIZE`, the following certified reference values must match within
1e-6 (independent builds agree to 1e-9 or better):

| State `(s_c, t_c, s_d, t_d)` | Expected value | Note |
| --- | --- | --- |
| (0, 0, 0, 0) | 0.08985007280951046 | the root of the whole game |
| (240, 0, 240, 0) | 0.3372132166291093 | independently derivable dead-band reference |
| (10, 60, 200, 0) | −0.7944428916469297 | |
| (150, 90, 30, 120) | 0.7244093036356785 | |
| (250, 300, 40, 0) | 0.9981152817381969 | |
| (100, 140, 100, 140) | 0.1877386378276193 | |

```text
procedure VERIFY():
    for (state, expected) in the anchor table:
        assert | V[ENCODE(state)] − expected | ≤ 1e-6
```

## 7. Provenance

The rules restated here are frozen by the repository (`docs/REVIVAL_MODEL.md`
owns the revival surface; `src/dth/` is the behavioral authority). The six
anchors are reference values of an earlier, independently implemented
artifact built by a different algorithm (two bordered 61×61 linear solves
per class and an LP residue of 190,995 classes). This recipe reproduces five
of them to within 2e-11 and the sixth, (250, 300, 40, 0), to 1.2e-9; the
finalize recheck's worst disagreement on the complete table is 8.9e-16.
