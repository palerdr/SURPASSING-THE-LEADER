DTH
This document is a self-contained, language-neutral recipe for computing the exact solution of pure Drop the Handkerchief: one certified game value for each of its 289,374,121 state classes. Every component is laid out as pseudocode; a validated ~300-line Python reference implementation of exactly this recipe lives in the repository at src/dth/docs/BUILD.md. You need nothing beyond a language with 64-bit floats, a linear system solver, and a linear programming solver (the reference uses NumPy and SciPy's bundled HiGHS).
1. What you are building
value — 289,374,121 float64 entries (2.16 GiB), one per state class: the exact game value from the current Dropper's perspective, in [-1, +1], certified to a saddle gap of at most 1e-6.
kind — one byte per class: which solver rung produced the value (0 pure, 1 support-certified, 2 LP).
checkpoint — a tiny progress record making the multi-hour build safely interruptible.
Optimal strategies are not stored and do not need to be: any class's equilibrium pair is recomputable in about a millisecond from the stored child values (Section 5.10).
2. Environment (for the Python reference)
uv init dth-tablebase
cd dth-tablebase
uv add numpy scipy
Then uv run python build_tablebase.py builds or resumes; a trailing integer bounds one session to that many layers; verify checks the anchors of Section 7.
3. The game
A live state is (s_c, t_c, s_d, t_d): squandered time (ST, 0..299) and accrued toxin time (TTD, 0..300) for the current Checker and current Dropper. Each turn, both players simultaneously pick a literal second in 1..60.
function SURVIVES(s, t):  # can this profile survive a failed check?
    dose ← s + 60
    return dose < 300 and dose + t ≤ 300
 
function REVIVAL(s, t):  # frozen revival surface
    if not SURVIVES(s, t): return 0
    return 0.95 · (1 - s/240) · 0.75^(t/60)
 
One simultaneous turn (Dropper picks d, Checker picks c, both in 1..60):
    if d ≤ c:  # successful check
        s'_c ← s_c + (c - d + 1)  # inclusive elapsed time
        if s'_c ≥ 300: DROPPER WINS
        else: next state ← (s_d, t_d, s'_c, t_c)  # roles swap
    else:  # failed check: the Checker takes dose s_c + 60
        with probability REVIVAL(s_c, t_c):
            next state ← (s_d, t_d, 0, t_c + s_c + 60)  # revived at ST 0, roles swap
        otherwise: DROPPER WINS
Payoffs are zero-sum: +1 to the winner, -1 to the loser. V(state) is the current Dropper's expected payoff. Because every transition swaps the roles, a child's value is negated wherever a parent reads it.
4. Why this is buildable: four ideas
A per-player quotient. TTD is read in exactly one place — the revival probability — and once a profile fails SURVIVES that is permanent, so all dead TTDs collapse to one sentinel per ST. TTD is also transition-closed over {0} ∪ [60, 300]. Result: 16,711 alive profiles + 300 dead sentinels = 17,011 profiles, and 17,011² = 289,374,121 classes — 18.2x fewer than the ~5.27 billion raw reachable states.
A potential every move strictly increases. With rho = TTD while alive and 301 once dead, define phi = s_c + s_d + rho_c + rho_d. No transition stays inside a phi layer, so solving layers in descending order phi = 1200..0 always reads finished children. The graph is never materialized.
61 numbers per class, not 3,600. The 60×60 payoff matrix has only 61 distinct entries: M[d][c] = success[c-d+1] when d ≤ c, else failed.
A certified solver ladder. Cheap rungs first, but every answer must pass the same test — saddle gap against the full matrix at most 1e-6 — and the stored value is always the certificate midpoint. Failing every rung aborts the build.
5. The components, in pseudocode
5.1 The quotient and its rule tables
Profile ids 0..16,710 are the alive profiles in a fixed, normative order; ids 16,711..17,010 are the dead sentinels by ST. All rules are evaluated once, here; the sweep only gathers.
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
    for s in 0..299:  # dead sentinels, by ST
        ST[16711 + s] ← s;  TTD[16711 + s] ← DEAD
 
    for each profile p in 0..17010:
        phi[p] ← ST[p] + TTD[p] if p is alive, else ST[p] + 301
        for lag in 1..60:  # successful-check children
            grown ← ST[p] + lag
            if grown ≥ 300:  succ[p][lag] ← WIN  # capacity reached: mover wins
            else if p is dead or not SURVIVES(grown, TTD[p]):
                succ[p][lag] ← dead sentinel at ST grown
            else:  succ[p][lag] ← alive_id[grown][TTD[p]]
        if p is alive:  # failed-check child
            rev[p] ← REVIVAL(ST[p], TTD[p])
            t' ← TTD[p] + ST[p] + 60
            fail[p] ← alive_id[0][t'] if SURVIVES(0, t'), else dead sentinel at ST 0
        else:
            rev[p] ← 0
            fail[p] ← WIN  # a dead Checker loses every failed check
 
    bucket[v] ← all profiles p with phi[p] = v,  for v in 0..600
5.2 Class indexing
function PROFILE(s, t):
    if not SURVIVES(s, t):  return 16711 + s  # dead: TTD is discarded exactly
    if alive_id[s][t] is undefined:  error  # alive TTD in 1..59 is off-domain
    return alive_id[s][t]
 
function ENCODE(s_c, t_c, s_d, t_d):
    return PROFILE(s_c, t_c) · 17011 + PROFILE(s_d, t_d)
5.3 Gathering one class's 61 continuation values
function CLASS_VALUES(pc, pd, V):  # pc: Checker profile, pd: Dropper profile
    for lag in 1..60:
        if succ[pc][lag] = WIN:  success[lag] ← +1
        else:  success[lag] ← -V[pd · 17011 + succ[pc][lag]]  # roles swap: negate
    if fail[pc] = WIN:  failed ← +1
    else:  failed ← rev[pc] · (-V[pd · 17011 + fail[pc]]) + (1 - rev[pc])
    abort the build if any value read above is UNSOLVED  # schedule bug
    return (success, failed)
 
full matrix (payoff to the Dropper), only when a rung needs it:
    M[d][c] ← success[c - d + 1] if d ≤ c, else failed  # d, c in 1..60
5.4 Rung 1 — pure saddle point, O(60)
Row d of M is (d-1) copies of failed followed by success[1 .. 61-d], so both reductions are prefix scans — no matrix is built.
function PURE_SCAN(success, failed):
    for d in 1..60:  rowmin[d] ← min( success[1 .. 61-d], and failed if d > 1 )
    for c in 1..60:  colmax[c] ← max( success[1 .. c], and failed if c < 60 )
    return ( maximin ← max(rowmin),  minimax ← min(colmax) )
5.5 Rung 2 — full-support equalizer
function SUPPORT_SOLVE(M):  # one 61×61 linear system per side
    solve  M q = v · ones,  sum(q) = 1  for (q, v)  # Checker mix: Dropper indifferent
    solve  Mᵀ p = w · ones,  sum(p) = 1  for (p, w)  # Dropper mix: Checker indifferent
    if either system is singular:  return FAIL
    if any entry of p or q < -1e-12:  return FAIL
    clip p and q to ≥ 0, renormalize each to sum 1
    upper ← max over d of (M q)[d]  # the Dropper's best reply to q
    lower ← min over c of (p M)[c]  # the Checker's best reply to p
    if upper - lower > 1e-6:  return FAIL  # the certificate gate
    return (lower + upper) / 2
5.6 Rung 3 — LP residue
function LP_SOLVE(M):
    for method in (dual simplex, interior point):  # retries change the solver, never the gate
        p ← argmax v  subject to  Mᵀ p ≥ v · ones,  sum(p) = 1,  p ≥ 0
        q ← argmin v  subject to  M q ≤ v · ones,  sum(q) = 1,  q ≥ 0
        clip p and q to ≥ 0, renormalize
        upper ← max(M q);  lower ← min(p M)
        if upper - lower ≤ 1e-6:  return (lower + upper) / 2
    abort the build  # nothing is ever stored uncertified
5.7 The ladder for one class
function SOLVE_CLASS(success, failed):
    (maximin, minimax) ← PURE_SCAN(success, failed)
    if minimax - maximin ≤ 1e-6:  return ( (maximin + minimax)/2, PURE )
    M ← full matrix from (success, failed)
    v ← SUPPORT_SOLVE(M)
    if v ≠ FAIL:  return (v, SUPPORT)
    return ( LP_SOLVE(M), LP )
5.8 One layer, and the whole sweep
The classes of potential P are exactly the bucket rectangles below. In practice, gather and pure-scan whole blocks at once (the reference batches ~32,768 classes per gather); only the mixed remainder touches a full matrix.
procedure SOLVE_LAYER(P, V, K):
    for a in max(0, P - 600) .. min(600, P):
        for pc in bucket[a],  pd in bucket[P - a]:
            (success, failed) ← CLASS_VALUES(pc, pd, V)
            (v, kind) ← SOLVE_CLASS(success, failed)
            V[pc · 17011 + pd] ← v;  K[pc · 17011 + pd] ← kind
 
procedure SWEEP():
    if no checkpoint exists:
        V[c] ← UNSOLVED for all 289,374,121 classes;  next ← 1200
    else:
        restore next (and the route counters) from the checkpoint
    while next ≥ 0:
        SOLVE_LAYER(next, V, K)
        flush V and K to disk
        atomically replace the checkpoint with { next - 1, updated route counters }
        next ← next - 1
    FINALIZE()
Interruption at any moment loses at most the layer in flight, which re-runs idempotently on the next start.
5.9 Finalize
procedure FINALIZE():
    for every class c:
        assert V[c] is solved,  |V[c]| ≤ 1 + 1e-9,  K[c] in {PURE, SUPPORT, LP}
    for 1,200 evenly strided classes c:
        assert | RECHECK(c) - V[c] | ≤ 1e-6  # independent re-derivation
    mark the checkpoint complete
5.10 Recheck — the audit primitive (and the policy oracle)
function RECHECK(c):
    (pc, pd) ← (c div 17011, c mod 17011)
    (success, failed) ← CLASS_VALUES(pc, pd, V)  # rebuilt from the stored children
    (v, kind) ← SOLVE_CLASS(success, failed)  # solved again, independently
    return v  # SUPPORT_SOLVE / LP_SOLVE also yield p, q: the optimal mixed strategies
6. Cost, measured (single core, the Python reference)
Region
Classes
Measured rate
phi ≤ 600 (the bulk; largest layer 1,678,715 classes at phi = 374)
283,297,201
~9,500 / s
phi 601..840 (mixed band, LP-heavier)
6,015,600
1,200–3,500 / s
phi 841..1200 (dead band)
61,320
2,000–5,000 / s

Total: about nine hours, ~2.5 GiB of disk, a few GiB of RAM. Expected routing mix at the end: ~334.6k pure, ~288.85M support-certified, ~191k LP (a few hundred borderline classes may route differently between builds; their values are certified either way).
7. Verification anchors
After FINALIZE, the following certified reference values must match within 1e-6 (independent builds typically agree to 1e-9 or better):
State (s_c, t_c, s_d, t_d)
Expected value
Note
(0, 0, 0, 0)
0.08985007280951046
the root of the whole game
(240, 0, 240, 0)
0.3372132166291093
independently derivable dead-band reference
(10, 60, 200, 0)
-0.7944428916469297


(150, 90, 30, 120)
0.7244093036356785


(250, 300, 40, 0)
0.9981152817381969


(100, 140, 100, 140)
0.1877386378276193



procedure VERIFY():
    for (state, expected) in the anchor table:
        assert | V[ENCODE(state)] - expected | ≤ 1e-6
8. Provenance
The rules restated here are frozen by the repository (docs/REVIVAL_MODEL.md owns the revival surface; src/dth/ is the behavioral authority). The Python reference implementing exactly this pseudocode was validated layer-by-layer against the repository's independently implemented certified artifact: sampled layers across every region agreed to at most 5.3e-7 (only on LP-routed classes; machine precision elsewhere), and all six anchors matched.

