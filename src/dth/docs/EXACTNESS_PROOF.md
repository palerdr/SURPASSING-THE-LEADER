# Pure DTH complete-game exactness

## Finite acyclic game

For a live role-relative state `x = (s_c,t_c,s_d,t_d)`, every live transition
strictly increases raw damage `s_c+t_c+s_d+t_d`: a successful check adds the
inclusive lag `c-d+1`, while a survived failed check adds the fixed 60-second
penalty. The complete dependency graph is therefore a finite DAG and admits
backward induction.

## Exact per-player quotient

The game reads a player's TTD only in `revival_model(s,t)`. Once
`survive_injection(s,t)` is false, that probability is identically zero and no
future transition can make the profile survivable again. All TTD values for a
dead profile at the same ST are behaviorally equivalent.

The transition-closed reachable TTD domain is `{0} ∪ [60,300]`. Quotienting
dead TTD produces 17,011 per-player profiles and
`17,011² = 289,374,121` two-player classes. Alive profiles with synthetic TTD
1..59 are outside the artifact domain and fail closed.

## Sweep order

The complete tablebase orders profiles by

```text
rho(s,t) = t     while revival remains possible
           301   after revival is impossible
Phi(x)   = s_c + s_d + rho(s_c,t_c) + rho(s_d,t_d)
```

Every live edge strictly increases `Phi`:

- an alive-to-alive or dead-to-dead successful lag adds the positive lag;
- a successful lag that crosses into dead status adds at least lag + 61;
- a survived failed check adds exactly 60;
- a failed profile crossing into dead status adds at least 61.

There are no same-layer edges. Solving layers from `Phi=1200` downward always
reads completed child values. `test_complete_potential.py` checks all 1,035,541
live profile transitions exhaustively.

## Certified stage games

Each state has 61 continuation classes: 60 successful lags and one common
failed-check class. The solver reconstructs the literal 60×60 zero-sum matrix
and follows a pinned ladder of pure, warm-support, full-support, and LP residue
solves. Every accepted policy pair is checked against the full matrix:

```text
max_d (M q)_d - min_c (M^T p)_c <= 1e-6
```

The stored value is the midpoint of those certified bounds. Policies need not
be stored: `CompleteTablebase.certificate()` reconstructs and recertifies them
from the state's 61 child values.

## Artifact integrity

The manifest binds the class encoding, frozen rule hash, sweep configuration,
array shapes and dtypes, SHA-256 digests, solver routing counts, and sampled
Bellman recertification. The reader rejects any mismatch. The independent
dead-band solver pins `V(240,0,240,0) = 0.3372132166291093`, and Python/Rust
synthetic sweeps must be byte identical under
[`DTH_COMPLETE_PARITY.md`](DTH_COMPLETE_PARITY.md).

The post-build `complete-audit` workflow targets the numerical tail rather
than repeating the finalizer's layer-stratified sample. It re-solves every
LP-routed class, screens full-size equalizers at deterministic class-stratified
anchors, and sends the lowest-mass strict-full-support cases to the independent
two-LP oracle. Its generated report binds the audited array digests.

These properties certify the complete pure-DTH quotient—not the leap-aware L2
game. In canonical arena play the only additional prospective action is Baku's
Dropper action 61 during the public leap window.
