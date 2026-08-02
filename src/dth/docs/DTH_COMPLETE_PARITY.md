# DTH Complete-Game Tablebase Parity Contract

`src/dth/packed.py` and `src/dth/complete_tablebase.py` are the behavioral
authority for the packed-quotient complete-game tablebase. The Rust kernel
`dth_complete_rs` (`src/crates/dth_complete`) is an opt-in accelerator and remains
opt-in until every gate in this contract passes; an unchecked Rust result must
never silently replace Python output. The contract version string is
`dth-complete-parity-v1`, exported by the extension and checked at import; a
mismatch refuses the backend outright.

## Class encoding

- A profile is one player's `(ST, TTD)` pair quotiented by the per-player
  TTD-dead rule of `tests/test_dead_ttd_quotient.py`: 16,711 alive profiles
  enumerated TTD-ascending over `{0} | [60, 240]` with ST ascending inside
  each TTD, then 300 dead sentinels by ST — 17,011 ids. This enumeration
  order is normative for `dth-packed-class-v1` and never changes without a
  version bump.
- A class is `checker_profile * 17_011 + dropper_profile`, one of
  289,374,121 indices; it fits `uint32`, and the kernel API carries it as
  `uint64`.
- A dead sentinel decodes to the representative TTD 300. An **alive** profile
  with TTD in 1..59 is a valid live state outside the artifact's
  transition-closed domain: `profile_id` and every state-facing lookup fail
  closed on it. A **dead** profile is accepted with any TTD, because the
  quotient discards dead TTDs exactly.
- The addressable domain `{alive TTD in {0} | [60, 300]}` is transition
  closed: revival is only possible when `checker_st + checker_ttd <= 240`, so
  a failure child's TTD `checker_ttd + checker_st + 60` never exceeds 300.

## Graph tables

- Python precomputes, once per build, the per-profile rule tables:
  `success_child` (17,011 x 60, -1 for cylinder overflow), `failure_child`
  (-1 for a dead checker), and `revival` (float64, 0.0 for dead). These are
  the bit-level authority: **the kernel performs no transcendental
  arithmetic** and only gathers from these arrays, which is what makes class
  assembly reproducible bit for bit across languages.
- The tables derive from `dth.solver.survive_injection` / `revival_model`
  only; their SHA-256 (with the encoding string) enters the build config
  digest, alongside `solver_schema_hash()`, so a rules change orphans every
  checkpoint fail-closed.

## Potential schedule

- `phi(profile) = ST + TTD` when alive, `ST + 301` when dead; a class's
  potential is the sum over both profiles, at most 1,200. Every live
  transition strictly increases it: success by the lag (alive to alive and
  dead to dead), by at least lag + 61 when the mover's profile dies; failure
  by exactly 60 when the revived profile is alive and by at least 61 when it
  is dead. `tests/test_complete_potential.py` checks all 1,035,541 live profile
  transitions exhaustively; the proof is in `docs/EXACTNESS_PROOF.md`.
- There are no same-layer edges, so the sweep solves whole layers descending
  from 1,200 with a barrier per layer, and layer-P LP residues need writing
  only before layer P-1 begins.
- A layer is the union of rectangles `bucket(a) x bucket(P - a)` over
  per-profile potential buckets, ascending in `a`; work items partition each
  rectangle's dropper bucket. The partition is a pure function of the
  configuration.
- Warm-start guesses are **precomputed per state** from the previous layer's
  support table (see below); kernels perform no chaining, so artifact bytes
  are independent of worker count and work-item partition by construction.
  `test_complete_rust_parity.py` verifies byte-identity against
  `RAYON_NUM_THREADS=1`.

## Assembly parity

- The 61 class values of a class `(p_c, p_d)` are `success[lag] = 1.0` on
  overflow else `-value[p_d * 17011 + success_child[p_c][lag]]`, and
  `failed = 1.0` for a dead checker else
  `revival * (-value[child]) + (1.0 - revival)` — one multiply, one negate,
  one subtract, one add, **no FMA**, in exactly that order in both backends.
- Reading an unsolved (NaN) child value aborts the layer with an error in
  both backends; it can only mean a schedule violation.
- The implied 60x60 matrix is `matrix[d, c] = success[c - d]` when `c >= d`
  else `failed` (0-based). Cross-backend class values are bit-identical by
  construction; the fallback bound, should a platform break bit-identity, is
  2e-14 per class value, and any use of the fallback must be reported.

## Routing and the solve ladder

The pinned rung order is `pure / warm-support / full-support / lp-v1`; the
config digest carries it. Every rung certifies against the full matrix at the
frozen `SADDLE_GAP_TOLERANCE = 1e-6` and returns the certificate midpoint
`(lower + upper) / 2`; nothing is accepted on a weaker test.

1. **Pure saddle**, O(60): `row_min[d] = min(prefix_min[59 - d], failed if
   d > 0)`, `col_max[c] = max(prefix_max[c], failed if c < 59)`; accept iff
   `min col_max - max row_min <= 1e-6`; the stored value is the midpoint of
   the two reductions. Min/max are exact, so this is bit-stable.
2. **Warm support**: guesses are the previous layer's recorded supports at
   the two lag-1 neighbours `succ1(p_c) * N + p_d` then `p_c * N + succ1(p_d)`,
   first certified hit wins. The square support is trimmed to
   `k = min(|rows|, |cols|)` leading ascending indices; both equalizer
   systems `(k+1) x (k+1)` are solved by Gaussian elimination pinned to:
   first-maximum partial pivoting (lowest index on ties), rejection below
   pivot magnitude 1e-12, explicit zeroing of eliminated cells, elementwise
   `a[r][x] -= factor * a[p][x]` row updates with no FMA, and sequential
   ascending back-substitution. Negative mass beyond -1e-12 fails closed;
   surviving mass is clipped, summed sequentially, and normalized by
   division. The O(60k) certificate sums support terms in ascending index
   order.
3. **Full support**: the identical mechanism at `k = 60` — this *is* the
   structured full-support solve, and it dominates the endgame regions
   (measured 2026-07-30: 7,021 of 7,021 mixed classes in the top 120 layers).
4. **LP residue**: kernels return the state's 61 class values; both backends
   route them through the same Python code in ascending class order,
   optionally on an LP worker pool: `support_solver.solve_matrix_single_lp`,
   then `solver.solve_matrix`, then HiGHS IPM, then the two-LP oracle under
   tightened HiGHS
   dual-simplex tolerances (1e-10 feasibility) — the retry tightens the
   *solver*, never the 1e-6 gate, and a matrix failing all four aborts the
   build. Identical code, so cross-backend parity cannot break here.

Recorded supports (for the next layer's guesses) use the pinned extraction:
mass above 1e-9, top-`max_support` by descending mass with ascending-index
ties, stored ascending. Routing parity is **exact**: the `solver_kind` byte
per class (0 pure, 1 support-certified, 2 lp) must be equal across backends,
and `test_complete_rust_parity.py` compares full synthetic sweeps byte for
byte, primitive calls bit for bit, and counter sets exactly.

## Artifact and resume parity

- Schema strings: artifact `dth.complete-tablebase.v1`, build checkpoint
  `dth.complete-tablebase-build.v1`. Arrays: `value.npy` float64 and
  `solver_kind.npy` uint8, both dense over class indices, NaN /
  implicit-zero before solving. No per-class certificates are stored:
  given the child values any certificate is re-derivable on demand
  (`recertify_class`), and finalize refuses artifacts with non-finite or
  out-of-range values or unknown routing bytes and re-checks deterministic
  per-layer samples against fresh solves at the frozen tolerance.
- Checkpoint order per layer: flush hot arrays, atomically replace the
  warm-support table (`warm-supports.npz`, tagged with its layer), then
  atomically replace the progress manifest. Resume re-runs the last
  incomplete layer idempotently; a warm-support table whose tag disagrees
  with the manifest fails closed rather than mis-route. Counters accumulate
  only at layer completion, so an interrupted layer never double-counts.
- Interrupt/resume must be byte-for-byte against an uninterrupted build of
  the same backend, verified with stops in at least two distinct layers
  (`test_complete_sweep_python.py`, `test_complete_rust_parity.py`).
- The finalize manifest embeds the build config digest inputs, per-array
  SHA-256, the routing counters, and a `code_config_digest` over
  `packed.py`, `complete_tablebase.py` and — when Rust executed — the crate's
  `Cargo.toml`, `lib.rs`, and the workspace `Cargo.lock`. The read facade
  re-verifies shape, dtype, schema, and digests on open and fails closed.

## External anchors

- `build_dead_band_reference()` solves the dead x dead sub-DAG without sweep
  machinery and independently pins `V(240,0,240,0) = 0.3372132166291093`
  within 1e-9.
- The completed artifact must reproduce that independent anchor, and sampled
  classes must pass Bellman recertification from their stored children.

## Promotion gate

The Rust backend is eligible as a default only after every test above passes
for both the Python fallback and the compiled extension, and after a
completed canonical build passes its finalize gates and external anchors.
Per `src/crates/AGENTS.md`, no Python runtime behavior may depend on an
unchecked Rust path.
