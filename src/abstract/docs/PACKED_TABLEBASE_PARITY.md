# Python/Rust packed-tablebase parity contract

Python in `src/abstract/rules.py`, `src/abstract/packed.py`, and
`src/abstract/packed_tablebase.py` is the behavioral authority. A Rust accelerator
may replace reachability enumeration or bottom-up backup only after satisfying
this contract. The accelerator is opt-in until all parity gates pass; an
unchecked Rust result must never silently replace Python output.

## State encoding

Let `C` be `load_cap_units` and `T = C + 1`. The four public fields are ordered
as:

```text
(checker_load, checker_ttd, dropper_load, dropper_ttd)
```

The packed unsigned integer index is:

```text
(((checker_load * T + checker_ttd) * C + dropper_load) * T) + dropper_ttd
```

Loads are in `0..C-1`; TTD fields are in `0..C`. The 5-second ruleset has
`C = 60`, so its `13,395,600` physical indices and all reachable ordinals fit
in `uint32`. Decode is the exact inverse sequence of Euclidean `divmod`
operations. Rust must reject the same out-of-domain API inputs; unchecked
internal helpers may be used only after the caller establishes those bounds.

## Transition parity

For every reachable packed state and every joint action in `1..N`:

- `check >= drop` uses `squandered = check - drop + 1`;
- successful overflow is terminal `+1` for the current Dropper;
- every live successor swaps roles exactly as `AbstractRuleset`;
- `check < drop` uses `dose = checker_load + N`;
- dose and `ttd + dose` fatality inequalities match Python exactly;
- live successors have strictly greater integer potential;
- chance branch probabilities, events, terminal values, and packed child
  indices match Python.

Revival calculations use IEEE-754 binary64. Cross-language absolute error must
be at most `1e-15` per branch probability. Probability mass must sum to one
within `1e-12`.

The `abstract-packed-parity-v3` contract covers the sole frozen probability
model, `linear_st_geometric_ttd_v1`. Python and Rust must agree on the linear
pre-failure-ST factor, geometric `0.75^(ttd / failed_check_penalty)` factor,
and fatality guards. The frozen model identifier and constants are part of
every checkpoint configuration digest and artifact manifest; other metadata
is rejected.

The public Rust binding validates the packed physical-domain size, exact array
shapes, state and child indices, child value/probability domains, and strictly
increasing potential before performing a backup. For this contract,
`action_size` must equal the failed-check penalty. Malformed inputs return a
Python exception rather than reaching an unchecked index or arithmetic path.

## Reachability parity

From packed root index zero, Rust must produce exactly the same bitset
membership and reachable count as Python. Queue order is not semantic and may
differ. Tests must compare:

- exhaustive closure for at least one small contract ruleset;
- the established 10-second closure count of `576,270`;
- deterministic sampled and boundary states for the 5-second ruleset;
- rejection of every non-increasing live edge.

Checkpoint commits are ordered: flush queue and bitset data, then atomically
replace the progress manifest. Resume must ignore writes beyond the committed
queue tail and reconstruct membership from the committed queue prefix.

## Backup and matrix parity

States are ordered by nondecreasing potential and solved in reverse order.
Every live child must map to a strictly higher-potential solved row. Rust must
match Python cell matrices within `2e-14` absolute error.

The pure-saddle gate is exact apart from Python's fixed comparison
`atol=1e-12, rtol=0`. Rust may take the fast path only when it identifies a
cell that simultaneously attains the maximum row minimum and minimum column
maximum and the resulting certified saddle gap is at most `2e-7`. All other
states are genuinely mixed and must be sent to the same LP contract. Neither
implementation may loosen feasibility, optimality, or saddle-gap tolerances.

For every parity state:

- value absolute error is at most `2e-10`;
- each policy is finite, nonnegative, and sums to one within `2e-7`;
- both policies' saddle gap is at most `2e-7`;
- dropper/checker win-probability absolute error is at most `2e-10`;
- pure-versus-LP routing is identical.

Gates cover all states of a small exhaustive ruleset, sampled states from every
reachable potential layer of both production rulesets, every terminal boundary,
and every state routed to LP in the sample.

## Artifact and resume parity

The artifact schema is `abstract.packed-tablebase.v5` and the resumable
checkpoint schema is `abstract.packed-tablebase-build.v4`. Required arrays,
shapes, dtypes, packed ordering, SHA-256 file digests, metadata, and lookup
results must match the Python build. Resume and read-time compatibility bind
the Python implementation sources and `uv.lock` and, when used, the Rust crate,
build script, and workspace lockfile. The loaded extension must expose the
compile-time source-bundle digest that Python independently recomputes from
those Rust inputs; stale binaries, partial layers, or artifacts fail closed
after an implementation change.
Per-state SHA IDs are forbidden in hot arrays and are derived from
`ruleset_id|state_fields` only at lookup/export.

Interrupt/resume parity tests must stop during reachability and during at least
two distinct backup chunks, then compare the resumed artifact byte-for-byte
with an uninterrupted build. A Rust backend is eligible as the default only
after these tests pass in CI for both Python fallback and compiled extension.

Set `STL_REQUIRE_RUST_PARITY=1` in CI or a release-validation shell. With that
gate enabled, a missing `abstract_solver_rs` extension is a hard
test-collection failure instead of an ordinary Python-development skip.
