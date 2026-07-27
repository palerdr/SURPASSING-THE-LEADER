# Exact role-relative bucket abstractions

`abstract` is a standalone exact solver for one finite stochastic game. It has
no neural evaluator, MCTS, self-play, CFR, private information, leap rule, or
identity-specific parameters.

These are ladder rung L0; see
[`docs/FORMULATION_LADDER.md`](../../../docs/FORMULATION_LADDER.md). Two
production discretizations implement the same rules:

| Ruleset | Bucket | Actions | Load cap | Failed-check dose | TTD half-life |
|---|---:|---:|---:|---:|---:|
| `bucket6_frozen95` | 10 seconds | `1..6` | 30 units | 6 units | 14.43 units |
| `bucket12_frozen95` | 5 seconds | `1..12` | 60 units | 12 units | 28.86 units |

The half-life is the frozen 144.3 seconds expressed in bucket units; it is a
derived quantity, not a free parameter.

The prior `bucket6_unified80`, `bucket12_unified80`, `bucket6_ttd_curve95`, and
`bucket12_ttd_curve95` rulesets remain addressable only to reproduce old
experiments. They are not CLI defaults and their artifacts are not
interchangeable with the production rulesets.

Actions are positive ordinal buckets and action zero is illegal. An action maps
to seconds only in artifact metadata. A check succeeds iff `check >= drop`, and
successful squandered time is inclusive in the selected discretization:

```text
ST = check - drop + 1
```

Each live public state is role-relative:

```text
(checker_load, checker_ttd, dropper_load, dropper_ttd)
```

For a cap of `C`, vial loads range from `0..C-1` and TTD ranges from `0..C`.
On every live transition, the current roles swap. Values are always from the
current Dropper's perspective, so a live child's value is negated during
backup.

## Frozen two-variable revival model

The revival surface is **not owned here**. It is frozen for the whole repository
in [`docs/REVIVAL_MODEL.md`](../../../docs/REVIVAL_MODEL.md); this section
restates it so the abstraction is readable on its own.

Let \(s\) be the ST already in the current Checker's vial immediately before a
failed check, and let \(t\) be that player's accrued TTD from prior survived
deaths, both in seconds. The failed check adds the fixed 60-second injection,
so \(q=s+60\). For an eligible death:

\[
P_{\mathrm{rev}}(s,t)
=0.95\left(1-\frac{s}{240}\right)\times 0.75^{\,t/60}.
\]

Thus \(P_{\mathrm{rev}}(0,0)=0.95\). Current ST acts linearly on the potency of
the impending dose, reaching zero exactly at the documentary lethal dose
\(q=300\) and nowhere else: the largest survivable injection is \(s=239\), where
the probability is \(0.95/240=0.003958\). Prior TTD decays geometrically at a
quarter per accrued death-minute, a 144.3-second half-life, keeping the state
role-relative and a function of exactly two variables.

`0.75` is a rounding of \(0.85\times0.88\), the product of `CARDIAC_DECAY` and
`REFEREE_DECAY` from the STL engine; the unified model is STL's own surface with
the dose term linearized and per-player physicality folded into the `0.95`
baseline. STL's `max(0.40, ...)` referee floor is omitted because it requires
more than 7.17 death-minutes while eligibility caps TTD at 4.0, so it can never
bind. Folding referee fatigue into accrued TTD is a modeling bridge: it does not
claim that accrued TTD is a literal observed CPR count.

In bucket units, let \(C=300/B\) and \(F=60/B\), where \(B\) is the bucket
width. If \(\ell=s/B\) and \(\tau=t/B\), the identical calculation is:

\[
P_{\mathrm{rev}}(\ell,\tau)
=0.95\left(1-\frac{\ell}{C-F}\right)\times 0.75^{\,\tau/F}.
\]

Because \(s/240=\ell/(C-F)\) and \(t/60=\tau/F\) identically, the surface is
**exactly** bucket-invariant: the same physical \((s,t)\) gives bit-identical
probabilities at \(B=10\), \(B=5\), and \(B=1\). This is what allows one play
interface to span every rung of the ladder.

A current dose of `C` units or more is fatal; `ttd + dose > C` is fatal; and
equality at `C` remains eligible. A survival resets the old Checker's vial and
adds the dose to their TTD.

The source record fixes the 300-second capacity and strict cumulative boundary,
but does not identify a numerical revival-probability surface. The shape and
parameters above are therefore an explicit solver model, not a transcription
of documentary odds. See the repository
[`EVIDENCE.md`](../../../docs/game-sources/EVIDENCE.md).

## Exact acyclic solution

The potential

```text
checker_load + checker_ttd + dropper_load + dropper_ttd
```

strictly increases on every nonterminal edge in both discretizations. The
reachable game graph is therefore finite and acyclic. Sorting reachable states
by potential and backing them up in descending order is the same exact dynamic
program as recursive memoization: every live child is already solved, terminal
branches are exact, and every simultaneous matrix is solved to the configured
saddle-gap tolerance. There is no depth horizon or approximate leaf value.

The production 10-second tablebase uses the packed, resumable builder:

```powershell
uv run python -m abstract exact --ruleset bucket6_frozen95
```

The full 5-second tablebase uses the same packed, resumable v3 builder:

```powershell
uv run python -m abstract exact `
  --ruleset bucket12_frozen95 `
  --checkpoint-states 10000
```

For the multi-hour production build, install the parity-versioned Rust kernel
and request it explicitly (or leave `--backend auto`, which selects it when
installed):

```powershell
cd src/crates/abstract_solver
uv run --project ../.. maturin develop --release
cd ../..
uv run python -m abstract exact `
  --ruleset bucket12_frozen95 `
  --backend rust
```

Rust expands bitset/queue chunks and constructs/certifies pure-saddle backup
batches. It returns only genuinely mixed payoff matrices to Python's persistent
HiGHS LP, so the Rust path cannot silently substitute a different matrix
solver.

The default outputs are `src/abstract/outputs/bucket6_frozen95/` and
`src/abstract/outputs/bucket12_frozen95/`. Re-running either command resumes from
`build-progress.json`. Reachability uses a bitset and a preallocated `uint32`
queue. Reachable packed indices are counting-sorted by potential; values,
policies, probabilities, and the packed-index-to-row map are contiguous
memory-mapped arrays. Backups checkpoint only after array pages are flushed.

Most late-potential matrices have a pure saddle. The builder certifies that
fast path directly and invokes HiGHS only when the minimax lower and upper
bounds differ, i.e. for a genuinely mixed `12 x 12` state.

The hot artifact contains a packed state index, never a per-row SHA string.
State IDs are derived only at lookup or export:

```powershell
uv run python -m abstract lookup `
  src/abstract/outputs/bucket12_frozen95 0 0 0 0
```

The regenerated production artifacts certify:

| Ruleset | Reachable states | Pure saddles | Mixed LP states | Initial Dropper value | Maximum persisted gap |
|---|---:|---:|---:|---:|---:|
| `bucket6_frozen95` | 576,270 | 95,294 | 480,976 | 0.09681321477839212 | \(3.579177532131439\times10^{-8}\) |
| `bucket12_frozen95` | 8,870,160 | 740,134 | 8,130,026 | 0.0927809531537424 | \(4.3575783537086465\times10^{-8}\) |

The physical transition graph is unchanged from the earlier probability
formulation. The 5-second closure occupies 8,870,160 of 13,395,600 physical
packed indices. Both persisted-policy gaps remain below the frozen
\(2\times10^{-7}\) gate; full per-array hashes and build digests remain in each
generated artifact manifest. Both production artifacts now bind
`linear_st_geometric_ttd_v1` with baseline `0.95` and TTD decay `0.75`.

The eligibility predicate is unchanged by the freeze, so the reachable counts
remain the same while values, policies, and saddle gaps are regenerated under
the frozen surface. The bucket6 reachable-state count was verified at exactly
576,270 before accepting the production artifact; bucket12 was regenerated at
exactly 8,870,160 reachable states.

The byte-level state, transition, numeric, and checkpoint contract that must be
satisfied before replacing a Python hot loop with Rust is specified in
[`PACKED_TABLEBASE_PARITY.md`](PACKED_TABLEBASE_PARITY.md).
