# Exact role-relative bucket abstractions

`abstract` is a standalone exact solver for one finite stochastic game. It has
no neural evaluator, MCTS, self-play, CFR, private information, leap rule, or
identity-specific parameters.

Two discretizations implement the same rules:

| Ruleset | Bucket | Actions | Load cap | Failed-check dose | TTD half-life |
|---|---:|---:|---:|---:|---:|
| `bucket6_ttd_curve95` | 10 seconds | `1..6` | 30 units | 6 units | 12 units |
| `bucket12_ttd_curve95` | 5 seconds | `1..12` | 60 units | 12 units | 24 units |

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

Failed checks apply one 60-second dose: 6 units in the 10-second formulation or
12 units in the 5-second formulation. For eligible deaths, revival uses the
same shared no-CPR, no-physicality curve. With bucket-specific cap `C` and
120-second half-life `H`:

```text
revive = 0.95 * (1 - (dose / C)^3) * 2^(-((ttd / H)^1.3))
```

A current dose of `C` units or more is fatal; `ttd + dose > C` is fatal; and
equality at `C` remains eligible. A survival resets the old Checker's vial and
adds the dose to their TTD.

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

The compact 10-second compatibility artifact can still be built as an NPZ:

```powershell
uv run python -m abstract exact --ruleset bucket6_ttd_curve95
```

The full 5-second tablebase uses the packed, resumable v2 builder:

```powershell
uv run python -m abstract exact `
  --ruleset bucket12_ttd_curve95 `
  --checkpoint-states 10000
```

For the multi-hour production build, install the parity-versioned Rust kernel
and request it explicitly (or leave `--backend auto`, which selects it when
installed):

```powershell
cd crates/abstract_solver
uv run --project ../.. maturin develop --release
cd ../..
uv run python -m abstract exact `
  --ruleset bucket12_ttd_curve95 `
  --backend rust
```

Rust expands bitset/queue chunks and constructs/certifies pure-saddle backup
batches. It returns only genuinely mixed payoff matrices to Python's persistent
HiGHS LP, so the Rust path cannot silently substitute a different matrix
solver.

The default output is
`abstract/outputs/bucket12_ttd_curve95/`. Re-running the same command resumes
from `build-progress.json`. Reachability uses a bitset and a preallocated
`uint32` queue. Reachable packed indices are counting-sorted by potential;
values, policies, probabilities, and the packed-index-to-row map are contiguous
memory-mapped arrays. Backups checkpoint only after array pages are flushed.

Most late-potential matrices have a pure saddle. The builder certifies that
fast path directly and invokes HiGHS only when the minimax lower and upper
bounds differ, i.e. for a genuinely mixed `12 x 12` state.

The hot artifact contains a packed state index, never a per-row SHA string.
State IDs are derived only at lookup or export:

```powershell
uv run python -m abstract lookup `
  abstract/outputs/bucket12_ttd_curve95 0 0 0 0
```

The reference full build has 8,870,160 reachable states out of 13,395,600
physical packed indices. It certifies 752,756 states by the pure-saddle fast
path and sends 8,117,404 genuinely mixed states to LP. The initial-state value
for the current Dropper is `0.08832751940455077`. After float32 policy storage,
an exhaustive second best-response audit finds a maximum saddle gap of
`4.3575783592597617e-08`, below the frozen `2e-7` gate.

The byte-level state, transition, numeric, and checkpoint contract that must be
satisfied before replacing a Python hot loop with Rust is specified in
[`PACKED_TABLEBASE_PARITY.md`](PACKED_TABLEBASE_PARITY.md).
