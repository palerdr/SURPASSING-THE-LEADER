# STL Project Instructions

This subtree contains the canonical state and clock helpers and the L2 leap
builder. We require the full artifact audit before claiming an L2 solve. We do
not provide a learned model, opponent model, or gameplay agent. Git history is the archive for
the removed experiments; do not restore them under `legacy`, `old`, or
version-suffixed names.

## Retained surfaces

- `cli.py` and `config/` are a neutral Hydra experiment harness. The default
  configuration performs no command and encodes no solver or learning choice.
- `engine/` is retained only as the compatibility interface currently consumed
  by `src/arena/`. It is a behavioral reference for already-frozen rules, not
  the canonical full STL formulation and not a foundation to extend casually.
- `tests/engine/` protects that compatibility interface. `test_cli_contract.py`
  protects the generic Hydra dispatcher.
- `solver/canonical.py` is the immutable state and leap-route skeleton specified
  by `docs/GAME_AND_SOLVER.md`; it is not a solved policy or planning system.
- `solver/leap_profiles.py` owns the STL quotient and child clocks.
  `solver/leap_oracle.py` certifies scalar stages; `solver/leap_build.py` owns
  reachability, calibration, and the full sweep checkpoint. Python remains the
  behavioral authority for the opt-in Rust leap kernel.
- Repository-wide canonical rules and formulation contracts belong in root
  `docs/`. Generated experiment data remains gitignored and STL-owned.

Read `docs/GAME_AND_SOLVER.md` for STL's full-game formulation and intended
solver architecture. It specializes the repository-wide contracts without
redefining their shared mechanics.

## Rebuild rule

Freeze the full game's state, observations, actions, transition/chance model,
and utility in the canonical documentation before adding implementation code.
Then add the smallest implementation that realizes that contract. Do not add
solver, learning, or play modules merely to preserve an earlier experiment.

Run `uv run python -m pytest src/stl/tests -q` for the retained surface.

## Leap calibration

Build the extension from `src/crates/stl_solver` with
`uv run --project ../../.. maturin develop --release`. From the repository root,
run `uv run python -m stl.cli command=leap`. The command checks the source and
DTH hashes, sweeps clocks 3600 through 3420, and retains dense tables and packed
copies under `outputs/leap/`. Resume uses the same command. Calibration rejects
other sweep floors. Pointwise tests below that floor do not sweep keys.

For the full public game, run `uv run python -m stl.cli command=leap_full`.
This command sweeps through the clock-720 opening and saves its artifact under
`outputs/leap-full/`. It preserves the calibration artifact. Resume uses the
same command and requires matching source, DTH, and checkpoint file hashes.
The builder commits packed-file hashes before removing each cold dense table.
You can read the opening value and LP-certified opening strategies in
`report.json`. The audit checks packed membership and finite values for all
reachable states, then checks sampled Bellman equations with HiGHS. It records
the audit sample count and residuals in `audit.json` and the final manifest.
For fallback stages, workers first solve the triangular packing LP in Rust
and check both saddle bounds on the original matrix. Python sends rejected
stages to `highspy`. Those workers reuse HiGHS bases and cache certified revival
values only when the child tables, profiles, and window flag match. The report
counts kernel failures, new LP solves, and cache hits as separate quantities.
The scalar LP retries with HiGHS' interior-point method if the first LP attempt
fails its status or saddle-gap check. If both methods fail, it retries IPM after
subtracting a common payoff constant. It checks each candidate against the
original matrix at the same 1e-6 gate.
The scalar retry sequence uses the failure payoff, then the first success
payoff, as common offsets if earlier attempts fail.

You can test reduced support reuse with `leap_lp.SupportFallback`. It seeds
each Checker row with a HiGHS policy, tries the recurrence-based reduced
system and paired edge moves, and checks the complete saddle gap at 1e-6.
It sends misses and systems above its dimension limit to HiGHS. The full
command leaves support reuse disabled; the measured H2 batches ran slower with support
reuse. The full sweep uses `NativeFallback` with a fresh packing basis per
stage. The Rust API accepts optional `supports` and `support_out` masks and
preserves its original behavior when you omit them. See
[`LEAP_CERTIFICATE.md`](../crates/docs/LEAP_CERTIFICATE.md) for the equations
and acceptance rule, and `solver/leap_support.py` for the Python authority.

You can reproduce the packing, native pivot, traversal, and REV reuse
benchmarks with `uv run python -m stl.solver.benchmark_leap_run`. This command
requires the retained `outputs/leap-full-centered/` children and the rung-2b
samples. It writes measurements under `outputs/leap-hypotheses/` and preserves
the sweep checkpoint. The native prototype lives in the Rust `leap_bench`
example. It certifies the original 60-action matrix after each packing solve;
the benchmark sends rejected stages to HiGHS. These benchmarks do not enable
a new full-sweep backend. Solve timings include process startup and fallback;
the REV experiment includes writes of sampled table rows and their flushes.

`solver/leap_resume.py` can retain a prefix from before this retry addition.
It accepts only that source change, verifies the original source snapshot and
file hashes, and compares a new reachability pass with the saved bitmaps. It
also audits the complete prefix's membership and finite values. The new
artifact retains the original source hash on each prefix record and archives
its source and checkpoint under `provenance/`.
`solver/leap_recover.py` applies the same source restrictions to the centered
retry. It archives the old partial table, reruns the kernel, and reuses its
certified LP values. It records recovered work and the missing LP timing scope.
Its `--native-review` option binds the packing upgrade to exact source hashes.
It rejects changes to protected game and dependency files, checks a fresh
reachability pass against the saved bitmaps, and audits the completed prefix.
The continuation archives both source versions and copies the unfinished table
before recovery. Normal checkpoint resume still requires an exact builder hash.

The packed bitmap excludes the implicit WIN column. Values follow its set
bits in row-major order. Each dense table keeps NaN in unreachable cells and
-1 in its last column. The manifest records file hashes and marks this segment
as incomplete. A complete L2 claim still requires the full audit.

The slow leap tests require `src/dth_compact/artifacts/V.npy`; kernel tests
require the extension. Run `uv run python -m pytest src/stl/tests -q` after the
build. The full reachability test checks the measured state totals without
performing a value sweep.
