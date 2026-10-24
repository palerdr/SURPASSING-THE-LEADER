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
  reachability, calibration, and the full sweep checkpoint.
  `solver/leap_kernel.py` is the numba leap kernel that the sweep calls.
  `solver/leap_oracle.py` and `solver/leap_support.py` remain its scalar
  authorities, and `tests/test_leap_golden.py` holds it to the outputs of the
  retired Rust kernel.
- `reader.py` is the public read path into a completed leap artifact
  (`stl.reader` in `docs/PROJECTS.toml`). Other projects and the paper's
  figure scripts read leap tables through it, not through `solver/`.
  `open_leap()` checks the manifest schema, the complete flag, and the SHA-256
  of the DTH tail before it returns a table. `verify="files"` also re-hashes
  each file that the manifest lists. `LeapTable.stage()` re-certifies one
  stored value with HiGHS and raises when the Bellman residual or the saddle
  gap exceeds 1e-6. Default paths resolve from this package, or from the
  absolute paths in `STL_LEAP_ARTIFACT` and `STL_LEAP_DTH`. The module opens
  the artifact read-only. It sits outside `solver/`, so the builder hash does
  not cover it; keep it there. `tests/test_reader.py` reproduces the paper's
  16 H1 records and skips when `outputs/leap-full-native/` is absent.
- `experiments/` holds the leap benchmarks: `benchmark_leap.py`,
  `benchmark_leap_run.py`, `benchmark_leap_replay.py`, and the rung 2b
  benchmarks in `experiments/rung2b/`. They read the builder and its
  artifacts and write no sweep table.
- `provenance/` holds `leap_resume.py` and `leap_recover.py`, the one-shot
  tools that carried earlier runs across reviewed source changes. They keep
  the bytes they had in `solver/`, so their allow-lists still name
  `src/stl/solver/`. `leap_recover.py` imports `stl.solver.leap_resume`, and
  `provenance/__init__.py` registers the moved module under that name. The
  builder hash no longer covers them, and they cannot certify the current
  source.
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

The leap kernel is `solver/leap_kernel.py`. numba compiles it on first use,
in 14 to 35 seconds, and caches the machine code in the ignored
`solver/__pycache__/`. A change to the kernel source or to the numba version
in `uv.lock` invalidates that cache. Set `NUMBA_CACHE_DIR` to move it, for
example to a disk that a container keeps. From the repository root, run
`uv run python -m stl.cli command=leap`. The command checks the source and
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
For fallback stages, workers first solve the triangular packing LP in the
numba kernel and check both saddle bounds on the original matrix. Python sends rejected
stages to `highspy`. Those workers reuse HiGHS bases and cache certified revival
values only when the child tables, profiles, and window flag match. The report
counts kernel failures, new LP solves, and cache hits as separate quantities.
The scalar LP retries with HiGHS' interior-point method if the first LP attempt
fails its status or saddle-gap check. If both methods fail, it retries IPM after
subtracting a common payoff constant. It checks each candidate against the
original matrix at the same 1e-6 gate.
The scalar retry sequence uses the failure payoff, then the first success
payoff, as common offsets if earlier attempts fail.

The full sweep solves H1 and H2 residue inside the kernel, as
`leap_build.RESIDUE` selects. Each 1024-class chunk moves the certified
support of its preceding residue class with the success-payoff kink and tries
it once. On a miss, the chunk runs the packing LP from the recurrence crash
basis. REV keys keep the Python fallback and its revival cache, and their
packing LP also starts from the crash basis. Every stored value still passes
the full 1e-6 matrix gate; [`LEAP_CERTIFICATE.md`](docs/LEAP_CERTIFICATE.md)
holds the equations. Records count kernel LP solves as `kernel_native_solves`
and support hits as `support_solves`, and `failures` still counts all
residue. The kernel time then includes the residue LPs, so compare residue
paths on sweep time. A change to `RESIDUE` changes the builder hash, so a
checkpoint cannot resume across it.

You can replay whole keys against the certified `outputs/leap-full-native/`
artifact before a full run:

```bash
uv run python -m stl.experiments.benchmark_leap_replay --keys H2_47 H2_35 --configs baseline kernel-crash kernel-crash-kink1
uv run python -m stl.experiments.benchmark_leap_replay --summarize --exclude H2_39
```

The replay restores each key's children, runs the production `sweep_key` once
per configuration, and aborts if a value differs from the artifact by more
than 1e-6. Run it on AC power: H1_44's baseline took 213.6 s on battery and
122.9 s on AC. On 2026-09-22 we replayed H2_47, H2_35, H2_31, H1_40, and
H1_44 on AC power, 155.5M residue classes in total. Per residue class, the
baseline took 8.5 µs of sweep time, the kernel crash path 1.7 µs, and one
kink attempt 1.05 µs. The seeds certified 70% to 75% of the residue. On H2_39,
two and four attempts ran slower than one. Every value matched the artifact
within 5.0e-7, and no class needed HiGHS. The projection gives about 1,160 s
of sweep time, against 5,070 s for the prior residue path and 12,195 s in the
recorded run, which solved its first keys with HiGHS. The reports and the
projection live under `outputs/leap-replay/`.

You can test reduced support reuse with `leap_lp.SupportFallback`. It seeds
each Checker row with a HiGHS policy, tries the recurrence-based reduced
system and paired edge moves, and checks the complete saddle gap at 1e-6.
It sends misses and systems above its dimension limit to HiGHS. The full
command leaves support reuse disabled; the measured H2 batches ran slower with support
reuse. The full sweep uses `NativeFallback` with a fresh packing basis per
stage. `leap_kernel.sweep_batch` accepts optional `supports` and `support_out`
masks and preserves its original behavior when you omit them. See
[`LEAP_CERTIFICATE.md`](docs/LEAP_CERTIFICATE.md) for the equations
and acceptance rule, and `solver/leap_support.py` for the Python authority.

`uv run python -m stl.experiments.benchmark_leap_run` reproduced the packing,
native pivot, traversal, and REV reuse benchmarks. It reads children from the
superseded partial runs `outputs/leap-full-centered/`, `leap-full-ipm/`,
`leap-full/`, `leap-full-initial/`, and `leap/`, the stopped H2_42 batch of
`leap-full-centered/`, and the rung 2b samples. We removed those five runs on
2026-09-25, so the command now stops and names the missing inputs. Git never
held the runs. Uncommitted builders made `leap-full/`, `leap-full-ipm/`, and
`leap-full-centered/` (`aea64aaf...`, `fcbe4f9c...`, `19bd5373...`), and
`outputs/leap-full-native/provenance/` keeps their sources. A new build from
any commit finishes H2_42 under another builder, so it cannot recreate the
stopped batch, and the benchmark cannot run again. The measurements of the
last run remain under `outputs/leap-hypotheses/`. The native prototype is
`leap_kernel.solve_packing_runs`, which the retired Rust `leap_bench` example
first implemented. It certifies the original 60-action matrix after each
packing solve; the benchmark sends rejected stages to HiGHS. These
benchmarks do not enable a new full-sweep backend. Solve timings include
process startup and fallback; the REV experiment includes writes of sampled
table rows and their flushes.

The rung 2b benchmarks of 2026-09-20 compare HiGHS with reduced-support
reuse. `uv run python -m stl.experiments.rung2b.benchmark` reads
`outputs/leap-rung2b-samples.npz`, then runs `rung2b/throughput_benchmark.py`
and `rung2b/batch_benchmark.py`. Those two read dense tables of the removed
partial runs, and `batch_benchmark.py` also needs the stopped H2_42 batch. A
new build does not recreate those inputs, so the two scripts cannot run again.
All three write their JSON reports to `outputs/`, where the 2026-09-20
reports remain.

`provenance/leap_resume.py` retained a prefix from before this retry addition.
It accepts only that source change, verifies the original source snapshot and
file hashes, and compares a new reachability pass with the saved bitmaps. It
also audits the complete prefix's membership and finite values. The new
artifact retains the original source hash on each prefix record and archives
its source and checkpoint in the artifact's own `provenance/` folder.
`provenance/leap_recover.py` applies the same source restrictions to the
centered retry. It archives the old partial table, reruns the kernel, and
reuses its certified LP values. It records recovered work and the missing LP
timing scope.
Its `--native-review` option binds the packing upgrade to exact source hashes.
It rejects changes to protected game and dependency files, checks a fresh
reachability pass against the saved bitmaps, and audits the completed prefix.
The continuation archives both source versions and copies the unfinished table
before recovery. Normal checkpoint resume still requires an exact builder hash.

The packed bitmap excludes the implicit WIN column. Values follow its set
bits in row-major order. Each dense table keeps NaN in unreachable cells and
-1 in its last column. The manifest records file hashes and marks this segment
as incomplete. A complete L2 claim still requires the full audit.

The slow leap tests require `src/dth_compact/artifacts/V.npy`. Run
`uv run python -m pytest src/stl/tests -q`; its first run compiles the kernel
in each worker. The full reachability test checks the measured state totals
without performing a value sweep.

## Numba kernel

On 2026-10-23 we replaced the Rust `stl_solver_rs` leap kernel with the numba
port in `solver/leap_kernel.py`. The port keeps the Rust operation order. It
writes `f64::mul_add`, `f64::min`, and `f64::max` as `llvm.fma`,
`llvm.minnum`, and `llvm.maxnum`, and it orders pivots by the IEEE total
order, as `f64::total_cmp` did. It keeps the 1024-class chunks, the 16-lane
batches, and the per-chunk packing tableau and kink seed.

Before the retirement we compared this kernel with `stl_solver_rs` on
113,175,708 rows in 725 cases: square and window stages under 13 option sets,
all 10 residue samples under `outputs/leap-residue/` in Dropper-major and
shuffled order, seeded stage families, random child tables, the support-seed
API, the packing LP, and the kink helpers. Values, kinds, failure counts, and
basis masks agreed bit for bit. Every sample value stayed within 4.97e-7 of
its certified value, inside the 1e-6 gate. The engines differed only on
malformed arguments: the kernel accepts an `int` for a `bool` and a `float`
for an integer, it rejects Fortran-ordered `supports`, `support_out`, and `s`
arrays, which Rust read in memory order, and its error messages differ.

After the port moved here, we ran it and the last `stl_solver_rs` build on
the H1_44 and H2_42 samples on AC power with 15 threads. Both engines wrote
the same bits in four workloads: the rungs alone, the production options of
`RESIDUE` in Dropper-major and shuffled order, and the window path. With the
production options, H1_44 took 1.08 s in numba against 1.26 s in Rust, and
H2_42 took 4.31 s against 7.64 s. An earlier run of the port on lower power
took 2.13 s against 2.70 s and 3.66 s against 6.74 s. The window path, where
the leap row wins every class, took 0.08 s against 0.04 s. A cold cache adds
a compile of 14 seconds on AC power, and up to 35 seconds on battery, to the
first calls.

`tests/data/leap_kernel_golden.npz` keeps 1,940 seeded stage classes and
1,100 random-table classes with the Rust outputs: square and window stages,
the rungs alone and the production options of `RESIDUE`, the packing LP from
the slack and crash bases, and the warm and cold packing runs of the
benchmarks. `tests/test_leap_golden.py` requires equal kinds, failure counts,
masks, and values. The commit that added the fixture holds
`tests/make_leap_golden.py`, which captured it.
