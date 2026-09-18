# C++ DTH solver instructions

This subtree is an independent, minimal C++ implementation of pure Drop the
Handkerchief. It must not import or link the Python, Rust, OCaml, STL, abstract,
or arena projects. Those projects may be used only as external validation
oracles while developing this implementation.

The complete solver implements the recurrence amendment in [`BUILD.md`](BUILD.md).
That document owns this project's construction gates and numerical design.

HiGHS `1.15.1` is the pinned numerical backend. CMake first accepts an exact,
toolchain-compatible installed package and otherwise fetches the pinned source
commit when `DTH_FETCH_HIGHS=ON` (the default). HiGHS supplies model execution
and simplex plumbing; DTH code still formulates every rung and independently
certifies every returned policy against the full game matrix.

## Frozen scope

- Actions are literal seconds `1..60`; action zero is illegal.
- Successful inclusive ST is `check - drop + 1`.
- The dose after a failed check is `ST + 60`.
- Capacity is 300 seconds. Equality at total damage 300 remains
  revival-eligible when the individual dose is below 300.
- The revival surface comes only from
  [`docs/REVIVAL_MODEL.md`](../../docs/REVIVAL_MODEL.md).
- This is pure DTH. It has no STL leap window, player identity, observation
  state, route mechanics, or arena dependency.
- Every accepted local value must carry a full-matrix saddle gap of at most
  `1e-6`. Solver failure is fatal to the build; no approximate fallback may
  weaken that gate.

## Project files

- `dth.hpp` declares the game and solver types shared by the exact sweep and
  matrix solver.
- `exact.cpp` owns rules, quotient profiles, transitions, potential layers,
  and backward induction.
- `storage/` owns persistent tablebase storage. Its `durable_store.hpp` declares
  mapped files, mapped arrays, checkpoints, and durable stores, while
  `durable_store.cpp` owns checkpoint serialization and store lifecycle.
- `storage/mapped_array.tpp` implements the typed mapped-array template. The
  `storage/mapped_file_posix.cpp` and `storage/mapped_file_win32.cpp` backends
  own the operating system calls used to map and flush files.
- `matrix_game.cpp` owns the implicit stage matrix, certificate, O(60) pure
  reduction, support selection, equalizer/LP formulations, and solver ladder.
- `highs_backend.cpp` owns HiGHS model execution, fixed numerical options,
  status translation, and raw solution extraction. It owns no game rules or
  certification logic.
- `solve_tablebase.cpp` owns the command-line executable.
- `tests.cpp` owns the dependency-ordered native test executable.
- `BUILD.md` is the complete implementation specification.

Generated builds, checkpoints, tablebases, reports, and benchmarks belong
under this subtree's ignored `build/` and `outputs/` directories.

After implementation, validate from the repository root with the commands in
the final section of `BUILD.md`, then run `graphify update .`.

## Build and run

```sh
cmake --preset release -S src/dth_cpp -B src/dth_cpp/build/recurrence-release
cmake --build src/dth_cpp/build/recurrence-release -j 12
ctest --test-dir src/dth_cpp/build/recurrence-release --output-on-failure
src/dth_cpp/build/recurrence-release/dth-solve-tablebase --fresh --output src/dth_cpp/outputs/complete-recurrence-v1 --threads 12
```

Use `--resume` with the same output directory to continue a checkpoint.
Use `--verify-only` to open completed arrays read-only and check their
hashes and certificates. `--stop-after-layers N` commits before stopping.
`--checkpoint-every N` defaults to 50; set 1 to commit each layer. SIGINT
and SIGTERM request a checkpoint at the next layer barrier. A crash can
require replay of the unfinished checkpoint group.

Exit codes: 0 means complete or verified; 2 means invalid arguments;
3 means solver failure; 4 means an I/O error; 5 means a requested checkpoint
stop; 6 means an incompatible or corrupt artifact on open or verification.

The native artifact contains `values.bin` and `solver_kind.bin`, plus a
source-bound build config, durable checkpoint, and SHA-256 manifest. It
uses the C++ schema. The arena continues to consume the Python provider's
artifact. Each implementation owns its code and storage contract.

## Full-build verification on this Mac

On 2026-09-11, the full builds took the following times, including final
scans, recertification, and manifest hashing. These are single local runs.
The 12-worker run includes a checkpoint stop and resume.

| Workers | Seconds |
| --- | ---: |
| 12 | 29.75 |
| 14 | 28.97 |
| 15 | 28.15 |

All runs produced identical value and route hashes. We compared all
289,374,121 values with the Python recurrence build and found no difference.
The root is `0.08985007281413855`. The native read-only verifier checked
4,792 classes and both file hashes. The generated benchmark report remains
at `outputs/recurrence-benchmark.json`.
