# C++ DTH solver instructions

This subtree is an independent, minimal C++ implementation of pure Drop the
Handkerchief. It must not import or link the Python, Rust, OCaml, STL, abstract,
or arena projects. Those projects may be used only as external validation
oracles while developing this implementation.

The implementation files are intentionally empty at scaffold creation. Build
them strictly in the chronological order specified by [`BUILD.md`](BUILD.md).
That document owns this project's construction sequence and numerical design.

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
- `durable_store.hpp` declares mapped files, mapped arrays, checkpoints, and
  durable stores. `durable_store.cpp` owns checkpoint serialization and store
  lifecycle.
- `mapped_array.tpp` implements the typed mapped-array template. The
  `mapped_file_posix.cpp` and `mapped_file_win32.cpp` backends own the operating
  system calls used to map and flush files.
- `matrix_game.cpp` owns the implicit stage matrix, certificate, O(60) pure
  reduction, equalizer systems, and dual-simplex fallback.
- `solve_tablebase.cpp` owns the command-line executable.
- `tests.cpp` owns the dependency-ordered native test executable.
- `BUILD.md` is the complete implementation specification.

Generated builds, checkpoints, tablebases, reports, and benchmarks belong
under this subtree's ignored `build/` and `outputs/` directories.

After implementation, validate from the repository root with the commands in
the final section of `BUILD.md`, then run `graphify update .`.
