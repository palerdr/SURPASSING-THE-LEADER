# Compact DTH Solver Instructions

This subtree is the solver that the paper
([`paper/dth_exact_solution.tex`](../../paper/dth_exact_solution.tex))
describes: one file, [`main.py`](main.py), that builds the complete pure-DTH
quotient tablebase — one certified value for each of the 289,374,121 state
classes — in a single backward pass over the 1,201 potential layers, in about
49 s on a fifteen-core laptop. [`architecture.md`](architecture.md) is the
language-neutral recipe it implements, and `tests/` holds the eighteen
structural and numerical checks the paper cites.

- Rules: pure DTH with literal seconds 1..60, inclusive ST, and the frozen
  revival surface of [`docs/REVIVAL_MODEL.md`](../../docs/REVIVAL_MODEL.md).
  No leap second and no STL information mechanics.
- Rungs: an O(60) pure-saddle test, the Toeplitz equalizer recurrence (both
  equilibrium strategies from one 59-step recurrence), and an LP residue.
  Every stored value is the midpoint of a full-matrix saddle-gap certificate
  at `1e-6`; failing every rung aborts the build.
- Output: `artifacts/V.npy` (17,011 × 17,012 float64; the extra column is the
  terminal-win sentinel) and `artifacts/K.npy` (the rung per class). Both are
  gitignored.

## Working in this subtree

- `main.py` is the whole implementation; keep it one file. The scalar ladder
  (`try_rung1`–`try_rung3`, `solve_class`) is the oracle for the fused numba
  kernel (`_ladder`, `_solve_layer_kernel`) and for the finalize recheck; keep
  the two independent.
- Never import `dth`, `stl`, `abstract`, or `arena`. This project is not the
  arena's policy provider and owns no artifact schema; `src/dth/` remains the
  behavioral authority for play and for `complete_full_v1`.
- The six `ANCHORS` in `main.py` are certified reference values shared with
  `src/dth/docs/BUILD.md`; do not edit them without a rules change.
- Changing the paper's numbers (timings, routing counts, line count) means
  re-rendering `paper/dth_exact_solution.pdf`.
- Keep numba out of the root `pyproject.toml` and `uv.lock`:
  `src/dth/complete_tablebase.py` digests the root lock, so any change there
  orphans the `complete_full_v1` artifact. This subtree is therefore its own
  `uv` project (`pyproject.toml`, `uv.lock`, an ignored `.venv/`).

Build and verify from inside `src/dth_compact/`:

```powershell
uv sync --dev
uv run main.py
uv run pytest -q
```

or from the repository root with `uv run --project src/dth_compact ...`.
