# Surpassing The Leader Solver Monorepo

This repository contains deliberately separate game-solving projects:

| Project | Purpose | Rules |
| --- | --- | --- |
| `src/stl/` | Full Surpassing the Leader engine, exact solver, search, learning, and play surface | Leap-aware public game; only Baku as Dropper may use second 61 |
| `src/dth/` | Completed pure Drop the Handkerchief solve and optional research tooling | Exact 289,374,121-class quotient tablebase; literal seconds 1..60 |
| `src/abstract/` | Exact bucket examples | Role-relative 10-second and packed 5-second TTD abstractions solved by exhaustive tablebases |
| `src/dth_ocaml/` | Minimal exact OCaml solver | Literal DTH seconds and the repository-wide frozen revival model |
| `src/crates/` | Shared Rust acceleration workspace | Python remains behavioral authority until explicit parity contracts |
| `src/arena/` | Neutral live-play surface | Canonical STL referee with pluggable policy-provider adapters |

Repository-wide game contracts live in [`docs/`](docs/). The mathematical
paper — a certified exact solution of the complete DTH game — lives at
[`paper/`](paper/); primary game evidence lives in
[`docs/game-sources/`](docs/game-sources/) and cited research in
[`docs/papers/`](docs/papers/). The OCaml project records its authorship in
[`src/dth_ocaml/AUTHORS.md`](src/dth_ocaml/AUTHORS.md).

## Setup

Use Python 3.12+ and `uv`:

```powershell
uv sync --dev
uv run python -m pytest --collect-only -q
uv run python -m pytest -q
cargo test --workspace
```

## Commands

```powershell
# Full STL Hydra command surface
uv run python -m stl.cli --help

# Pure DTH complete solution and optional research tools
uv run python -m dth --help

# Complete exact DTH quotient tablebase
uv run python -m dth complete
uv run python -m dth dataset --help
uv run python -m dth train --help

# Exact abstract example (optional: `arena play` builds this automatically when absent)
uv run python -m abstract --help
uv run python -m abstract exact

# Canonical STL referee; complete DTH is the default Hal policy
uv run python -m arena play
uv run python -m arena play --hal-agent abstract
uv run python -m arena play --hal-agent abstract --buckets 5
uv run python -m arena play --tui
```

Each project owns its `config/`, `docs/`, tests, checkpoints, and outputs.
Generated artifacts are gitignored and must not be mixed between projects or
across incompatible schema versions.

## Exact DTH paper

The completed paper is available as both the rendered
[`paper/dth_exact_solution.pdf`](paper/dth_exact_solution.pdf) and its
[`paper/dth_exact_solution.tex`](paper/dth_exact_solution.tex) source. It gives
a certified exact solution of the complete finite DTH game, including the root
value and equilibrium strategies.

With the exact value table available, the paper and its figures can be
reproduced from the repository root:

```powershell
uv run python paper/generate_figure_data.py
uv run --with matplotlib --with seaborn --with pandas python paper/make_figures.py
Push-Location paper
tectonic dth_exact_solution.tex --synctex --keep-logs
Pop-Location
```

## Canonical contracts

- [`docs/ACTION_TIMING.md`](docs/ACTION_TIMING.md) owns literal action seconds,
  successful-check timing, and leap-second action legality.
- [`docs/CANONICAL_EXTENSIVE_FORM.md`](docs/CANONICAL_EXTENSIVE_FORM.md) owns the
  full-game state, transition, death, revival, clock, and terminal boundaries.
- [`docs/FOUNDATIONS.md`](docs/FOUNDATIONS.md) owns the shared zero-sum matrix
  and exact-solver foundations without repeating game rules.
- [`docs/game-sources/EVIDENCE.md`](docs/game-sources/EVIDENCE.md) records
  the documentary basis for frozen rule choices.

## Repository policy

Root guidance applies everywhere. Read the nearest binding subtree `README.md` before
working inside a project. Folder-specific context belongs beside that folder,
not in the repository root. Root Markdown is limited to this overview and the
global agent contract; project Markdown stays under its owning `src/` subtree.
