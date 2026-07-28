# Surpassing The Leader Solver Monorepo

This repository contains deliberately separate game-solving projects:

| Project | Purpose | Rules |
| --- | --- | --- |
| `src/stl/` | Full Surpassing the Leader engine, exact solver, search, learning, and play surface | Leap-aware public game; only Baku as Dropper may use second 61 |
| `src/dth/` | Pure Drop the Handkerchief solver | Literal seconds 1..60; no leap second or STL-only rules |
| `src/abstract/` | Exact bucket examples | Role-relative 10-second and packed 5-second TTD abstractions solved by exhaustive tablebases |
| `src/ocaml/` | Minimal exact OCaml solver | Literal DTH seconds and the repository-wide frozen revival model |
| `src/crates/` | Shared Rust acceleration workspace | Python remains behavioral authority until explicit parity contracts |
| `arena/` | Neutral live-play surface | Canonical STL referee with pluggable policy-provider adapters |

Repository-wide game contracts live in [`docs/`](docs/). The whitepaper,
primary game evidence, and cited research are kept separately in
[`papers/`](papers/). The OCaml project records its authorship in
[`src/ocaml/AUTHORS.md`](src/ocaml/AUTHORS.md).

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

# Pure DTH target generation and training
uv run python -m dth --help
uv run python -m dth dataset --help
uv run python -m dth train --help

# Exact abstract example (optional: `arena play` builds this automatically when absent)
uv run python -m abstract --help
uv run python -m abstract exact

# Canonical referee with a pluggable Hal policy provider
uv run python -m arena play --hal-agent abstract
uv run python -m arena play --hal-agent abstract --buckets 5
```

Each project owns its `config/`, `docs/`, tests, checkpoints, and outputs.
Generated artifacts are gitignored and must not be mixed between projects or
across incompatible schema versions.

## Canonical contracts

- [`docs/ACTION_TIMING.md`](docs/ACTION_TIMING.md) owns literal action seconds,
  successful-check timing, and leap-second action legality.
- [`docs/CANONICAL_EXTENSIVE_FORM.md`](docs/CANONICAL_EXTENSIVE_FORM.md) owns the
  full-game state, transition, death, revival, clock, and terminal boundaries.
- [`docs/FOUNDATIONS.md`](docs/FOUNDATIONS.md) owns the shared zero-sum matrix
  and exact-solver foundations without repeating game rules.
- [`papers/game-sources/EVIDENCE.md`](papers/game-sources/EVIDENCE.md) records
  the documentary basis for frozen rule choices.

## Repository policy

Root guidance applies everywhere. Read the nearest nested `AGENTS.md` before
working inside a project. Folder-specific context belongs beside that folder,
not in the repository root. Root Markdown is limited to this overview and the
global agent contract; project Markdown stays under its owning `src/` subtree.
