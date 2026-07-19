# Surpassing The Leader Solver Monorepo

This repository contains three deliberately separate game-solving projects:

| Project | Purpose | Rules |
| --- | --- | --- |
| `stl/` | Full Surpassing the Leader engine, exact solver, search, learning, and play surface | Leap-aware public game; only Baku as Dropper may use second 61 |
| `dth/` | Pure Drop the Handkerchief solver | Literal seconds 1..60; no leap second or STL-only rules |
| `toy/` | Small exact example | Fixed finite abstraction intended for complete enumeration |

Repository-wide game contracts live in [`docs/`](docs/). The whitepaper,
primary game evidence, and cited research are kept separately in
[`papers/`](papers/). Rust acceleration lives in [`crates/`](crates/) and does
not replace the Python behavioral authorities.

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

# Exact toy example
uv run python -m toy --help
uv run python -m toy exact
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
not in the repository root. The only root Markdown files are this overview and
the global agent contract.
