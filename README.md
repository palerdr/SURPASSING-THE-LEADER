# Surpassing The Leader

Compact Python research implementation of the *Usogui* Drop the Handkerchief
game. The project is organized around one current path:

```text
engine rules -> exact/search anchors -> value/policy learning -> gated self-play
```

The goal is not to preserve every historical script. The live tree keeps the
engine, exact solver, playable agents, training bridge, and Hydra command
surface in `stl/`. Obsolete pre-consolidation packages are kept only under
`legacy/old_layout/` for reference.

## Current Layout

| Path | Role |
|---|---|
| `stl/engine/` | Rule authority: constants, players/referee, game clock, half-round resolution, death/revival, leap-second legality. |
| `stl/solver/` | Three-file solver core: `exact.py`, `search.py`, `tablebase.py`. |
| `stl/learning/` | Target generation, value/policy model, training, gates, and RL/self-play support. |
| `stl/play/` | Actual playable surface: `SolverAgent`, `CanonicalHal`, opponents, teachers, environment wrapper, terminal CLI. |
| `stl/commands/` | Importable command implementations used by Hydra. |
| `configs/` | Hydra command and experiment configs. |
| `tests/{engine,solver,learning,play}/` | Regression tests grouped by subsystem. |
| `docs/whitepaper/` | Current solver whitepaper source/PDF. |
| `docs/papers/` | Reference papers. |
| `legacy/old_layout/` | Obsolete old layout only. Do not add new runtime code here. |

Generated corpora, checkpoints, logs, demos, `outputs/`, `multirun/`, and
`target/` should stay out of source review unless the artifact itself is under
discussion.

## Current Action Core

Normal action seconds are `1..60`. During the leap window, only Baku as dropper
may use second `61`; Hal never uses `61`, and checkers are always capped at
`60`. Dense policy/action tensors have length `62`: index `0` is always-illegal
padding, and index `s` is literal second `s`.

Checks succeed iff `check_time >= drop_time`. Successful ST is exactly
`check_time - drop_time`, so same-second checks succeed with `ST=0`; failed
checks remain `check_time < drop_time` with the flat `+60` penalty.

Artifacts generated under the old action semantics are stale. See
`docs/REGEN2RL.md` for the reset and regeneration order.

## Setup

Use Python 3.12+ with `uv`:

```bash
uv sync --dev
uv run pytest --collect-only -q
uv run pytest -q
```

The latest verified full suite after the action-core reset was:

```text
641 passed
```

## Playing A Match

Play against the current solver-backed Hal:

```bash
uv run python -m stl.cli command=play 'command.args=["--use-tier-a"]'
```

Useful variants:

```bash
# Faster, weaker
uv run python -m stl.cli command=play 'command.args=["--iterations","50","--use-tier-a"]'

# Slower, stronger
uv run python -m stl.cli command=play 'command.args=["--iterations","800","--use-tier-a"]'

# Reproducible sampling
uv run python -m stl.cli command=play 'command.args=["--use-tier-a","--seed","123"]'
```

By default the CLI uses `stl.play.agent.SolverAgent`: value/policy checkpoint
plus matrix-game MCTS. `--use-tier-a` enables certified Tier A tablebase lookup
where available. If `--seed` is omitted, the CLI picks a fresh random seed so
human games are not replaying the same sampled line.

## Hydra Commands

Hydra is the public command surface:

```bash
uv run python -m stl.cli --help
uv run python -m stl.cli command=tier_a --cfg job
uv run python -m stl.cli command=self_play --cfg job
```

Use `--cfg job` to inspect a config without starting a long run.

Common commands:

```bash
# Focused engine/solver checks
uv run pytest tests/engine -q
uv run pytest tests/solver -q

# Build a small Tier A smoke slice
uv run python -m stl.cli command=tier_a command.stage=1 command.limit=1 command.workers=1

# AlphaZero-style bridge commands
uv run python -m stl.cli command=targets experiment=smoke
uv run python -m stl.cli command=train experiment=alphazero_smoke
uv run python -m stl.cli command=self_play experiment=alphazero_smoke
uv run python -m stl.cli command=eval
uv run python -m stl.cli command=promote
```

## Solver Shape

`stl/solver/` is intentionally compact:

```text
stl/solver/
  exact.py       # public state, LP minimax, half-round helpers, timing, diagnostics
  search.py      # leaf evaluators, selective solve, subgame resolve, matrix-game MCTS
  tablebase.py   # tactical fixtures, pinned registry, backward map, Tier A lookup/build
```

The exact tower solves finite-horizon stochastic zero-sum public states by
building exact-second payoff matrices and solving minimax LPs. A bounded CFR+
matrix solver is available for critical runtime resolves, but certified exact
labels and audits keep using the LP path. Chance branches come from the engine.
The solver core must not import reward shaping, observation features,
route-stage labels, or bucketed abstractions.

The search tower adds selective candidate sets, matrix-game MCTS, optional
learned leaf evaluation, bounded critical-state subgame resolve, and Tier A
lookup. Critical resolve defaults to the current half-round and uses the
leaf evaluator at the boundary. The live playable agent uses a broader
candidate grid at the root than the tight selective solver so human-facing play
is less pattern-readable, while training/exact audits keep the smaller candidate
set by default.

## Learning And Self-Play

`stl/learning/` contains the exact-to-self-play bridge:

```text
targets.py      # feature extraction, exact labels, corpus IO, bootstrap target sources
model.py        # value/policy net and checkpoint-compatible model pieces
train.py        # supervised value/policy training and checkpoint loading
gates.py        # calibration, audit, ladder, strength and promotion gate surface
support/        # route features, reanalysis, bootstrap, self-play, tournaments, strength internals
```

The accepted promotion path is:

```text
exact/tablebase anchors -> value/policy training -> MCTS/self-play targets
-> calibration/strength/exploitability gates -> promotion
```

A checkpoint should not be promoted just because held-out MSE improves.
Promotion evidence should include calibration, deterministic ladder results,
exploitability or best-response evidence where available, and policy drift
checks when the policy head is involved.

## Architecture Rules

- Engine truth belongs in `stl.engine`.
- `stl.engine` must not depend on solver, learning, reward shaping, or play.
- Exact/search solver work belongs in `stl.solver`.
- `stl.solver` must preserve exact integer-second action spaces, including the
  Baku-only 61-second leap-window drop.
- `stl.solver` must not import reward shaping, observations, route-stage labels,
  or bucketed abstractions.
- Learning and self-play bridge code belongs in `stl.learning`.
- Playable agents, scripted opponents, environment wrappers, and terminal UI
  belong in `stl.play`.
- New runtime code does not belong in `legacy/`.

## Documentation

- `docs/whitepaper/stl_solver_whitepaper.tex`
- `docs/whitepaper/stl_solver_whitepaper.pdf`
- `docs/papers/`

The whitepaper is the current long-form architecture and rigor statement. The
papers directory holds reference PDFs and a literature assessment.
