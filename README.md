# Surpassing The Leader

Compact research implementation of the *Usogui* Drop the Handkerchief game.

The live project is organized around one path: exact engine rules -> exact/search
anchors -> value/policy learning -> gated self-play style improvement. Older
bucketed abstractions, one-off probes, and historical script layouts are kept
under `legacy/` for reference.

## Repository Map

| Path | Role |
|---|---|
| `stl/engine/` | Engine truth: constants, player/referee state, game clock, death/revival, leap-second rules, legal actions. |
| `stl/solver/` | Exact minimax LP solver, selective search, matrix-game MCTS, pinned scenarios, Tier A tablebase helpers. |
| `stl/learning/` | Exact target generation, value/policy net, training, RL/self-play support, calibration, strength and promotion gates. |
| `stl/play/` | Actual playable game surface: SolverAgent, CanonicalHal, scripted/model opponents, teachers, environment wrapper, terminal play. |
| `stl/commands/` | Importable command implementations used by the Hydra dispatcher. |
| `configs/` | Hydra command and experiment configs. |
| `tests/{engine,solver,learning,play}/` | Regression tests grouped by subsystem. |
| `legacy/old_layout/` | Obsolete pre-consolidation package layout kept for reference only. |

Generated corpora, checkpoints, logs, demos, `outputs/`, `multirun/`, and
`target/` should stay out of source review unless a specific artifact is the
subject of the task.

## Setup

Use Python 3.12+ with `uv`:

```bash
uv sync --dev
uv run pytest --collect-only -q
uv run pytest -q
```

Hydra is the public command surface:

```bash
uv run python -m stl.cli --help
uv run python -m stl.cli command=tier_a --cfg job
uv run python -m stl.cli command=self_play --cfg job
```

Use `--cfg job` to inspect a config without starting a long run.

## Common Commands

Smoke exact engine and solver:

```bash
uv run pytest tests/engine -q
uv run pytest tests/solver/test_cfr_firewall.py tests/solver/test_cfr_rigorous_exact.py -q
```

Tier A tablebase config:

```bash
uv run python -m stl.cli command=tier_a command.stage=1 command.limit=1 command.workers=1
```

AlphaZero-style bridge configs:

```bash
uv run python -m stl.cli command=targets experiment=smoke
uv run python -m stl.cli command=train experiment=alphazero_smoke
uv run python -m stl.cli command=self_play experiment=alphazero_smoke
uv run python -m stl.cli command=eval
uv run python -m stl.cli command=promote
uv run python -m stl.cli command=play
```

## Architecture Rules

- Engine truth belongs in `stl.engine`; it must not depend on solver, learning,
  reward shaping, or playable adapters.
- Exact/search solver work belongs in `stl.solver`; it must preserve exact
  integer-second action spaces, including the 61-second leap-window case.
- `stl.solver` must not import reward shaping, observation features, route-stage
  labels, or bucketed abstractions.
- Learning and self-play bridge code belongs in `stl.learning`; it may depend on
  engine, solver, and play adapters.
- Playable agents, scripted opponents, and terminal UI belong in `stl.play`.
- Historical bucketed/Gym/SB3 work belongs under `legacy/`, not the live solver
  core.

## Solver Shape

The exact tower solves finite-horizon stochastic zero-sum public states by
building exact-second payoff matrices and solving minimax LPs. Chance branches
come from the engine, and the solver firewall tests prevent shaping or bucketed
abstractions from entering the rigorous core.

The approximation tower uses a compact value/policy network as an accelerator:
exact labels and tablebase anchors train the model, MCTS can use it as a leaf
evaluator, and promotion requires calibration, deterministic ladder, and
certified exploitability evidence. This keeps the door open for true
AlphaZero-style self-play without making learned policy the source of engine or
solver truth.
