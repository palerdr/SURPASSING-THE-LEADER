# Surpassing The Leader

Compact Python research implementation of the *Usogui* Drop the Handkerchief
game as a fully observed, two-player, simultaneous-move stochastic zero-sum
game. The project is organized around one current path:

```text
engine rules -> exact/search anchors -> value/policy learning -> gated self-play
```

The goal is not to preserve every historical script. The live tree keeps the
engine, exact solver, playable agents, training bridge, and Hydra command
surface in `stl/`. The disconnected pre-consolidation mirror was removed after
the pushed `a4b03db` archival baseline; `legacy/README.md` maps old paths to
their live owners.

## Current Layout

| Path | Role |
|---|---|
| `stl/engine/` | Rule authority: constants, players/referee, game clock, half-round resolution, death/revival, leap-second legality. |
| `stl/solver/` | Exact/search/tablebase core plus conformance and MCTS audit packs. |
| `stl/learning/` | Target generation, value/policy model, training, gates, and RL/self-play support. |
| `stl/play/` | Actual playable surface: `SolverAgent`, `CanonicalHal`, opponents, teachers, environment wrapper, terminal CLI. |
| `stl/commands/` | Importable command implementations used by Hydra. |
| `configs/` | Hydra command and experiment configs. |
| `tests/{engine,solver,learning,play}/` | Regression tests grouped by subsystem. |
| `docs/whitepaper/` | Current solver whitepaper source/PDF. |
| `docs/papers/` | Reference papers. |
| `legacy/README.md` | Archive commit and old-to-live path map; no runtime code. |

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
`docs/REGEN2RL.md` for the audited Python-first plan from exact regeneration
through genuine simultaneous-move MCTS self-play and gated promotion.

## Setup

Use Python 3.12+ with `uv`:

```bash
uv sync --dev
uv run pytest --collect-only -q
uv run pytest -q
```

The latest verified full suite after executing REGEN2RL through P3, the P4
pipeline smoke, and distillation D1 was:

```text
719 passed, 4 skipped
```

## Playing A Match

The checked-in headline checkpoint predates the V2 feature/action contract and
is intentionally rejected by the strict loader. Until P4 produces a reviewed
V2 checkpoint, the immediately runnable local match uses the quarantined legacy
agent:

```bash
uv run python -m stl.cli command=play 'command.args=["--agent","legacy"]'
```

After a V2 checkpoint exists, run the solver-backed Hal explicitly:

```bash
uv run python -m stl.cli command=play 'command.args=["--checkpoint","outputs/regen2rl/gen0_checkpoint_v2/best.pt","--use-tier-a","--seed","123"]'
```

The default `solver` agent fails loudly on a missing or incompatible checkpoint;
it never silently falls back to handcrafted evaluation. `--use-tier-a` enables
certified Tier A tablebase lookup where available. If `--seed` is omitted, the
CLI picks a fresh random seed so human games do not replay the same sampled
line.

## Hydra Commands

Hydra is the public command surface:

```bash
uv run python -m stl.cli --help
uv run python -m stl.cli command=tier_a --cfg job
uv run python -m stl.cli command=gen0_targets --cfg job
uv run python -m stl.cli command=train_gen0 --cfg job
```

Use `--cfg job` to inspect a config without starting a long run.

Common commands:

```bash
# Focused engine/solver checks
uv run pytest tests/engine -q
uv run pytest tests/solver -q

# Build a small Tier A smoke slice
uv run python -m stl.cli command=tier_a command.stage=1 command.limit=1 command.workers=1

# Reconstructable Generation-Zero pipeline smoke (short)
uv run python -m stl.cli command=gen0_targets experiment=gen0_smoke
uv run python -m stl.cli command=train_gen0 experiment=gen0_train_smoke

# Inspect the first long generation and training jobs without running them
uv run python -m stl.cli command=gen0_targets --cfg job
uv run python -m stl.cli command=train_gen0 --cfg job

# Evaluation and promotion surfaces
uv run python -m stl.cli command=eval command.checkpoint=/path/to/candidate.pt
uv run python -m stl.cli command=promote \
  command.calibration_report=/path/to/calibration.json \
  command.ladder_report=/path/to/ladder.json \
  command.exploitability_report=/path/to/exploitability.json
```

The current unchecked REGEN2RL action is the full Generation-Zero anchor corpus:
`uv run python -m stl.cli command=gen0_targets`. It is intentionally a long
generation boundary; inspect its resolved config and output scope before
starting it, then review both manifests before `command=train_gen0`.

There is intentionally no public `command=self_play` before P5. The old broken
Hydra wrapper was removed; the tested compatibility module still runs the
legacy bucketed actor against scripted opponents. Phase P5 of
`docs/REGEN2RL.md` replaces that module with matrix-game MCTS self-play before
adding a command again.

## Solver Shape

`stl/solver/` is intentionally compact:

```text
stl/solver/
  exact.py       # public state, LP minimax, half-round helpers, timing, diagnostics
  search.py      # leaf evaluators, selective solve, subgame resolve, matrix-game MCTS
  tablebase.py   # tactical fixtures, pinned registry, backward map, Tier A lookup/build
  conformance.py # exact/search conformance audit helpers
  mcts_conformance.py # frozen simultaneous-MCTS convergence pack
```

The exact tower solves finite-horizon stochastic zero-sum public states by
building exact-second payoff matrices and solving minimax LPs. A bounded CFR+
matrix solver is available for critical runtime resolves, but certified exact
labels and audits keep using the LP path. Chance branches come from the engine.
The solver core must not import reward shaping, observation features,
route-stage labels, or bucketed abstractions.

The search tower adds selective candidate sets, matrix-game MCTS, optional
learned leaf evaluation, bounded critical-state subgame resolve, and Tier A
lookup. MCTS exposes candidate, playable-candidate, and full-width modes. Its
canonical improvement object is a pair of legal role marginals formed from a
linearly weighted average of empirical mean-Q equilibria; optimistic policies
select exploratory samples but are not mislabeled as training targets. Root
priors and optional Dirichlet noise remain role-factorized. Critical resolve
defaults to the current half-round and uses the leaf evaluator at the boundary.

## Learning And Self-Play

`stl/learning/` contains the pieces of the exact-to-self-play bridge:

```text
contracts.py    # finite episode, role, config, and horizon-sensitivity contracts
replay.py       # reconstructable TrainingRecordV2 shards and grouped splits
targets.py      # feature extraction, exact labels, corpus IO, bootstrap target sources
model.py        # 52-field V2 feature encoder and value/two-role-policy network
train.py        # grouped supervised training and strict V2 checkpoint bundles
gates.py        # calibration, audit, ladder, strength and promotion gate surface
support/        # route features, reanalysis, bootstrap, self-play, tournaments, strength internals
```

V2 replay stores the exact public state, feature and action schema versions,
target semantics, legal masks, both role policies, truncation state, provenance
digests, and RNG seeds. Shards are pickle-free and hash-verified. The default
trainer accepts these manifests and writes restorable bundles containing model,
optimizer, scheduler, schemas, provenance, resolved config, history, and RNG
state. Bare legacy checkpoints and stale length-61/122-logit artifacts require
an explicit legacy path and are rejected by default.

The current supervised/expert-iteration promotion path is:

```text
exact/tablebase anchors -> value/policy training -> MCTS/self-play targets
-> calibration/strength/exploitability gates -> promotion
```

A checkpoint should not be promoted just because held-out MSE improves.
Promotion evidence should include calibration, deterministic ladder results,
exploitability or best-response evidence where available, and policy drift
checks when the policy head is involved.

This is not yet a closed AlphaZero reinforcement-learning loop. P0--P3 of the
audited bridge are implemented, and the short P4 Generation-Zero corpus/trainer
smoke passes. Full anchor generation is in progress; the first full training
run has not started. The compatibility self-play module still runs legacy
`CanonicalHal` against scripted opponents; genuine two-role MCTS self-play
begins in P5. The phase gates and current stopping point are in
`docs/REGEN2RL.md`.

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
- `docs/DISTILLATION_AUDIT.md`
- `docs/papers/`

The whitepaper is the current long-form architecture and rigor statement. The
papers directory holds reference PDFs and a literature assessment.
