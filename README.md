# Surpassing The Leader

This is a Python research/playground implementation of the *Usogui* "Drop the
Handkerchief" death game. The long-term vision is still a faithful playable duel
where the human plays Baku against a layered Hal, but the codebase currently
provides something more specific:

- a deterministic rules engine for the game clock, cylinders, death, revival,
  and leap-second timing;
- exact-second minimax/search tools for proving local positions;
- generated tablebases and value targets used to train/evaluate a small Hal
  value model;
- a CLI/playable baseline and scripted teachers for experiments.

The active continuation point is `scripts/run_tier_a.py`.

## Current Direction

The project started as a rules simulator, then grew into scripted opponents,
route-math tests, self-play experiments, exact minimax solvers, value-net
training, and finally tablebase construction. That history left several valid
but overlapping paths in the repo.

The direction now is:

1. Keep the engine exact and testable.
2. Use classical exact methods first: minimax LPs, backward induction,
   tablebases, and pinned scenario audits.
3. Use learned models only as approximation tools around those exact anchors.
4. Build Hal upward from inspectable strategy layers, not from pure end-to-end
   RL alone.

So the answer to "is this RL?" is: not primarily. RL/self-play exists in the
repo, but the strongest current path is classical exact solving plus tablebase
generation, with neural value models or MCTS used as accelerators where a full
exact solve is too expensive.

## Repository Map

| Path | Role |
|---|---|
| `src/` | Core game engine: players, referee, constants, half-round resolution, clock. |
| `environment/cfr/` | Exact/search solver core: minimax LPs, matrix-game MCTS, tablebase scenarios. |
| `training/tablebase/` | Certified interval tablebase epochs used by `run_tier_a.py`. |
| `training/` | Target generation, value-net training, calibration, audits, tournaments. |
| `hal/` | Playable Hal wrappers, value net, search adapter, legacy CanonicalHal support. |
| `environment/` | Gym wrapper, route features, legal actions, scripted opponents, abstractions. |
| `scripts/` | Human entry points. Start here for runs. |
| `tests/` | Regression suite for engine, solver, route math, AI wrappers, and training glue. |
| `docs/` | Canon sources, architecture notes, solver charter, reports, whitepaper. |
| `checkpoints/`, `models/`, `logs/`, `demos/` | Generated artifacts. Large and mostly ignored. |

## Continue From Tier A

Use Python 3.12 if available, then install dependencies:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On this machine `python` may not exist; use `python3` or `.venv/bin/python`.

Smoke the Tier A path without starting the long run:

```bash
python3 scripts/run_tier_a.py --help
python3 scripts/run_tier_a.py --stage 1 --limit 1 --workers 1
```

The real run is long and writes to `checkpoints/tablebase/tier_a/`:

```bash
python3 scripts/run_tier_a.py --workers 20
```

`run_tier_a.py` builds:

- 480 one-death epochs: `d1_{hal|baku}_{60..299}.npz`;
- one zero-death opening epoch: `d0.npz`;
- `manifest.json` with hashes.

## Development Narrative

The project has moved through these phases:

1. **Rules engine.** The initial value was making the gamble executable:
   check/drop resolution, Squandered Time, cylinder overflow, death/revival,
   referee fatigue, and leap-second clock behavior.
2. **Route and character modeling.** The next layer added LSR/deviation math,
   scripted Baku/Hal teachers, and tests for canonical route pressure. This
   made the project more than a generic timing game.
3. **Approximate playable Hal.** `hal/` and the early self-play scripts added a
   playable/searching Hal with a small value net. Useful, but not enough as the
   foundation for "Perfect Hal."
4. **Exact solver tower.** `environment/cfr/` became the rigorous path:
   exact-second joint actions, minimax LPs, engine-derived chance branches, and
   tests that prevent bucketed or shaped logic from leaking into exact solving.
5. **Training and gates.** `training/` generated exact labels, trained small
   value models, and enforced calibration gates against held-out/pinned states.
6. **Tablebase push.** `training/tablebase/` and `scripts/run_tier_a.py` are the
   current next step: build certified post-leap epoch tables so the solver has a
   stronger exact backbone.

Once Tier A exists, the project is closer to a real "Hal" because the AI can
reason from exact arithmetic instead of only local heuristics or self-play
imitation. The eventual playable experience should combine that exact backbone
with belief, perception, persona, and memory layers.

## What To Prefer

- Prefer `src/` for engine truth.
- Prefer `environment/cfr/` and `training/tablebase/` for exact solver work.
- Prefer `training/` for generated labels, gates, and audit logic.
- Treat `hal/` self-play and older scripts as useful experiments, not the core
  proof path.
- Keep generated artifacts out of source review unless a specific artifact is
  the subject of the work.

## Quick Checks

```bash
python3 -m pytest tests/test_clock.py tests/test_death_mechanics.py tests/test_tablebase_epoch.py
python3 -m pytest --collect-only -q
```
