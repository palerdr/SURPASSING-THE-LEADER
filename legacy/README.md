# `legacy/`

There is no second source tree. The obsolete `legacy/old_layout/` mirror was
deleted during the Python distillation audit after commit `a4b03db` was pushed.
Use that commit for exact historical contents.

Old-to-live ownership is:

| Old path | Live owner |
|---|---|
| `src/` | `stl/engine/` |
| `environment/cfr/` | `stl/solver/` |
| `training/` | `stl/learning/` |
| `hal/` and non-rigorous environment code | `stl/play/` |
| `scripts/` | `stl/commands/` plus Hydra `configs/` |

Do not restore runtime code under `legacy/`. Recover a historical file from
Git, then deliberately migrate only the still-needed behavior to its live
owner with tests.
