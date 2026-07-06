# `legacy/`

This is not a second source tree. It only holds the obsolete pre-consolidation layout under `old_layout/` for audit/reference.

Active AlphaZero/RL/game functionality lives in `stl/`:

- engine truth in `stl.engine`
- exact/search/tablebase work in `stl.solver`
- target generation, model training, self-play support, and gates in `stl.learning`
- playable Hal, solver agents, opponents, and the CLI in `stl.play`
- Hydra commands in `stl.commands`

Do not add new runtime code here. If an old file becomes useful again, promote it into the appropriate `stl/` package.
