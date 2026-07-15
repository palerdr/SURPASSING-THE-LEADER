# `stl.engine/`

Core engine truth: constants, players, referee CPR/survival logic, game clock, half-round resolution, death/revival, leap-second behavior, and action legality. This package must stay free of solver, training, reward shaping, and learned-model dependencies.

Action legality is centralized in `stl.engine.actions`: normal seconds are `1..60`, only Baku as leap-window dropper may use `61`, checkers are capped at `60`, and dense action index `0` is illegal padding. A same-second check succeeds with `ST=0`.
