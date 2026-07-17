# `stl.engine/`

Core engine truth: constants, players, referee CPR/survival logic, game clock, half-round resolution, death/revival, leap-second behavior, and action legality. This package must stay free of solver, training, reward shaping, and learned-model dependencies.

The five-minute boundaries are intentionally asymmetric: current dose `>= 300` is fatal, while resulting cumulative TTD is fatal only when `prior_ttd + dose > 300`. Exactly 300 cumulative TTD remains revival-eligible. [`docs/CANONICAL_EXTENSIVE_FORM.md`](../../docs/CANONICAL_EXTENSIVE_FORM.md) freezes the source evidence and repository-wide semantics.

Action legality is centralized in `stl.engine.actions`: normal seconds are `1..60`, only Baku as leap-window dropper may use `61`, checkers are capped at `60`, and dense action index `0` is illegal padding. Action `1` is the immediate first-second action; there is no pass action. A same-second check succeeds with inclusive `ST=1`, using `ST = check - drop + 1`. [`docs/ACTION_TIMING.md`](../../docs/ACTION_TIMING.md) is the repository-wide documentary authority.
