# `stl.play/`

Compact playable surface.

- `cli.py` - terminal play implementation.
- `agent.py` - solver/value-net-backed playable agent (`SolverAgent`, Hal/Baku wrappers).
- `canonical_hal.py` - canonical Hal adapter for the older bucketed playable baseline.
- `env.py` - Gym-style environment wrapper used by RL/self-play tests.
- `action_model.py`, `search.py`, `state.py` - bucket baseline mechanics for playable Hal.
- `opponents/` - scripted opponents, teacher policies, pattern reader, and weighted scripted leagues.

This layer may use the engine, solver, and learning model. Engine and rigorous solver code must not depend on play code.

The Gym-style wrapper exposes `Discrete(62)` so action index equals literal second; action `0` is illegal padding. The bucket baseline files are legacy playable references only and must not define solver-core legality.
