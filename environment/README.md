# `environment/`

Glue around the engine for experiments, learning, and route analysis.

Important areas:

- `dth_env.py` - Gymnasium-style wrapper for agent training/evaluation.
- `legal_actions.py` - action legality for Hal/Baku, role, and leap window.
- `observation.py` - feature encoding for learning agents.
- `route_math.py`, `route_stages.py`, `routing_features.py`,
  `strategy_features.py` - LSR, round-start deviation, pressure, and safe-budget
  features.
- `scenarios.py` - named game starts.
- `opponents/` - scripted teachers and baseline opponents.
- `abstractions/` - older bucketed/abstract game solvers.
- `cfr/` - rigorous exact/search solver core.

Keep engine truth in `src/`. Keep exact unshaped solving in `environment/cfr/`.
This top-level package may contain learning features and route labels.
