# `environment/`

Glue around the engine for experiments, learning, route analysis, and solver
interfaces. Engine truth stays in `src/`; exact unshaped solving stays in
`environment/cfr/`.

Important areas:

- `cfr/` - rigorous exact/search solver core: exact-second minimax, selective
  search, matrix-game MCTS, tablebase, leaf evaluators, and subgame resolve.
- `dth_env.py` - Gymnasium-style wrapper for agent training/evaluation.
- `legal_actions.py` - action legality for Hal/Baku, role, and leap window.
- `observation.py` - feature encoding for learning agents.
- `route_math.py`, `route_stages.py`, `routing_features.py`,
  `strategy_features.py` - LSR, round-start deviation, pressure, and safe-budget
  features.
- `scenarios.py` - named game starts.
- `opponents/` - scripted teachers, baseline opponents, and ladder factories.
- `abstractions/` - legacy/diagnostic abstractions kept outside the rigorous CFR
  firewall.

The top-level package may contain learning features and route labels; do not
move those concerns into the core engine.
