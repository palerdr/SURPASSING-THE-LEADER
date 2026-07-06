# `stl.solver/`

Exact and search solver core.

Live kernel files:

- `exact.py` - exact public state, snapshots, minimax LP, bounded CFR+ matrix solve, half-round matrix helpers, timing predicates, diagnostics, expansion, finite-horizon solve.
- `search.py` - leaf evaluators, selective solve, bounded critical-state subgame resolve, matrix-game MCTS.
- `tablebase.py` - tactical scenarios, pinned registry, analytic backward map, certified Tier A generation/runtime lookup.

No bucketed abstractions, reward shaping, or value-net heuristics belong in the exact solver core. Learned evaluators are only leaf-evaluator adapters around exact/search machinery. Certified exact labels and audits use the LP solver; CFR+ is for bounded local resolves anchored by a frontier evaluator.
