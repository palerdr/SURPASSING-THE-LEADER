# `stl.solver/`

Exact and search solver core.

Live kernel files:

- `exact.py` - exact public state, snapshots, minimax LP, expansion, finite-horizon solve.
- `search.py` - compact search facade for MCTS, evaluator glue, selective solve, subgame resolve, diagnostics.
- `tablebase.py` - pinned scenarios and Tier A lookup/build primitives.
- `mcts.py`, `evaluator.py`, `selective.py`, `subgame_resolve.py` - search and leaf-evaluation machinery.
- `backward.py`, `epoch_sweep.py`, `epoch_tablebase.py`, `tier_a.py` - certified Tier A tablebase generation/runtime lookup.

No bucketed abstractions, reward shaping, or value-net heuristics belong in the exact solver core. Learned evaluators are only leaf-evaluator adapters around exact/search machinery.
