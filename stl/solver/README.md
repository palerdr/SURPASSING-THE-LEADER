# `stl.solver/`

Exact and search solver core.

Live kernel files:

- `exact.py` - exact public state, snapshots, minimax LP, bounded CFR+ matrix solve, half-round matrix helpers, timing predicates, diagnostics, expansion, finite-horizon solve.
- `search.py` - leaf evaluators, selective solve, bounded critical-state subgame resolve, matrix-game MCTS.
- `tablebase.py` - tactical scenarios, pinned registry, analytic backward map, certified Tier A generation/runtime lookup.
- `conformance.py` - reusable action, transition, role-orientation, solver-parity, candidate, and chance conformance reports.
- `mcts_conformance.py` - the frozen horizon-one MCTS convergence pack and report writer.

No bucketed abstractions, reward shaping, or value-net heuristics belong in the exact solver core. Learned evaluators are only leaf-evaluator adapters around exact/search machinery. Certified exact labels and audits use the LP solver; CFR+ is for bounded local resolves anchored by a frontier evaluator.

Solver payoff matrices use literal action seconds. Normal grids are `1..60` by `1..60`; the leap asymmetry is Baku-dropper-only `61` against checker cap `60`. Success is `check >= drop` with inclusive `ST = check - drop + 1`, including diagonal `ST=1`; failure is the strict triangle `check < drop` with the flat `+60` penalty. See [`docs/ACTION_TIMING.md`](../../docs/ACTION_TIMING.md).

MCTS supports `candidate`, `candidate_playable`, and `full_width` action modes.
Its canonical output is `improved_dropper_policy` plus
`improved_checker_policy`: legal role marginals formed by a linearly weighted
average of per-iteration empirical mean-Q equilibria. Optimism-augmented
strategies select exploratory cells only. The final mean-Q saddle point is
reported separately as a diagnostic. Priors and optional root Dirichlet noise
are factorized by role, and full-width mode propagates to descendants.
