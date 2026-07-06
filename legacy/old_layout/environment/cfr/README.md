# `environment/cfr/`

Rigorous exact/search solver core.

Despite the name, this directory is mainly exact-second minimax and matrix-game
MCTS, not full information-set CFR. The important contract is:

- exact integer seconds, including the 61-second Baku drop case in the leap
  window;
- terminal utility from Hal's perspective;
- chance branches derived from the engine;
- no reward shaping, observation features, or bucketed abstractions.

Start with:

- `exact.py` - exact finite-horizon minimax LP solver.
- `backward.py` - analytic post-leap transition map used by tablebase epochs.
- `mcts.py` - matrix-game MCTS over exact action sets.
- `evaluator.py` - terminal/tablebase/value-net leaf evaluators.
- `tablebase.py` and `tactical_scenarios.py` - pinned scenarios and fixtures.

See `MODULES.md` for the detailed module index and firewall rules.
