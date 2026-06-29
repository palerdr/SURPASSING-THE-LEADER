# Environment And Solver Interfaces

`environment/` contains wrappers and strategy-facing helpers around the rules
engine.

- `cfr/`: exact-second minimax, selective search, matrix-game MCTS, tablebase,
  and subgame resolve.
- `opponents/`: scripted opponents and ladder factories.
- `abstractions/`: legacy/diagnostic abstractions, kept outside the rigorous CFR
  firewall.
- top-level modules: Gymnasium environment, legal actions, observations, route
  features, and reward utilities.
