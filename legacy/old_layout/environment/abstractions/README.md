# `environment/abstractions/`

Legacy bucketed solver experiments.

These files predate the exact-second solver and are still useful for regression
tests and quick intuition:

- `game_state.py` - bucketed public state representation.
- `tree.py` - abstract successor generation and solve loop.
- `full_round.py` - older half/full-round abstractions.

Do not use this package as the foundation for new exact solver work. New exact
logic belongs in `environment/cfr/`; new engine logic belongs in `src/`.
