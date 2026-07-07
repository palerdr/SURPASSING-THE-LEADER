# `tests/`

Regression tests for the compact `stl/` package.

Subdirectories:

- `engine/` - game clock, half-round resolution, death/revival, leap behavior.
- `solver/` - exact minimax, selective search, MCTS, tablebase, firewall.
- `learning/` - targets, value/policy net training, calibration, gates.
- `play/` - playable adapters, teachers, self-play, route/play glue.

Useful focused checks:

```bash
uv run pytest tests/engine -q
uv run pytest tests/solver/test_cfr_firewall.py tests/solver/test_cfr_rigorous_exact.py -q
uv run pytest tests/learning -q
uv run pytest tests/play -q
```

`uv run pytest --collect-only -q` is a cheap sanity check after structural
changes.

Action-core tests should cover legal seconds `1`, `60`, Baku-only leap `61`,
same-second success with `ST=0`, strict fail triangle `check < drop`, dense
length-62 policy vectors, and the always-illegal padding index `0`.
