# `tests/`

Regression tests for the compact `stl/` package.

Subdirectories:

- `engine/` - game clock, half-round resolution, death/revival, leap behavior.
- `solver/` - exact minimax, selective search, MCTS, frozen conformance packs, tablebase, firewall.
- `learning/` - episode contracts, replay V3, reachable targets, value/policy training, calibration, gates.
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

P1/P3 certification checks live in `test_solver_conformance.py` and
`test_mcts_conformance.py`. Replay/checkpoint boundary checks live in
`test_replay_v2.py` (retained filename, current V3 contract), while
`test_gen0_corpus.py` exercises the structural
Generation-Zero shard-to-checkpoint path.
