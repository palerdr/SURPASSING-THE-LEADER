# Pure DTH Project Instructions

This subtree owns pure Drop the Handkerchief with literal actions 1..60,
inclusive ST, and no leap-second or STL-only route/information mechanics.

- `solver.py` is the transition and finite-horizon authority.
- `cfr.py`, `mcts.py`, data generation, and learning depend only on DTH code.
- All configs and artifacts remain under `dth/`.
- Never import `stl` or `toy`.
- Schema mismatches fail closed; do not load legacy `pure-*` artifacts.

Read `docs/GAME_AND_SOLVER.md` for the model and `docs/EXACTNESS_PROOF.md` for
the proof boundary. Run `uv run python -m pytest dth/tests -q`.

