# Pure DTH Project Instructions

This subtree owns pure Drop the Handkerchief with literal actions 1..60,
inclusive ST, and no leap-second or STL-only route/information mechanics.

- `solver.py` is the pure-DTH transition and Bellman authority.
- `complete_tablebase.py` owns the completed 289,374,121-class exact sweep.
- `agent.py` is the play-time exact facade and has no approximate fallback.
- `cfr.py`, `mcts.py`, data generation, and learning depend only on DTH code.
- All configs and artifacts remain under `src/dth/`.
- Never import `stl` or `abstract`.
- Schema mismatches fail closed; do not load legacy `pure-*` artifacts.

## Workflow and module hygiene

- Do not add top-level `dth/` modules for one-off iteration scripts or
  experiment-specific fixtures.
- Reproducible experiment workflows must be parameterized through the existing
  Hydra entry points and versioned configs; prefer extending those workflows
  over adding a new script.
- Keep transient runners, reports, checkpoints, and fixtures in their existing
  generated/ignored locations or in focused tests. Add a new module only when
  it has a durable role in the DTH training, evaluation, solver, or artifact
  contract.

Read `docs/GAME_AND_SOLVER.md` for the model, `docs/EXACTNESS_PROOF.md` for
the completed proof, and `docs/AGENT_GOAL.md` for the exact play contract.
Dataset, training, self-play, and MCTS workflows remain optional research
surfaces downstream of the solved game. The packed tablebase's Rust kernel must satisfy
`docs/DTH_COMPLETE_PARITY.md` before it can become a default backend. Run
`uv run python -m pytest src/dth/tests -q`.
