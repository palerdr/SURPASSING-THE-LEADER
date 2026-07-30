# Pure DTH Project Instructions

This subtree owns pure Drop the Handkerchief with literal actions 1..60,
inclusive ST, and no leap-second or STL-only route/information mechanics.

- `solver.py` is the transition and finite-horizon authority.
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

Read `docs/GAME_AND_SOLVER.md` for the model and `docs/EXACTNESS_PROOF.md` for
the proof boundary. `docs/AGENT_GOAL.md` owns the sub-project's end goal,
gates, milestone order, and any open decision; `docs/PROMOTION_CAMPAIGN.md`
owns the measured history behind it and is reference, not instruction. The
packed backup tablebase's Rust kernel must satisfy
`docs/DTH_BACKUP_PARITY.md` before it can become a default backend. Run
`uv run python -m pytest src/dth/tests -q`.
