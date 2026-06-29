# `scripts/`

Command-line entry points. Prefer adding new long-run, evaluation, or reporting
workflows here rather than in the repository root.

Current primary entry point:

- `run_tier_a.py` - builds the Tier A certified tablebase epochs. This is where
  current work resumes.

Core exact/training pipeline:

- `run_tablebase_pilot.py` - validates tablebase cost/invariants.
- `run_phase_g_corpus.py` - generates exact-LP value targets.
- `run_phase_h_regen.py`, `run_phase_h_reanalysis.py` - later target rescue and
  reanalysis path.
- `run_ceiling_holdout.py`, `make_clean_holdout.py` - held-out ruler creation.
- `run_gen_iteration.py` - one value-net generation plus calibration gate.
- `run_ceiling.py` - orchestrated full pipeline.
- `run_corpus_validation.py`, `run_strength_gate.py`, `run_exploitability.py` -
  validation/evaluation gates.

Playable and diagnostic tools:

- `play_cli.py` - terminal game.
- `play_vs_cfr.py` - older terminal game against a precomputed abstract solver.
- `evaluate.py`, `evaluate_teachers.py`, `eval_canonical_hal.py` - opponent
  evaluation.
- `benchmark_mcts_convergence.py`, `plot_survival.py`, `sweep_physicality.py` -
  diagnostics.

Directory conventions:

- `run_*` - generation, training, and decision events.
- `compare_*` - checkpoint/runtime comparisons.
- `train_*` - direct training helpers.
- `probes/` - one-off exploratory probes kept for historical context.
- `archive/` - older patch-validation scripts that are no longer primary entry
  points.
