# `training/`

Training, calibration, audit, and tablebase support.

Main files:

- `value_targets.py` - generate/load/save exact-labeled target records.
- `train_value_net.py` - train the small Hal value/policy network.
- `bootstrap_loop.py` - generation loop and calibration gate.
- `calibration.py` - held-out evaluation metrics.
- `reanalysis.py` - re-solve ambiguous/high-variance targets.
- `audit_pack.py` - pinned audit scenario runner.
- `tournament.py` - scripted/model match evaluation.
- `curriculum.py` and `teacher_demos.py` - curriculum starts and teacher data.
- `tablebase/` - certified epoch solver used by `scripts/run_tier_a.py`.
- `strength/` - match-gate and best-response strength checks.

Generated corpora and checkpoints should go under `checkpoints/` or `models/`,
not inside this source directory.

This package may depend on `src/`, `environment/`, and `hal/`, but runtime engine
code should not import it.
