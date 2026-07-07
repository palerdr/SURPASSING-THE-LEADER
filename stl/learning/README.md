# `stl.learning/`

Compact learning kernel for the exact-to-self-play bridge.

Live files:

- `targets.py` - feature extraction, exact label generation, corpus IO, bootstrap/reanalysis target sources.
- `model.py` - value/policy network, feature vector contract, checkpoint-compatible model pieces.
- `train.py` - supervised value/policy training, checkpoint loading, prediction helpers.
- `gates.py` - promotion-facing gate surface: calibration, audit, ladder, strength checks.
- `support/` - core RL/self-play helpers: route features, calibration, reanalysis, bootstrap generation, tournament play, teacher demos, and strength internals.

Support modules remain importable as `stl.learning.<name>` through the package search path, but the files live under `stl/learning/support/` so the top-level learning kernels stay readable. Generated corpora, checkpoints, and Hydra outputs stay out of source control.

Policy targets and model heads use dense length-62 vectors. Index `0` is illegal padding; index `s` is second `s`. Old length-61 corpora and 122-logit checkpoints were produced under stale action semantics and must be regenerated rather than migrated.
