# `stl.learning/`

Compact learning kernel for the exact-to-self-play bridge.

Live files:

- `contracts.py` - versioned episode caps, outcome/perspective mapping, canonical config hashes, and horizon-sensitivity reports.
- `replay.py` - reconstructable `TrainingRecordV3` rows, pickle-free hash-verified shards, collision audits, and grouped splits.
- `reachable.py` - legal-engine Generation-Zero trajectories and pre-label episode isolation.
- `targets.py` - exact label generation, legacy corpus IO, and conversion into V3 replay records.
- `model.py` - value/two-role-policy network with an explicit 0/2/3-half-round horizon input.
- `bellman.py` - exact V3-root/V2-successor closure artifacts and Bellman consistency gates.
- `certificates.py` and `holdout.py` - exact policy evidence and one-use sealed ruler controls.
- `train.py` - grouped supervised value/policy training, strict V3 checkpoint bundles, and prediction helpers.
- `gates.py` - promotion-facing gate surface: calibration, audit, ladder, strength checks.
- `support/` - core RL/self-play helpers: route features, calibration, reanalysis, bootstrap generation, tournament play, teacher demos, and strength internals.

Support modules remain importable as `stl.learning.<name>` through the package search path, but the files live under `stl/learning/support/` so the top-level learning kernels stay readable. Generated corpora, checkpoints, and Hydra outputs stay out of source control.

Policy targets and model heads use dense length-62 vectors. Index `0` is illegal padding; index `s` is second `s`. Old length-61 corpora and 122-logit checkpoints were produced under stale action semantics and must be regenerated rather than migrated.

The active `stl.features.v2` input has 52 named fields: a lossless numeric
encoding of every value-relevant `ExactPublicState` field followed by documented
relative and safe-budget features. Replay rows also retain the serialized exact
state, so collision audits and reanalysis never depend on features alone.

Default training requires a validated V3 replay manifest and writes
`stl.checkpoint.v3` bundles containing model, optimizer, scheduler, horizon schema,
provenance, resolved config, training history, and RNG state. Bare state dicts
and manifestless corpora are legacy inputs and require explicit opt-in.

The short Generation-Zero smoke is:

```bash
uv run python -m stl.cli command=gen0_targets experiment=gen0_smoke
uv run python -m stl.cli command=train_gen0 experiment=gen0_train_smoke
```

The next long boundary is `command=gen0_bellman_cycle` with
`experiment=gen0_v5_bellman_full`; review all three split manifests and the
candidate-action representability report before merging or training.
