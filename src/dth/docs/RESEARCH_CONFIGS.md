# DTH research configuration catalogue

Pure DTH is solved by `complete_full_v1`; none of the configurations catalogued
here changes that value authority. Dataset, training, self-play, CFR, and MCTS
presets are reproducible research inputs downstream of the completed tablebase.
Tracked configuration names do not provide compatibility with old generated
artifacts: current schema and provenance checks still apply, and mismatches fail
closed.

## Production authority

- `complete_full_v1.yaml` builds or verifies the canonical dense quotient
  tablebase used by `CompleteDTHAgent`.

## Current command defaults

- `dataset.yaml` generates exact or explicitly labelled research targets.
- `train.yaml` trains policy/value approximators against declared targets.
- `self_play.yaml` runs the retained empirical self-play workflow.
- `mcts_audit.yaml` evaluates finite-budget MCTS against exact matrices.

These four defaults are the supported entry points exposed by `python -m dth`.

## Reproducible research families

- `strategic_exact_*`, `boundary_tablebase_*`, `anchor_leaf_closure_*`,
  `capacity_rows_*`, `class_head_*_corpus`, `resolve_labeled_*`, and
  `paired_boundary_orientation_*` preserve target-generation experiments.
- `train_*` presets preserve model, loss, coverage, widening, composition, and
  policy-generalization comparisons. They produce approximators, never exact
  DTH values.
- `boundary_self_play_*` and `self_play_readiness_*` preserve empirical
  self-play and readiness comparisons.
- `mcts_*` and `readiness_*` preserve finite-budget search audits, depth gates,
  and development/reference comparisons.
- `boundary_lift_*_artifacts` records packaging parameters for a research
  lineage; it is not a production tablebase or compatibility loader.

Versioned suffixes record experiment lineage so a result can name the preset
that produced it. They are not a promotion ladder, and a higher suffix is not
implicitly preferred. Research reports must identify the exact configuration,
input digests, target semantics, and measured saddle gaps.
