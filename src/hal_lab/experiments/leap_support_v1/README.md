# Leap support prediction v1

This study asks whether a small network can name the exact equilibrium
support of a hard STL leap stage, 60 bits per player, from the stage's 61
payoffs, on keys that it never saw. It also asks whether training on easy
stages helps. It changes no solver, rule, or artifact. It ran on 2026-10-16
and failed its gate.

## Measured ground

Sources: `src/stl/outputs/leap-full-native/report.json` and the replay
figures in [`src/stl/README.md`](../../../stl/README.md).

| Quantity | Value |
| --- | ---: |
| Reachable state classes in the leap sweep | 9,453,333,117 |
| Residue classes, which need a packing LP | 609,698,652 (6.45 %) |
| Kink-seed hit rate on the residue | 70 % to 75 % |
| Sweep time that a perfect seed inside the kernel saves | about 130 s of a projected 5,200 s |

A perfect seed lowers the residue cost from 1.05 µs to about 0.84 µs per
class, so speed is not a goal, and the Rust kernel is unchanged.

## Data

| Item | Value |
| --- | --- |
| Keys | ten H keys with at least 1,000,000 residue classes each |
| Hidden keys, for evaluation alone | `H2_39`, `H1_44` |
| Training keys | four H1 and four H2 keys, drawn by the protocol seed `20261016` |
| Rows per key | 1,000,000 residue rows and 1,000,000 easy rows |
| Label | the Dropper and Checker supports of the packing LP basis, certified at 1e-6 |
| Exporter | [`src/stl/experiments/export_leap_residue.py`](../../../stl/experiments/export_leap_residue.py) |
| Export folder (ignored) | `src/stl/outputs/leap-residue/` |

The exporter reruns `sweep_key` on each key, aborts above a 1e-6 difference
from the artifact, and writes nothing to the artifact. The runner asserts that
no hidden-key row enters training, validation, or model selection.

## Runs

Each row scores the 1,000,000 exported residue rows of each hidden key. The
last two columns count the seeds that certify after the paired edge moves of
`leap_support.solve_supported`, on the first 3,000 rows. The pilots ran from a
scratch script that the repository does not keep, and no pilot is a record.

| Run | Status | Training rows | Budget | `H1_44` exact | `H2_39` exact | `H1_44` certifies after edge moves | `H2_39` certifies after edge moves |
| --- | --- | ---: | --- | ---: | ---: | ---: | ---: |
| Revision 1, Net A: 61 inputs, 64-64 | protocol run | 8,000,000 | 20,000 steps of 4,096; 26 s | 3.0 % | 0.05 % | not measured | not measured |
| Revision 1, Net B: residue plus easy | protocol run | 15,865,659 | 20,000 steps of 4,096; 28 s | 1.5 % | 0.01 % | not measured | not measured |
| Revision 2, Net A: 120 inputs, 256-256-256, 193,400 parameters | frozen record | 8,000,000 | 1,000,000 steps of 4,096; 1,916 s | 72.8 % | 20.8 % | 88 % | 43 % |
| Revision 2, Net B: residue plus easy | frozen record | 15,865,659 | 1,000,000 steps of 4,096; 2,046 s | 71.4 % | 15.8 % | not measured | not measured |
| xgboost trees, depth 8, 600 rounds | pilot | 1,000,000 | 3,624 s on 14 cores | 51.0 % | 14.4 % | 65 % | 29 % |
| Dilated 1-D convolution, width 64, 125,378 parameters | pilot | 2,000,000 | 60,000 steps of 2,048; 2,254 s | 53.7 % | 2.1 % | 71 % | 6 % |
| Perceptron plus recurrence features, 241 inputs | pilot | 2,000,000 | 300,000 steps of 4,096; 1,009 s | 70.8 % | 17.2 % | 87 % | 41 % |
| Transformer, bit-wise head, 104,962 parameters | pilot | 2,000,000 | 7,000 steps of 8,192; 3,646 s | 70.2 % | 6.9 % | 84 % | 20 % |
| Transformer, hole-boundary head, 106,896 parameters | pilot | 2,000,000 | 7,000 steps of 8,192; 3,497 s | 67.9 % | 10.1 % | 82 % | 23 % |

## Diagnosis

- Exact match of 120 coupled bits is all-or-nothing. A per-bit accuracy of
  97.6 % to 99.6 % leaves 27 % of `H1_44` rows and 79 % of `H2_39` rows
  wrong. The net reaches 39 % to 74 % on its own training keys.
- The misses are late, wide, and multiple holes. A one-hole row whose hole
  starts at second 30 or later misses 92 % on `H1_44` and 100 % on `H2_39`.
  Each extra hole multiplies the miss rate. On `H2_39`, 10 % of rows have a
  one-hole Dropper support.
- The misses are not label ambiguity. Each LP label of the 3,000-row samples
  certifies. Of the net's wrong supports, 0 of 2,386 on `H2_39` and 1 of 795
  on `H1_44` certify as given.
- The misses are not a gap in state-coordinate coverage. `H2_39` rows whose
  (ST/10, TTD) cell the training keys covered miss 81 %. Uncovered rows miss
  54 %.
- Certificate slack separates the misses on `H1_44`, where a miss is a
  near-tie: median slack 4.3e-4 for a miss against 1.5e-3 for a hit. The
  `H2_39` misses are not near-ties: 1.1e-3 against 1.8e-3.
- The kink claim of the first draft was wrong. Between 2 % and 10 % of
  residue holes start at the most negative success difference.

## Decision

| Item | Result |
| --- | --- |
| Gate: exact match on each hidden key at least 90 % | failed: 72.8 % on `H1_44`, 20.8 % on `H2_39` |
| Gates: at most 1,000,000 parameters and 2 hours of training | passed |
| Experiment 2: Net B minus Net A, exact match | −1.4 points on `H1_44`, −5.1 points on `H2_39` |
| Experiment 2 verdict under the 2-point margin | no measurable effect at this budget |

Revision 2 is frozen. Version 1 gets no further revision. A v2 would let the
network propose a support, let the certifier's edge moves place its hole
boundaries, and score the seed by certification after edge moves.

## Record and provenance

[`leap_support_v1_results.json`](leap_support_v1_results.json) is a frozen
record in `RECORDS` of [`src/hal_lab/provenance.py`](../../provenance.py). It
binds by SHA-256 the ten exports, both weights files, the ignored
`report.json`, the runner, and the exporter. `evidence/verification_baseline.json`
holds its 15 verified bindings. The protocol hash is unverifiable by design.
The weights and the report live under `src/hal_lab/outputs/leap_support_v1/`.
The docstrings of [`run.py`](run.py) and the exporter name `docs/PREDICTION.md`,
which this README replaces. The record binds the bytes of both files, so the
docstrings stay as written.

## Commands

The `export` phase needs `src/stl/outputs/leap-full-native/` and the Rust
extension. The `record` phase rewrites the frozen record; do not run it again.

```bash
uv run python -m hal_lab.experiments.leap_support_v1.run keys
uv run python -m hal_lab.experiments.leap_support_v1.run export
uv run python -m hal_lab.experiments.leap_support_v1.run study
uv run python -m hal_lab.experiments.leap_support_v1.run record
uv run python -m pytest src/hal_lab/tests/test_leap_support_v1.py -q
```

[`test_leap_support_v1.py`](../../tests/test_leap_support_v1.py) checks the
protocol invariants and runs a smoke study on fabricated rows.
[`HAL_RESEARCH.md`](../../docs/HAL_RESEARCH.md) summarizes this study.
