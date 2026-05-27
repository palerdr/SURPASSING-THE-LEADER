# Phase G + Phase I-1 validation result

First generation to pass the strict calibration gate (Phase 8 design) on
the held-out v2 ruler (Phase F-expanded). Documented here because the
`checkpoints/` directory is gitignored and this result is the milestone
the rest of the iteration sequence builds on.

## Phase G corpus (LP-labeled training data)

- Wall-clock: 113,394 s (31.5 h, 8 workers)
- Grid: 7168 candidates (14 baku × 8 hal × 8 clock × 2 half × 2 deaths × 2 cpr)
- Records emitted: **4,812** (vs prior 196 — 24.6× expansion)
- Pass rate: 67.1%
- Source breakdown:
  - exact_horizon_3: 2,938
  - exact_horizon_2: 1,871
  - tablebase: 3
- Memory: held flat at ~11 GiB used / ~265 MiB swap for the entire run
  thanks to the bounded LRU cache (`STL_SOLVE_CACHE_MAXSIZE=30000`).
  Previous unbounded runs OOM'd within 2-3 hours.

## Phase I-1 net architecture

- `hidden_dim=128` (vs default 64) → 35.5K params (vs 13.7K)
- Same trunk + heads structure; only the width changed.
- Training: 300 epochs, early-stopping patience 80, source weights
  (terminal 30, exact_horizon_2 10, exact_horizon_3 10), policy loss
  weight 1.0.
- Best val MSE 0.00701 at epoch 262 (still improving when budget ran out).
- Per-source val MSE: exact_horizon_2 0.00687, exact_horizon_3 0.00710.

## Held-out v2 calibration (the strict gate)

| Source | n | gen-2 v2 | gen-3 v2 | Phase I (small corpus) | **Phase G + I-1** |
|---|---|---|---|---|---|
| terminal | 24 | 0.553 | 0.522 | 0.593 | 0.557 |
| **tablebase** | 19 | 0.147 | 0.117 | 0.008 | **0.005** ✅ |
| exact_horizon_2 | 128 | 0.046 | 0.082 | 0.447 ❌ | **0.021** ✅ |
| exact_horizon_3 | 64 | 0.056 | 0.090 | 2.166 ❌ | **0.003** ✅ |
| **Overall MSE** | — | 0.108 | 0.132 | 0.895 | **0.070** ✅ |

**vs gen-2 v2 baseline 0.10841: +35.7% improvement.**

## Gate verdict

```
STRICT GATE PASSED:
  tablebase_mse_threshold       0.01       — passed (0.005)
  exact_horizon_2 threshold     0.10       — passed (0.021)
  exact_horizon_3 threshold     0.15       — passed (0.003)
  terminal threshold            1.0        — passed (0.557)
  max_unresolved_per_source     0.35       — passed (max 0.30)
  required sources              all 4      — present
```

First generation to clear every per-source threshold simultaneously.

## Why earlier attempts failed (and why this one didn't)

- **gen-2 v2** (hidden=64, 196 records): tablebase MSE 0.147 — too few
  pinned anchors and too small a net for fitting the diverse tablebase
  feature vectors after Phase F expanded from 2 → 19 pins.
- **Phase I-1 alone** (hidden=128, 196 records): tablebase MSE 0.008 ✅
  but h3 MSE 2.166 ❌ — bigger net + too-small h2/h3 corpus → catastrophic
  overfit. Train-side val h3=0.020 vs held-out h3=2.166 (100× gap = textbook).
- **Phase G + I-1** (hidden=128, 4,812 records): 24.7× more h2/h3 data
  let the bigger net generalize instead of memorize. Held-out h3 dropped
  from 2.166 → 0.003 (720× improvement) on the same architecture, same
  hyperparameters, more data. The bias-vs-overfit transition the original
  Phase 9 plan predicted.

## What unblocks next

The Phase G corpus is the new anchor set for downstream MCTS-bootstrap
iterations. `gen_phaseG_h128/best.pt` is the current promotion-quality
evaluator.

## Reproducibility pointers

- Corpus generation: `scripts/run_phase_g_corpus.py`
- Training: `training/train_value_net.py` (default `TrainConfig` plus
  `hidden_dim=128` and the source weights above).
- Held-out: `checkpoints/holdout_v2_targets.npz` (122 → 235 records via
  Phase F expansion).
- Gate: `training/bootstrap_loop.enforce_calibration_gate` with
  `per_source_mse_thresholds` populated.
- LRU cache fix: `environment/cfr/exact.py:_SOLVE_CACHE` (OrderedDict
  with `move_to_end` + `popitem(last=False)`).

---

## Phase G iter 2 — policy-improvement loop continues

Second iteration. Same architecture (`hidden_dim=128`), same anchor set
(Phase G's 4812-record LP corpus), but `gen_phaseG_h128/best.pt` is now
the MCTS evaluator instead of gen-2 v2. Tests whether the loop has
signal left after the Phase G corpus expansion.

### Run characteristics

- MCTS bootstrap: 8,092.9 s (2 h 15 m) at 1000 iters.
- CPU 260-274% sustained — hidden=128's 128×128 matmul saturates
  PyTorch's intra-op BLAS threading much better than hidden=64 did
  (~140%). Effective wall-clock per iter is *faster* than the smaller
  net's, despite ~2.6× more arithmetic per forward.
- Final training corpus: 6,357 records
  (mcts_bootstrap 864 + terminal 24 + tablebase 660 [22 × 30 replicate]
   + exact_horizon_3 2,938 + exact_horizon_2 1,871).
- Best train val MSE 0.03220 at epoch 205 (early-stopped from 300).

### Held-out v2 calibration

| Source | n | iter 1 (G + I-1) | **iter 2** | Δ |
|---|---|---|---|---|
| terminal | 24 | 0.557 | **0.505** | **−9.3%** ✅ |
| tablebase | 19 | 0.005 | 0.006 | +0.001 (passes 0.01) |
| exact_horizon_2 | 128 | 0.021 | **0.012** | **−43%** ✅ |
| exact_horizon_3 | 64 | 0.003 | 0.004 | +0.001 (passes 0.15) |
| **Overall MSE** | — | 0.070 | **0.060** | **−14.1%** ✅ |

Strict gate passed on all axes (tablebase 0.01, h2 0.10, h3 0.15,
terminal 1.0, max_unresolved 0.35).

### What this confirms

The policy-improvement loop *is still extracting signal* after the
Phase G corpus expansion. The pre-Phase-G plateau (gen-2 v2 → gen-3:
0.108 → 0.132, noise-level) was data-imposed, not algorithm-imposed.
With 24.7× the LP-labeled data, each subsequent iteration can still
pull genuine improvement: 0.070 → 0.060 in one step is bigger than
any improvement in the gen-2/gen-3 era.

The biggest gains came on classes the prior generation was weakest at
(terminal −9.3%, h2 −43%). The classes already near-perfect at iter 1
(h3 at 0.003) didn't move much because there's no room left.

### Best checkpoint after iter 2

`checkpoints/gen_phaseG_iter2/best.pt`. Same architecture as iter 1
(hidden_dim=128, 35.5K params).

---

## Phase G iter 3 — convergence

Third iteration, `gen_phaseG_iter2/best.pt` as MCTS evaluator. Same
anchors, same architecture, same gates.

### Run characteristics

- MCTS bootstrap: 8,100.7 s (basically identical to iter 2's 8,092.9 s).
- Final training corpus: 6,357 records (identical breakdown to iter 2;
  the bootstrap grid + anchor merge produces deterministic-shape output).
- Best train val MSE 0.03405 at epoch 139 (early-stopped from 300).
- Training-side terminal MSE jumped 0.84 (iter 2) → 1.23 (iter 3),
  the first signal that the net is starting to thrash on its own
  training-set terminals — characteristic of convergence on a fixed
  corpus.

### Held-out v2 calibration

| Source | n | iter 2 | **iter 3** | Δ |
|---|---|---|---|---|
| terminal | 24 | 0.505 | **0.498** | −1.4% |
| tablebase | 19 | 0.006 | 0.007 | basically tied |
| exact_horizon_2 | 128 | 0.012 | 0.012 | tied |
| exact_horizon_3 | 64 | 0.004 | 0.004 | tied |
| **Overall MSE** | — | 0.06000 | **0.05934** | **−0.9%** ✅ |

Strict gate passed.

### Convergence rate across the full phase chain

| Transition | Δ overall MSE |
|---|---|
| gen-0 → gen-2 v2 | −46% (rapid early gains, pre-Phase-G) |
| gen-2 v2 → gen-3 v2 | +22% (plateau pre-Phase-G) |
| Phase G + I-1 → iter 2 | −14.1% (Phase G data unlock) |
| iter 2 → iter 3 | **−0.9% (convergence)** |

The classic AlphaZero curve, twice: rapid gains until the corpus is
exhausted, plateau at the data-imposed ceiling, rapid recovery once
the corpus expands, plateau again at a much lower ceiling. Two
plateaus, two ceilings — both data-imposed.

### Best checkpoint after iter 3

`checkpoints/gen_phaseG_iter3/best.pt`. Overall MSE 0.05934 on
held-out v2 — a **45% improvement over the pre-Phase-G best**
(gen-2 v2 at 0.108).

### Status: converged within current data + architecture

The Phase 9 chain (F + G + I-1 + iter 2 + iter 3) has fully landed.
The system is now at the data-imposed performance ceiling under the
current setup. Pushing further requires one of:

- **Phase F-2**: expand REGISTRY beyond 19 pinned scenarios so the
  anchor diversity isn't dominated by the cyl=299 forced-overflow
  family.
- **Phase H**: selective high-iter MCTS reanalysis on the highest-
  variance states from the Phase G corpus.
- **Phase I-2**: hidden=192 with 50K param-count guard raised; risk
  of returning to the overfit regime unless paired with more data.
- **Downstream applications**: the `gen_phaseG_iter3/best.pt`
  checkpoint is ready for RL self-play, manga validation, CLI play,
  or whatever the supervised-learning checkpoint enables next.

The Phase 9 plan ended at "the bottleneck shifts from 'is the data
sufficient' to 'is the game model sufficient'" — and that's where we
are. The supervised-learning loop has extracted what's available.
