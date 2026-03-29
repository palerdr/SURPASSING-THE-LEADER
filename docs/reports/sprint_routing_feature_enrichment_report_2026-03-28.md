# Sprint Report: Routing Feature Enrichment (p=0.88)
**Date:** 2026-03-28
**PHYSICALITY_BAKU:** 0.88
**Verdict:** NO_IMPROVEMENT
**Overlap broken:** True
**Tradeoff broken:** False

## Executive Summary

This sprint targets the observation-space limitation identified in the dual-head
opening sprint: bridge_pressure and native Hal produce numerically identical
20-dim observations at T2, making classification impossible. The fix: append
4-dim public opponent-action-history features to the classifier input (24-dim
total) while keeping the policy on the original 20-dim observation shape.

**Result:** The observation overlap IS broken — enriched features differ between bp
and native Hal at T2. However, the difference is **too small** (0.017 in one
dimension) for the classifier to exploit. Bridge_pressure drops at second 2 while
native Hal drops at second 3 on T0; both check at second 2 on T1. The 1-second
difference in one early action is insufficient signal. All enriched T2 classifiers
reproduce the same tradeoff as unenriched ones.

**Critical finding:** The enriched T4 classifier achieves **100% accuracy** on all
three opponent types (bp, hal_dt, native). By T4, game state divergence from R1
outcomes creates strong separation. This proves the history-feature approach works —
the bottleneck is that only 2 opponent actions exist at T2 and they're too similar.

**Verdict:** NO_IMPROVEMENT. Routing-only enrichment at T2 does not break the
tradeoff. The correct next step is a full observation-v2 retrain embedding history
features directly in the policy input, allowing the model to leverage opponent
action patterns across all turns.

## Optimization Target

Break the bp-vs-native observation overlap by giving the routing/classifier
path the public opponent-action-history features that Baku would actually
remember. The policy observation shape must NOT change.

## Why Routing-Only Enrichment Is the Right Intermediate Step

A full observation-v2 retrain would change the policy input shape and require
retraining all models from scratch. By enriching ONLY the classifier/routing
input, we can test the hypothesis (opponent history breaks the overlap) without
any retrain risk. If enrichment works, it validates the features for a future
observation-v2 migration. If it doesn't, we learn cheaply.

## Reference Baselines (Reproduced)

| Metric | Ref 1: Old T2 clf | Ref 2: Native-aware T2 clf |
|--------|-------------------|----------------------------|
| bridge_pressure r7 | 88% | 50% |
| hal_death_trade r7 | 50% | 50% |
| hal_pressure r7 | 50% | 50% |
| seeded r9 | 100% | 100% |
| prior_4096 WR | 70% | 70% |
| native WR | 0% | 20% |
| native r7 | 0% | 50% |

## Enriched Classifier Diagnostics

### Accuracy

| Classifier | bp acc | hal acc | native acc |
|------------|--------|---------|------------|
| enriched_old_t2 | 100% | 100% | 0% |
| enriched_native_t2 | 0% | 100% | 100% |
| enriched_native_t4 | 100% | 100% | 100% |

### Confidence Profiles (Enriched Features)

| Classifier | Opponent | P(bp) | P(hal) | std | History Features |
|------------|----------|-------|--------|-----|------------------|
| enriched_old_t2 | bp | 0.949 | 0.051 | 0.000 | ['0.033', '0.000', '0.033', '1.000'] |
| enriched_old_t2 | hal_dt | 0.051 | 0.949 | 0.000 | ['1.000', '0.000', '0.600', '1.000'] |
| enriched_old_t2 | native | 0.947 | 0.053 | 0.000 | ['0.033', '0.000', '0.050', '1.000'] |
| enriched_native_t2 | bp | 0.491 | 0.509 | 0.000 | ['0.033', '0.000', '0.033', '1.000'] |
| enriched_native_t2 | hal_dt | 0.027 | 0.973 | 0.000 | ['1.000', '0.000', '0.600', '1.000'] |
| enriched_native_t2 | native | 0.482 | 0.518 | 0.000 | ['0.033', '0.000', '0.050', '1.000'] |
| enriched_native_t4 | bp | 0.638 | 0.362 | 0.000 | ['0.033', '0.000', '0.033', '1.000'] |
| enriched_native_t4 | hal_dt | 0.082 | 0.918 | 0.000 | ['0.083', '0.000', '0.583', '1.000'] |
| enriched_native_t4 | native | 0.280 | 0.720 | 0.000 | ['0.033', '0.000', '0.317', '1.000'] |

### Key Finding: Overlap Broken = **True** (but signal too weak at T2)

- bp history features:     `['0.0333', '0.0000', '0.0333', '1.0000']`
- native history features: `['0.0333', '0.0000', '0.0500', '1.0000']`

**Decoded:** At T2, the history contains 2 opponent actions from R0:
- T0 (opponent as dropper): bp drops at second 2 (0.033×60), native Hal drops at second 3 (0.050×60)
- T1 (opponent as checker): both check at second 2 (0.033×60)

The **only discriminating signal** is a 1-second difference in T0 drop time (feature index 2:
0.0333 vs 0.0500, delta=0.0167). This is because bridge_pressure and native Hal both use
aggressive early-action strategies — they're behaviorally similar in R0. The classifier
coefficient for this feature produces only a 0.002 shift in P(bp) (0.949 vs 0.947), which
is not enough to flip classification.

**T4 enriched classifier proves the approach:** By T4, the game state has diverged
(different R1 outcomes — death trades, ST accumulation), and the T4 history features show
clear separation: native Hal's prev_action_norm jumps to 0.317 (vs bp's 0.033). The
enriched_native_t4 classifier achieves 100% accuracy on all opponent types. The feature
approach is validated; the limitation is T2 timing, not the features themselves.

## Tested Selector Variants

| Candidate | Type | Classifier | Gate |
|-----------|------|------------|------|
| E1_enriched_old_modular | enriched_modular | enriched_old_t2 | PASS |
| E2_enriched_native_modular | enriched_modular | enriched_native_t2 | FAIL |
| E3_enriched_native_dual_head | enriched_dual_head | enriched_native_t2 | FAIL |
| E4_enriched_stacked | enriched_stacked | enriched_old_t2 | PASS |
| D1_delayed_t4_hal_univ | enriched_delayed_t4 | enriched_native_t4 | FAIL |

## Full Validation

| Candidate | bp r7 | ht r7 | hp r7 | r9 | prior WR | native WR | native r7 | Level |
|-----------|-------|-------|-------|----|----------|-----------|-----------|-------|
| E1_enriched_old_modular | 88% | 50% | 50% | 100% | 70% | 0% | 0% | None |
| E4_enriched_stacked | 88% | 50% | 50% | 100% | 70% | 0% | 0% | None |

## Tests Added

- `tests/test_enriched_routing.py`: unit tests for enriched selector classes
  - TestEnrichedModularSelector: routing with enriched classification
  - TestEnrichedDualHeadSelector: dual-head routing with enriched classification
  - TestEnrichedStackedSelector: stacked routing with enriched classifiers
  - TestEnrichedDelayedSelector: delayed T4 classification routing
  - TestBuildEnrichedClassifier: classifier training on 24-dim features

## Final Verdict

**NO_IMPROVEMENT**

- Overlap broken: **True**
- Tradeoff broken: **False**
- Best candidate: **E1_enriched_old_modular**
- Best level: **None**
- Candidates evaluated (full): 2

## Best Next Move

The observation overlap WAS broken (features differ by 0.017 at T2), but the
signal is too weak for classifier exploitation. T4 enriched classifier proves
100% accuracy — the features work, the timing doesn't.

**Why delayed classification fails:** Using a universal opening for T0-T3
sacrifices bp opening quality. The bp and hal openings differ at T2-T3
(bp checks at 60 to accumulate ST; hal checks at 1 to trigger death trade).
No universal T2 action is safe for both opponent types. The D1 variant failed
gate with bp=50% and r9=0%.

**The definitive answer:** Routing-only enrichment cannot solve this at T2
because the opponents are behaviorally too similar in R0. The classifier needs
either (a) more history (T4+, which has structural costs) or (b) the policy
itself to adapt over time.

**Recommended next steps (in order):**
1. **Observation-v2 retrain** — embed opponent action history features directly
   in the 24-dim policy input. This lets the policy learn opponent-adaptive
   behavior across ALL turns, not just at a classification point. The
   routing_features.py scaffold is ready for this.
2. **Recurrent policy** — an LSTM/GRU policy that accumulates opponent behavior
   over the episode, eliminating the need for a classification seam entirely.
3. **Accept the ceiling** — the current 88/50/50 baseline against scripted
   opponents is solid; native Hal resistance may require fundamentally
   different training (curriculum with native Hal in the loop).

Sprint time: ~78s
