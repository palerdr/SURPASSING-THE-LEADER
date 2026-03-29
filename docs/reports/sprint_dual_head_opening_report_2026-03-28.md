# Sprint Report: Dual-Head Opening Architecture (p=0.88)
**Date:** 2026-03-28
**PHYSICALITY_BAKU:** 0.88
**Verdict:** NO_IMPROVEMENT
**Tradeoff broken:** False

## Executive Summary

This sprint targeted the bp-vs-native routing tradeoff identified in the previous
sprint. Three dual-head opening architecture families (12 candidates) were built
to break the tradeoff by splitting opening paths. **None succeeded.**

**Root cause identified with precision:** Bridge_pressure and native Hal produce
**literally identical observations** at the T2 classify turn. Classifier confidence
profiling confirms P(bp)=0.643 for BOTH opponents with std=0.000 across all games.
This means no classifier algorithm, threshold, confidence band, or routing
architecture can distinguish them at T2 — the input signal does not exist.

This is NOT a classifier weakness or a routing architecture gap. It is a
**feature-space limitation**: at T2, the observation vector contains no information
about opponent identity because opponent actions have not yet affected Baku's
observable game state.

## Optimization Target

Preserve bp opening strength (r7>=0.84) AND native-Hal resistance (native_wr>=0.06
or native_r7>=0.10) simultaneously by using separate opening paths for each
opponent family.

## Two Reference Baselines (Reproduced)

| Metric | Ref 1: Old T2 clf | Ref 2: Native-aware T2 clf |
|--------|-------------------|---------------------------|
| bp r7 | **88%** | 50% |
| bp WR | 0% | 20% |
| ht r7 | 50% | 50% |
| hp r7 | 50% | 50% |
| seeded r9 | 100% | 100% |
| prior_4096 WR | 70% | 70% |
| native WR | 0% | **20%** |
| native r7 | 0% | **50%** |

Both baselines reproduce exactly. The tradeoff is confirmed: old clf preserves bp
at the cost of native; native-aware clf preserves native at the cost of bp.

## Classifier Confidence Diagnostic (New Finding)

| Classifier | Opponent | P(bp) | P(hal) | std |
|------------|----------|-------|--------|-----|
| Old T2 | bridge_pressure | 0.643 | 0.357 | 0.000 |
| Old T2 | hal_death_trade | 0.357 | 0.643 | 0.000 |
| Old T2 | **native Hal** | **0.643** | 0.357 | **0.000** |
| Native T2 | bridge_pressure | 0.393 | 0.607 | 0.000 |
| Native T2 | hal_death_trade | 0.232 | 0.768 | 0.000 |
| Native T2 | **native Hal** | **0.393** | 0.607 | **0.000** |

**Key finding:** Bridge_pressure and native Hal have **identical** classifier
probabilities (both P(bp)=0.643 with old clf, both P(bp)=0.393 with native clf).
The std=0.000 means EVERY game produces the exact same probability. The observation
vectors at T2 are numerically identical for both opponents because:

1. At T2 (end of R0), Baku has played 2 fixed actions (from BASELINE_OVERRIDES)
2. The opponent has played 2 actions (D on T0, C on T1) but those actions don't
   appear in Baku's 20-dim observation vector — obs only tracks game state, not
   opponent action history
3. Both opponents produce the same game state at this point (no deaths, same clock,
   same cylinder levels)

**Implication:** No confidence threshold, dual-head architecture, or routing logic
can separate bp from native Hal at T2. The classifier literally receives the same
input for both opponents.

## Opening Route Comparison

| Turn | Role | bp_specialist opening | hal FRC opening | Match? |
|------|------|----------------------|-----------------|--------|
| T0 | C, R0, H1 | 1 | 1 | YES |
| T1 | D, R0, H2 | 1 | 1 | YES |
| T2 | C, R1, H1 | **60** | **1** | NO |
| T3 | D, R1, H2 | **1** | **10** | NO |
| T4 | C, R2, H1 | 60 | 60 | YES |
| T5 | D, R2, H2 | 1 | 1 | YES |
| T6 | C, R3, H1 | 60 | 60 | YES |
| T7 | D, R3, H2 | 1 | 1 | YES |

The opening routes differ only at T2-T3 (R1). bp_specialist checks at 60 on R1
(accumulates ST to pressure bridge_pressure), while hal FRC checks at 1 (minimizes
ST against hal). Using bp opening against hal is catastrophic (B2: ht=0% hp=0%).
Using hal opening against bp degrades bp to 50% (A3, B1).

## Architecture Families

### Family A: Explicit Dual-Head Opening
Two opening heads (bp_opening_frc from bp_specialist behavior, hal_opening_frc
from FRC route). Classifier selects which head during opening, then routes to
bp_specialist or late_model for late game.

| Candidate | Gate | bp r7 | native WR | native r7 | Notes |
|-----------|------|-------|-----------|-----------|-------|
| A1_dual_old_clf | PASS | 88% | 0% | 0% | Same as ref1 — native gets bp path |
| A2_dual_native_clf | FAIL | 50% | — | — | Same as ref2 — bp gets hal path |
| A3_universal_hal_old_clf | FAIL | 50% | — | — | hal opening hurts bp |

A1 and A2 reproduce the exact same tradeoff as the reference baselines because
the classifier gives **identical predictions** for bp and native Hal.

### Family B: Stacked/Delayed Routing
Opening and late phases use independent classifiers. Tested 4 combinations of
opening classifier x late classifier.

| Candidate | Gate | bp r7 | native WR | native r7 | Notes |
|-----------|------|-------|-----------|-----------|-------|
| B1_univ_hal_old_late | FAIL | 50% | — | — | Universal hal opening hurts bp |
| B2_univ_bp_old_late | FAIL | 75%* | — | — | ht=0% hp=0% — bp opening kills hal |
| B3_old_open_native_late | PASS | 88% | 0% | 0% | Old opening routes native to bp path |
| B4_native_open_old_late | FAIL | 50% | — | — | Native clf routes bp to hal path |

B3 is interesting: old clf gives correct bp opening routing → bp=88%. But native Hal
also gets bp opening (misclassified as bp) → enters late game in a suboptimal state
→ even the native-aware late clf can't recover → native=0%.

### Family C: Confidence-Based Routing
Uses predict_proba() with configurable thresholds. Routes high-confidence to
specialized paths, ambiguous to fallback.

| Candidate | Gate | bp r7 | Notes |
|-----------|------|-------|-------|
| C1 (old, th=0.7, fb=hal) | FAIL | 50% | P(bp)=0.643 < 0.7 → all fallback |
| C2 (old, th=0.9, fb=hal) | FAIL | 50% | P(bp)=0.643 < 0.9 → all fallback |
| C3 (old, bp=0.8/hal=0.5, fb=hal) | FAIL | 50% | P(bp)=0.643 < 0.8 → all fallback |
| C4 (native, th=0.7, fb=hal) | FAIL | 50% | P(bp)=0.393 < 0.7 → all fallback |
| C5 (old, th=0.7, fb=bp) | FAIL | 75% | All fallback to bp → ht/hp crash |

**All confidence variants fail** because bp and native Hal have identical
classifier probabilities. Every threshold either routes both to the same path
or sends both to fallback.

## Tests Added

20 unit tests in `tests/test_dual_head_opening.py`:
- TestDualHeadOpeningSelector (6 tests): opening/late routing, classifier-gated
  head selection, pre-classification defaults, reset behavior
- TestStackedRoutingSelector (4 tests): universal opening mode, independent
  opening/late classifiers, late-phase routing, reset behavior
- TestConfidenceRoutingSelector (7 tests): high/low/ambiguous confidence routing,
  fallback behavior, asymmetric thresholds, late-phase routing, reset behavior
- TestControllerAsModel (3 tests): action mapping, fallback, game_turn extraction

All 20 tests pass.

## Full Validation

Only 2 candidates passed the gate (bp>=60% AND ht>=30% AND hp>=30% AND r9=100%):

| Candidate | bp r7 | ht r7 | hp r7 | r9 | prior WR | native WR | native r7 | Level |
|-----------|-------|-------|-------|----|----------|-----------|-----------|-------|
| A1_dual_old_clf | 88% | 50% | 50% | 100% | 70% | 0% | 0% | --- |
| B3_old_open_native_late | 88% | 50% | 50% | 100% | 70% | 0% | 0% | --- |

Both are functionally identical to Reference 1 (old classifier baseline). Neither
achieves any native Hal improvement.

## Final Verdict

**NO_IMPROVEMENT** — The bp-vs-native routing tradeoff was NOT broken.

**Reason:** The tradeoff is not caused by a routing architecture limitation — it is
caused by an **observation-space limitation**. At T2, bridge_pressure and native Hal
produce numerically identical 20-dim observation vectors. No routing architecture
(dual-head, stacked, confidence-based) can overcome identical inputs.

Promotion gate: **NOT MET** (no candidate achieves bp>=0.82 AND native improvement).

## Best Next Move

The routing architecture is sound — all 3 families work correctly when the
classifier can distinguish opponents (verified by unit tests). The fix must come
from outside the routing layer:

1. **Observation enrichment (RECOMMENDED):** Add opponent action history to the
   observation vector. At T2, the opponent has taken 2 actions that ARE different
   between bp and native Hal but are NOT in the current obs. Adding
   `last_opponent_action / 60` or a running average would break the feature overlap.

2. **Delayed classification with enriched obs:** Classify at T4+ using an obs
   vector that includes opponent actions from T0-T3. By T4, different opponent
   strategies produce distinguishable game states.

3. **Eliminate the classifier entirely:** Train a single unified model (opening +
   late) that handles all opponent types. This removes the routing problem at the
   cost of potential specialization loss.

4. **Accept intermediate regression:** Deploy native-aware classifier (bp=50%,
   native=50%) and retrain bp response from the new baseline.

Sprint time: ~70s
