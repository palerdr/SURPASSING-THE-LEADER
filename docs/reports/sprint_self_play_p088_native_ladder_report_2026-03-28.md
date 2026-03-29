# Sprint Report: Scaffold-Native Self-Play Ladder at p=0.88
**Date:** 2026-03-28
**PHYSICALITY_BAKU:** 0.88
**Verdict:** NO_IMPROVEMENT

## Objective

Fix the key confound from the previous ladder sprint: Hal opponents were
trained against BASE_MODEL alone, not the actual modular p=0.88 Baku scaffold.
This sprint trains scaffold-native Hal candidates against the full
ModularBakuSelector, then uses them in a graded opponent ladder.

## Baseline Reproduction

- Route: `[1, 1, 1, 10, 60, 1, 60, 1]`
- BC accuracy: 100.0% (1930/1930), misaligned=0
- bp r7: 88% | ht r7: 50% | hp r7: 50%
- seeded r9: 100%
- vs prior_4096 win rate: 32%

## Scaffold-Native Hal Candidates

These Hals were trained against the **full modular Baku selector** (BC opening
+ classifier + late model), not BASE_MODEL alone.

| Candidate | Steps | Seed | Degenerate | Unique Actions | Baku WR |
|-----------|-------|------|------------|----------------|---------|
| native_hal_16k | 16384 | 42 | no | 4 | 0% |
| native_hal_32k | 32768 | 42 | no | 6 | 0% |
| native_hal_32k_s2 | 32768 | 137 | no | 9 | 18% |

**Selected best native Hal:** native_hal_16k (Baku WR=0%)
- Baseline Baku WR vs native: 0%

## Scaffold-Native Self-Play Signal Assessment

Native Hal provides strong pressure (Baku WR < 10%).

Compared to prior_4096 (Baku WR=32%), the native Hal is **stronger**.

## Graded-Ladder Families

| Family | Mechanism | Steps | Gate |
|--------|-----------|-------|------|
| E_phased_native_ladder | phased_native_ladder | 24576 | PASS |
| F_native_dominant_curriculum | native_dominant_curriculum | 20480 | PASS |
| G_balanced_native_prior | balanced_native_prior | 20480 | PASS |
| H_native_heavy_light_shaping | native_heavy_light | 24576 | FAIL |

### Family Descriptions

**E_phased_native_ladder:** 3-phase (scripted->+prior->+native@2.0), bridge shaping, 24k

**F_native_dominant_curriculum:** Native-dominant (4x) + scripted(0.5), LateCurriculum p=0.3, r7+r8+r9, bridge, 20k

**G_balanced_native_prior:** Balanced (native@2.0 + prior@2.0 + scripted@1.0), OpeningAutoPlay, bridge, 20k

**H_native_heavy_light_shaping:** Native heavy (5x) + scripted(0.5), OpeningAutoPlay, light shaping, 24k

## Full Evaluation Results (50-game)

| Candidate | bp r7 | ht r7 | hp r7 | r9 | prior WR | native WR | mis |
|-----------|-------|-------|-------|----|----------|-----------|-----|
| **baseline** | 88% | 50% | 50% | 100% | 32% | 0% | 0 |
| E_phased_native_ladder | 88% | 50% | 50% | 100% | 32% | 0% | 0 |
| F_native_dominant_curriculum | 88% | 50% | 50% | 100% | 32% | 0% | 0 |
| G_balanced_native_prior | 88% | 50% | 50% | 100% | 32% | 0% | 0 |

### Deltas vs Baseline

| Candidate | bp r7 | ht r7 | hp r7 | prior WR | native WR | Score | Level |
|-----------|-------|-------|-------|----------|-----------|-------|-------|
| E_phased_native_ladder | +0% | +0% | +0% | +0% | +0% | 0.0000 | --- |
| F_native_dominant_curriculum | +0% | +0% | +0% | +0% | +0% | 0.0000 | --- |
| G_balanced_native_prior | +0% | +0% | +0% | +0% | +0% | 0.0000 | --- |

## Promotion Criteria

| Bar | bp r7 | ht r7 | hp r7 | r9 | mis | prior WR delta | native WR delta |
|-----|-------|-------|-------|----|-----|----------------|-----------------|
| Strong | >=84% | >=48% | >=48% | 100% | 0 | +6% | +6% |
| Acceptable | >=82% | >=48% | >=48% | 100% | 0 | +4% | +4% |

## Verdict

**NO_IMPROVEMENT**

- Best candidate: E_phased_native_ladder
- Level: None
- Composite score: 0.0
- Any improvement signal: NO
- Sprint time: 2046s

## Analysis

### Does scaffold-native self-play yield real improvement signal?

**NO** — training Hal against the full modular Baku scaffold produced
no measurable improvement in Baku adaptation, despite fixing the confound
of training against BASE_MODEL alone. The MlpPolicy deterministic argmax
ceiling persists regardless of opponent provenance.
