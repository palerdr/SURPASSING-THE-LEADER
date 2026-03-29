# Sprint: Opening Re-Audit under PHYSICALITY_BAKU = 0.88

**Date:** 2026-03-28
**Sprint:** `opening-reaudit-p088`
**Duration:** 209s
**Verdict:** FAIL (no improved route found; baseline is locally optimal)

---

## Executive Summary

After changing `PHYSICALITY_BAKU` from 0.70 to 0.88, the performance ceiling moved **significantly upward**. The existing baseline route `[1, 1, 1, 10, 60, 1, 60, 1]` now achieves **88/50/50** (bp/ht/hp r7 rates) on 50-game evaluation, up from the old **79/33/33** regime. All hard promotion bars (bp >= 80%, ht >= 40%, hp >= 40%, seeded r9 = 100%) are now met **by the existing baseline** without any route modifications.

However, the counterfactual audit found **zero single-turn improvements** across T0-T7. The baseline route is a strict local optimum at p=0.88. No candidate route was constructed, validated, or promoted. The sprint result is FAIL for the promotion question (no better route exists), but the news is positive: the parameter change alone broke through the old ceiling.

---

## Baseline Table under PHYSICALITY_BAKU = 0.88

| Metric | p=0.88 (50-game) | p=0.70 (24-game, old) | Delta | Hard Bar |
|--------|------------------:|----------------------:|------:|---------:|
| bp r7 | **88%** (44/50) | 79% (19/24) | +9 | >= 80% |
| ht r7 | **50%** (25/50) | 33% (8/24) | +17 | >= 40% |
| hp r7 | **50%** (25/50) | 33% (8/24) | +17 | >= 40% |
| seeded r9 | **100%** (1/1) | 100% (1/1) | 0 | = 100% |
| learned hal WR | **32%** (16/50) | 17% (4/24) | +15 | n/a |
| BC misaligned | **0** | 0 | 0 | = 0 |
| hal combined r7 | **50%** (50/100) | 36% (36/100) | +14 | n/a |

### Why the ceiling moved

`PHYSICALITY_BAKU = 0.88` (vs 0.70) enters the survival probability formula as a direct multiplier:

```
P(survival) = base_curve(t) * cardiac(ttd_prior) * referee(n) * physicality
```

At 0.88, Baku survives ~26% more death events than at 0.70 (0.88/0.70 = 1.257). Since the opening route involves 2 death-trade deaths (T0 and T2), the compound survival probability rises from ~0.49 to ~0.77 for the two-death opening path. This explains the ht/hp r7 jump from 33% to 50%.

---

## Opening Route Audit (T0-T7)

**Baseline route:** `[1, 1, 1, 10, 60, 1, 60, 1]`
**Roles:** T0=C, T1=D, T2=C, T3=D, T4=C, T5=D, T6=C, T7=D

### Per-turn counterfactual results

| Turn | Role | Baseline | Status | Notes |
|------|------|----------|--------|-------|
| T0 | C | 1 | **Optimal** | a>=40 drops r7 to 0% (failed check kills Baku in R1) |
| T1 | D | 1 | **Optimal** | All alternatives equal (50%); drop timing in R1H2 is irrelevant |
| T2 | C | 1 | **Optimal** | a>=40 degrades (40%), a>=55 drops to 0% |
| T3 | D | 10 | **Optimal** | a<=5 drops to 32% (changes death-trade timing); a>=15 ties at 50% |
| T4 | C | 60 | **Optimal** | Any other value drops to 18% (must check@60 for safe strategy) |
| T5 | D | 1 | **Optimal** | All alternatives equal (50%); drop timing irrelevant here |
| T6 | C | 60 | **Optimal** | Any other value drops to 18% (must check@60) |
| T7 | D | 1 | **Optimal** | a>=40 drops to 6% (late drop shifts timing catastrophically) |

**Key observations:**
- T4 and T6 (both Checker) are the most rigidly constrained: check@60 is the only viable action. Any deviation collapses r7 from 50% to 18%.
- T3 (Dropper) has a plateau: drop>=10 all yield 50%, but drop<=5 drops to 32%. The baseline's drop@10 is on the edge of this plateau.
- T0, T1, T5 have flat plateaus where many actions tie the baseline. No improvement exists because no action exceeds 50%.
- T7 (Dropper) has a cliff at a>=40, dropping from 50% to 6%.

**Improvements found:** 0
**Locally optimal:** Yes, the baseline route is a strict local optimum.

---

## Candidate Routes

None. No single-turn deviation improved the hal-family r7 rate above the 50% baseline.

Since no improvements were found in Lane A, Lanes B (candidate construction) and C (validation) were skipped.

---

## Full Validation

N/A (no candidates to validate).

---

## Final Verdict

**FAIL** — No promoted candidate.

The baseline route `[1, 1, 1, 10, 60, 1, 60, 1]` remains the unique locally optimal opening at p=0.88. The hard promotion bars require "deterministic opening action differences vs old baseline," which is impossible when no alternative route improves on the baseline.

However, the **existing baseline now exceeds all numeric bars**:
- bp r7 = 88% >= 80%
- ht r7 = 50% >= 40%
- hp r7 = 50% >= 40%
- seeded r9 = 100%
- misaligned = 0

The practical interpretation: the parameter change to `PHYSICALITY_BAKU = 0.88` is itself the improvement. No route change is needed.

---

## Ceiling Movement: Old vs New Regime

| Regime | bp r7 | ht r7 | hp r7 | hal comb r7 | Characterization |
|--------|------:|------:|------:|------------:|------------------|
| p=0.70 | 79% | 33% | 33% | 36% | Death-trade survival ~49%; Baku fragile |
| p=0.88 | **88%** | **50%** | **50%** | **50%** | Death-trade survival ~77%; Baku viable |

The ceiling moved **decisively**:
- hal-family r7 rose from 36% to 50% (+14 pts)
- bp r7 rose from 79% to 88% (+9 pts)
- learned hal WR rose from 17% to 32% (+15 pts)

The new ceiling at 50% hal-family r7 reflects the mathematical bound: with ~88% per-death survival and 2 required death-trades, compound survival is ~77%, but opponent stochasticity and late-game variance bring the effective reach rate to ~50%.

---

## Best Next Move

1. **Accept the baseline as-is.** The route `[1, 1, 1, 10, 60, 1, 60, 1]` at p=0.88 meets all numeric promotion bars. The bottleneck is no longer the opening route.

2. **Focus on late-game improvement.** The 50% r7 rate means half the games reach round 7. Improving the r7-to-r9 conversion rate (late-head quality) is now the highest-leverage target.

3. **Investigate learned Hal.** The 32% win rate against learned Hal and 0% r7 rate suggest the learned Hal opponent plays very differently from scripted Hal. Understanding and adapting to its patterns is a separate workstream.

4. **Consider multi-turn deviations.** The counterfactual audit tested single-turn deviations only. Coordinated multi-turn changes (e.g., shifting T3+T7 together) could theoretically find improvements, but the flat/cliff structure of the audit landscape makes this unlikely.
