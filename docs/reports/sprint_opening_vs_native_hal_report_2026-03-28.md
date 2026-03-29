# Sprint Report: Opening vs Native Hal at p=0.88
**Date:** 2026-03-28
**PHYSICALITY_BAKU:** 0.88
**Verdict:** NO_IMPROVEMENT

## Executive Summary

This sprint targets Baku's OPENING response (T0-T7) against the strongest
scaffold-native Hal (native_hal_16k). The prior sprint showed Baku WR=0%
and r7_rate=0% against native Hal, with late-head tuning producing no signal.
Three materially different opening-improvement families were tested:
counterfactual-guided search, population-based evolution, and greedy
coordinate descent.

## Optimization Target

Improve opening response to raise Baku win rate or r7_rate vs native Hal,
while preserving scripted baseline (bp>=0.84, ht>=0.48, hp>=0.48, r9=100%).

## Why Opening Progress Matters

The strongest native Hal defeats Baku before round 7 (r7_rate=0%). Late-head adaptation produced zero improvement across 3+ families. If the opening can be improved, it would unlock meaningful self-play iteration. If not, the opening action space is saturated under current calibration.

## Baseline Table

| Metric | Value |
|--------|-------|
| Route | `[1, 1, 1, 10, 60, 1, 60, 1]` |
| BC accuracy | 100.0% |
| Misaligned | 0 |
| bp r7 | 88% |
| ht r7 | 50% |
| hp r7 | 50% |
| seeded r9 | 100% |
| prior_4096 WR | 32% |
| native Hal WR | 0% |
| native Hal r7 | 0% |

## Opening Failure Analysis vs Native Hal

- Avg half-rounds: 8.7
- Avg Baku deaths: 2.3
- Avg Hal deaths: 0.6
- First Baku death turn: median=0, range=[0, 0]
- Opening death fraction: 100%

## Counterfactual Audit vs Native Hal

Found 6 single-turn improvement(s):

- T1 (D): baseline=1 -> 5 (+0.0244)
- T2 (C): baseline=1 -> 30 (+0.2356)
- T4 (C): baseline=60 -> 10 (+0.4900)
- T5 (D): baseline=1 -> 5 (+0.1633)
- T6 (C): baseline=60 -> 10 (+0.4900)
- T7 (D): baseline=1 -> 5 (+0.1789)

## Search/Evolution/Distillation Families

### Family A: Counterfactual-Guided

- **A1_all_cf**: diffs=['T1=5', 'T2=30', 'T4=10', 'T5=5', 'T6=10', 'T7=5'] wr=80% r7=7%
- **A2_top1_T4eq10**: diffs=['T4=10'] wr=37% r7=47%
- **A2_top2_T6eq10**: diffs=['T6=10'] wr=37% r7=47%
- **A2_top3_T2eq30**: diffs=['T2=30'] wr=13% r7=67%
- **A3_T4eq10_T6eq10**: diffs=['T4=10', 'T6=10'] wr=37% r7=47%
- **A3_T4eq10_T2eq30**: diffs=['T4=10', 'T2=30'] wr=33% r7=40%
- **A3_T6eq10_T2eq30**: diffs=['T6=10', 'T2=30'] wr=33% r7=40%

### Family B: Population Evolution

- **B_evo_best**: route=[20, 5, 40, 20, 15, 10, 30, 40] wr=87% r7=13%

### Family C: Greedy Descent

- **C_greedy**: route=[5, 5, 5, 10, 5, 1, 5, 1] wr=60% r7=60%

## Full Validation

| Candidate | bp r7 | ht r7 | hp r7 | r9 | prior WR | native WR | native r7 | mis | Level |
|-----------|-------|-------|-------|----|----------|-----------|-----------|-----|-------|
| baseline | 88% | 50% | 50% | 100% | 32% | 0% | 0% | 0 | --- |
| A2_top1_T4eq10 | 88% | 18% | 18% | 100% | 10% | 0% | 0% | 0 | --- |
| A2_top2_T6eq10 | 88% | 18% | 18% | 100% | 10% | 0% | 0% | 0 | --- |
| A2_top3_T2eq30 | 88% | 50% | 50% | 100% | 32% | 0% | 0% | 0 | --- |
| A3_T4eq10_T2eq30 | 88% | 18% | 18% | 100% | 10% | 0% | 0% | 0 | --- |
| A3_T6eq10_T2eq30 | 88% | 18% | 18% | 100% | 10% | 0% | 0% | 0 | --- |

## Final Verdict

**NO_IMPROVEMENT**

- Best candidate: A2_top3_T2eq30
- Level: None
- Score: 0.0
- Any improvement signal: NO
- Sprint time: 648s

## Critical Analysis: Fast Eval vs Full Eval Gap

The most important finding of this sprint is the stark gap between the fast
evaluation (direct action overrides during opening, base model for late game)
and the full modular selector evaluation:

| Route | Fast Eval WR | Fast Eval r7 | Full Eval WR | Full Eval r7 |
|-------|-------------|-------------|-------------|-------------|
| Baseline [1,1,1,10,60,1,60,1] | 13% | 47% | 0% | 0% |
| A1 all CF combined | 80% | 7% | GATE FAIL | - |
| B evo [20,5,40,20,15,10,30,40] | 87% | 13% | GATE FAIL | - |
| C greedy [5,5,5,10,5,1,5,1] | 60% | 60% | GATE FAIL | - |

**Root cause:** The BC opening model is trained on observations collected
from games against *scripted* opponents (hal_death_trade, hal_pressure).
When deployed against native Hal — whose drop patterns differ substantially —
the observations shift outside the training distribution. Despite 100%
accuracy on training data, the model produces different actions against
native Hal than intended.

**Key implication:** The opening action space is NOT saturated. Theoretical
routes exist that would achieve 60-87% WR against native Hal. The bottleneck
is the BC model's inability to generalize across opponent distributions.

**Evidence from diagnosis:** Baku's first death occurs at T0 in 100% of
traced games. Native Hal learned to drop late on T0, exploiting Baku's
deterministic check@1. The BC model, having never seen native Hal's
observation profile during training, cannot adapt.

## Best Next Move

The opening route *can* theoretically beat native Hal (up to 87% WR with
direct overrides). The delivery mechanism (BC model) fails due to
distribution shift. Priority actions:

1. **Train BC on native-Hal observations** — collect opening samples by
   playing the override route against native Hal (not just scripted opponents),
   so the model sees the actual observation distribution it will face.
2. **Observation-agnostic opening** — use a lookup table indexed by
   (round_index, half_index) instead of a neural network, making the opening
   immune to observation distribution shift.
3. **Mixed-opponent BC corpus** — collect samples from both scripted AND
   native Hal opponents during BC training, improving generalization.
4. **If none of the above work:** consider stochastic openings, recurrent
   opening policies, or temperature-based action selection.
