# DTH promotion campaign record

The measured history behind [`AGENT_GOAL.md`](AGENT_GOAL.md): the evidence the
goal was built on, the five expert-iteration generations, the four
champion-or-ceiling attempts, and the diagnostics that closed the campaign.
The goal file owns the deliverable, the gates, and the open decision; this file
owns the narrative and the numbers. Generated artifacts are gitignored, so
every load-bearing figure is snapshot here with its artifact name. Commands are
in [`WORKFLOWS.md`](WORKFLOWS.md).

## Evidence basis — recorded 2026-07-28

Generated artifacts are gitignored, so the load-bearing numbers are snapshot
here with their artifact names. Re-verify before relying on them; none of
these is a frozen claim.

- **Budget bought nothing.** Every retained network-leaf readiness ladder is
  bit-identical at MCTS budgets 0 and 4096 (e.g.
  `mcts_h5_gate_network_guarded_equilibrium_v11_d1.json`: max saddle gap
  `0.078460…` at both ends). Production configs ran `max_depth: 1` with
  `root_warmup_cells: 3600`, so the warmup enumerated every joint cell and
  simulations added no information. No retained result ever measured search.
- **The search core is sound.** The same ladder with exact leaves reaches
  max saddle gap `5.55e-16` (`mcts_h5_gate_exact_v6_d1.json`). The entire
  deficit is frontier value quality.
- **Learned-leaf quality is the open problem.** Network-leaf gaps sit at
  0.02–0.27 across the 11-root pack. Orientation-paired roots built by
  `generate_dataset.mirror_state` (coordinate swap `(s_c,t_c,s_d,t_d) →
  (s_d,t_d,s_c,t_c)`) differ about 5× in learned gap: `[179,60,59,180]`
  0.2734 vs `[59,180,179,60]` 0.0547; `[179,60,119,120]` 0.2220 vs
  `[119,120,179,60]` 0.0410. The purpose-built probe
  (`paired_boundary_orientation_v1`) has no recorded result.
- **The capacity ladder stalled at v19.** Width 128, `boundary_v1`/`v2`
  feature lifts, the 3600-cell residual, and the action-input Q MLP all
  failed to beat `strategic_matrix_policy_generalization_balanced_v19`; the
  v22 product-fit run regressed max gap 0.2386 → 0.2693 while training on
  its own eval roots. No checkpoint survives on disk; v19 is reproducible
  in principle from its config chain only.
- **Full closure is out of reach.** Pilot reports project the universal
  envelope at ~8.15e9 states, 51–70 LP solves/s observed, ~3.7–5.1 years
  and ~8.1 TB of certificate payload (`full_closure_projection` blocks in
  the `exact_agent_*` reports). The opening census is bounded-incomplete
  with the root interval still `[-1, 1]`, and observed transposition
  elimination is 93.2% (`exact_opening_consolidated_v1_report.json`).
- **The promotion gate has never run.** No `dth-self-play-readiness-v1`
  artifact exists; `readiness.compare_ladders` has never produced a
  promote/no-promote decision.


## Expert iteration — the five generations

### Generations one to three, recorded 2026-07-29

The `resolve_labeled` emission closed the coverage loop: self-play under the
current agent, one depth-three resolve per unlabeled trajectory state, every
interior solution harvested as a training row (64,280 rows from 30 resolves
in 5.2 minutes at generation one). Selection worst dev-root gap fell
0.5161 (baseline) → 0.2701 (v1) → 0.1985 (gen1) → 0.2310 (gen2 final epoch),
evaluation median gap improved 49–70% every generation, and the
anchor-gap-regression gate that failed v1/v2 passes from generation one on.
Promotion still says no-promote (`promotion_gen1..3.json`): the worst-anchor
bar (0.0624) plateaued at 0.0657–0.0671, and anchor value errors *rose* as
coverage grew — including in generation three, which seeded half its games
inside the anchor neighbourhood. The diagnosis: play-coverage rows are
labeled at the query horizon with depth-amplified complete-game estimates,
which semantically collide with the anchors' finite-horizon-exact values at
the same horizon feature; near states carrying both label kinds, value
accuracy against the finite-exact reference degrades even as play strength
improves. The next lever is separating the two value semantics — a distinct
horizon channel or head for play values, or exact finite-horizon labels
inside the anchor region — before spending further generations.

### Generation four — value semantics separated, recorded 2026-07-29

The first half of the lever is built and structural: every target row carries
`value_semantics` (0 finite-horizon-exact, 1 resolve-play), play rows
supervise a dedicated play head (migrated as a copy of the value head), the
agent's scalar resolve leaves answer from the play head, `ExactTargetStore`
refuses play rows as exact authority (so decision matrices and class targets
can never compose a play value), and models without a play head are scored
on finite rows only. Generation four retrained the generation-three
coverage (484,950 rows: 268,344 play, 216,606 finite) under the split.

The heads separated cleanly — validation finite value MSE 0.0043, play value
MSE 0.0032 — and the evaluation roots posted the loop's best result: median
gap 0.0837 → 0.0210 (74.9% improvement), max gap no-regression, seed-stable.
Promotion still says no-promote (`promotion_gen4.json`): worst anchor gap
0.0703 against the 0.0624 bar, and anchor value errors regressed
+0.017/+0.016/+0.027 against the 0.01 allowance. With the collision
structurally removed, the anchor failure now has a cleaner reading: play
rows had been giving the finite head incidental near-anchor coverage, and
the shared trunk now also serves play values, so finite-head accuracy at the
anchor leaf horizons is the binding constraint — a coverage problem, not a
labeling conflict. The next lever is the second half of the recorded pair:
exact finite-horizon closures generated inside the anchor neighbourhood
(and/or heavier anchor frontier replay) so the finite head has literal
supervision where the gates bind.

### Generation five — coverage disproved, fit confirmed, recorded 2026-07-29

The closure half of the lever falsified its own premise: the full exact
anchor closure at horizon five (`anchor_leaf_closure_v1`, 68,346 states,
h5..h1) turned out to already be inside the training reference — every leaf
the depth-two anchor resolve queries had a finite training row all along.
The anchor failure was never coverage; it is fit under multi-task trunk
pressure. The replay half then delivered: generation five added the two
leaf layers the anchor resolves actually evaluate (7,442 h3 rows at
repeats 4, 3,721 h2 rows at repeats 8) to the exact-frontier replay stage.

Result (`promotion_gen5.json`): **seven of eight promotion gates pass** —
anchor value-error regression passes for the first time in the campaign
(deltas +0.0098/−0.0124/+0.0014 against the 0.01 allowance), the mirror h5
anchor resolves to gap 0.0, the primary h5 anchor to 0.0096, and eval
median improved 64.5% with no max regression. The sole failing gate is
anchor-worst-gap improvement: the h4 anchor sits at 0.0676 against the
0.0624 bar (a 2.5% improvement where 10% is required). A refinement probe
(`promotion_gen5b.json`, h2 replay tripled, fine-tuned from generation
five) moved the gap only to 0.0673 while re-breaking the value-error gate
and costing eval-median improvement — replay weight is past its optimum,
and generation five remains the selected candidate of the loop.

The stuck quantity deserves naming: the depth-two gap of the h4 anchor has
sat in the 0.066–0.070 band for the baseline and every candidate across
five generations, while depth-three resolve — the depth the flagship agent
actually plays — equalizes the same anchors to ≤ 0.0073 (G1 ladder). The
plausible next levers, in evidence order: a trunk-capacity probe (the
`train_capacity_probe_128x2` recipe exists for exactly this question), or
a rules-level decision by the maintainer about whether the depth-two bar
measures the deployed agent — which is not a change this goal may make.


## The promotion goal — champion or ceiling (frozen 2026-07-29)

Standing state when this goal was frozen: generation five
(`class_head_gen5/best.pt`) is the best candidate — seven of eight G3 gates
pass (`promotion_gen5.json`). The sole failing gate is anchor-worst-gap
improvement, bound entirely by the depth-two gap of anchor
`(239, 0, 0, 240)` h4: candidate 0.0676 against the 0.0624 bar (baseline
0.0694 × 0.9). That gap has held 0.066–0.070 for every checkpoint since the
baseline, while depth-three resolve equalizes the same anchor to ≤ 0.0073.

This goal has exactly two exits, both terminal; iterating past the budget
is a failure of the goal, not diligence.

- **Exit A — champion.** A candidate passes every G3 gate, unmodified,
  through `readiness.py promotion`. Then finish the promotion: point the
  arena default dth checkpoint at the new champion, re-run the G4
  paired-seat league and the latency/certified-play smoke with it, record
  every verdict here, and update the measured-status table.
- **Exit B — ceiling.** The lever budget exhausts without a promotion.
  Then record the measured band of the h4 depth-two gap across all
  attempts alongside the existing five-generation history, and close by
  putting the rules question to the maintainer: should a depth-two bar
  gate an agent that plays depth-three resolve inside its budget? Changing
  the gate, its threshold, or the promotion code is out of scope.

Budget: at most **six** full candidate evaluations (train → depth-two
ladder → promotion), spent on these levers in evidence order:

1. **Capacity probe.** A 128×2 trunk (precedent:
   `train_capacity_probe_128x2.yaml`) trained on the generation-five corpus
   with the generation-five leaf-layer replay groups, class head and play
   head enabled. No migration path exists from 64-wide checkpoints, so
   stage it like the M1 baseline: a from-scratch rows stage, then the
   decision recipe. This lever is first because fit at the anchor leaf band
   plateaued at 64 wide under three independent data levers.
2. **Horizon-batch emphasis.** The existing `training.horizon_batch`
   counts knob weighted toward the h2/h3 leaf horizons, applied to the
   best trunk from lever one (or to generation five if lever one
   regresses).
3. **Init lineage.** The best recipe so far retrained from scratch
   (rows stage, then decision stage) instead of fine-tuned from the
   generation-four lineage, in case the fine-tuning chain itself carries
   the plateau.

Standing rules, restated because a fresh session will read this cold:
gates, tolerances, and readiness thresholds are untouchable; exact
authority stays finite-only and play rows never supervise the finite head;
every experiment runs through a versioned config; artifacts stay
gitignored; each attempt's verdict is recorded here before the next
attempt starts.

### Attempt 1 — capacity probe, 128x2 trunk (recorded 2026-07-30)

Configs `train_capacity_rows_v1` (from-scratch rows stage, both heads live
from the first step because the corpus carries resolve-play rows) and
`train_class_head_capacity_v1` (the generation-five decision recipe
unchanged: same corpus, same leaf-layer replay groups, same thresholds;
only trunk width and init differ). Stage A stopped at epoch 57 selecting
epoch 7; stage B selected epoch 74.

Verdict (`promotion_capacity_v1.json`): **no-promote, and the binding gate
moved the wrong way.** Seven of eight gates pass again, but anchor worst
gap *regressed* to 0.07624 against the baseline's 0.06937 — an improvement
fraction of −9.9% where +10% is required — driven entirely by the h4
anchor. Everything else improved: evaluation median gap 0.0837 → 0.0210
(74.9%, tied for the campaign's best), evaluation max 0.2794 → 0.2153,
both h5 anchors within tolerance, value-error regression passing.

The interesting part is what this falsifies. Width tripled the finite
head's fit — validation finite value MSE 0.0032 against generation five's
0.0099, with play 0.0048 and class 0.0072 — and the h4 anchor's depth-two
gap still got *worse*. Fit at the anchor leaf band was the standing
diagnosis after generation five disproved coverage; attempt one says the
h4 depth-two gap does not track finite-head accuracy either. Two of the
three recorded explanations for this quantity are now spent.

Because lever one regressed, lever two applies the horizon-batch knob to
generation five, per its own conditional.

### Attempt 2 — horizon-batch emphasis on generation five (recorded 2026-07-30)

Config `train_class_head_horizon_v1`: generation five fine-tuned with
`training.horizon_batch` composing every batch as 24/96/95/40/1 across
horizons 1–5 over 400 batches per epoch, so the h2 and h3 finite leaf
layers take about 37% of the gradient mass each instead of the ~20%
inverse-count loss balancing gives them. Counts cover exactly the training
horizons, h5 included with its single row. Selected epoch 58, dev-root
worst gap 0.2425.

Verdict (`promotion_horizon_v1.json`): **no-promote, six of eight gates.**
Anchor worst gap 0.06758 — a 2.6% improvement against the required 10%,
landing within 0.00005 of generation five's own 0.06762. Evaluation median
improved 46.2% and max fell to 0.1999, but the anchor value-error gate
broke this time: the mirror h5 anchor regressed +0.0158 against the 0.01
allowance while its gap fell to 0.0001, the same trade the campaign has
seen since generation one.

Emphasising the leaf horizons the anchor resolves query moved the anchor
gap by 0.00004. Together with attempt one this is now three independent
data levers and one capacity lever that leave this quantity inside the
same band.

### Attempt 3 — init lineage from scratch (recorded 2026-07-30)

Configs `train_lineage_rows_v1` and `train_class_head_lineage_v1`: the
generation-five decision recipe unchanged at its own 64 width, but
initialised from a from-scratch rows stage on the generation-five corpus
instead of the chain that runs generation five ← four ← … ← the M1
baseline. Holding width at 64 also makes this the width control for
attempt one. Rows stage stopped at epoch 57 selecting epoch 7; decision
stage selected epoch 66 with dev-root worst gap 0.2103, the closest any
attempt here came to generation five's 0.2078.

Verdict (`promotion_lineage_v1.json`): **no-promote, seven of eight
gates**, and the binding gate regressed hardest of all: anchor worst gap
0.08270, an improvement fraction of −19.2%. Everything else was the
campaign's best: evaluation median gap 0.0837 → 0.0139 (83.4%),
evaluation max 0.2010, anchor value-error regression passing.

The width control is worth keeping: the 64-wide from-scratch rows stage
reached validation finite value MSE 0.003173 against the 128-wide stage's
0.003128. Width bought 1.4% of corpus fit. Whatever holds the h4 anchor,
it is neither trunk capacity nor initialisation lineage.

### Attempt 4 — the two non-capacity levers composed (recorded 2026-07-30)

Config `train_class_head_lineage_horizon_v1`: lever two's leaf-horizon
batch emphasis applied to lever three's from-scratch trunk rather than to
generation five, so the anchor is measured under leaf emphasis on a
second, independent trunk. Selected epoch 39, dev-root worst gap 0.2192.
One run died mid-training on a transient `cudaErrorLaunchFailure` and was
restarted; the restart reproduced every logged epoch bit-for-bit through
the failure point, so the fault was the accelerator, not the recipe.

Verdict (`promotion_lineage_horizon_v1.json`): **no-promote, six of eight
gates.** Anchor worst gap 0.07655 (−10.4%), evaluation median 79.2%,
evaluation max 0.1996. The composition inherits the from-scratch trunk's
anchor regression rather than the emphasis lever's neutrality.


## Exit B evidence

### The measured band of the h4 anchor's depth-two gap

Every promotion attempt in the campaign, sorted by the quantity that
decides the gate — the depth-two saddle gap of anchor `(239, 0, 0, 240)`
at horizon 4. The bar is 0.06243 (baseline 0.06937 × 0.9).

| Attempt | h4 depth-two gap | worst-gap improvement | eval median improvement |
|---|---:|---:|---:|
| gen2 | 0.06569 | +5.30% | 49.13% |
| gen3 | 0.06672 | +3.81% | 66.51% |
| gen1 | 0.06705 | +3.34% | 69.68% |
| gen5b | 0.06726 | +3.04% | 21.30% |
| horizon_v1 | 0.06758 | +2.58% | 46.21% |
| gen5 | 0.06762 | +2.52% | 64.54% |
| gen4 | 0.07032 | −1.38% | 74.87% |
| v2 | 0.07448 | −7.36% | 97.26% |
| v1 | 0.07505 | −8.19% | 94.71% |
| capacity_v1 | 0.07624 | −9.90% | 74.90% |
| lineage_horizon_v1 | 0.07655 | −10.36% | 79.20% |
| lineage_v1 | 0.08270 | −19.22% | 83.37% |

Twelve attempts, band 0.06569–0.08270, best improvement 5.30% against a
required 10%. The levers that have now been spent on this one number:
three coverage and labeling generations, value-semantics separation, an
exact anchor closure, leaf-layer replay at two weights, trunk capacity,
leaf-horizon batch emphasis, and initialisation lineage. Two of those
disproved their own premise outright — the closure showed coverage was
never missing, and the width control showed the 64-wide and 128-wide
trunks fit the corpus to within 1.4% of each other.

The campaign also shows a consistent tension the maintainer should see:
across those twelve attempts the h4 depth-two gap correlates *positively*
with evaluation-median improvement (Spearman 0.73, n = 12). Every attempt
that improved the evaluation roots most — v2 at 97.3%, v1 at 94.7%,
lineage_v1 at 83.4% — scored worst on this anchor, and the attempt with
the best anchor (gen2) had the second-weakest evaluation improvement.
This is an observed association over non-independent attempts, not a
causal claim, but no attempt has yet escaped it.

### What depth-three resolve says about the same anchor

`class_head_gen5_d3.json` measures the standing best candidate on the
same 11-root pack at the depth the flagship agent actually plays inside
its 2.0 s budget (G6 measured p95 0.609 s at that depth):

| Root | h | depth-two gap | depth-three gap |
|---|---:|---:|---:|
| `(239,0,0,240)` | 4 | 0.06762 | **0.00588** |
| `(239,0,0,240)` | 5 | 0.00960 | 0.00557 |
| `(0,240,239,0)` | 5 | 0.00000 | 0.00000 |

Whole-pack maximum at depth three is 0.0196, with zero LP fallbacks. The
anchor that blocks promotion is 11.5× smaller at play depth than at the
depth the gate scores.


### The pathology check — run 2026-07-30

Run after the four attempts, on the standing best candidate. Method: rebuild
the depth-two resolve at the h4 anchor with the ladder's own seeds
(`make_node`, `expand`, `warmup_node` at 3600 cells, `max_depth=2`,
`NetworkEvaluator` on `class_head_gen5/best.pt`), take the root's
`mean_q_matrix`, and compare it against `payoff_from_exact_targets` for the
same root. The reproduced gap is 0.06762293792521135, identical to the
ladder record, so the analysis is of the same object the gate scores.

**The exploitability is diffuse, not pivotal.** Injecting the 100 largest
cell errors into the exact matrix leaves the gap at 2.2e-16; about 500 of
3600 cells are needed to recreate it. A random-cell control inverts the
naive expectation — 100 randomly chosen errors give a mean gap of 0.0117
against the largest 100's exactly zero — so the biggest errors sit on cells
no equilibrium touches.

**The anchor is a zero-margin classification.** In the exact matrix 33 of 60
checker replies and 31 of 60 dropper actions are tied at the game value to
within 1e-12, while the remaining actions lose 0.0069–0.1779 (checker) and
0.0083–0.1660 (dropper). Half the action space is exactly optimal and the
other half is badly wrong, with no margin between them. Against that, the
evaluator's cell error is 0.0011 median, 0.0104 mean, 0.0233 on the
equilibrium's own support — straddling the 0.0069 cheapest mistake. The
resolve answers with a 21x21 support where the exact equilibrium is 4x2,
and the gap splits almost evenly across seats (checker 0.0373, dropper
0.0303).

**The maximum over anchors always selects h4.** The same measurement at
`(239,0,0,240)` h5 finds 57 of 60 checker replies and 58 of 60 dropper
actions tied, with a worst pure deviation against equilibrium play costing
0.0062 — a nearly flat payoff surface, not a bound on exploitability
(uniform play is exploitable by 0.096 at that root). Empirically, across
the thirteen checkpoints measured at depth two, the two h5 anchors stayed
within 0.0000–0.0234 while the h4 anchor never left 0.0657–0.0827. The
maximum is therefore always attained at h4, which makes
`anchor_worst_gap` a single-state test in practice.

**The gate asks for most of the quantity's dynamic range.** That h4 band
is 0.0170 wide across every checkpoint the campaign produced, from the M1
baseline to a 128-wide trunk to a from-scratch lineage. The gate requires
a 0.00694 improvement — 41% of the total observed variation of the
quantity, in one direction, at one state.

This refutes the artifact hypothesis (reading 3 of the open decision in
[`AGENT_GOAL.md`](AGENT_GOAL.md)) and refines the
weak-evaluator hypothesis (reading 1). The evaluator's typical cell is well
inside tolerance; it fails only where a tied-optimal set abuts a losing set
across a zero-margin boundary, which carries no gradient to learn from.
That is consistent with every result in the campaign: no data or capacity
lever sharpens a step function, and depth-three resolve avoids the problem
by reaching horizon-one leaves whose values are terminal expectations.

The diagnostic ran from a scratch script, not a repository module, per this
subtree's rule against one-off experiment modules; the method above is
enough to reproduce it.


### A backtested candidate replacement (recorded 2026-07-30, not adopted)

Swapping the aggregate alone is not safe: CVaR(alpha 0.5) over the full
eleven-root pack improves 14–38% for *every* checkpoint ever measured,
including `promo_candidate` and `promo_candidate_v2`, which regressed the
h4 anchor. A 10% bar on that aggregate alone is a rubber stamp.

Pairing it with a firewall restores strictness. Under "CVaR(0.5) over all
eleven roots improves ≥ 10% **and** no single root regresses by more than
0.01" — the same tolerance `anchor_gap_regression` already applies to the
three anchors, extended pack-wide — eleven of the twelve candidates are
rejected, most of them on the evaluation root `(119,120,179,60)` h3
rather than on an anchor. Only `class_head_lineage_horizon_v1` passes,
at CVaR +37.8% with a worst per-root regression of 0.0096.

Two properties recommend this shape over the current one: the aggregate
has real dynamic range (0.0913–0.1477 across checkpoints, against the h4
anchor's 0.0170), and the no-regression clause, not the improvement
clause, does the rejecting — which is the conservative direction.

Whether it should also move to play depth is separate and untested: no
depth-three ladders exist for the candidates, so that variant cannot be
backtested without running them.

Changing the gate, its threshold, or the promotion code was out of scope
for this goal, and none of it was touched. The six new configs are
`train_capacity_rows_v1`, `train_class_head_capacity_v1`,
`train_class_head_horizon_v1`, `train_lineage_rows_v1`,
`train_class_head_lineage_v1`, and
`train_class_head_lineage_horizon_v1`; every artifact named above is
regenerable from them through the commands in
[`WORKFLOWS.md`](WORKFLOWS.md).

