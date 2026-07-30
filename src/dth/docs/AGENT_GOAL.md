# DTH agent goal

This file owns the end goal of `src/dth`: the deliverable, the hypothesis it
bets on, the gates that define done, and the milestone order. It does not own
rules or proofs. The model lives in [`GAME_AND_SOLVER.md`](GAME_AND_SOLVER.md),
the exactness boundary in [`EXACTNESS_PROOF.md`](EXACTNESS_PROOF.md), commands
and exact-result status in [`WORKFLOWS.md`](WORKFLOWS.md), and the claim ladder
in [`FORMULATION_LADDER.md`](../../../docs/FORMULATION_LADDER.md). The error
bounds cited below are the matrix-value Lipschitz and saddle-gap propositions
in the whitepaper
(`docs/papers/whitepaper/stl_solver_whitepaper.tex`).

## Goal

Ship the flagship playable agent for pure DTH (rung L1) behind
`arena play --hal-agent dth`: a **certified bounded-resolve player**. On every
move it expands the full 61-class stage game to the deepest horizon its
per-move budget allows, solves every interior matrix exactly, evaluates the
frontier with exact values wherever they exist and a learned evaluator only
where they do not, and emits a per-move provenance record naming what was
proven about that move. The agent grows the certified tablebase along the
states it actually visits.

A complete-game solution of L1 is explicitly **not** the goal. The exact
tier's role under this goal is to certify the play frontier, not to close the
game.

## Hypothesis

> A per-move full-width depth-limited resolve — expand all 61 transition
> classes to depth `h`, solve every interior matrix exactly, and evaluate
> frontier states by (1) committed exact values, (2) the solved failure-dead
> quotient band, (3) a calibrated 61-output transition-class network — plays
> with one-step saddle gap at most `2·ε_h` at every visited state, where
> `ε_h` is the frontier value error. `ε_h` falls with depth because every
> live edge strictly increases raw damage, so deeper frontiers sink toward
> the certified near-terminal region where the error is exactly zero.

Consequences the milestones test directly:

- Depth, not simulation count, is the search variable that matters. Sampled
  joint-cell MCTS is demoted to self-play data ordering and to depths beyond
  the full-width budget; it is not the play-time algorithm.
- The network's job is 61 numbers per state. With a transition-class head,
  matrix error is bounded by the worst class error and the `2·ε` gap bound
  applies to exactly the quantity the loss optimizes. This is the head the
  regeneration plan already concluded must be tried before further LP time
  ([`REGENERATION_PLAN.md`](../../../docs/REGENERATION_PLAN.md)).

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

## The agent contract

Play-time behavior, in priority order at each decision:

1. **Exact lookup.** If the state has an exact committed value with a
   reconstructable policy (including the failure-dead quotient band), play it.
   Provenance: `complete-game-exact`.
2. **Bounded resolve.** Otherwise run iterative-deepening full-width class
   expansion under a wall-clock budget the agent owns: interior matrices are
   solved by `solver.solve_matrix` (structured path, LP fallback); frontier
   leaves come from exact store, then quotient band, then the network.
   Provenance: `finite-horizon-exact(h)` when every frontier leaf is exact,
   else `approximate(h, gap)` with the measured root saddle gap.
3. **Deposit.** Optionally commit deadline-bounded durable closures for
   visited states through the existing `ExactDTHAgent` budgets, so exactness
   accumulates where play actually goes. This is the sanctioned direction of
   [`WORKFLOWS.md`](WORKFLOWS.md): approximate methods may order selective
   expansion; they never provide certification.

Integration rules the agent must satisfy (owned by `arena/AGENTS.md` and the
root instructions):

- The provider returns a mixed policy as a weight map; only arena's
  `PolicyDrivenAgent` masks and samples. The provider never samples and never
  raises past its own fallback ladder — arena has no timeout and no exception
  handling.
- The projection from `CanonicalDecision` is the identity at one-second
  resolution plus defensive `int()` and range clamps.
- Documented approximations, following the abstract adapter's precedent: the
  opponent's leap-window second 61 is unmodeled; the live STL engine's
  revival surface is the pre-freeze model until its pending migration
  ([`REVIVAL_MODEL.md`](../../../docs/REVIVAL_MODEL.md)) — same eligibility
  zero-set, different probabilities.
- Artifacts stay under `src/dth/`; `dth` never imports `arena`; no
  auto-build of exact artifacts from arena.

## Definition of done

| Gate | Condition |
|---|---|
| G1 search effectiveness | At a fixed checkpoint, the upper-tail (CVaR, alpha 0.5) saddle gap on the readiness root pack strictly decreases across resolve depths 1 → 2 → 3 and max gap does not increase. Amended after the first measured ladder (2026-07-28): the originally gated median is zero-inflated — several roots sit at exactly zero gap at depth one and deeper resolve spreads sub-0.03 gaps onto them while every materially wrong root improves — so the gate uses the repository's existing CVaR tail vocabulary and the median stays report-only. The LP-fallback path in `mcts.py` `refresh_policies` is counted and reported; ladders with uncounted fallbacks are invalid. |
| G2 orientation consistency | The paired-orientation audit runs end to end; worst orientation-pair gap ratio ≤ 1.5 (provisional threshold, revisit with the first result). |
| G3 promotion | `readiness.compare_ladders` runs end to end at least once and the champion is frozen through it; all its existing gate conditions pass unmodified. |
| G4 empirical strength | Paired-seed arena league with a predeclared SPRT: beats the `abstract` bucket12 adapter and the scripted opponents in both available seatings. |
| G5 certified play | Every arena game reports the fraction of moves carrying `complete-game-exact` or `finite-horizon-exact` provenance; 100% by construction once both ST coordinates are ≥ 240. |
| G6 latency | p95 per-move wall time within the agent's declared budget (default 2.0 s) on CPU in arena. |

Done means: adapter merged, champion frozen through G3, all six gates green,
and the claim boundary below intact.

## Milestones

**M1 — make depth real.** Materialize a reference checkpoint from the v19
recipe (from-scratch init if the `strategic_mixed_v7` lineage is
unrecoverable) and freeze it as baseline. Extend the depth-gate config into a
depth ladder (depths 1–3, fixed checkpoint, fallback counting) and run it.
Exit: G1 measured — pass or fail. A fail falsifies the hypothesis cheaply and
stops this file from wasting further effort.

**M2 — transition-class head and orientation.** Add the 61-output
continuation-class head to `network.py`, reconstruct matrices through
`reconstruct_transition_class_matrix`, train with the existing
Bellman-composition objectives, and select on the M1 ladder. Run the
paired-orientation probe. Before enforcing any symmetry architecturally,
test empirically against exact finite-horizon values whether an algebraic
mirror-value identity holds — none is currently frozen, and none may be
assumed. Exit: a checkpoint that beats the M1 baseline on the depth ladder,
plus a recorded G2 audit.

**M3 — the agent module.** Implement the bounded-resolve player inside
`src/dth` on top of `exact_agent.py`'s deadline machinery: iterative
deepening, leaf priority, provenance record, optional closure deposits.
Exit: unit tests for budget adherence, provenance correctness on states with
known scope, and determinism at fixed seed.

**M4 — arena wiring.** `arena/dth_adapter.py` plus the two `arena/cli.py`
dispatch edits (add `"dth"` to the choices and an explicit branch before the
stl-mcts fallback), flags for database path, checkpoint, deadline, and
fallback horizon, with the three documented approximations and monkeypatch
tests in the existing arena style. Exit: `arena play --hal-agent dth`
completes full matches; G5 and G6 reported per game.

**M5 — first promotion and champion freeze.** Run the promotion pipeline end
to end (M2 candidate vs M1 baseline), freeze the champion, then and only
then loop self-play for data ordering. Exit: G3 and G4 green; goal met.

## Measured status — recorded 2026-07-29

Every milestone was executed and every gate measured; artifacts are gitignored
so the verdicts are snapshot here with their artifact names.

| Gate | Verdict | Evidence |
|---|---|---|
| G1 | **pass** — depth-effective | `depth_gate_v1.json`: CVaR gap 0.1675 → 0.1477 → 0.0033 and max gap 0.2844 → 0.0073 across depths 1–3, zero LP fallbacks. |
| G2 | **pass at play depth, fail at leaf depth** | `orientation_gate_baseline_d3.json`: consistent, every pair ≤ 0.0073. At depth-2 leaves the imbalance is real for every checkpoint (`orientation_gate_*_d2.json`, ratios to 100×). The paired probe ran for the first time: single-orientation training raised held-out mirror value MSE 0.053 → 0.133. No algebraic mirror identity exists (43,189 exact pairs checked). |
| G3 | **fail — no-promote, twelve times; closed at Exit B 2026-07-30** | `promotion_v1.json` … `promotion_lineage_horizon_v1.json`: evaluation median gap improved 21–97% and every infrastructure gate passed, but anchor-worst-gap improvement never reached 10% — best 5.30%, band 0.06569–0.08270 on the h4 anchor. No champion is frozen; the flagship default remains `depth_baseline_v1`. See "Exit B — ceiling" below for the band, the depth-three comparison, and the open rules question. |
| G4 | **fail** for the deployed default | `arena_league_bucket12_v2.json`: 24–26, SPRT accept-h0. `arena_league_bucket6_v1.json`: 30–30. Strong seat asymmetry both ways (≈60–67% dropping first, ≈33–36% second). |
| G5 | reported per match | Every match prints certified-move fraction and provenance counts; 100% by construction once both STs ≥ 240 via `exact_band_v1.sqlite`. |
| G6 | **pass** | `arena_latency_smoke_v1.json`: p95 0.609 s, max 1.024 s against the 2.0 s budget, after the predictive depth gate landed. |

One durable exact claim was produced on the way: the complete failure-dead
quotient band is solved and certified, giving the complete-game value
`V(240, 0, 240, 0) = 0.3372132166291093` at saddle gap 1.7e-16
(`exact_band_v1_report.json`, Bellman-recertified on reopen).

The open problems, in the order the evidence ranks them: the promotion
anchors need a candidate that improves them at depth-2 resolve rather than
only the evaluation roots; the leaf-level orientation imbalance persists
through orientation-balanced training; and the flagship needs stronger leaves
or deeper in-budget resolve before it decisively beats the bucketed exact
tablebases from both seats.

### Expert iteration — generations one to three, recorded 2026-07-29

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

## Exit B — ceiling (closed 2026-07-30)

The goal takes **Exit B**. Four candidate evaluations covered all three
sanctioned levers and the composition of the two that did not act through
capacity; none produced a promotion, and none brought the binding gate
within reach. No champion is frozen. The flagship default remains
`depth_baseline_v1`, and generation five (`class_head_gen5/best.pt`)
remains the best candidate at seven of eight gates.

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

**The h5 anchors cannot discriminate.** The same measurement at
`(239,0,0,240)` h5 finds 57 of 60 checker replies and 58 of 60 dropper
actions tied, with a worst mistake of 0.0062. Maximum available
exploitability there is about 0.006, which is why every checkpoint passes
those two anchors. `anchor_worst_gap` is therefore a single-state gate in
practice: the h4 anchor is the only one of the three carrying strategic
content.

This refutes the artifact hypothesis (reading 3 below) and refines the
weak-evaluator hypothesis (reading 1). The evaluator's typical cell is well
inside tolerance; it fails only where a tied-optimal set abuts a losing set
across a zero-margin boundary, which carries no gradient to learn from.
That is consistent with every result in the campaign: no data or capacity
lever sharpens a step function, and depth-three resolve avoids the problem
by reaching horizon-one leaves whose values are terminal expectations.

The diagnostic ran from a scratch script, not a repository module, per this
subtree's rule against one-off experiment modules; the method above is
enough to reproduce it.

### The rules question for the maintainer

**Should a depth-two bar gate an agent that plays depth-three resolve
inside its budget?**

The G3 promotion gate compares candidates at `mcts.max_depth=2`. The
deployed agent resolves to depth three within its declared budget, and
G1 established that depth is the search variable that matters: the same
anchors that sit at 0.066–0.083 at depth two equalize to ≤ 0.0073 at
depth three (`depth_gate_v1.json`), and generation five reaches 0.00588
there. Twelve attempts have failed to move the depth-two number by the
required margin while five of them improved the evaluation roots by 65%
or more.

The pathology check above narrows what is still open. It refuted the
artifact reading — the exploitability is diffuse across hundreds of cells,
not one pivot — and it refined the weak-evaluator reading, because the
evaluator's typical cell error (0.0011) is well inside tolerance and only
its behaviour at a zero-margin boundary fails. What remains is a question
about the standard, not about the facts:

1. Demanding equilibrium-support recovery on a zero-margin boundary is
   the right promotion standard. Half the h4 anchor's action space is
   exactly optimal and half loses up to 0.18, with nothing in between; a
   learned evaluator with 0.0104 mean cell error cannot place that
   boundary, and an agent that cannot should not be promoted. Then the
   ceiling is real, no training lever will pass it, and the next work is
   a different evaluator class — exact values, or a certified bound at
   the leaf — not another training generation.
2. The bar measures the wrong configuration — the deployed agent never
   plays depth two, and at depth three the same anchor is 0.00588 because
   its frontier reaches horizon-one leaves whose values are terminal
   expectations. Then the gate should be re-declared at play depth. That
   is a change to the gate, explicitly out of scope here.
3. The gate's aggregate is the problem rather than its threshold.
   `anchor_worst_gap` is a maximum over three anchors, two of which admit
   at most 0.006 of exploitability and therefore cannot discriminate
   between checkpoints. The bar is in practice a single-state test, which
   may be more sensitivity than a promotion decision should rest on.

Changing the gate, its threshold, or the promotion code was out of scope
for this goal, and none of it was touched. The six new configs are
`train_capacity_rows_v1`, `train_class_head_capacity_v1`,
`train_class_head_horizon_v1`, `train_lineage_rows_v1`,
`train_class_head_lineage_v1`, and
`train_class_head_lineage_horizon_v1`; every artifact named above is
regenerable from them through the commands in
[`WORKFLOWS.md`](WORKFLOWS.md).

## Claim boundary

- Nothing under this goal claims a complete-game L1 solution; the root
  interval narrows only through the interval machinery of
  [`EXACTNESS_PROOF.md`](EXACTNESS_PROOF.md), never through play results.
- No global exploitability claim. Strength claims use the whitepaper's four
  categories — exact, bounded, approximate, empirical — and per-move
  provenance; a promotion report may say no sampled case is certified worse,
  not that the agent is globally better.
- Network outputs never provide leaf values or bounds for certification, and
  the agent never weakens a firewall, tolerance, or schema check to pass a
  gate.
