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
| G3 | **fail — no-promote, twice** | `promotion_v1.json`, `promotion_v2.json`: evaluation median gap improved 94.7% / 97.3% and every infrastructure gate passed, but the anchor-improvement criteria bind on anchor gaps of at most 0.075 that depth-2 resolve has already equalized. No champion is frozen; the flagship default remains `depth_baseline_v1`. |
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
