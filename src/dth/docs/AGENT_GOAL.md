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


## Standing state — closed at Exit B (2026-07-30)

The champion-or-ceiling promotion goal is closed at its ceiling exit. **No
champion is frozen**; the flagship default remains `depth_baseline_v1`, and
generation five (`class_head_gen5/best.pt`) is the best candidate at seven of
eight G3 gates. Twelve promotion attempts across seven distinct levers left the
binding quantity in a 0.06569–0.08270 band against a 0.06243 bar.

One decision is open and is the maintainer's: see
[the open decision](#the-open-decision) below. The full campaign record —
evidence basis, five generations, four attempts, and the diagnostics that
closed it — is in [`PROMOTION_CAMPAIGN.md`](PROMOTION_CAMPAIGN.md).

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
| G3 | **fail — no-promote, twelve times; closed at Exit B 2026-07-30** | `promotion_v1.json` … `promotion_lineage_horizon_v1.json`: evaluation median gap improved 21–97% and every infrastructure gate passed, but anchor-worst-gap improvement never reached 10% — best 5.30%, band 0.06569–0.08270 on the h4 anchor. No champion is frozen; the flagship default remains `depth_baseline_v1`. The band, the depth-three comparison, and the diagnosis are in [`PROMOTION_CAMPAIGN.md`](PROMOTION_CAMPAIGN.md); the open rules question is below. |
| G4 | **fail** for the deployed default | `arena_league_bucket12_v2.json`: 24–26, SPRT accept-h0. `arena_league_bucket6_v1.json`: 30–30. Strong seat asymmetry both ways (≈60–67% dropping first, ≈33–36% second). |
| G5 | reported per match | Every match prints certified-move fraction and provenance counts; 100% by construction once both STs ≥ 240 via `exact_band_v1.sqlite`. |
| G6 | **pass** | `arena_latency_smoke_v1.json`: p95 0.609 s, max 1.024 s against the 2.0 s budget, after the predictive depth gate landed. |

One durable exact claim was produced on the way: the complete failure-dead
quotient band is solved and certified, giving the complete-game value
`V(240, 0, 240, 0) = 0.3372132166291093` at saddle gap 1.7e-16
(`exact_band_v1_report.json`, Bellman-recertified on reopen).


The diagnosis behind the G3 row is recorded in
[`PROMOTION_CAMPAIGN.md`](PROMOTION_CAMPAIGN.md): the gating h4 anchor is a
zero-margin equilibrium boundary — 33 of 60 checker replies and 31 of 60
dropper actions tied at the game value to within 1e-12, the rest losing up to
0.18 — which no learned evaluator places reliably, and which no data or
capacity lever sharpened. G2's leaf-depth imbalance and G4's seat asymmetry
remain open on their own terms.

## The open decision

The goal took **Exit B**, its ceiling exit. Four candidate evaluations covered
all three sanctioned levers and the composition of the two that did not act
through capacity; none produced a promotion, and none brought the binding gate
within reach. What remains is one question, and it belongs to the maintainer
because answering it means changing a gate.

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

The pathology check recorded in [`PROMOTION_CAMPAIGN.md`](PROMOTION_CAMPAIGN.md)
narrows what is still open. It refuted the
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
   `anchor_worst_gap` is a maximum over three anchors whose levels never
   overlap, so it is a single-state test on the h4 anchor, and it demands
   41% of that state's entire observed dynamic range. That may be more
   sensitivity than a promotion decision should rest on.

A candidate replacement for the gate's aggregate was backtested against all
thirteen measured checkpoints and is recorded in
[`PROMOTION_CAMPAIGN.md`](PROMOTION_CAMPAIGN.md). It is not adopted, because
adopting it is exactly the decision above.

## The exactness alternative — verified 2026-07-30

The campaign's diagnosis argues against buying leaf quality with training at
all: a zero-margin boundary has no gradient, so the levers this goal is
allowed to pull cannot reach it. The alternative is to make the leaves exact
instead, which the recorded closure projection — 3.7–5.1 years, ~8.1 TB —
appeared to forbid. That projection is priced on a design that three
measurements say need not be used.

- **Per-player TTD-dead quotient, verified at 1-second granularity.**
  `survive_injection(s, t)` is exactly `s <= 239 and s + t <= 240` over all
  72,600 profiles; all 55,889 failing profiles have `revival_model == 0.0`;
  and no failing profile is revived by any ST increase, which is the only
  motion available to it. So a dead player's TTD may be canonicalized:
  17,011 profiles, **289,374,121 classes against 5,267,489,760 reachable
  live states (18.2x)**. The shipped `failure_dead_quotient` needs *both*
  STs ≥ 240 and delivers 1.04x. Empirically the quotient has zero value
  spread over 47,448 multi-member classes covering 125,314 of 256,553
  certified `(state, horizon)` pairs on disk, while a control that also
  collapses live TTDs finds 63 classes with spread to 0.0558 — so the test
  has power. Locked in `tests/test_dead_ttd_quotient.py`.
- **Equilibrium supports are tiny, and the fast path almost never fires.**
  Over 400 sampled h2/h3 matrices, `solve_full_support_structured_matrix`
  applies to 5%; the other 95% run two generic 60x60 LPs to find equilibria
  of median support 2x2 (84.7% are 6x6 or smaller). Throughput on one core:
  744 solves/s structured, 186 LP. (Superseded 2026-07-30: the double oracle
  measured 45x *slower* — `support_solver.py` records why — and the
  complete-game endgame band is 96% mixed with near-full supports, so the
  h2/h3 support statistics do not transfer there. The speedup that
  materialized is pure-saddle screening, single-LP dual extraction, and the
  full-support equalizer solve.)
- **TTD-layer decomposition** (recorded, not re-verified here) puts the
  whole census in ~142 MB resident, or ~25 MB quotiented.

Re-priced on measured throughput, 289M classes is 18 core-days at today's LP
rate and 4–5 at the structured rate. This is not a claim that closure is
done or scheduled — it is a claim that its price is now an open question
rather than a settled impossibility, and that the next scoping decision
should be made against the re-priced figure. That scoping decision was taken
on 2026-07-30 and is the goal below.

## Goal 2 — certified complete-game closure of the quotient space

Adopted 2026-07-30. Solve every one of the 289,374,121 per-player TTD-dead
quotient classes exactly, as one dense certified artifact
(`dth.backup-tablebase.v1`: a float64 value and a solver-routing byte per
class, digest manifest, no per-class certificate — certificates are
re-derived on demand from stored children, the standard
[`EXACTNESS_PROOF.md`](EXACTNESS_PROOF.md) applies to queried roots). The
solver is the descending-potential backup sweep of `dth.backup_tablebase`
(Hydra entry `dth backup`, config `backup_full_v1`), with the Rust kernel
`dth_backup_rs` behind the parity contract
[`DTH_BACKUP_PARITY.md`](DTH_BACKUP_PARITY.md). The existing SQLite interval
pipeline is untouched and remains the authority for its own artifacts.

The closure claim is valid **only** when all four gates pass on the finished
canonical artifact:

- **BG1 — completeness.** Every class value is finite, in `[-1, 1]`, with a
  known routing byte; the finalize scan refuses anything else.
- **BG2 — sampled recertification.** Deterministic per-layer samples
  re-derive their matrices from stored children and re-solve within the
  frozen 1e-6 saddle tolerance.
- **BG3 — external anchors.** The artifact reproduces all 3,541
  `exact_band_v1` classes within certificate width and
  `V(240,0,240,0) = 0.3372132166291093`; the independent
  `build_dead_band_reference` agrees on the full dead-dead band.
- **BG4 — backend parity.** Every gate of
  [`DTH_BACKUP_PARITY.md`](DTH_BACKUP_PARITY.md) is green for both the
  Python authority and the compiled kernel.

Measured markers (2026-07-30, build in progress): the top 150 real layers
solve with zero LP calls (the full-support equalizer rung takes 96% of them),
the swept root matches the band value to 6.9e-12, and the Rust kernel
reproduces the Python authority bit for bit at 59.7x its speed. The claim
covers exactly the addressable domain — alive TTDs in `{0} | [60, 300]`,
which is transition closed; off-domain states fail closed at lookup.

## Claim boundary

- The bounded-resolve goal claims no complete-game L1 solution; the root
  interval narrows only through the interval machinery of
  [`EXACTNESS_PROOF.md`](EXACTNESS_PROOF.md), never through play results.
  Goal 2's complete-game closure claim exists **only** once BG1–BG4 pass on
  a finished canonical artifact, and covers only the transition-closed
  quotient domain; until then, nothing claims a complete-game solution.
- No global exploitability claim. Strength claims use the whitepaper's four
  categories — exact, bounded, approximate, empirical — and per-move
  provenance; a promotion report may say no sampled case is certified worse,
  not that the agent is globally better.
- Network outputs never provide leaf values or bounds for certification, and
  the agent never weakens a firewall, tolerance, or schema check to pass a
  gate.
