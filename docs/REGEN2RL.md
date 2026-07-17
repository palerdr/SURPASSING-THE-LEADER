# REGEN2RL: Python-First Bridge to Simultaneous-Move AlphaZero

Status: canonical implementation checklist, executed through P3 and the V4
static P4 holdout. V4 passed its one-use value/policy ruler but its first
learned-MCTS audit used a horizon-one terminal oracle against a multi-step
learned frontier. That run remains rejected, but it does not isolate successor
generalization. The horizon-aware Bellman-closure cycle is implemented; its
full exact generation, model ladder, fresh sealed evaluation, and aligned MCTS
run remain pending. P5 is blocked.

Checklist legend:

- [x] implemented and verified in the current Python working tree;
- [ ] pending, or dependent on a later phase/run;
- a checked implementation item does not imply that later empirical training
  or promotion evidence already exists.

## Current Execution Boundary

- [x] **P0 objective and episode contract.** Versioned 64/128 caps, role
  orientation, sensitivity reporting, Hydra validation, and wrapper tests are
  implemented.
- [x] **P1 Python oracle/search conformance.** Actor-aware `60 x 60` and
  `61 x 60` legality, RNG-safe state restoration, LP/CFR+ parity,
  omitted-action audits, and chance-frequency tests are implemented.
- [x] **P2 reconstructable schemas.** `TrainingRecordV3`, V2 features, safe
  shards, grouped state/episode/feature splits, strict checkpoint bundles, and
  stale-artifact rejection are implemented. V3 supersedes the rejected Gen-0
  V2 rows without changing the V2 feature encoder or V2 checkpoint format.
- [x] **P3 MCTS improvement operator.** Canonical linearly weighted mean-Q
  policies, factorized priors/noise, full-width mode, and the frozen
  convergence gate pass. Report:
  `outputs/regen2rl/p3_mcts_conformance.json`, canonical report-payload SHA-256
  `245037e1bb11d8d2bef0e30a7ae10f78751fe541d24a5c938b6aaa4bae432eb1`.
- [x] **P4 scaled implementation and pilot.** Legal-engine trajectories, exact
  source quotas, independent development/ruler roles, analytic tablebase
  breadth, root-matrix certificates, four-worker chunking, source-balanced
  training, development selection, external evaluation, and evaluator-in-MCTS
  audit commands are implemented and smoke-tested.
- [ ] **STOP: horizon-aware Bellman cycle.** The V4 static ruler is consumed.
  Do not begin P5 until exact V2 successor closure, direct/backed V3 gates, a
  fresh sealed calibration ruler, and the aligned bounded-frontier MCTS audit
  all pass for one selected horizon-aware checkpoint.

This document replaces the action-core reset memo. The reset remains a
non-negotiable input, but it is no longer the roadmap. The roadmap is now to
turn the current exact/search stack into a reproducible AlphaZero-style
policy/value reinforcement-learning system for a fully observed,
two-player, zero-sum game with simultaneous actions and engine chance.

The implementation scope through Phase 8 is Python only. Existing Rust code
may remain as an opt-in parity fixture, but no phase may depend on it and no
new work under `crates/` is part of this plan. A Rust kernel follows only after
the Python baseline meets the Phase 8 handoff gate.

## Source-of-Truth Documents

Every ticket and gate below inherits these contracts:

- `README.md`, especially **Current Action Core**, **Solver Shape**,
  **Learning And Self-Play**, and **Architecture Rules**.
- `stl/engine/README.md` for rule authority and actor-aware legality.
- `stl/solver/README.md` for exact-second minimax, terminal-only utility, and
  the solver firewall.
- `stl/learning/README.md` for the length-62 policy contract and the current
  target/model/training boundary.
- `stl/play/README.md` for the quarantine boundary around bucketed playable
  code.
- `docs/papers/README.md` and `docs/papers/literature_assessment.md` for the
  AlphaZero, simultaneous-move MCTS, CFR, DeepStack, and ReBeL lineage.
- `docs/whitepaper/stl_solver_whitepaper.tex` for the mathematical model,
  guarantees, limitations, and target architecture.
- `tests/README.md` for subsystem ownership of regression tests.

If a phase conflicts with one of these documents, update the contradiction in
the same change. Do not let an experiment silently redefine the engine,
utility, action space, or promotion rule.

## Audit Verdict

### What is already a sound foundation

| Capability | Current evidence | Verdict |
|---|---|---|
| Engine authority | `stl/engine/game.py`, `stl/engine/actions.py`, engine tests | Keep unchanged except for discovered rule bugs. |
| Exact local game | `stl/solver/exact.py` enumerates literal seconds, expands engine chance, and solves LP minimax | Certification oracle for bounded horizons. |
| Search | `stl/solver/search.py`, `conformance.py`, and `mcts_conformance.py` | P1/P3 conformance complete; candidate mode remains approximate and audited. |
| Exact anchors | `stl/solver/tablebase.py` and `stl/learning/targets.py` | Retain as immutable replay strata. |
| Policy/value model | `stl/learning/model.py` emits one Hal value and two length-62 role heads from 52 named V2 features | P2 schema complete; empirical fit remains a P4 gate. |
| Replay/checkpoints | `stl/learning/replay.py` and `train.py` provide reconstructable V3 shards, grouped splits, and strict V2 checkpoint bundles | P2 integrity boundary complete; rejected V2 Gen-0 shards are not migrated. |
| Supervised learner | `stl/learning/train.py` has masked role-policy losses, weighted value loss, grouped selection, and V3 replay loading | The first corrected full candidate trained successfully but failed its external ruler. |
| Evaluation pieces | `stl/commands/gen0_eval.py` now provides the P4 external-ruler gate; calibration, audit, ladder, best-response intervals, SPRT, and promotion modules remain separate | Integrate the post-P4 pieces into one mandatory promotion gate with precise claim labels. |

### Why the current tree is not yet full AlphaZero reinforcement learning

1. The audit found that the former `configs/command/self_play.yaml` dispatched
   to `stl.commands.alphazero_loop` with two parser-invalid flags. Both the
   misleading command and its legacy wrapper were removed in distillation D1.
   There is intentionally no public self-play command until P5 implements the
   joint MCTS actor and passes the command-contract test.
2. `stl/learning/support/self_play.py` runs bucketed `CanonicalHal` against a
   rotating scripted opponent list. It records only `(features, outcome)` and
   never calls matrix-game MCTS. This is useful legacy/league data, not
   AlphaZero self-play.
3. `generate_mcts_bootstrap_targets` searches a hand-built state grid and uses
   the MCTS root estimate as the value label. AlphaZero trains the value head
   against a completed self-play outcome and the policy head against the search
   policy at each visited state.
4. P0--P3 now provide an explicit capped objective, reconstructable replay,
   grouped splits, strict checkpoint bundles, and one canonical MCTS policy
   object. Those are bridge foundations, not evidence that a learned
   Generation-Zero checkpoint or self-play improvement loop exists.
5. The corrected full P4 corpus and first 100-epoch training run completed. The
   external ruler rejected that candidate, and the search-in-the-loop audit has
   not run; therefore Generation Zero is not frozen or promotion-eligible.
6. Promotion can consume policy drift as an optional report without enforcing
   a policy-quality threshold, and its “certified exploitability” evidence is
   a finite-depth best-response interval, not a global exploitability proof.
7. Tests cover solver mechanics and the P0--P4-smoke boundaries, but the
   current self-play test only
   verifies rejection of an illegal action. There is no end-to-end test from
   champion checkpoint through MCTS trajectory, replay shard, training,
   arena, and promotion decision.

Graphify corroborates the integration gap: the active self-play module has no
path to `mcts_search` or `promote`, while `generate_mcts_bootstrap_targets` is
connected directly to MCTS. The first objective is therefore one coherent
Python loop, not more disconnected training scripts.

## Non-Negotiable Mathematical and Software Contracts

1. Treat the game as a fully observed stochastic simultaneous-move game. The
   public state is known; the two current actions are chosen without observing
   one another. Do not model it as an alternating-action AlphaZero game.
2. Keep value in Hal perspective everywhere: terminal Hal win `+1`, terminal
   Baku win `-1`, and a declared training-horizon draw `0`.
3. Preserve literal action indices: `0` is illegal padding, normal actions are
   `1..60`, and only Baku as leap-window dropper may use `61`. Checkers never
   use `61`.
4. A simultaneous policy is a product of two independent legal role
   marginals, `pi_dropper` and `pi_checker`. Do not train or sample a correlated
   joint policy unless the mathematical game definition is deliberately
   changed and re-audited.
5. Exact labels and certification use full-width LP minimax. Candidate sets,
   CFR+, MCTS, neural values, and replay are approximate layers.
6. Engine chance comes only from `stl.engine`. No learned survival model and no
   reward shaping may enter `stl.solver`. The dose and cumulative-TTD boundary
   is frozen in [`CANONICAL_EXTENSIVE_FORM.md`](CANONICAL_EXTENSIVE_FORM.md).
7. Search policy targets use the linearly weighted average of per-iteration
   empirical mean-Q equilibria, not optimism-augmented selection strategies or
   an arbitrary final LP vertex over a nearly flat mean-Q matrix.
8. RL value targets use terminal or explicitly truncated episode outcomes.
   Root search values remain diagnostics or separately tagged bootstrap data.
9. No metric is called “global exploitability” unless it covers the full game.
   Existing best-response outputs must carry scenario, depth, support mass,
   frontier interval, and width.
10. A phase is incomplete until its tests, manifests, and failure behavior are
    checked in. A successful long run without reproducible metadata is not an
    accepted result.

## Phase Dependency Chain

```text
P0 contract freeze
  -> P1 Python oracle/search conformance
  -> P2 reconstructable data and checkpoint schemas
  -> P3 simultaneous MCTS improvement operator
  -> P4 anchor-supervised generation zero
  -> P5 genuine MCTS self-play actor
  -> P6 replay, learner, and reanalysis loop
  -> P7 statistical promotion gate
  -> P8 stable Python baseline
  -> R1 Rust kernel handoff (future plan, not current scope)
```

Phases may overlap only where their entry gates are already satisfied on the
same commit. No long self-play job starts before P5 passes its smoke gate.

## P0 - Freeze the Objective and Episode Contract [x]

### Goal

Define exactly which game the learner optimizes before changing data or model
code.

### Python scope

- `stl/learning/` for the RL wrapper contract.
- `configs/` for a versioned baseline configuration.
- Documentation and tests only; do not change engine termination semantics.

### Work packages

- [x] **P0.1 State the game class.** Document a fully observed, two-player,
  zero-sum stochastic Markov game with simultaneous legal role actions and
  engine chance. Use the whitepaper notation.
- [x] **P0.2 Define `G_H`.** Add a training/evaluation wrapper cap
  `max_half_rounds`. For the first baseline, freeze `H_train=64` and
  `H_eval=128`. A nonterminal state at the cap is a training draw `z=0`, tagged
  `truncated=true`; it is not an engine terminal.
- [x] **P0.3 Define horizon sensitivity.** Every promotion pack reports evaluation
  at both 64 and 128 half-round caps. If the paired score changes by more than
  2 percentage points or policy total variation on shared states exceeds
  `0.10`, the cap is outcome-relevant and must be raised or the objective
  revised.
- [x] **P0.4 Freeze perspective and role mapping.** Specify how Hal/Baku map to
  maximizing/minimizing roles in each half-round and test both halves.
- [x] **P0.5 Freeze baseline config.** Create one resolved config manifest with all
  seeds, MCTS budget, exploration parameters, episode caps, model schema,
  replay mix, and promotion thresholds. No hidden module constants may change
  a run.

### Exit gate

All of the following are required:

- [x] Whitepaper and README terminology says “fully observed simultaneous-move”
  and does not call the state imperfect-information.
- [x] Tests demonstrate an infinite-safe-line fixture is truncated by the wrapper
  without setting `game.game_over`.
- [x] Terminal wins remain `+1/-1`; capped nonterminals are `0` and carry a
  distinct truncation flag.
- [x] A resolved Hydra config round-trips to a canonical JSON form and stable hash.
- [x] Default Python configs select only LP or Python CFR+; no default imports or
  executes `stl_solver_rs`.

### Stop condition

If cap sensitivity fails at both 64 and 128, do not start self-play scaling.
First choose a larger finite game, a discounted objective, or another explicit
adjudication rule and update the proof contract.

## P1 - Certify the Python Oracle and Search Boundary [x]

### Goal

Make the engine, exact solver, tablebase, and approximate search agree on the
rules and expose measurable approximation error.

### Python scope

- `stl/engine/`
- `stl/solver/`
- matching tests under `tests/engine/` and `tests/solver/`

### Work packages

- [x] **P1.1 Action conformance.** Exhaustively compare engine legality,
  `enumerate_joint_actions`, policy masks, and environment actions on normal
  turns and every leap-window role assignment.
- [x] **P1.2 Transition conformance.** For a stratified state/action fixture,
  compare `expand_joint_action` with direct engine resolution, including
  no-death, survival, fatality, same-second success with inclusive ST=1, failed check, overflow,
  and Baku drop `61`.
- [x] **P1.3 Matrix orientation.** On both halves, compare the drop/check matrix
  orientation to an explicitly Hal-row/Baku-column matrix and verify both
  equilibrium marginals and Hal value.
- [x] **P1.4 LP/CFR+ parity.** Keep LP as oracle. On deterministic matrix fixtures
  and sampled exact-second matrices, record CFR+ saddle gap and value error;
  do not accept a runtime speed claim without this error report.
- [x] **P1.5 Candidate audit.** For every state in the tactical pack and a fixed
  stratified random sample, compare candidate search with full width at horizon
  one. Report omitted-action best-response gain, not only value difference.
- [x] **P1.6 Chance sampling.** Across fixed seeds, verify empirical MCTS survival
  frequencies fall inside a 99% binomial interval around the engine
  probability.

### Exit gate

- [x] Normal full-width enumeration is exactly `60 x 60`; the leap Baku-dropper
  grid is exactly `61 x 60`.
- [x] No legal mask activates index `0` or checker index `61`.
- [x] Transition probabilities sum to `1` within `1e-12`, and state restoration is
  exact after every probe.
- [x] LP primal/dual values agree within `1e-9` on all oracle fixtures.
- [x] Python CFR+ value error is at most `0.01` and saddle gap at most `0.02` on
  every frozen parity matrix at its frozen iteration budget.
- [x] Candidate search has no omitted-action gain above `0.02` on the frozen
  tactical audit pack. Any failure either expands candidates or forces
  full-width search for that state class.
- [x] Existing engine, firewall, rigorous exact, MCTS, tablebase, and subgame
  resolve test groups pass without tolerance weakening.

### Required commands

```bash
uv run pytest tests/engine -q
uv run pytest tests/solver/test_cfr_firewall.py tests/solver/test_cfr_rigorous_exact.py -q
uv run pytest tests/solver/test_cfr_candidates.py tests/solver/test_cfr_mcts.py -q
uv run pytest tests/solver/test_cfr_tablebase.py tests/solver/test_subgame_resolve.py -q
```

## P2 - Introduce Reconstructable Data and Checkpoint Schemas [x]

### Goal

Make every training row reproducible, reanalyzable, and rejectable when its
semantics do not match the current engine/model.

### Python scope

- `stl/learning/targets.py`
- a focused replay/schema module under `stl/learning/`
- `stl/learning/model.py` and `stl/learning/train.py`
- learning tests

### Work packages

- [x] **P2.1 `TrainingRecordV3`.** Store feature vector, serialized
  `ExactPublicState`, Hal value target, target kind, both role distributions,
  both legal masks, source, episode ID, half-round index, truncation flag,
  parent checkpoint digest, search-config digest, engine/action/feature schema
  versions, and all RNG seeds.
- [x] **P2.2 Target kinds.** Distinguish at least `exact_value`, `tablebase_value`,
  `search_bootstrap_value`, and `terminal_outcome`. Never infer semantics from
  `source` or overload `horizon` with MCTS iterations.
- [x] **P2.3 Safe shards.** Write numeric arrays plus a JSON manifest with
  `allow_pickle=False`. Write to a temporary filename, validate counts and
  hashes, then atomically rename.
- [x] **P2.4 Exact reconstruction.** Rebuild a `Game` from every stored state and
  require `exact_public_state(rebuilt) == stored_state` before reanalysis.
- [x] **P2.5 Feature schema V2.** Encode all value-relevant fields of
  `ExactPublicState`, then append documented relative/safe-budget features.
  Keep V1 loadable only through an explicit legacy adapter. Checkpoints declare
  their input schema.
- [x] **P2.6 Leakage-safe splits.** Group by exact-state hash and episode ID before
  train/validation split. Replicated anchors remain in one split. The fixed
  external ruler is never sampled into training or replay.
- [x] **P2.7 Checkpoint bundle.** Save model, optimizer, scheduler, model schema,
  parent digest, corpus digests, resolved config, git commit, seed, and training
  history. Loading a bare incompatible `state_dict` must fail loudly unless an
  explicit migration is selected.
- [x] **P2.8 Gen-0 V3 provenance repair.** Preserve exact value horizon and
  cutoff probability separately from Tier A lower/upper value bounds. Record
  state origin, source artifact and digest, and a replayable legal action trace
  for engine-derived rows. Bind shards to a canonical generation plan, the
  actual Python/config source-tree digest, runtime versions, split rule, Tier A
  manifest digest, and ordered parent chunks. V2 Gen-0 shards fail schema load.

### Exit gate

- [x] A 100-record mixed-source fixture round-trips byte-for-byte in manifest data
  and numerically within dtype tolerance in tensors.
- [x] All 100 reconstructed states equal their stored `ExactPublicState`.
- [x] Corrupt shard, mismatched action size, feature schema, or checkpoint parent
  digest fails before training begins.
- [x] Train and validation exact-state hash sets have empty intersection.
- [x] V3 collision audit reports zero divergent-value collisions on the full
  pinned/tablebase corpus. Any remaining collision class is documented and
  routed away from bare-net evaluation.
- [x] Old length-61 or 122-logit artifacts are rejected, as required by
  `stl/learning/README.md`.
- [x] Engine-trajectory rows replay to the stored exact state; impossible forced
  chance outcomes, inconsistent CPR/death/TTD counters, active uncertified Tier
  A policies, and malformed Tier A inventories/arrays fail before publication.

## P3 - Make Matrix-Game MCTS the Tested Improvement Operator [x]

### Goal

Produce one well-defined search policy/value object suitable for play,
training targets, reanalysis, and audit.

### Python scope

- `stl/solver/search.py`
- thin learning adapters only; no learning imports into the solver
- solver and learning search tests

### Work packages

- [x] **P3.1 Canonical root output.** Name the linearly weighted average of
  per-iteration empirical mean-Q role
  strategies `improved_dropper_policy` and `improved_checker_policy`. Keep the
  final mean-Q saddle point as a diagnostic, not the default play/target policy.
- [x] **P3.2 Role-factorized priors.** Mask and normalize dropper/checker priors
  separately, then form their product for joint cell exploration. Assert that
  each marginal is legal and normalized.
- [x] **P3.3 Self-play exploration hook.** Perturb legal role marginals separately
  at the root. The initial frozen choice is `epsilon=0.25` and
  `alpha=10/|A_role|`; record it in the config. Search used for evaluation has
  noise disabled.
- [x] **P3.4 Preserve mixed play.** Do not argmax or temperature-collapse a
  simultaneous equilibrium. Evaluation samples from the noise-free improved
  marginals; self-play samples from the explicitly mixed exploration
  marginals.
- [x] **P3.5 Full-width audit mode.** Add an audit switch that uses all legal
  seconds. Runtime candidate mode remains allowed only with P1.5 omitted-action
  evidence.
- [x] **P3.6 Convergence pack.** For exact horizon-one fixtures, run budgets
  `64, 256, 1024` over seeds `0..9`; report value error, saddle gap, policy
  total variation where the oracle equilibrium is unique, entropy, and cells
  visited.
- [x] **P3.7 Remove ambiguous config.** Eliminate or implement inert
  `MCTSConfig` fields so the resolved config describes actual behavior.

### Exit gate

- [x] The same root policy object is used by `SolverAgent`, target generation,
  reanalysis, and policy audit.
- [ ] P5 genuine self-play must consume that same canonical policy object; the
  legacy scripted-opponent module is not evidence for this item.
- [x] Every returned distribution is finite, nonnegative, legal, and sums to one
  within `1e-6`.
- [x] At 1024 iterations, the frozen convergence pack has median absolute value
  error at most `0.05`, 95th-percentile error at most `0.10`, and saddle gap at
  most `0.05`; no fixture worsens between 256 and 1024 by more than `0.02`.
- [x] Root-value standard deviation across the ten seeds is at most `0.03` on each
  pinned fixture.
- [x] Chance, transposition, and state-restoration tests pass under sanitizing
  assertions.
- [x] Search reports candidate/full-width mode and never labels candidate output
  “exact.”

### Stop condition

If this convergence pack fails, do not compensate with a larger neural net.
Fix the simultaneous selection/backup algorithm or replace it with a better
validated regret-matching SM-MCTS variant first.

## P4 - Train and Freeze Generation Zero from Exact Anchors [ ]

### Goal

Create a reproducible policy/value checkpoint that is good enough to seed
MCTS without claiming reinforcement learning has begun.

### Python scope

- `stl/learning/model.py`
- `stl/learning/train.py`
- exact/tablebase target generators
- calibration and audit modules

### Work packages

- [x] **P4.0 Rejected-pipeline diagnosis and repair.** The original V2 smoke and
  long run proved that the command path executed, but their data failed the
  reachability, terminal-mechanics, split-isolation, Tier A, and provenance
  audit. Obsolete artifacts were removed. The replacement V3 path is
  implemented and tested without publishing a corrected dataset.
- [x] **P4.1 Fixed anchor corpus.** Regenerate terminal, full-width finite-horizon,
  tactical tablebase, and certified Tier A rows under V3 replay schemas. Exact
  and terminal rows must replay from the canonical opening; tactical fixtures
  must pass counter invariants; Tier A rows retain interval bounds and have no
  active policy label. Store corpus, ruler, source-tree, plan, and parent-chunk
  digests.
- [x] **P4.2 Policy labels.** Use LP equilibrium role marginals on exact rows. Rows
  without a certified policy carry an inactive policy-loss mask rather than a
  fabricated uniform target.
- [x] **P4.3 Loss contract.** Train
  `lambda_v * MSE + lambda_d * CE_drop + lambda_c * CE_check + L2`, with all
  weights in the resolved config. Log each component by source.
- [x] **P4.4 Three-way selection implementation.** Optimizer batches may load
  only `replay`; epochs and the 3-by-3 model ladder are selected on the immutable
  `development` shard; the `external_ruler` role is rejected by the trainer.
- [x] **P4.5 Search-in-the-loop audit implementation.** Compare terminal-only and
  tablebase-then-candidate MCTS on the frozen horizon-one fixtures. Empirical
  acceptance remains pending a selected scaled candidate.

### Execution checklist

- [x] Retire the failed V2 smoke/full shards, dated traces, and non-authoritative
  checkpoints while preserving raw Tier A and the canonical P3 report.
- [x] Structural code tests: replayable trajectories, non-zero-probability chance
  branches, physical terminal counters, V3 round trip, source/episode/feature
  split isolation, strict Tier A preflight, and chunk resume mismatch refusal.
- [x] **Corrected tiny generation:**
  `uv run python -m stl.cli command=gen0_targets experiment=gen0_smoke`.
- [x] Strict-load both V3 outputs; review plan/source/Tier A/parent-chunk digests,
  exact-state and feature isolation, source coverage, interval widths, inactive
  Tier A policies, and collision report.
- [x] One-epoch V3-replay/V2-checkpoint trainer/reload smoke:
  `uv run python -m stl.cli command=train_gen0 experiment=gen0_train_smoke`.
- [x] **First full anchor run:** `uv run python -m stl.cli command=gen0_targets`.
- [x] First full training run:
  `uv run python -m stl.cli command=train_gen0`.
- [x] Implement and test strict V3 external-ruler evaluation with provenance,
  train/ruler isolation, per-source value errors, exact matrix re-solves,
  illegal-mass checks, saddle-gap checks, and deterministic reload checks.
- [x] Run the selected seed-4 checkpoint on the ruler. Best grouped-validation
  MSE was `0.15749897` at epoch 99. External-ruler MSE was `0.117329`; the
  candidate was rejected for tablebase MSE `0.055174 > 0.01`, interior MSE
  `0.090261 > 0.05`, and maximum exact-policy saddle gap
  `0.765666 > 0.05`. The saved local report is
  `outputs/regen2rl/gen0_checkpoint_v3_full_s4/external_ruler_report.json`.

### Scaled Generation-Zero replacement

The rejected candidate had only about 70 training rows and used an internal
grouped split for epoch selection. The replacement freezes three independently
generated roles before any expensive label solve:

| Source class | Training | Development | External ruler | Policy target |
|---|---:|---:|---:|---|
| exact horizon 2 | 2,048 | 384 | 384 | exact LP marginals |
| exact horizon 3 | 1,024 | 192 | 192 | exact LP marginals |
| tablebase boundary | 192 | 48 | 48 | inactive |
| tablebase interior | 96 | 24 | 24 | inactive |
| terminal | 128 | 32 | 32 | inactive |
| certified Tier A | 384 | 0 | 0 | inactive |
| **Total** | **3,872** | **680** | **680** | |

Every engine row begins at the canonical opening and stores its complete legal
action/chance trace. Paired Hal/Baku episode families are assigned by a seeded
hash before execution. The final publisher rejects overlap in exact-state hash,
episode ID, or V2 feature bytes for every pair of roles.

The scalable boundary tablebase sets the current checker cylinder to 300. Every
legal matrix cell then has death duration 300 and engine survival probability
zero, so its value is certified without a learned or search-derived label:

\[
V(s)=
\begin{cases}
+1,&\text{Baku is the current checker},\\
-1,&\text{Hal is the current checker}.
\end{cases}
\]

The scalable interior tablebase uses the Baku-dropper leap window. Baku's legal
second 61 forces Hal to fail every check. If the engine revival probability for
death duration \(q=x_H+60<300\) is \(p_{\mathrm{rev}}(q,n)\), the fatal branch
is a Baku win and the revived branch reaches Baku-at-300, a certain Hal win.
Therefore

\[
V(s)=p_{\mathrm{rev}}(+1)+(1-p_{\mathrm{rev}})(-1)
    =2p_{\mathrm{rev}}-1.
\]

Tests independently compare generated examples against the exact finite-horizon
LP solver. The original registry pins remain on the external ruler.

Each accepted exact row also publishes a hash-bound, pickle-free certificate

\[
\mathcal{C}(s)=
\left(H,\,A_s,\,\sigma_D^*,\,\sigma_C^*,\,V(s),\,u(s)\right),
\]

where \(H\) is the exact-state hash plus search-config digest, \(A_s\) is the
full root payoff matrix, and \(u(s)\le 0.5\) is unresolved probability. Candidate
selection and ruler evaluation compute full-width saddle gaps from \(A_s\)
without re-running thousands of horizon-3 solves. The final ruler still
re-solves eight certificates by default and rejects any matrix difference above
\(10^{-6}\).

The frozen optimizer batch has 128 rows:

| Source | Rows/batch | Value contribution | Policy contribution |
|---|---:|---:|---:|
| exact horizon 2 | 48 | 0.20 | 0.55 |
| exact horizon 3 | 24 | 0.20 | 0.45 |
| Tier A | 16 | 0.10 | 0 |
| tablebase boundary | 16 | 0.15 | 0 |
| tablebase interior | 12 | 0.20 | 0 |
| terminal | 12 | 0.15 | 0 |

For source \(k\), raw per-row weights are the declared contribution divided by
its batch count. Thus the implemented value objective is

\[
\mathcal{L}_V=
\frac{\sum_k\alpha_k\,\frac{1}{n_k}
      \sum_{i\in k}(\hat v_i-v_i)^2}{\sum_k\alpha_k},
\qquad \sum_k\alpha_k=1,
\]

and the active policy objective is

\[
\mathcal{L}_\pi=
0.55\,\mathrm{CE}(\pi_{H2}^*,\hat\pi_{H2})
+0.45\,\mathrm{CE}(\pi_{H3}^*,\hat\pi_{H3}).
\]

That imitation objective produced low average error but failed the worst-case
policy gate. Let (A_s) be the stored Hal-payoff matrix, (x_s) the network's
dropper distribution, and (y_s) its checker distribution. The implemented
certificate-backed gap is exactly

\[
g_s(x_s,y_s)=
\begin{cases}
\max_a (A_sy_s)_a-\min_b(x_s^\top A_s)_b,
  &\text{Hal drops},\\[3pt]
\max_b (x_s^\top A_s)_b-\min_a(A_sy_s)_a,
  &\text{Baku drops}.
\end{cases}
\]

This equals the sum of the two players' one-step best-response advantages and
is zero at a matrix equilibrium. The correction run minimizes

\[
\mathcal{L}_{\mathrm{gap}}
=\frac{\sum_{s\in B}w_sg_s}{\sum_{s\in B}w_s}
+\beta\max_{s\in B}g_s,
\qquad \beta=1,
\]

over each optimizer batch (B). Only training certificates enter this loss.
Development certificates measure the same maximum for checkpoint selection but
never produce gradients. Rows without exact matrices are excluded before the
softmax, and checkpoint provenance includes both replay and certificate hashes.

Each epoch log records unweighted value MSE, dropper/checker cross entropy, and
mean/maximum certificate gap in addition to the weighted optimizer losses.

#### Scaled execution checklist

- [x] **S1 candidate enumeration.** Construct the full unsolved pool and verify
  exact counts: train 3,200 reachable rows, development 608, ruler 608; add the
  exact tactical quotas and training-only Tier A to reach 3,872/680/680.
- [x] **S2 bounded generation pilot.** Run
  `uv run python -m stl.cli command=gen0_targets experiment=gen0_scaled_pilot`.
  It completed in 179.2 seconds with 68/26/46 rows. A repeat validated every
  replay/certificate bundle and completed in 0.5 seconds without an LP solve.
- [x] **S3 bounded trainer/evaluator pilot.** Run the one-epoch pilot trainer,
  certificate-backed development selector, external evaluator, and 4/8-visit
  MCTS audit. The intentionally untrained candidate failed the quality gates;
  this is expected and proves that the gates are active.
- [x] **S4 frozen full generation.** Inspect, then run
  `uv run python -m stl.cli command=gen0_targets experiment=gen0_scaled_full`.
  Acceptance requires exact source counts, all manifests/certificate hashes,
  cutoff at most 0.5, zero cross-role overlap, no feature collisions, and a
  second no-solve resume completing successfully. The published counts were
  3,872/680/680 with 4,224 exact matrix certificates, zero overlap/collisions,
  and a verified no-solve resume in 13.6 seconds.
- [x] **S5 bounded model ladder.** Train widths 64, 128, and 192 at seeds 0, 4,
  and 8 for at most 100 epochs with patience 20. Every optimizer batch must
  match the table above exactly. Every checkpoint must name both the training
  and development payload digests and must not name the ruler digest. All nine
  runs completed; best development value MSE was `0.00396585` (`h192_s0`).
- [x] **S6 development-only selection -- failed closed.** Run
  `command=gen0_select` over all nine
  `best.pt` files. A candidate is selectable only if tablebase MSE is at most
  0.01, tablebase-interior MSE at most 0.05, maximum exact saddle gap at most
  0.05, illegal mass is zero, and all required sources are present. Break ties
  by development MSE, then saddle gap, then checkpoint digest. Every candidate
  passed value, tablebase, provenance, reload, cutoff, and illegal-mass checks;
  all failed only saddle gap (`0.073055`--`0.111589`). Horizon-2 maximum gaps
  were below `0.0027`; all violations were horizon-3 Baku-pressure rows.
- [x] **S6.1 policy-objective correction.** Use the stored exact
  payoff matrices to train or fine-tune against row/column exploitability,
  rather than relying only on cross-entropy to one LP equilibrium marginal.
  The frozen `h192_s0` trunk and value head remained bit-identical while its
  policy head was fine-tuned. Worst development gap fell from `0.101532` to
  `0.037826`; the independent selector passed it without changing the `0.05`
  limit.
- [x] **S7 one external-ruler evaluation -- failed closed on value only.** Run
  `command=gen0_eval` only for the
  selected S6 checkpoint. The same gates apply; reload prediction drift must be
  at most \(10^{-7}\), certificate spot checks at most \(10^{-6}\), and the
  ruler digest must be absent from checkpoint training provenance. Policy gap
  passed at `0.037919`, tablebase-interior MSE passed at `0.020828`, and every
  integrity check passed. Tablebase boundary MSE failed at
  `0.012818 > 0.01`; therefore S8 was not run.
- [x] **S7.1 consumed-ruler diagnosis and V4 contract.** The failed ruler is
  development evidence only. Its boundary error was concentrated in the two
  both-cylinder-overflow pins; its interior aggregate hid large errors on the
  three canonical Hal-revival states. The V4 generator therefore crosses
  checker role, both cylinder states, clock class, both players' valid
  death/TTD histories, Hal dose, and referee CPR count. It publishes a
  state-hash taxonomy, moves all inspected canonical pins to development, and
  excludes every V3 state/episode/feature hash from the new holdout.
- [x] **S7.2 seal the V4 full holdout before correction.** Published 4,483
  training, 871 development, and 849 holdout rows. The holdout contains fresh
  384/192 exact-horizon rows, 32 terminals, and 176/65 boundary/interior rows.
  Freeze replay, certificate, taxonomy, plan, source-tree, prior-artifact, and
  gate digests. No model inference occurred before seal
  `6bcb62688e5ccb8d4d73b86f6935432ea781f75a64235719751099b99cb307e7`.
  Pairwise state, episode, and feature overlap is zero; a no-solve resume
  revalidated the published result in 13.7 seconds.
- [x] **S7.3 development-only staged correction.** Tried seeds 0/4/8 with only
  the value head trainable. Require boundary/interior development MSE at most
  0.005/0.025, per-family gates, maximum errors 0.10/sqrt(0.05), exact/terminal
  preservation, maximum saddle gap 0.04, and byte-identical trunk/policy. If
  none pass, run the frozen trunk/value LR-by-seed ladder, distill only
  exact/terminal/Tier-A rows, then repair the policy head from certificates.
  All value-head-only trials failed. Of six trunk/value trials, only
  `lr=0.0003, seed=4` passed every value and preservation gate; its policy gap
  was `0.112627`. Policy-only repair seeds 0 and 8 passed, and deterministic
  selection chose seed 0 with development gap `0.028483`.
- [x] **S7.4 one-use V4 evaluation -- passed.** Bound the sealed holdout to the selected
  checkpoint and resolved thresholds before inference. A resume must use the
  same checkpoint/config digests. Any failed completed report consumes the
  holdout and returns to a new development cycle; it does not authorize a
  second candidate or threshold change. The completed ledger binds checkpoint
  `735c2409af113bde71fe885096db7b8d9a541f9fdcf7f573067f3d349365df39`.
  Holdout overall MSE was `0.002281`; boundary/interior tablebase MSE was
  `0.000177`/`0.000838`; maximum exact saddle gap was `0.028476`; all gates
  passed.
- [x] **S8 first candidate-guided MCTS audit -- rejected and superseded.** Ran
  `command=gen0_mcts_audit` at budgets 64, 256, and 1,024 over seeds 0 through 9.
  The frozen P3 median/p95 value errors, maximum saddle gap, seed stability, and
  budget-worsening gates must pass. Candidate-vs-terminal-only value-error and
  saddle-gap worsening must each be at most 0.02. The candidate's 1,024-budget
  median/p95/maximum value errors were `0.183849`/`0.910766`/`0.911683`, and
  maximum saddle gap was `0.804233`. Maximum paired worsening was `0.911683`
  for value and `0.775828` for saddle gap. Forced fatal fixtures remained
  exact. Subsequent diagnosis found that the safe fixtures have (V_1=0/0.5)
  but (V_3\approx0.737/0.75), while candidate child predictions were closer
  to V2 continuation values. The audit compared depth-one MCTS with learned
  multi-step leaves against a terminal-only V1 oracle. It rejects that
  search/evaluator configuration but cannot prove missing successor rows were
  the sole cause.
- [x] **S8.1 explicit horizon contract.** Checkpoint V3 conditions the shared
  trunk and all heads on (h\in\{0,2,3\}). Prediction requires an explicit
  horizon. MCTS carries `root_value_horizon=3`, decrements to V2 children, and
  keys transpositions by `(exact public state, horizon)`. Horizon-blind V2
  bundles are rejected for inference and have a separate zero-column warm-start
  importer.
- [x] **S8.2 Bellman-closure implementation.** The fixed causal grid crosses
  checker role, cylinders `180/239/240/241/242/269/270/299`, other cylinder,
  six clock values, and eight valid history profiles. The balanced allocation
  is 96 training roots, 32 generated development roots plus the four disclosed
  fixtures, and 32 concealed roots. Every legal cell and chance branch is
  stored; successor V2 states are hash-deduplicated; roots carry V2 and V3.
  Generation fails unless
  \[
    Q_3(s,a,b)=\sum_{s'}P(s'\mid s,a,b)V_2(s'),\qquad
    V_3(s)=\operatorname{val}(Q_3(s))
  \]
  holds to (10^{-8}). Candidate subsets are rejected before training when
  restricted value error exceeds `0.02` or lifted saddle gap exceeds `0.05`.
  The smoke exposed that `safe_st=0` also requires literal full width; that
  correction changed the failing root from value error `0.188889` and saddle
  gap `0.222222` to numerical zero. Every root is now an independently
  hash-committed Bellman/replay/certificate chunk. Final artifacts are assembled
  from committed bytes, exact evidence is preserved if the candidate gate
  fails, and a holdout seal is written only after that gate passes. Split
  allocation reserves the root plus its complete one-transition successor
  closure before any exact label solve. The immutable plan records a digest and
  count for each reservation. Committed roots can be imported lazily into a new
  split by a binding over the old plan, role/index, root hash, and commit digest.
- [x] **S8.3 development-only training/selection implementation.** Existing V4
  training remains training evidence; V4 development and its consumed holdout
  are disclosed development evidence. The fixed width-192 ladder crosses
  `scratch/V4-import`, learning rates `1e-4/3e-4`, and seeds `0/4/8`. Value
  training precedes policy-only certificate gap repair. Selection requires
  direct V3 error, backed V3 error, successor V2 error, Bellman residual,
  horizon contrast, and exact saddle-gap gates.
- [x] **S8.4 fresh seal and one-use evaluator implementation.** A full V5 run
  first generates a new 849-row calibration ruler and the 32-root concealed
  Bellman closure. One seal binds both, their certificates/taxonomy, all gates,
  prior-artifact exclusions, and an eight-root MCTS subset. One ledger binds
  that seal to one checkpoint and configuration before inference.
- [x] **S8.5 aligned bounded-frontier audit implementation.** Static calibration
  and Bellman gates run first. Only a static pass starts depth-one MCTS with
  exact V3 root matrices and exact V2 frontier lookup as the baseline. Budgets
  remain `64/256/1024`, seeds remain `0..9`, P3 ceilings remain unchanged, and
  paired candidate worsening remains at most `0.02`.
- [x] **S8.6a execute the V5 smoke.** The real CLI smoke completed in `50.8 s`
  with one isolated root per split, producing `190` replay rows and `184`
  unique successor states. It exercised full-cell branch storage, manifests,
  policy certificates, and representability reporting without creating or
  claiming a holdout seal. After the zero-safe-budget correction, its previously
  failing replay root passed with zero value error and numerical-zero saddle
  gap; a second run resumed every root without solving in `5.3 s`.
- [x] **S8.6b1 execute closure-separated full exact generation.** The first
  root-only split failed closed after all roots because training/development,
  development/external, and training/external closures shared `95`, `89`, and
  `247` physical states. The corrected preflight reserved `5,512` training,
  `2,147` development, and `1,874` external-ruler states with zero overlap,
  reused `146/164` old root commits, and published `5,608`/`2,183`/`1,906`
  records. The external ruler was not sealed or consumed.
- [ ] **S8.6b2 repair candidate-action representability.** The full exact pack
  failed the unchanged pre-training gate on 25 training and 11 development
  roots, concentrated at checker cylinders `239--242`. Diagnose and correct
  candidate construction against the preserved exact matrices; do not weaken
  the `0.02` value-error or `0.05` saddle-gap limits. Then rerun publication
  validation, merge disclosed evidence, train the fixed ladder, select on
  development, and perform exactly one sealed bounded audit.
- [ ] **S9 P4 handoff.** Freeze Generation Zero only after S7 and S8 pass. A
  failed gate returns to data/model diagnosis; it does not authorize threshold
  changes or P5 self-play.

### Exit gate

- [x] Existing tablebase boundary MSE is at most `0.01` and tablebase-interior MSE
  at most `0.05`, preserving the current calibration contract.
- [x] Every exact source class appears in train, development, and external
  ruler reports; none crosses split hashes.
- [x] Masked policy illegal mass is exactly zero after inference.
- [x] On the exact policy holdout, the induced full-width one-step saddle gap is at
  most `0.05`. On the subset whose equilibrium is unique under a predeclared
  numerical tolerance, median role-policy total variation is at most `0.15`;
  total variation is report-only on non-unique equilibrium sets.
- [ ] The audit pack's maximum MCTS value drift from pinned values is at most
  `0.05` across seeds `0..9`.
- [x] The checkpoint bundle reloads and reproduces predictions within `1e-7` on
  CPU.
- [x] Train/development/ruler exact-state hashes, episode IDs, and feature hashes
  are pairwise disjoint; every engine-derived row replays exactly; every terminal
  loser has a fatal death, positive TTD, and matching CPR count.
- [x] An interrupted run reuses valid committed chunks, refuses a changed plan or
  corrupt committed chunk, and yields the same ordered per-array digests as an
  uninterrupted run.

## P5 - Build Genuine MCTS Self-Play [ ]

### Goal

Generate complete trajectories in which one frozen champion network guides
matrix-game MCTS for both roles, producing search policies and terminal
outcomes.

### Python scope

- replace or explicitly rename `stl/learning/support/self_play.py`
- a new joint-MCTS command; do not restore the removed legacy wrapper
- new `self_play` command and experiment configs only after the parser contract exists
- self-play tests

### Work packages

- [ ] **P5.1 Joint actor.** At each public state, run one MCTS, obtain both improved
  role marginals, independently sample drop and check seconds, validate both,
  and let the engine sample/resolve chance.
- [ ] **P5.2 Outcome targets.** Buffer state/search rows during the game. At engine
  terminal, attach the same Hal-perspective result to every row. At the P0 cap,
  attach `0` and `truncated=true`.
- [ ] **P5.3 No legacy path.** Core self-play may not import `CanonicalHal`,
  `stl/play/action_model.py`, `stl/play/search.py`, scripted opponents, reward
  helpers, or bucket abstractions. Those remain league/evaluation tools.
- [ ] **P5.4 Determinism.** Separate and record episode, root-noise, action-sampling,
  and engine-chance seeds. Replaying a manifest reproduces the exact trajectory.
- [ ] **P5.5 Runnable Hydra surface.** Config keys must map to parser fields. The
  experiment config must actually override command defaults; `--cfg job` and a
  no-training smoke run are tested.
- [ ] **P5.6 Operational safety.** Enforce cap, wall-clock timeout, shard flush
  interval, clean worker shutdown, and resumable completed-shard discovery.

### Exit gate

- [ ] A four-game CPU smoke run completes through Hydra and writes valid V3 shards
  without a checkpoint or parser error.
- [ ] Re-running the same smoke manifest produces identical state/action/outcome
  hashes; changing only the root-noise seed changes at least one trajectory.
- [ ] Across a 100-game, five-seed pilot: zero illegal actions, zero NaN/empty
  policies, zero unhandled worker failures, and fewer than 1% truncated games.
- [ ] Every nonterminal row has active legal role-policy targets; every row has an
  engine terminal outcome or an explicit truncation tag.
- [ ] No self-play row uses MCTS root value as `terminal_outcome`.
- [ ] A test demonstrates that both Hal-dropper and Baku-dropper states, including
  legal Baku `61`, occur and round-trip.

## P6 - Close the Replay, Learner, and Reanalysis Loop [ ]

### Goal

Run reproducible generations from frozen champion to candidate without
allowing approximate data to overwrite exact anchors.

### Python scope

- replay/schema modules under `stl/learning/`
- learner orchestration under `stl/commands/`
- `stl/learning/support/reanalysis.py`
- Hydra configs and learning tests

### Work packages

- [ ] **P6.1 Synchronous baseline.** Freeze champion digest, generate all shards,
  validate them, build replay, train one candidate, then evaluate. Do not begin
  asynchronous actors until the synchronous baseline is stable.
- [ ] **P6.2 Replay policy.** Keep at most the latest five accepted generations or
  one million self-play positions, whichever is smaller. The initial minibatch
  mix is 25% immutable exact/tablebase anchors and 75% self-play; log realized
  source ratios every epoch.
- [ ] **P6.3 Duplicate control.** Group repeated exact states for split placement,
  cap any one state at 1% of a self-play epoch, and retain independent outcomes
  as weighted observations rather than silently deduplicating them.
- [ ] **P6.4 Reanalysis.** Re-run current champion MCTS on reconstructable replay
  states to refresh policy targets. Preserve terminal outcomes. Never relabel
  exact/tablebase rows with MCTS.
- [ ] **P6.5 Candidate isolation.** A candidate never generates its own promotion
  data before it passes integrity and calibration gates. All training rows name
  the champion/search digest that produced them.
- [ ] **P6.6 Resume semantics.** Resume from checkpoint plus optimizer/scheduler and
  verified replay digests; refuse partial or mismatched state.

### Exit gate

- [ ] Three consecutive seeded end-to-end generations complete on the smoke
  profile, including at least one rejected candidate, without corrupting the
  champion pointer or replay manifest.
- [ ] Source ratios stay within 1 percentage point of the configured 25/75 mix.
- [ ] Exact-anchor predictions do not regress beyond P4 ceilings in any generation.
- [ ] Reanalysis changes policy targets on at least one sampled state, leaves all
  terminal outcomes unchanged, and is deterministic under fixed seeds.
- [ ] Interrupted training resumes to the same final checkpoint digest as an
  uninterrupted deterministic smoke run.
- [ ] Peak CPU memory, positions/second, search iterations/second, and shard
  throughput are recorded; no Rust acceleration is used.

## P7 - Make Promotion Statistical, Mandatory, and Claim-Safe [ ]

### Goal

Promote only a candidate that is calibrated, search-compatible, statistically
stronger, no more exploitable on the certified finite-depth probes, and
operationally reproducible.

### Python scope

- `stl/learning/gates.py` and support modules
- `stl/commands/compare_*`, `strength_gate.py`, and `promote.py`
- promotion configs and tests

### Work packages

- [ ] **P7.1 Integrity gate.** Verify candidate/champion/checkpoint/replay/config
  digests match across every report before reading scores.
- [ ] **P7.2 Calibration gate.** Preserve exact-source thresholds and require no
  degradation on the fixed external ruler. Report value and role-policy
  calibration separately.
- [ ] **P7.3 Search gate.** Re-run the P3 convergence pack with the candidate leaf
  evaluator. Reject if search error or seed variance breaches P3 ceilings.
- [ ] **P7.4 Paired arena.** Pair seeds and chance streams. Evaluate candidate-Hal
  versus champion-Baku and champion-Hal versus candidate-Baku so asymmetric
  player physics/legality are represented. Use SPRT with `H0=0 Elo`,
  `H1=+20 Elo`, `alpha=0.05`, `beta=0.05`, and a hard cap of 800 paired games.
- [ ] **P7.5 League robustness.** Require non-regression against random, safe,
  teacher, leap-specialist, and `pattern_reader` opponents on paired seeds.
  Scripted opponents are canaries, not self-play actors.
- [ ] **P7.6 Finite-depth best response.** Compare candidate and champion at the
  same scenarios, depth, support mass, state cap, and Tier A frontier. Reject
  any `candidate_certified_worse` case; report overlapping intervals as
  inconclusive.
- [ ] **P7.7 Policy report.** Make the policy-drift/readability report mandatory.
  On exact anchors, reject any candidate whose induced saddle gap exceeds
  `0.05`; elsewhere, report total variation and entropy without pretending
  that small drift is automatically good.
- [ ] **P7.8 Atomic promotion.** Update the champion pointer only after all reports
  pass and share hashes. Preserve the prior champion and a rollback manifest.

### Exit gate

- [ ] Integrity, calibration, search, paired arena, league, finite-depth best
  response, and policy reports all exist and identify the same artifacts.
- [ ] SPRT accepts `H1`; an inconclusive result is not a promotion.
- [ ] Overall and per-opponent paired results do not regress; `pattern_reader` is
  explicitly present.
- [ ] No best-response scenario is certified worse, and interval widths are
  reported. Wide/overlapping intervals are not described as proof.
- [ ] The candidate passes both P0 horizon caps and the sensitivity threshold.
- [ ] A forced failure in each gate leaves the champion digest unchanged.

## P8 - Freeze the Stable Python Baseline [ ]

### Goal

Produce the reference behavior and evidence package that a later Rust kernel
must reproduce exactly.

### Required baseline package

- [ ] champion checkpoint bundle and digest;
- [ ] resolved training, self-play, search, arena, and promotion configs;
- [ ] exact/tablebase corpus and external-ruler digests;
- [ ] three-generation replay manifests;
- [ ] P1/P3 oracle and convergence reports;
- [ ] calibration, arena/SPRT, league, policy, and finite-depth BR reports;
- [ ] deterministic trajectory fixtures for normal, leap, survival, and fatal
  branches;
- [ ] CPU throughput and memory benchmark;
- [ ] updated README, REGEN2RL, whitepaper source/PDF, and graphify graph.

### Exit gate

- [ ] `uv run pytest -q` passes, followed by `uv run pytest --collect-only -q`.
- [ ] The documented Gen-0 fixture generation, four-game self-play smoke, and one
  full promotion dry-run work from a clean checkout with no undocumented local
  checkpoints assumed.
- [ ] At least one candidate is accepted by P7 after a prior candidate is rejected,
  demonstrating both sides of the gate.
- [ ] Ten thousand recorded decisions contain zero illegal actions, zero invalid
  distributions, and zero unreconstructable states.
- [ ] Three runs with the same manifest reproduce trajectory, corpus, checkpoint,
  and report digests; a different training seed is recorded as a distinct run.
- [ ] The default runtime imports no Rust extension and all required evidence was
  produced by Python.
- [ ] The whitepaper labels exact finite-horizon results, approximate search
  results, and empirical RL results separately and makes no full-game solution
  claim.

Only after this gate is satisfied is the Python behavior frozen as the kernel
oracle.

## R1 - Future Rust Kernel Handoff (Explicitly Out of Current Scope) [ ]

R1 is a follow-on plan, not permission to modify `crates/` during P0-P8.

The first Rust candidates are pure, profiled kernels with narrow array/state
interfaces: payoff assembly, CFR+ matrix iterations, batched engine transition
expansion, and MCTS statistics updates. Python remains orchestration, config,
replay, training, gates, and the reference oracle.

Rust work may start only when:

- [ ] P8 is signed off on one immutable Python commit;
- [ ] every proposed kernel has a Python fixture and property test;
- [ ] parity tolerances are specified before benchmarking;
- [ ] fallback to Python is tested; and
- [ ] no Rust optimization changes legal actions, chance probabilities, role
  orientation, policy normalization, target semantics, or promotion results.

The Rust acceptance rule is behavioral parity first, speed second. Any speedup
that changes a P8 digest outside an explicitly approved floating-point
tolerance is a failed port.

## Definition of “Full AlphaZero/AlphaGo RL” for This Repository

The bridge is complete when the accepted champion is produced by this closed
loop:

```text
frozen champion policy/value net
  -> simultaneous matrix-game MCTS for both roles
  -> complete engine self-play trajectories
  -> (state, improved role policies, terminal/truncated outcome) replay
  -> masked policy/value optimization with exact-anchor retention
  -> reanalysis and fixed-ruler evaluation
  -> statistical arena + finite-depth exploitability + robustness gates
  -> atomic champion promotion
```

A static MCTS state sweep is expert iteration. Scripted-opponent outcome
training is curriculum learning. A value net behind legacy bucket search is an
approximate playable agent. All can be useful, but none alone satisfies this
definition.

## Per-Change Checklist

- [ ] Keep each pull request inside one numbered work package where possible.
- [ ] Add the exit-gate test in the same change as the behavior.
- [ ] Record thresholds before running the candidate being judged.
- [ ] Never weaken a gate because a candidate fails it; open a documented contract
  change with new evidence instead.
- [ ] Keep generated shards, checkpoints, reports, and logs outside source review
  unless the artifact itself is under audit.
- [ ] After every code phase, run `graphify update .` as required by the project
  instructions.
