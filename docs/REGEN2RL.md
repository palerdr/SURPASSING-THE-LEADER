# REGEN2RL: Python-First Bridge to Simultaneous-Move AlphaZero

Status: canonical implementation checklist, executed through P3 on 2026-07-10.

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
- [x] **P2 reconstructable schemas.** `TrainingRecordV2`, V2 features, safe
  shards, grouped splits, strict checkpoint bundles, and stale-artifact
  rejection are implemented.
- [x] **P3 MCTS improvement operator.** Canonical linearly weighted mean-Q
  policies, factorized priors/noise, full-width mode, and the frozen
  convergence gate pass. Report:
  `outputs/regen2rl/p3_mcts_conformance.json`, canonical report-payload SHA-256
  `245037e1bb11d8d2bef0e30a7ae10f78751fe541d24a5c938b6aaa4bae432eb1`.
- [x] **P4 preparation smoke.** The V2 Gen-0 Hydra smoke generated 26 training
  and 26 external-ruler records, and a one-epoch trainer smoke reloaded its
  checkpoint successfully.
- [ ] **STOP: P4.1 full anchor generation.** This is the first long generation
  run. Execute the command in P4 only after reviewing compute/storage scope.

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
| Replay/checkpoints | `stl/learning/replay.py` and `train.py` provide reconstructable V2 shards, grouped splits, and strict bundles | P2 integrity boundary complete. |
| Supervised learner | `stl/learning/train.py` has masked role-policy losses, weighted value loss, grouped selection, and V2 loading | P4 smoke complete; full anchor generation/training pending. |
| Evaluation pieces | calibration, audit, ladder, best-response intervals, SPRT, and promotion modules | Integrate into one mandatory gate with precise claim labels. |

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
5. The P4 smoke proves only structural interoperability. The full fixed anchor
   corpus, external-ruler review, training run, calibration ceilings, and
   search-in-the-loop audit have not run.
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
   reward shaping may enter `stl.solver`.
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
  no-death, survival, fatality, same-second success, failed check, overflow,
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

- [x] **P2.1 `TrainingRecordV2`.** Store feature vector, serialized
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

### Exit gate

- [x] A 100-record mixed-source fixture round-trips byte-for-byte in manifest data
  and numerically within dtype tolerance in tensors.
- [x] All 100 reconstructed states equal their stored `ExactPublicState`.
- [x] Corrupt shard, mismatched action size, feature schema, or checkpoint parent
  digest fails before training begins.
- [x] Train and validation exact-state hash sets have empty intersection.
- [x] V2 collision audit reports zero divergent-value collisions on the full
  pinned/tablebase corpus. Any remaining collision class is documented and
  routed away from bare-net evaluation.
- [x] Old length-61 or 122-logit artifacts are rejected, as required by
  `stl/learning/README.md`.

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

- [x] **P4.0 Pipeline smoke.** `command=gen0_targets experiment=gen0_smoke`
  writes reconstructable V2 train/ruler shards, and a one-epoch CPU trainer
  smoke reloads the resulting checkpoint bundle.
- [ ] **P4.1 Fixed anchor corpus.** Regenerate terminal, full-width finite-horizon,
  tactical tablebase, and certified Tier A rows under V2 schemas. Store corpus
  and holdout digests.
- [ ] **P4.2 Policy labels.** Use LP equilibrium role marginals on exact rows. Rows
  without a certified policy carry an inactive policy-loss mask rather than a
  fabricated uniform target.
- [ ] **P4.3 Loss contract.** Train
  `lambda_v * MSE + lambda_d * CE_drop + lambda_c * CE_check + L2`, with all
  weights in the resolved config. Log each component by source.
- [ ] **P4.4 Grouped selection.** Select the best epoch only on grouped validation;
  score the frozen external ruler once after selection.
- [ ] **P4.5 Search-in-the-loop audit.** Compare MCTS with the Gen-0 evaluator to
  terminal/tablebase and exact horizon-one fixtures at all convergence budgets.

### Execution checklist

- [x] Structural generation smoke:
  `uv run python -m stl.cli command=gen0_targets experiment=gen0_smoke`.
- [x] One-epoch V2 trainer/reload smoke:
  `uv run python -m stl.cli command=train_gen0 experiment=gen0_train_smoke`.
- [ ] **First long run -- current stopping point:**
  `uv run python -m stl.cli command=gen0_targets`.
- [ ] Review both V2 manifests, source coverage, collision report, and external
  ruler isolation before training.
- [ ] First full training run:
  `uv run python -m stl.cli command=train_gen0`.

### Exit gate

- [ ] Existing tablebase boundary MSE is at most `0.01` and tablebase-interior MSE
  at most `0.05`, preserving the current calibration contract.
- [ ] Every exact source class appears in train, grouped validation, and external
  ruler reports; none crosses split hashes.
- [ ] Masked policy illegal mass is exactly zero after inference.
- [ ] On the exact policy holdout, the induced full-width one-step saddle gap is at
  most `0.05`. On the subset whose equilibrium is unique under a predeclared
  numerical tolerance, median role-policy total variation is at most `0.15`;
  total variation is report-only on non-unique equilibrium sets.
- [ ] The audit pack's maximum MCTS value drift from pinned values is at most
  `0.05` across seeds `0..9`.
- [ ] The checkpoint bundle reloads and reproduces predictions within `1e-7` on
  CPU.

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

- [ ] A four-game CPU smoke run completes through Hydra and writes valid V2 shards
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
