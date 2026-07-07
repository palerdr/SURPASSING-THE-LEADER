# REGEN2RL Action-Core Reset

This is the canonical forward plan after the action-core reset.

## Corrected Action Core

- Normal action seconds are `1..60`.
- In the leap window, only Baku as dropper may choose second `61`.
- Hal never uses second `61`, and checkers are always capped at `60`.
- Dense policy/action tensors have length `62`. Index `0` is always-illegal padding; index `s` is literal second `s`.
- A check succeeds iff `check_time >= drop_time`.
- Successful squandered time is exactly `check_time - drop_time`; same-second checks succeed with `ST=0`.
- Normal successful ST is therefore `0..59`.
- Failed checks remain `check_time < drop_time` and apply the flat `+60` death-duration penalty.

## Invalidated Artifacts

All artifacts generated under the old action semantics are stale. This includes:

- `checkpoints/*.pt`
- `*_targets.npz`
- `checkpoints/tablebase/tier_a/*`
- promotion reports and calibration packs
- policy-drift reports
- ladder and exploitability comparisons

Old Tier A tables and policy checkpoints are not migration targets. They are historical baselines only.

## Regeneration Order

1. Regenerate exact/tablebase anchors from the corrected engine action core.
2. Regenerate exact-label corpora with length-62 dense policy targets.
3. Retrain value/policy checkpoints from regenerated labels.
4. Rebuild calibration, ladder, policy-drift, and exploitability evidence.
5. Resume RL/self-play only after exact anchors and supervised checkpoints are rebuilt.

No candidate agent should be treated as valid until it has evidence from the corrected pipeline.

## Tractability Extensions

Full RL should not try to make the 23-float feature projection perfectly
lossless. The exact public state remains the source of truth; learned features
are an approximation layer whose uncertainty should control when search spends
more work.

### 1. Collision-Gated Cascading Evaluation

Use the corpus collision diagnostic as a feature-engineering and search-routing
signal. A collision means two distinct `ExactPublicState` values share the same
feature vector; if their exact values disagree, the value head is being asked to
average incompatible states.

Action:

- Store a collision score per feature hash: collision count, value range, value
  variance, and worst source class.
- At MCTS/search leaf evaluation, look up the leaf's feature hash before
  trusting the value head.
- If the score crosses a threshold, cascade to a stronger evaluator:
  Tier A lookup if available, then bounded subgame resolve, then deeper exact
  reanalysis or high-iteration MCTS fallback.
- Feed cascaded results back into the target corpus with the collision metadata
  attached, so high-collision regions receive more exact labels over time.

Justification:

- This treats collisions as a calibrated uncertainty signal, not as a failure
  requiring a perfect embedding.
- It spends expensive exact/search budget only where the learned projection is
  demonstrably under-representing the state.
- It preserves the architecture rule that solver truth is non-lossy, while
  allowing learning features to stay compact and useful.

### 2. Relative Feature V2

The current feature vector is mostly absolute, normalized state. Keep those
features for backward compatibility during the reset, but add a v2 feature set
that emphasizes strategic relationships.

Candidate additions:

- `delta_cylinder = (hal_cylinder - baku_cylinder) / 300`
- `delta_ttd = (hal_ttd - baku_ttd) / 300`
- `delta_deaths = (hal_deaths - baku_deaths) / max_deaths`
- `hal_safe_budget = max(0, 299 - hal_cylinder) / 299`
- `baku_safe_budget = max(0, 299 - baku_cylinder) / 299`
- `safe_budget_ratio = hal_safe_budget / (hal_safe_budget + baku_safe_budget + eps)`
- `rounds_until_leap`, using current clock and half-round role schedule as a
  decaying feature rather than only raw proximity-to-leap seconds
- `leap_actor_pressure`, indicating whether the upcoming leap asymmetry favors
  Baku as dropper while the checker remains capped at `60`

Action:

- Introduce the feature v2 extractor beside the existing extractor; do not
  silently change checkpoint input shape.
- Train v1 and v2 candidates against the same regenerated labels.
- Promote v2 only if it reduces collision value spread, calibration error, and
  policy drift without weakening exact/tablebase gates.

Justification:

- Simultaneous timing decisions depend on relative pressure more than absolute
  totals alone.
- Safe-budget and leap-relative features expose the actual tactical pivots:
  overflow threshold, checker safety, and the Baku-only second-61 threat.
- Versioning avoids accidental compatibility with stale checkpoints.

### 3. Depth-Limited Continual Re-Solving

Unbounded exact search is not the target. The practical runtime path should
follow the DeepStack/ReBeL pattern already reflected in the solver architecture:
continually re-solve the current public state with a hard depth limit, then use
a learned or tablebase-backed frontier value.

Action:

- Default critical-state resolve to the current half-round or current round,
  not to match end.
- At the frontier, evaluate in this order:
  exact terminal, Tier A interval, collision-gated deeper solve, value net.
- Track unresolved mass and interval width at every frontier; reject promotion
  if runtime strength depends on wide or uncalibrated leaves.
- Treat public belief/state features as inputs to future value heads where
  hidden or opponent-policy uncertainty matters; do not collapse engine state
  inside the rigorous solver core.

Justification:

- The exact 60x60 or 61x60 action grid is tractable for one local resolve but
  explodes across unconstrained horizons.
- A hard frontier makes runtime predictable and turns the value net into a
  bounded approximation rather than an implicit terminal oracle.
- Public-belief/value learning is the right abstraction boundary for imperfect
  information; exact engine state still governs legal actions and chance.

### 4. CFR+ For Bounded Local Matrices

Keep LP minimax as the certification path for exact labels and audits. Use
vector-form alternating CFR+ for bounded local resolves where runtime matters.

Action:

- Keep CFR+ regret tables on literal second actions, including index `0`
  exclusion and Baku-only leap `61`.
- Clamp negative cumulative regrets to zero and average strategies over
  iterations for stable runtime policies.
- Compare CFR+ against LP on sampled exact matrices as a regression gate.
- Prefer CFR+ only where the local matrix is bounded by the candidate set or
  one-half-round frontier; do not use it to justify unbounded search.

Justification:

- CFR+ converges faster than vanilla CFR in large imperfect-information games
  and has a compact nonnegative-regret state.
- LP remains the cleaner proof path, while CFR+ is the practical local solver
  for repeated runtime re-solving.
- Keeping literal seconds in the regret tables prevents action abstraction from
  leaking into the rigorous solver core.

### 5. Safe Abstractions

Abstractions are allowed only as routing, training, or diagnostic tools unless
explicitly quarantined as legacy playable behavior.

Useful abstractions:

- collision-risk bands for evaluator trust
- safe-budget pressure bands for sampling and diagnostics
- leap-phase bands for curriculum and stratified validation
- public-history opponent buckets for routing/play, not exact solving
- interval-width bands for deciding when Tier A values are precise enough

Forbidden in the rigorous solver core:

- replacing exact seconds with buckets
- representative-value substitution
- feature-vector state compression
- route-stage reward shaping

Justification:

- These abstractions reduce training and runtime cost without changing engine
  truth or exact payoff assembly.
- They make promotion evidence more interpretable by reporting where a model is
  trusted, where it cascades, and where it remains unresolved.
- They preserve the firewall between exact solver correctness and playable or
  learned approximations.
