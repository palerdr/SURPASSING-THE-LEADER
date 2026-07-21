# DTH build and experiment workflows

`dth/solver.py` is the complete mathematical implementation of the first
no-leap game. Keep it compact and import-only: rules, exact transition
expansion, finite-horizon recursion, and the matrix LP solve remain there.

`dth/generate_dataset.py` owns reachable-state enumeration and NPZ writing.
`dth/config/dataset.yaml` is its Hydra configuration. Dataset concerns must
not move back into the solver.

`dth/docs/GAME_AND_SOLVER.md` is the mathematical specification. Code and documentation must
change together whenever a rule changes. The canonical engine remains separate.

## Frozen rules

- Both simultaneous action sets are exactly `1..60`.
- There is no action zero, pass, or no-check action.
- A check succeeds iff `check >= drop`.
- Successful ST is `check - drop + 1`, so every success adds at least one ST.
- The cylinder stores exact integer ST from `0` through `299`.
- Reaching or exceeding 300 accumulated ST on a successful check immediately
  dumps the cylinder.
- The injected overflow dose is capped at 300 and is inclusively fatal.
- A failed check creates dose `current_st + 60`.
- Failed-check revival is impossible when the dose is at least 300 or prior TTD
  plus the dose is greater than 300. Resulting total TTD exactly 300 remains
  revival-eligible.
- Otherwise:

  ```text
  p_revive = (1 - (dose / 300)^3) * 2^(-prior_ttd / 240)
  ```

- Revival resets the failed Checker's ST to zero and adds the dose to their TTD.
- Every surviving half-turn swaps Checker and Dropper roles.
- There is no leap second, absolute clock, physicality, or referee fatigue.

The old `-1` fatal-on-next-failure sentinel is forbidden. ST values `240..299`
cannot be merged because they differ in how much additional successful ST is
needed to dump the cylinder.

## State and value

The role-canonical live state is:

```python
(checker_st, checker_ttd, dropper_st, dropper_ttd)
```

The value is always from the current Dropper's perspective:

- permanent current-Checker death: `+1`;
- current-Dropper loss: `-1`;
- live horizon cutoff: `0`.

Because roles swap after a live transition, continuation values are negated.
The remaining horizon indexes the dynamic-programming layer and is not stored
inside the physical state tuple.

## Exact solve

For every live state and positive horizon:

1. build the full `60 x 60` matrix with Dropper rows and Checker columns;
2. expand deterministic or revival-chance branches exactly;
3. use terminal reward or `-V(child, horizon - 1)`;
4. solve Dropper-maximizing and Checker-minimizing LPs with SciPy HiGHS;
5. normalize both marginals and reject saddle gap above `1e-6`;
6. memoize the value and both policies.

Do not add MCTS, neural-network, canonical-engine, Hydra, or artifact-writing
dependencies to `dth/solver.py`.

## Reachable shallow targets

Run the dedicated Hydra entry point from the repository root:

```powershell
uv run python -m dth.generate_dataset
```

The default parameters live in `dth/config/dataset.yaml`. Any field can be
overridden without editing code, for example:

```powershell
uv run python -m dth.generate_dataset horizon=2 output=dth/artifacts/exact_h2.npz
uv run python -m dth.generate_dataset horizon=1 root_states='[[239,0,0,0],[240,0,0,0]]' output=dth/artifacts/boundary_h1.npz
```

The first strategically informative root-only fixture has its own frozen
configuration and command:

```powershell
uv run python -m dth.generate_dataset --config-name strategic_exact_v1
```

Unlike the opening plumbing fixture, this emits only the configured stratified
roots. Recursive children are solved as dependencies but are not allowed to
swamp the training rows with horizon-1 states.

Train the compact exact-target policy/value network with:

```powershell
uv run python -m dth.train
```

The default parameters live in `dth/config/train.yaml`. Training uses a
physical-state-grouped 80/20 split, horizon-balanced losses, a scalar
current-Dropper value head, and independent 60-action Dropper and Checker
policy heads. The best checkpoint and deterministic report are written under
`dth/checkpoints/strategic_exact_v1/`.

The richer horizon-3 follow-up is reproducible with:

```powershell
uv run python -m dth.generate_dataset --config-name strategic_exact_v2
uv run python -m dth.train dataset=dth/artifacts/strategic_exact_v2.npz output_dir=dth/checkpoints/strategic_exact_v2
```

The default config starts from `(0,0,0,0)`, uses transition-equivalent
representatives to enumerate every distinct live successor, and writes:

```text
dth/artifacts/exact_h3.npz
```

The artifact contains the opening at horizon 3, its live successors at horizon
2, and their live successors at horizon 1. Its schema is
`dth-v1-ttd-strict-overflow`.

Horizon 3 is intentionally only a plumbing fixture. From an empty opening, it
cannot expose the eventual cost of repeatedly choosing `check=60`: the initial
Checker can act as Checker only twice and accumulate at most 120 ST. The first
possible five-check, 300-ST overflow for that player occurs on half-turn 9.
Do not treat the horizon-3 equilibrium as a long-horizon strategy claim.

## Matrix-induced policy experiment

The v13 experiment directly supervises the policy induced by matrices rebuilt
from learned child values:

```powershell
uv run python -m dth.train --config-name train_matrix_policy_cfr_v13
```

`dth/torch_cfr.py` unrolls a fixed, differentiable CFR+ budget over the three
approximate root matrices.  The resulting policies are evaluated on exact LP
target matrices, and checkpoint selection minimizes their worst exact saddle
gap across both H5 roots and the H4 guard root.  The H4 gap and induced value
error also have baseline-relative hinge losses and hard promotion guards.
Exact-corpus value/policy loss and repeated H4/H5 frontier rows remain
supporting supervision.  The unrolled solver is not an exact target generator;
LP targets remain authoritative.

The seed-4, 64x2 warm start in
`dth/checkpoints/strategic_matrix_policy_cfr_v13/` selected epoch 200.  With 64
CFR+ iterations, worst exact induced-policy gap improved from `0.119546` to
`0.096643`.  The H4 guard gap improved from `0.119546` to `0.096643`, and its
value error improved from `0.029139` to `0.024856`.  A separate LP solve of each
learned approximate matrix also improved the worst exact gap from `0.130042` to
`0.106001`.  NumPy CFR+ at the same iteration budget matches the Torch audit;
it is an audit/runtime approximation only and is not used for gradients.

## Matrix-policy generalization readiness v2

The v2 readiness sprint keeps the eight roots in
`readiness_exact_h3_v1.yaml` and `readiness_exact_h4_v1.yaml` frozen. Its
development pack uses six asymmetric boundary families, both role
orientations, and both horizons 3 and 4: 24 root identities in total. The H4
closure is generated in two bounded batches and merged into
`dth/artifacts/readiness_development_v2.npz`.

Three 64x2 configurations were run. The final v19 configuration adds a
per-root saddle-gap weight for the three retained guards and performs 16
unrolled decision updates per epoch. It improved the development worst CFR
gap from `0.476041` to `0.421691`, but did not pass the unchanged frozen gate:
the exact LP worst-anchor improvement was `2.12%`, evaluation median
improvement was `0.86%`, and evaluation maximum gap regressed from `0.235416`
to `0.240992`. The machine report is
`dth/artifacts/self_play_readiness_v2_report.json`; its recommendation is
`no-promote`. Do not use v19 as the self-play champion.

## Function-preserving width probe v20

The bounded v20 capacity probe changes only the hidden width from 64 to 128.
Create its initialization by embedding the selected v19 tensors into the
leading blocks of a 128x2 network:

```powershell
uv run python -m dth.widen_checkpoint --source dth/checkpoints/strategic_matrix_policy_generalization_balanced_v19/best.pt --destination dth/checkpoints/strategic_matrix_policy_capacity_widened_v20_initial/widened.pt --target-width 128
uv run python -m dth.train --config-name train_matrix_policy_capacity_widened_v20
```

`dth/widen_checkpoint.py` duplicates every hidden neuron and divides its
outgoing weight between copies with deterministic unequal shares. The widened
value and both policy logits therefore equal v19 within floating-point
roundoff at initialization, while the unequal shares let duplicate neurons
receive different gradients immediately. The checkpoint records its source
hash and widening method.
The training config inherits the v19 data, roots, depth, losses, optimizer,
selection rules, and seed unchanged.

The primary run selected epoch 45 and improved the development worst gap from
`0.476041` to `0.411942`, with its internal guard passing. It did not nearly
pass the unchanged frozen readiness gate, so the conditional second seed was
not run. Against mixed-v7, worst-anchor gap improved `5.88%` (required `10%`),
evaluation median improved `3.32%` (required `10%`), and evaluation maximum
regressed from `0.235416` to `0.254937`. The H4 anchor value-error regression
was `0.009187`, inside the allowed `0.01`.

The favorable root `[179,60,119,120]` at H3 improved from `0.235416` to
`0.153602`, while `[179,60,59,180]` at H4 worsened from `0.213575` to
`0.254937`. Torch/NumPy CFR+ parity and the two-run, three-episode replay audit
passed. The machine report is
`dth/artifacts/self_play_readiness_v20_report.json`; its recommendation is
`no-promote`. This isolates the remaining failure as cross-root interference
that raw width alone did not remove. Do not increase width again or launch bulk
self-play from v20.

## Regression requirements

At minimum, tests must pin:

1. `drop=1, check=1` succeeds for one ST;
2. successful accumulation to 299 remains live;
3. successful accumulation to exactly 300 is terminal;
4. successful accumulation above 300 is terminal and the physical dose remains
   capped at 300 by the rule contract;
5. a failed-check dose of exactly 300 is terminal;
6. prior TTD plus current failed-check dose of exactly 300 remains
   revival-eligible;
7. prior TTD plus current failed-check dose above 300 is terminal;
8. live successful and revived transitions swap role coordinates;
9. horizon zero returns value zero;
10. a known matrix game has normalized minimax marginals and a small saddle gap;
11. target artifacts contain only ST values `0..299`, live TTD values `0..300`,
    legal policies, value bounds, schema `dth-v1-ttd-strict-overflow`, and
    horizons `1..3`.

## Next learning step

After the contract tests and artifact audit pass, train a small network with:

- input `(sC, tC, sD, tD, remaining_horizon)`;
- one current-Dropper value output;
- one 60-action Dropper marginal;
- one 60-action Checker marginal.

For strategically informative exact pretraining, either extend the opening
horizon far enough to expose overflow or add shallow exact roots already near
the cylinder boundary. Keep those datasets explicitly distinguished from the
compact horizon-3 plumbing artifact.

Plan:
Generate exact reachable targets for horizons 1–3.
Supervise a tiny network on those targets.
Validate MCTS against the exact solver.
Use that network as the MCTS leaf evaluator.
Run capped MCTS self-play.
Train the next network on MCTS policies and terminal/truncated outcomes.
