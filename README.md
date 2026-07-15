# Surpassing The Leader

Compact Python research implementation of the *Usogui* Drop the Handkerchief
game as a fully observed, two-player, simultaneous-move stochastic zero-sum
game. The project is organized around one current path:

```text
engine rules -> exact/search anchors -> value/policy learning -> gated self-play
```

The goal is not to preserve every historical script. The live tree keeps the
engine, exact solver, playable agents, training bridge, and Hydra command
surface in `stl/`. The disconnected pre-consolidation mirror was removed after
the pushed `a4b03db` archival baseline; historical paths remain available from
that commit rather than as a second runtime tree.

This scope is deliberate. The rigorous solver currently solves a fully
observed public-state baseline abstraction. It does not yet represent private
Leap Second knowledge, memory loss, echolocation observations, signaling, or
nested beliefs. Those mechanics require information sets or public beliefs and
define a larger imperfect-information game; they cannot be added by giving the
current minimax recursion more depth.

## Current Layout

| Path | Role |
|---|---|
| `stl/engine/` | Rule authority: constants, players/referee, game clock, half-round resolution, death/revival, leap-second legality. |
| `stl/solver/` | Exact/search/tablebase core plus conformance and MCTS audit packs. |
| `stl/learning/` | Target generation, value/policy model, training, gates, and RL/self-play support. |
| `stl/play/` | Actual playable surface: `SolverAgent`, `CanonicalHal`, opponents, teachers, environment wrapper, terminal CLI. |
| `stl/commands/` | Importable command implementations used by Hydra. |
| `crates/stl_solver/` | Opt-in Rust parity/acceleration seam; not baseline authority before Python P8. |
| `configs/` | Hydra command and experiment configs. |
| `tests/{engine,solver,learning,play}/` | Regression tests grouped by subsystem. |
| `docs/whitepaper/` | Current solver whitepaper source/PDF. |
| `docs/papers/` | Reference papers. |

Generated corpora, checkpoints, logs, demos, `outputs/`, `multirun/`, and
`target/` should stay out of source review unless the artifact itself is under
discussion.

## Current Action Core

Normal action seconds are `1..60`. During the leap window, only Baku as dropper
may use second `61`; Hal never uses `61`, and checkers are always capped at
`60`. Dense policy/action tensors have length `62`: index `0` is always-illegal
padding, and index `s` is literal second `s`. This statement assumes canonical
Hal/Baku identities; the generic engine predicate is currently spelled as a
non-Hal dropper check and should not be generalized into a theorem about
arbitrary actor names.

Checks succeed iff `check_time >= drop_time`. Successful ST is exactly
`check_time - drop_time`, so same-second checks succeed with `ST=0`; failed
checks remain `check_time < drop_time` with the flat `+60` penalty. A failed
check resets the checker's cylinder only after successful revival; fatality is
terminal and does not perform that reset.

Artifacts generated under the old action semantics are stale. See
`docs/REGEN2RL.md` for the audited Python-first plan from exact regeneration
through genuine simultaneous-move MCTS self-play and gated promotion.

## Foundational Solver Model

One half-round is one simultaneous matrix node, not a max node followed by a
min node. The dropper and checker independently commit to legal seconds; every
matrix cell contains the expected Hal-perspective continuation value after the
engine transition.

| Component | Actual job | Current implementation |
|---|---|---|
| Transition model | Defines legal actions and advances cylinders, death/revival chance, roles, the absolute clock, and Leap Second behavior. | `stl/engine/actions.py` and `stl/engine/game.py` are authoritative. |
| Exact matrix construction | Evaluates each legal drop/check outcome and recursively obtains its continuation value. | `solve_exact_finite_horizon` in `stl/solver/exact.py`. |
| Minimax | Supplies the Bellman--Shapley backup and local equilibrium policies. | A fail-closed SciPy/HiGHS LP is the certification oracle. |
| CFR+ | Approximately solves one already-populated local zero-sum matrix. | Optional bounded runtime resolve and LP-conformance target; it does not perform tree search. |
| Matrix-game MCTS | Chooses public states and joint-action cells to simulate, estimates their continuation values, and emits two role marginals. | `mcts_search` in `stl/solver/search.py`; empirically gated, not claimed as a general convergence theorem. |
| Value/policy network | Approximates Hal value plus independent dropper/checker policy marginals around exact anchors. | Checkpoint V3 conditions every head on value horizon 0, 2, or 3; a reviewed full checkpoint is pending. |

The finite-horizon operator is

```text
V_h(s) = max_x min_y sum[a,b] x(a) y(b)
         sum[s'] P(s' | s,a,b) V_(h-1)(s').
```

Immediate squandered time is state transition data, not solver utility. The
certified objective is terminal Hal value, with explicitly tagged unresolved
mass at a finite exact horizon and explicitly tagged draws at the RL episode
cap. Alternating pure-action minimax would reveal one player's current choice
to the other and therefore solve a different game.

## Setup

Use Python 3.12+ with `uv`:

```bash
uv sync --dev
uv run pytest --collect-only -q
uv run pytest -q
```

The latest verified full suite, run on 2026-07-15 after the horizon-aware
Bellman-closure implementation, was:

```text
779 passed, 4 skipped
```

## Playing A Match

The checked-in headline checkpoint predates the V2 feature/action contract and
is intentionally rejected by the strict loader. Until P4 produces a reviewed
V2 checkpoint, the immediately runnable local match uses the quarantined legacy
agent:

```bash
uv run python -m stl.cli command=play 'command.args=["--agent","legacy"]'
```

After a reviewed checkpoint exists, run the solver-backed Hal explicitly:

```bash
uv run python -m stl.cli command=play 'command.args=["--checkpoint","outputs/regen2rl/gen0_checkpoint_v3/best.pt","--use-tier-a","--seed","123"]'
```

The default `solver` agent fails loudly on a missing or incompatible checkpoint;
it never silently falls back to handcrafted evaluation. `--use-tier-a` enables
certified Tier A tablebase lookup where available. If `--seed` is omitted, the
CLI picks a fresh random seed so human games do not replay the same sampled
line.

## Hydra Commands

Hydra is the public command surface:

```bash
uv run python -m stl.cli --help
uv run python -m stl.cli command=tier_a --cfg job
uv run python -m stl.cli command=gen0_targets --cfg job
uv run python -m stl.cli command=train_gen0 --cfg job
uv run python -m stl.cli command=gen0_select --cfg job
uv run python -m stl.cli command=gen0_eval --cfg job
uv run python -m stl.cli command=gen0_mcts_audit --cfg job
uv run python -m stl.cli command=gen0_bellman_cycle --cfg job
uv run python -m stl.cli command=gen0_bellman_select --cfg job
uv run python -m stl.cli command=gen0_bounded_audit --cfg job
```

Use `--cfg job` to inspect a config without starting a long run.

Common commands:

```bash
# Focused engine/solver checks
uv run pytest tests/engine -q
uv run pytest tests/solver -q

# Build a small Tier A smoke slice
uv run python -m stl.cli command=tier_a command.stage=1 command.limit=1 command.workers=1

# Scaled Generation-Zero preflight
uv run python -m stl.cli command=gen0_targets experiment=gen0_scaled_pilot
uv run python -m stl.cli command=train_gen0 experiment=gen0_scaled_pilot_train

# Frozen full corpus and 3 x 3 model ladder (long jobs; inspect first)
uv run python -m stl.cli command=gen0_targets experiment=gen0_scaled_full --cfg job
uv run python -m stl.cli -m command=train_gen0 experiment=gen0_scaled_train \
  command.hidden_dim=64,128,192 command.seed=0,4,8
uv run python -m stl.cli command=train_gen0 experiment=gen0_scaled_gap_finetune

# Older minimal V3 smoke commands
uv run python -m stl.cli command=gen0_targets experiment=gen0_smoke
uv run python -m stl.cli command=train_gen0 experiment=gen0_train_smoke

# Inspect the first long generation and training jobs without running them
uv run python -m stl.cli command=gen0_targets --cfg job
uv run python -m stl.cli command=train_gen0 --cfg job

# Evaluation and promotion surfaces
uv run python -m stl.cli command=gen0_select \
  'command.checkpoint=[/path/to/h64_s0/best.pt,/path/to/h128_s4/best.pt]'
uv run python -m stl.cli command=gen0_eval \
  command.checkpoint=/path/to/gen0-candidate.pt
uv run python -m stl.cli command=gen0_mcts_audit \
  command.checkpoint=/path/to/selected-gen0.pt
uv run python -m stl.cli command=eval command.checkpoint=/path/to/candidate.pt
uv run python -m stl.cli command=promote \
  command.calibration_report=/path/to/calibration.json \
  command.ladder_report=/path/to/ladder.json \
  command.exploitability_report=/path/to/exploitability.json
```

The July 10--13 V2 Gen-0 run completed computationally but failed its data audit:
most exact rows were manually assembled, terminal rows omitted their fatal death
mechanics, all tactical pins appeared in both outputs, certified Tier A was
absent, and exact horizon/cutoff metadata was dropped at the replay boundary.
Those generated shards and their obsolete checkpoints have been removed.

The replacement is `TrainingRecordV3`. Exact and terminal candidates now come
from replayable calls to the engine, whole episodes are assigned before any LP
labeling, tactical pins are partitioned once, Tier A interval rows are
training-only with inactive uncertified policies, and the final gate rejects
state, episode, or feature overlap. Every run has a canonical plan, source-tree
digest, Tier A manifest digest, immutable component chunks, and fail-closed
resume behavior.

The earlier minimal V3 full dataset and seed-4 candidate were generated and
audited. That candidate reached grouped-validation MSE `0.15749897` at epoch 99,
but the immutable external ruler rejected it:
tablebase MSE was `0.055174` (ceiling `0.01`), tablebase-interior MSE was
`0.090261` (ceiling `0.05`), and maximum exact-policy saddle gap was `0.765666`
(ceiling `0.05`). It is not a Generation-Zero champion and must not seed P5.

The replacement scaled contract is implemented, generated, and audited. The
pilot published 68 training, 26 development, and 46 ruler rows plus exact
root-matrix certificate sidecars; the full run then published the frozen counts
below. A no-solve rerun validated every component and final payload in 13.6
seconds. The initial nine-model ladder failed the exact-policy saddle-gap gate.
A certificate-backed policy-head fine-tune then reduced the development gap
from `0.101532` to `0.037826` and passed independent selection. Its one permitted
external-ruler evaluation confirmed policy gap `0.037919`, but rejected the
unchanged value head on tablebase MSE `0.012818 > 0.01`. It remains ineligible
for MCTS/P5.

| Source | Train | Development | External ruler |
|---|---:|---:|---:|
| exact horizon 2 | 2,048 | 384 | 384 |
| exact horizon 3 | 1,024 | 192 | 192 |
| tablebase boundary | 192 | 48 | 48 |
| tablebase interior | 96 | 24 | 24 |
| terminal | 128 | 32 | 32 |
| certified Tier A | 384 | 0 | 0 |
| **Total** | **3,872** | **680** | **680** |

That external ruler is now a consumed diagnostic artifact: its predictions
identified missing double-overflow and Hal-history revival combinations, so it
cannot be reused for promotion. The V4 correction path constructs a causal
tablebase grid with 704/195 training boundary/interior rows, 176/65 generated
development rows plus the 22 known canonical regressions, and a fresh sealed
176/65 tablebase holdout. A hash-bound taxonomy gates double overflow, checker
role, and every interior history separately so easy rows cannot average-mask a
failed family. Fresh exact-horizon and terminal episodes make the sealed
holdout a full evaluator test rather than a tablebase-only test.

Correction is development-only and staged. It first updates only the value
head and verifies byte-identical trunk and policy parameters. If that fails,
it updates trunk/value with source-restricted distillation and then reruns the
certificate saddle-gap policy repair. The sealed holdout is atomically bound
to one checkpoint/config digest; MCTS remains blocked until that single
evaluation passes every value, policy, provenance, and reload gate.

That V4 sequence has now run. The sealed corpus contains 4,483 training, 871
development, and 849 holdout rows with zero state, episode, or feature overlap
between any pair. A no-solve resume revalidated every chunk and final digest in
13.7 seconds. V4 reused already-certified V3 exact/terminal labels only in
training and development, generated all 608 exact/terminal holdout labels from
fresh states, and replaced the defective tablebase grid in all three splits.
Value-head-only repair failed development. The trunk/value ladder
produced one value-eligible checkpoint (`lr=0.0003`, seed 4), and policy-only
repair seed 0 reduced its development saddle gap to `0.028483`. Its one-use
sealed-holdout evaluation passed: overall MSE `0.002281`, boundary/interior
tablebase MSE `0.000177`/`0.000838`, exact-policy saddle gap `0.028476`, and no
failed integrity gate.

| V4 source | Train | Development | Sealed holdout |
|---|---:|---:|---:|
| exact horizon 2 | 2,048 | 384 | 384 |
| exact horizon 3 | 1,024 | 192 | 192 |
| tablebase boundary | 704 | 195 | 176 |
| tablebase interior | 195 | 68 | 65 |
| terminal | 128 | 32 | 32 |
| certified Tier A | 384 | 0 | 0 |
| **Total** | **4,483** | **871** | **849** |

The checkpoint is nevertheless rejected for search. At 1,024 MCTS simulations,
the candidate evaluator had median/p95/maximum root-value errors
`0.183849`/`0.910766`/`0.911683` and maximum saddle gap `0.804233`. The failure
is concentrated in the two nonterminal safe-budget fixtures; the forced fatal
fixtures remain exact. Exact diagnosis then exposed a contract mismatch: that
audit compared a horizon-one terminal oracle with MCTS whose learned leaves
approximated longer-horizon continuation. It rejects the configuration but does
not by itself prove successor generalization failure.

The implemented V5 correction makes the contract explicit. The network takes
the unchanged 52 physical features plus a one-hot value horizon, and prediction
requires (h\in\{0,2,3\}). Depth-one candidate MCTS requests V3 at the root and
V2 after one transition; transpositions include the remaining horizon. The
Bellman generator stores every legal root cell, every engine chance branch,
every unique V2 successor, and the exact V3 root matrix. Development gates both
the direct root prediction and the independently backed value

```text
B3_hat(s) = val(sum[s'] P(s'|s,a,b) V2_hat(s')).
```

The initial V5 CLI smoke completed in `50.8 s`, producing `190` rows over three
isolated splits and `184` unique successor states. Its sampled replay root
proved that zero-safe-budget states cannot use the compact candidate set: the
omitted-action value error was `0.188889` and saddle gap was `0.222222`. Those
states now use literal full-width actions; the stored matrix recheck reduced
both errors to numerical zero. Bellman generation commits each root with
hash-bound replay/certificate sidecars, merges those commits deterministically,
and preserves exact artifacts even if the later candidate gate blocks sealing.
A verified no-solve smoke resume completed in `5.3 s`. The first full
allocation exposed 431 shared successor states across its three nominal root
splits; no corpus was published. The corrected allocator now reserves each
root's complete one-transition closure before solving. Its preflight certified
zero cross-split overlap across `5,512` training, `2,147` development, and
`1,874` external-ruler physical states, and reused `146/164` committed roots.

The corrected full exact artifacts contain `5,608` training, `2,183`
development, and `1,906` external-ruler records. They are intentionally
unsealed: the representability gate found 25 training and 11 development roots
whose compact candidate actions cannot reproduce the exact matrices, mainly at
checker cylinders `239--242`. P5 stays blocked until that candidate-action
defect is corrected, the preserved exact artifacts pass the gate, and one
candidate passes the joint calibration/Bellman seal and aligned MCTS audit.

There is intentionally no public `command=self_play` before P5. The old broken
Hydra wrapper was removed; the tested compatibility module still runs the
legacy bucketed actor against scripted opponents. Phase P5 of
`docs/REGEN2RL.md` replaces that module with matrix-game MCTS self-play before
adding a command again.

## Solver Shape

`stl/solver/` is intentionally compact:

```text
stl/solver/
  exact.py       # public state, LP minimax, half-round helpers, timing, diagnostics
  search.py      # leaf evaluators, selective solve, subgame resolve, matrix-game MCTS
  tablebase.py   # tactical fixtures, pinned registry, backward map, Tier A lookup/build
  conformance.py # exact/search conformance audit helpers
  mcts_conformance.py # frozen simultaneous-MCTS convergence pack
```

The exact tower solves finite-horizon stochastic zero-sum public states by
building exact-second payoff matrices and solving minimax LPs. A bounded CFR+
matrix solver is available for critical runtime resolves, but certified exact
labels and audits keep using the LP path. Chance branches come from the engine.
Rows remain drop seconds and columns remain check seconds, while entries and
backups stay in Hal perspective; role-aware transpose/negation handles which
role Hal currently occupies. The LP rejects invalid inputs and optimizer
failure instead of silently substituting a heuristic. Exact result arrays are
immutable once cached.
Finite-horizon results use a bounded per-process LRU; eviction changes runtime,
not the value recomputed for a state.

Full timing matrices remain literal `60 x 60` or `61 x 60` games. Matrix
assembly may share recursive evaluation across public-transition-equivalent
outcome classes--normally at most 61 success/failure signatures--without
merging their strategically distinct matrix cells or claiming their literal
history records are identical. The solver core must not import reward shaping,
observation features, route-stage labels, or bucketed abstractions.

The search tower adds selective candidate sets, matrix-game MCTS, optional
learned leaf evaluation, bounded critical-state subgame resolve, and Tier A
lookup. MCTS exposes candidate, playable-candidate, and full-width modes. Its
canonical improvement object is a pair of legal role marginals formed from a
linearly weighted average of empirical mean-Q equilibria; optimistic policies
select exploratory samples but are not mislabeled as training targets. Root
priors and optional Dirichlet noise remain role-factorized. Critical resolve
defaults to the current half-round and uses the leaf evaluator at the boundary.

One MCTS simulation observes only one joint cell. The implementation updates
only that cell and repeatedly solves the current empirical matrix; it does not
apply full-information CFR+ regret updates to unobserved counterfactuals. The
frozen P3 conformance pack currently certifies horizon-one fixtures. Deeper
search, learned frontiers, zero-prior exploration, candidate omissions, and
coverage remain explicit conformance and promotion obligations rather than
proved facts.

The finite-horizon proof applies only to the full legal action sets, engine
transition kernel, public Markov state, and declared cutoff actually enumerated
by exact recursion. It reports unresolved frontier mass separately. It does not
prove the unrestricted-duration game, finite-budget MCTS convergence,
finite-iteration CFR+ exactness, or global exploitability from a local saddle
gap.

## Learning And Self-Play

`stl/learning/` contains the pieces of the exact-to-self-play bridge:

```text
contracts.py    # finite episode, role, config, and horizon-sensitivity contracts
reachable.py    # legal-engine Gen-0 trajectories and pre-label split ownership
replay.py       # reconstructable TrainingRecordV3 shards and grouped splits
bellman.py      # exact V3-root/V2-successor closure and consistency gates
targets.py      # exact/bootstrap labels, legacy corpus IO, and V3 target conversion
model.py        # 52-field V2 encoder plus explicit horizon-conditioned heads
train.py        # grouped supervised training and strict V3 checkpoint bundles
gates.py        # calibration, audit, ladder, strength and promotion gate surface
support/        # route features, reanalysis, bootstrap, self-play, tournaments, strength internals
```

V3 replay stores the exact public state, feature and action schema versions,
target semantics, exact-search horizon, cutoff probability, value bounds, state
origin, source artifact, legal engine trace where applicable, legal masks, both
role policies, truncation state, provenance digests, and RNG seeds. Shards are
pickle-free and hash-verified. The default
trainer accepts these manifests and writes restorable bundles containing model,
optimizer, scheduler, schemas, provenance, resolved config, history, and RNG
state. Horizon-blind V2 bundles, bare legacy checkpoints, and stale
length-61/122-logit artifacts are rejected for inference. V2 bundles have a
separate, provenance-recorded warm-start importer that copies physical-feature
weights and initializes horizon columns to zero.

The current supervised/expert-iteration promotion path is:

```text
exact/tablebase anchors -> value/policy training -> MCTS/self-play targets
-> calibration/strength/exploitability gates -> promotion
```

A checkpoint should not be promoted just because held-out MSE improves.
Promotion evidence should include calibration, deterministic ladder results,
exploitability or best-response evidence where available, and policy drift
checks when the policy head is involved.

This is not yet a closed AlphaZero reinforcement-learning loop. P0--P3 and the
V4 static P4 ruler are complete, but no horizon-aware checkpoint has passed the
new Bellman and bounded-MCTS gates. The next boundary is executing the V5 exact
closure, training ladder, fresh one-use holdout, and aligned audit. The
compatibility self-play
module still runs legacy `CanonicalHal`
against scripted opponents; genuine two-role MCTS self-play begins in P5. The
phase gates and current stopping point are in `docs/REGEN2RL.md`.

## Architecture Rules

- Engine truth belongs in `stl.engine`.
- `stl.engine` must not depend on solver, learning, reward shaping, or play.
- Exact/search solver work belongs in `stl.solver`.
- `stl.solver` must preserve exact integer-second action spaces, including the
  Baku-only 61-second leap-window drop.
- `stl.solver` must not import reward shaping, observations, route-stage labels,
  or bucketed abstractions.
- Public states with different private knowledge or observation histories must
  not be merged if an imperfect-information solver is introduced later.
- Learning and self-play bridge code belongs in `stl.learning`.
- Playable agents, scripted opponents, environment wrappers, and terminal UI
  belong in `stl.play`.
- New runtime code does not belong in `legacy/`.
- Python remains the behavioral baseline through REGEN2RL P8; Rust kernels are
  deferred until that baseline and its parity fixtures are frozen.

## Documentation

- `docs/whitepaper/stl_solver_whitepaper.tex`
- `docs/whitepaper/stl_solver_whitepaper.pdf`
- `docs/REGEN2RL.md`
- `docs/README.md`
- `docs/papers/`

The whitepaper is the current long-form architecture and rigor statement.
`REGEN2RL.md` is the executable phase checklist and claim boundary. The papers
directory holds reference PDFs and a literature assessment.
