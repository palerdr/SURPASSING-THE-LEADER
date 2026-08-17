# Arena Project Instructions

`src/arena/` is the neutral executable surface for matches between peer projects.
It may import public interfaces from `stl`, `dth`, and `abstract`; peer projects
must not import one another in return.

- The STL engine remains the only canonical live-game referee.
- The completed DTH tablebase is the default Hal policy provider.
- Providers return policy distributions; `PolicyDrivenAgent` alone masks and
  samples literal legal seconds.
- `adaptive_dth.py` is the one-step exploit layer: callers supply population
  Dirichlet priors, revealed actions update separate role posteriors, and the
  complete DTH matrix constrains every selected policy under a per-game
  epsilon budget. It falls back to equilibrium when evidence is weak or an
  opponent action lies outside DTH's 1..60 contract.
- `--adaptive-prior-json` accepts a versioned pair of learned role means or a
  mixture of role archetypes and strengths; the CLI never fits that prior
  during live play. Session
  transcripts include bounded per-game exploit, epsilon, fallback, and
  saddle-gap diagnostics for offline validation.
- `arena play --games N` is the repeated-opponent surface: one Hal provider is
  retained across games while each game receives a fresh canonical referee and
  seed. `--start-clock-sequence` can counterbalance canonical and leap-focused
  starts inside that same posterior. `--transcript` records only public
  pre-decision states, revealed actions, and outcomes for generated experiment
  data.
- `--public-hal-label` permits blinded policy comparisons without changing the
  provider recorded in the transcript. `--conceal-hal-details` keeps provider
  summaries and diagnostics in the transcript without printing them to the
  player.
- `arena play` opens with the ordinary-turn rules and waits for Enter on an
  interactive terminal before the first action. The full-screen and plain-text
  modes share the same rules text. Piped sessions do not consume an action as
  acknowledgement; automation may suppress the screen with `--skip-rules`. The
  opening screen intentionally does not disclose the leap-window advantage.
- DTH projection is exact for the shared state and actions 1..60. The only
  prospective mismatch is Baku's legal Dropper action 61 in the public leap
  window; arena keeps that canonical action even though DTH has no 61 policy.
- Projection adapters may not alter canonical game state or transitions.
- Keep generated artifacts in the owning project, never under `src/arena/`.

## Play surfaces and the session

`session.py` owns the phase machine every interactive surface drives:
`RULES -> AWAITING_ACTION -> AWAITING_ACK -> GAME_OVER`. It performs no I/O and
no rendering; it only sequences the referee calls. `cli.py` and `web/app.py` are
both thin adapters over it, so a rules change lands in one place.

Hal's action is chosen inside `PlaySession.submit`, after the human's second has
been accepted and validated. That ordering is the hidden-information guarantee,
not a convenience: while a client is deciding, Hal's second does not exist in
the process, so no snapshot can leak it. Do not hoist that call earlier to
"prepare" a move.

`web/app.py` serves the TypeScript client in `webclient/`. It builds its
provider once at startup — provider construction memory-maps a
multi-gigabyte artifact and the `abstract` provider can build a tablebase
outright, so neither may happen on a request path; `python -m arena.web`
refuses `--hal-agent abstract` for that reason. `web/schema.py` holds the only
serializer that faces the browser, so the seat-scoping rule has exactly one
place to be enforced and one place to be tested.

Engine identities remain exactly `Hal` and `Baku`. `--human-name` and the web
session's `human_name` are presentation labels only; they never replace Baku's
rule-bearing identity. `Hal` is reserved as a display label. Browser session
replacement is a sequenced mutation, is allowed only before play or after a
terminal acknowledgement, and advances the sequence across the replacement.
The browser server's live provider set is `dth`, `adaptive-dth`, and
`exploit-hal`; terminal `arena play` additionally offers `abstract`. The
retired `stl-mcts` surface is not advertised. Browser snapshots carry
server-owned character, role, and winner-seat fields, so the client never
infers identity from presentation labels.

## Exact, Adaptive, Exploit, and Aggro Hal

All policy providers return a distribution over literal seconds;
`PolicyDrivenAgent` remains the only component that masks and samples a legal
action.

- **Exact Hal** is the default. It plays the complete pure-DTH equilibrium.
- **Adaptive Hal** generates DTH-certified opponent-directed candidates and
  applies the hand-written posterior-confidence gate.
- **Exploit Hal** gives the same fixed candidates to a feed-forward
  actor-critic. The network chooses a candidate index, never a literal second
  or an arbitrary action distribution. Its public observation contains the
  canonical state, role-separated Bayesian posterior, per-game safety budget,
  candidate diagnostics, and current tablebase certificate.
- The **one-step oracle** is evaluation-only. It sees a synthetic opponent's
  true current role distribution and chooses the best valid member of the same
  candidate family. It is a ceiling for that family, not a dynamic best
  response.

Exploit Hal is deliberately not a new game solver. The completed DTH tablebase
still supplies the exact minimax value and equilibrium. The actor-critic learns
only a meta-policy over the certified candidate menu: when to stay at exact
equilibrium, when an observed opponent bias is worth targeting, and which
local epsilon allowance to spend. It does not learn transitions, literal
actions, the tablebase, or the opponent posterior.

The candidate family always starts with exact equilibrium, followed by one
LP-constrained response for each configured epsilon. The tracked grid is
`0, .0025, .005, .01, .02, .05, .10`; the per-game budget is configured
separately and is never increased to admit a larger candidate. Independent
post-solve checks recompute every candidate's worst-case loss, and cumulative
declared epsilon cannot exceed the game budget.

Opponent posteriors persist across games in one repeated-opponent session;
epsilon resets for each fresh game. Arena match and interactive loops deliver
each public reveal through explicit `reset_game`, `observe`, and `end_game`
hooks. The old history-scraping path remains only as compatibility for callers
that have not migrated.

Exploit Hal falls back to exact DTH and spends no epsilon whenever either
relevant action space includes Baku's leap-only Dropper action 61. The reveal
remains in public transcripts and diagnostics but is not inserted into a
60-action posterior. No candidate is claimed to optimize action 61.

**Aggro Hal** is the separate unrestricted pure-DTH path. It does not select
from the certified epsilon menu: a two-layer GRU predicts the opponent's next
literal action, the exact continuation-adjusted stage matrix converts that
forecast into `M @ q` as Dropper or `-M.T @ q` as Checker, a learned residual
adjusts all 60 action logits, and a learned gate may move completely away from
equilibrium. Its hidden state persists for one repeated-opponent session and
resets only when the opponent identity changes. This is intentionally an
empirical exploit policy, not a maximin certificate.

`--aggro-hal-fast-adaptation` adds an optional public-history fast path for the
provider. Role-separated, exponentially decayed action evidence is blended
into the learned forecast only when that evidence is concentrated; diffuse
opponents leave authority with the network. The blend is cleared at session
boundaries and never sees an unrevealed simultaneous action. It is an explicit
hand-written adapter, not evidence that the GRU learned to adapt, and should be
selected only by validation rather than assumed to be stronger.

Aggro training and evaluation use `PureDTHGame`, which reuses the shared
canonical resolution, revival, load, and clock mechanics while permanently
fixing the turn to actions `1..60`. It therefore never inherits STL's
leap-window action 61, even if a long game crosses that wall-clock interval.
The provider fails closed before tablebase or model inference outside that
contract. Aggro is not exposed by canonical `arena play`; agent matches must
opt in with `arena match --pure-dth`.

## Training, checkpoints, and evaluation

Tracked configurations live in `src/arena/config/`; generated checkpoints,
trajectories, and reports belong under gitignored `outputs/exploit-hal-v2/`.
The supported training-protocol schema is v2. The three v1 configuration files
are retained byte-for-byte as historical declarations, but are intentionally
incompatible because they name scripted opponents from the removed STL play
stack; the loader fails before opening an exact artifact. They are not silently
rewritten to describe a different experiment.
The live checkpoint schema remains independently versioned, while v2 protocol
runs and their recovery state stay under `outputs/exploit-hal-v2/`.
Checkpoints bind model shapes to the named observation schema and exact ordered
feature list, epsilon grid, safety budget, activation, action count, and DTH
artifact compatibility. Loading is strict: missing, partial, incompatible, or
randomly initialized live actors are rejected.

Exploit Hal training accepts only the in-tree opponent kinds `categorical`,
`switching`, `uniform`, and `exact`. Oracle-backed validation and evaluation
accept `categorical`, `switching`, and `uniform`; `exact` is excluded because it
does not expose the state-independent truth callback required by the one-step
oracle. Removed STL scripted opponents are rejected while loading the
configuration, before an exact artifact or training runtime is initialized.

Training uses Gymnasium plus `sb3-contrib` `MaskablePPO`. The Gym action is a
candidate-policy index (`Discrete(8)` in the tracked configuration), not a
literal second. Its dynamic mask rejects candidates that failed certification,
would exceed the remaining per-game epsilon budget, or are outside pure DTH's
action contract. In the leap-only 61 case that leaves exact candidate zero as
the safe fallback. Each canonical game is one Gym episode; the opponent
posterior persists across all games in its repeated-opponent session.

The trainer preserves the configured experimental unit exactly: it collects
`sessions_per_update * games_per_session` games into an SB3 maskable rollout
buffer before each optimizer update. It uses `gamma=.99`, GAE lambda `.95`,
clip `.20`, value coefficient `.50`, entropy coefficient `.01`, and gradient
norm `.50` by default. Its dense counterfactual term compares the selected
policy distribution with equilibrium against the revealed opponent action.
This term is intentionally policy-shaping, not policy-invariant; safety comes
from the candidate certificates and cumulative epsilon budget.

Every update writes `maskable-ppo.zip`, `trainer-state.json`, and
`rng-state.pt` for exact recovery, and exports `checkpoint.pt` in the strict v1
live schema used by evaluation and arena play. A resumed run treats `updates`
as the total target, restores Python/NumPy/PyTorch random state, preserves its
update history, and runs only the unfinished updates. Training resumes should
use the SB3 archive; the live checkpoint path is a warm start rather than the
overnight recovery path. Each completed update also commits an immutable bundle
under `recovery/update-NNNN` and atomically advances `recovery/latest.json`, so
the overnight orchestrator ignores partially written latest files after an
interruption. The every-25-update snapshots are longer-lived experiment
landmarks rather than the recovery transaction.

The tracked v2 overnight protocol runs four independent 500-update seeds
sequentially. Each update still contains 24 repeated-opponent sessions of eight
games. It writes a complete snapshot every 25 updates and evaluates every 50
updates on 64 held-out sessions of eight games. Best-checkpoint eligibility
requires zero candidate-certificate violations and zero per-game epsilon-budget
violations. Eligible checkpoints are ranked lexicographically by worst and
mean paired win-rate improvement over Exact/Adaptive Hal, mean one-step oracle
regret, and the weakest seat/start-clock slice. PPO entropy, KL, clip fraction,
candidate frequencies, epsilon spend, and safety maxima remain in the training
report.

All four runs and their validation-based best snapshots finish before Arena
writes `selected/selection-commitment.json`. Only then does it open the
separate 128-session final-test seed namespace. Final-test results are reported
for all four seeds to measure training stability, but they cannot change the
committed selected seed. Re-running the overnight command resumes incomplete
runs and reuses compatible completed final reports.

Reproducible commands from the repository root:

```bash
# 1. Throughput-calibration smoke training.
uv run python -m arena.policies.train_exploit_hal train --config src/arena/config/exploit_hal_smoke_v2.yaml --output-dir outputs/exploit-hal-v2/smoke

# 2. One resumable v2-protocol seed (500-update total target).
uv run python -m arena.policies.train_exploit_hal train --config src/arena/config/exploit_hal_v2.yaml --output-dir outputs/exploit-hal-v2/v2 --resume outputs/exploit-hal-v2/v2/maskable-ppo.zip

# 3. Complete/resume the predeclared four-seed overnight protocol.
uv run python -m arena.policies.train_exploit_hal overnight --config src/arena/config/exploit_hal_v2.yaml --output-dir outputs/exploit-hal-v2/overnight-4x500

# 4. Validation evaluation of the smoke checkpoint.
uv run python -m arena.policies.train_exploit_hal evaluate --config src/arena/config/exploit_hal_smoke_v2.yaml --checkpoint outputs/exploit-hal-v2/smoke/checkpoint.pt --output-dir outputs/exploit-hal-v2/smoke-validation

# 5. Paired Exact/Adaptive/Exploit/oracle v2-protocol final benchmark.
uv run python -m arena.policies.train_exploit_hal benchmark --config src/arena/config/exploit_hal_v2.yaml --checkpoint outputs/exploit-hal-v2/v2/checkpoint.pt --output-dir outputs/exploit-hal-v2/v2-benchmark

# 6. Checkpoint metadata inspection.
uv run python -m arena.policies.train_exploit_hal inspect --checkpoint outputs/exploit-hal-v2/v2/checkpoint.pt

# 7. Interactive deterministic Exploit Hal play.
uv run python -m arena play --hal-agent exploit-hal --exploit-hal-config src/arena/config/exploit_hal_v2.yaml --exploit-hal-checkpoint outputs/exploit-hal-v2/v2/checkpoint.pt
```

For a fresh run, omit `--resume` from command 2. The supported outcome-only
ablation is `src/arena/config/exploit_hal_outcome_only_smoke_v2.yaml` and sets
the exploit shaping weight to zero. Evaluation uses opponent seeds and latent
parameters disjoint from training, pairs seats/seeds/start clocks, and treats a
repeated-opponent session as the independent bootstrap unit. Small smoke runs
are throughput checks, not significance or equivalence evidence.

### Aggro Hal training and evaluation

Aggro Hal's custom recurrent trainer treats a whole repeated-opponent session
as one episode. Warm-start actions come from a seeded `75% exact + 25% uniform`
behavior policy and therefore do not reveal simulator truth through Hal's own
history. Only after an action is committed does the training record expose the
synthetic opponent distribution. The predictor uses ordinary categorical
cross-entropy, while the actor maximizes expected payoff under that target in
the exact DTH stage matrix. Optional recurrent PPO then uses terminal outcomes
across the complete session while retaining prediction and tactical losses.
There is no forced-open gate loss. The default device is CPU; CUDA is used only
when explicitly requested.

The opponent league has immutable disjoint train, validation, and test family
manifests plus a registered audit manifest with unseen parameter seeds. It
includes stationary shapes, deterministic and periodic policies, copy/counter
rules, switches, retreat after detected exploitation, and bait-then-reverse
behavior. Evaluation pairs common scenarios across both seats, compares Exact
DTH, reports NLL/Brier against uniform, and treats opponent family plus
parameter seed as the independent unit. Its primary uplift statistic scores a
win/loss/cap as `1/0/.5` and bootstraps opponent identities; pooled decisive
rates and Wilson intervals remain secondary diagnostics.

Generated artifacts belong under `outputs/aggro-hal-v1/`:

```bash
# CPU smoke train.
uv run python -m arena.policies.train_aggro_hal train --config src/arena/config/aggro_hal_smoke_v1.yaml --output-dir outputs/aggro-hal-v1/smoke

# Full tracked warm start plus recurrent PPO target.
uv run python -m arena.policies.train_aggro_hal train --config src/arena/config/aggro_hal_v1.yaml --output-dir outputs/aggro-hal-v1/v1

# CPU-only validation. Model shape is read from the strict checkpoint.
uv run python -m arena.policies.evaluate_aggro_hal --checkpoint outputs/aggro-hal-v1/v1/checkpoint.pt --split validation --output outputs/aggro-hal-v1/v1/validation-report.json

# Causal recurrent-memory probe at one- and eight-cover delays.
uv run python -m arena.policies.evaluate_aggro_hal_memory --checkpoint outputs/aggro-hal-v1/v1/checkpoint.pt --output outputs/aggro-hal-v1/v1/memory-latent-twin-report.json --protocol-output outputs/aggro-hal-v1/v1/memory-latent-twin-protocol.json --twin-seeds 32 --cover-games 1 8 --bootstrap-replicates 5000

# Pure-DTH agent match. Canonical STL play intentionally does not offer Aggro.
uv run python -m arena match --candidate aggro-hal --opponent dth --pure-dth --games 50 --aggro-hal-checkpoint outputs/aggro-hal-v1/v1/checkpoint.pt --output outputs/aggro-hal-v1/v1/vs-exact.json
```

The audit manifest is a one-way door: predeclare checkpoint and ablations,
record checkpoint/manifest hashes, run the memory, reset, adapter, and
history-free conditions together, and do not tune the selected model after
reading those results. A strong exploit win rate alone does not establish that
recurrent memory or long-horizon adaptation caused the wins.

The latent-twin probe isolates recurrent state from the ordinary provider
lifecycle. It builds two legal, equal-length public histories whose old
opponent-action cues differ, then evaluates correct, swapped, and zero hidden
states on one byte-identical target observation. `--cover-games N` means the
policy and forecast have processed `N` identical cover reveals before the
measured output. Expected payoff is scored exactly against two checkpoint-
independent target distributions with conflicting best responses, and paired
bootstrap intervals resample twin seeds. Merely detecting different hidden
states or outputs is history sensitivity; an adaptation claim additionally
requires correctly directed, practically meaningful gains in every role and
latent mode.

### Adaptive exploitation capability gate

`aggro_hal_adaptive_exploitation_goal_v1.json` freezes the promotion ladder,
including a sealed, unopened audit seed reservation. Validation can select a
candidate, but only a later hash-bound audit may establish the capability. The
warm-start experiment config binds that goal by canonical JSON hash, and the
trainer carries the binding into every checkpoint and training report.

`aggro_hal_tactical_baseline_v1.json` freezes `corrected-v1` as the tactical
baseline by checkpoint, training, DTH, and evidence hashes. Its supported
claim is deliberately narrow: strong unrestricted exploitation in the audited
synthetic pure-DTH league. It does not claim learned-memory advantage,
forecast calibration, long-horizon adaptation, or human-opponent
generalization.

`aggro_memory_curriculum.py` provides an injectable memory-necessity task.
Matched mode A/B sessions are legal public histories with byte-identical
current target tensors; only older revealed actions identify the mode. The two
modes have conflicting unique exact best responses in both roles. Prefix and
cover tokens update the GRU, while an objective mask restricts privileged
supervision to the target decision. Train and validation namespaces are
immutable and disjoint; training uses cover delays 2/4/6 and validation uses
an unseen eight-game cover.

The tracked first capability experiment keeps the existing 128-wide,
two-layer GRU, initializes weights from `corrected-v1`, resets the optimizer
and counters, performs 60 supervised warm-start updates, and performs zero PPO
updates:

```bash
uv run python -m arena.policies.train_aggro_hal train --config src/arena/config/aggro_hal_adaptive_memory_v1.yaml --output-dir outputs/aggro-hal-v1/adaptive-memory-v1 --initial-checkpoint outputs/aggro-hal-v1/corrected-v1/checkpoint.pt

uv run python -m arena.policies.evaluate_aggro_hal_adaptive --checkpoint outputs/aggro-hal-v1/adaptive-memory-v1/checkpoint.pt --output outputs/aggro-hal-v1/adaptive-memory-v1/candidate-validation.json --protocol-output outputs/aggro-hal-v1/adaptive-memory-v1/validation-protocol.json --split validation --bootstrap-replicates 5000 --bootstrap-seed 20260809
```

Promotion has no pooled escape hatch. In every Dropper/Checker by mode A/B
cell, the 95% lower bound for correct memory versus swapped, zero, and
history-free controls must exceed `.02` normalized exact payoff and `.01` nat
forecast NLL after the eight-game cover. The first warm-start candidate failed
that gate: correct and swapped histories remained nearly indistinguishable in
all four cells. That development result remains with its generated reports
under gitignored `outputs/`; it is not a tracked final-source experiment. PPO
remains locked, and `corrected-v1` remains the tactical policy. A new
optimization hypothesis must use a versioned protocol and fresh held-out
seeds; the failed validation namespace is not retried or weakened.

In this fixed-target harness, reset-before-every-token history-free evaluation
is required to equal the direct zero-hidden target intervention. It verifies
the provider/reset path but is not a separately trained, capacity-matched
history-free architecture; an architecture comparison would require that
additional model.
