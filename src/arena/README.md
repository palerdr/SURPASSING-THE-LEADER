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

## Exact, Adaptive, and Exploit Hal

All three live providers return a distribution over literal seconds;
`PolicyDrivenAgent` remains the only component that masks and samples a legal
action.

- **Exact Hal** is the default. It plays the complete pure-DTH equilibrium.
- **Adaptive Hal** generates DTH-certified opponent-directed candidates and
  applies the hand-written posterior-confidence gate.
- **Exploit Hal v1** gives the same fixed candidates to a feed-forward
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
LP-constrained response for each configured epsilon. The tracked v1 grid is
`0, .0025, .005, .01, .02, .05, .10`; the per-game budget is configured
separately and is never increased to admit a larger candidate. Independent
post-solve checks recompute every candidate's worst-case loss, and cumulative
declared epsilon cannot exceed the game budget.

Opponent posteriors persist across games in one repeated-opponent session;
epsilon resets for each fresh game. Arena match and interactive loops deliver
each public reveal through explicit `reset_game`, `observe`, and `end_game`
hooks. The old history-scraping path remains only as compatibility for callers
that have not migrated.

Exploit Hal v1 falls back to exact DTH and spends no epsilon whenever either
relevant action space includes Baku's leap-only Dropper action 61. The reveal
remains in public transcripts and diagnostics but is not inserted into a
60-action posterior. No candidate is claimed to optimize action 61.

## Training, checkpoints, and evaluation

Tracked configurations live in `src/arena/config/`; generated checkpoints,
trajectories, and reports belong under gitignored `outputs/exploit-hal-v1/`.
Checkpoints bind model shapes to the named observation schema and exact ordered
feature list, epsilon grid, safety budget, activation, action count, and DTH
artifact compatibility. Loading is strict: missing, partial, incompatible, or
randomly initialized live actors are rejected.

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

The tracked overnight protocol runs four independent 500-update seeds
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
uv run python -m arena.policies.train_exploit_hal train --config src/arena/config/exploit_hal_smoke_v1.yaml --output-dir outputs/exploit-hal-v1/smoke

# 2. One resumable v1 seed (500-update total target).
uv run python -m arena.policies.train_exploit_hal train --config src/arena/config/exploit_hal_v1.yaml --output-dir outputs/exploit-hal-v1/v1 --resume outputs/exploit-hal-v1/v1/maskable-ppo.zip

# 3. Complete/resume the predeclared four-seed overnight protocol.
uv run python -m arena.policies.train_exploit_hal overnight --config src/arena/config/exploit_hal_v1.yaml --output-dir outputs/exploit-hal-v1/overnight-4x500

# 4. Validation evaluation of the smoke checkpoint.
uv run python -m arena.policies.train_exploit_hal evaluate --config src/arena/config/exploit_hal_smoke_v1.yaml --checkpoint outputs/exploit-hal-v1/smoke/checkpoint.pt --output-dir outputs/exploit-hal-v1/smoke-validation

# 5. Paired Exact/Adaptive/Exploit/oracle v1 final benchmark.
uv run python -m arena.policies.train_exploit_hal benchmark --config src/arena/config/exploit_hal_v1.yaml --checkpoint outputs/exploit-hal-v1/v1/checkpoint.pt --output-dir outputs/exploit-hal-v1/v1-benchmark

# 6. Checkpoint metadata inspection.
uv run python -m arena.policies.train_exploit_hal inspect --checkpoint outputs/exploit-hal-v1/v1/checkpoint.pt

# 7. Interactive deterministic Exploit Hal play.
uv run python -m arena play --hal-agent exploit-hal --exploit-hal-config src/arena/config/exploit_hal_v1.yaml --exploit-hal-checkpoint outputs/exploit-hal-v1/v1/checkpoint.pt --adaptive-prior-json outputs/luna-adaptive/learned-role-prior.json
```

For a fresh run, omit `--resume` from command 2. The tracked outcome-only
ablation is `src/arena/config/exploit_hal_outcome_only_smoke_v1.yaml` and sets
the exploit shaping weight to zero. Evaluation uses opponent seeds and latent
parameters disjoint from training, pairs seats/seeds/start clocks, and treats a
repeated-opponent session as the independent bootstrap unit. Small smoke runs
are throughput checks, not significance or equivalence evidence.
