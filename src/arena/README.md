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

The browser server is one repeated-opponent series, the same unit as
`arena play --games N`: one Hal is retained across games, game `N` is seeded
with the base seed plus `N`, and every finished game is appended to a public
transcript in the CLI's `arena-public-play-session-v1` shape. `GET
/api/transcript` serves that transcript plus the live game's resolved
half-rounds, and `--transcript PATH` rewrites the same JSON after every
finished game. `--public-hal-label`, `--conceal-hal-details`, `--pure-dth`,
and every agent option of `arena play` are accepted with the same meaning.
When `webclient/dist/` has been built, the Python server serves it at `/`, so
one process is the whole game.

Engine identities remain exactly `Hal` and `Baku`. `--human-name` and the web
session's `human_name` are presentation labels only; they never replace Baku's
rule-bearing identity. `Hal` is reserved as a display label. Browser session
replacement is a sequenced mutation, is allowed only before play or after a
terminal acknowledgement, and advances the sequence across the replacement.
The browser server's live provider set is `dth`, `adaptive-dth`,
`exploit-hal`, and, behind `--pure-dth`, `perfect-hal` and `pm-hal`; terminal
`perfect-hal --perfect-hal-model translated-v1` also supports canonical play
through its leap fallback. Terminal `arena play` offers `abstract`. The
retired `stl-mcts` surface is not advertised. Browser snapshots carry
server-owned character, role, and winner-seat fields, so the client never
infers identity from presentation labels.

The Vercel entrypoint uses `web/hosted.py` and `web/production.py` to isolate
players with secure cookies and Redis command logs. It replays accepted commands
through the same local HTTP adapter and commits each mutation before returning
a reveal. A process keeps the game it last served and rebuilds it from the
command log only when Redis shows that the game moved on elsewhere.
`web/ledger.py` then writes the game's public history to Supabase,
and `GET /api/leaderboard` ranks each player's best win by the winner's
seconds of life left. A restart and a next game each replace the record with fresh seeds, an empty
command list, and the next sequence number, so replay covers only the current
game. The hosted exact Hal keeps no memory. You can opt into translated Hal
with `STL_HAL_POLICY=translated-v1`. The hosted adapter then stores a private
opponent checkpoint with each game record and restores it before command
replay. Reload and next-game requests retain that evidence. Separate cookies
keep separate models; workers share the immutable tablebase. The local server
keeps one repeated-opponent series.
See [deployment instructions](web/DEPLOYMENT.md).

The browser requests a sequenced restart on page load. This abandons the active
game and clears the visible series before returning the title page. Ordinary
session reads still recover state for stale-request handling and worker replay.

## Exact, Adaptive, Exploit, Aggro, Perfect, and PM Hal

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

**Perfect Hal v2** uses a Bayesian opponent model for pure DTH. We keep
separate evidence for Dropper and Checker. We update latent predictor weights
with Bayes' rule under a fixed switching prior. We integrate categorical
predictions over possible change times and a grid of change hazards. We retain
64 run hypotheses per role and report the posterior mass discarded by pruning.
Context predictors learn action offsets and responses to Hal's past actions;
they use public load bands and periodic patterns. Their shrinkage prior uses
the observed prefix, an empirical Bayes approximation.

We score a forecast before its reveal and update it after resolution. We use
the posterior predictive mean in the certified DTH stage matrix. At zero
response temperature, we assign mass to the best-response action set. We keep
the tablebase as equilibrium continuation authority. This remains a one-step
exploit policy; it does not plan future learning or certify full-game safety.
We retain `PerfectHalOpponentModel` as the v1 reference for PM and comparisons.
The provider API defaults to `BayesianHalOpponentModel`. The terminal and local
browser keep v1 as their play default after the v2 evaluation. You can select
v2 with `--perfect-hal-model bayesian-v2`.

The name describes the policy's character, not a mathematical guarantee. It
is an unrestricted empirical exploiter, not a second DTH solver, a maximin
certificate, or a claim of unbeatable play. Its opponent memory persists
across one repeated-opponent session and is cleared only by `reset_session`.
Like Aggro Hal, it fails closed outside literal actions `1..60` and is exposed
only by explicit pure-DTH Arena surfaces.

### Frozen translated Hal candidate

You can select `--hal-agent perfect-hal --perfect-hal-model translated-v1`.
We preserve Old, Bayesian, and both ensemble variants. The new model extends
Old's expert mixture with offsets from Hal's previous action in the same role
and from Hal's last revealed action across roles. Each reference has copy and
mirror forecasts, with full-history and decayed counts. We score forecasts
before each reveal and use the same unrestricted matrix best response.

Validation selected expert-weight retention `.97` and offset retention `.97`.
The learning rate remains `1.25`. The frozen configuration and source hashes
live in `config/translated_hal_v1_selection.json`. Generic Perfect tuning
flags do not alter this selected variant.

The first holdout improved synthetic wins from 88.62% to 90.43% across 2,048
games per policy. Human-fitted results tied at 89.06% across 512 games per
policy. Its uncertainty interval failed the declared one-point regression
limit. We retained that failure and froze a larger confirmation without
changing the model or threshold.

The confirmation passed. Translated Hal won 7,493/8,192 synthetic games
(91.47%); Old won 7,320 (89.36%). The paired identity-bootstrap gain was
2.11 percentage points, with a 95% interval of [1.55, 2.66]. Against
human-fitted emulators it won 4,623/5,120 (90.29%); Old won 4,617 (90.18%).
That paired interval was [-0.33, 0.61] points. The emulator comparison has
eight source browser identities and does not establish a human win-rate gain.
Both providers completed these games without a stop.

Translated-response stress wins rose from 88.02% to 97.27% across 768 games.
The first holdout also compared Bayesian and the policy ensembles, plus an
offset-free ablation. Removing offsets lost the synthetic gain. Small family
regressions remain in confirmation; the largest was 0.59 points on multimodal
opponents. The candidate remains an empirical exploiter with no whole-game
safety guarantee.

The canonical adapter uses DTH equilibrium during leap turns and leaves Baku's
Dropper action 61 legal. It skips those reveals in the 60-action model and
clears sequence references while retaining prior evidence. It makes no
action-61 optimization claim. The generated reports live under
`outputs/translated-hal-v1/`; `config/translated_hal_v1_results.json` binds
their hashes. See [deployment instructions](web/DEPLOYMENT.md) for activation,
memory lifetime, and local runtime checks. No production deployment formed
part of this evaluation.

### External human repetition prior experiment

You can reproduce the research experiment from the repository root with a new
output directory:

```bash
uv run python -m arena.policies.evaluate_reward_prior fit --output outputs/external-hal-prior-v1-reproduction
uv run python -m arena.policies.evaluate_reward_prior evaluate --output outputs/external-hal-prior-v1-reproduction
```

The evaluator downloads the authors' human repeated-game CSV at a pinned
revision and checks its SHA-256. We used 118 people for fitting, 43 for
validation, and 34 for the external holdout. We excluded the Psych-201 text
conversion after finding inconsistent action/payoff labels. We fit repetition
after zero or positive reward, adjust repetition odds for 60 actions, and add
one expert to frozen translated Hal. Two neutral-prior controls separate the
adapter effect from the external prior effect.

We froze the model before target evaluation. The external adapter won
913/1,024 synthetic games; frozen translated Hal won 914/1,024. Both won
190/192 offset stress games. On 1,470 logged moves from ten existing browser
identities, the external prior produced no clear value gain over translated
Hal and worse prediction loss than the weak neutral prior. The external
holdout NLL gain over neutral was 0.0176 nats, with a paired 95% interval that
included zero.

`config/external_hal_prior_v1_results.json` binds the evidence and fitted
parameters. This experiment has no deployment integration. You need fresh
STL human participants to test human win-rate gains. Keep the translated
deployment candidate unchanged.

### Neural research pilots

You can train and test the four neural experiments from the repository root:

```bash
uv run python -m arena.policies.run_neural_pilots train --output outputs/neural-pilots-v1-reproduction
uv run python -m arena.policies.run_neural_pilots evaluate --output outputs/neural-pilots-v1-reproduction
```

Use a new output directory. We record the protocol before training and bind
selected checkpoints to their source hashes before test. We train on generated
public histories, with four opponent families excluded from both fitting and
validation. We preserve the existing Aggro experiments and their gates.

We compare a 42,444-parameter transformer with a 32,748-parameter GRU on the
same 16-token next-action task. We add each neural forecast to translated Hal
through online prediction scores. A 7,256-parameter selector learns weights for
the 24 existing statistical experts. Its inputs include their past errors.

We train a 42,749-parameter GRU actor against frozen translated Hal to search
for exploitable behavior. We train a second actor on rewards across four-game
sessions as a probing experiment. The actor retains hidden state across games.
We compare it with an immediate-payoff learner and a cleared-GRU control.
Transformer controls shuffle or clear its history window while retaining the
current public token and statistical memory. Higher session wins alone do not
establish active information seeking.

These research surfaces require pure DTH actions 1..60. We keep the canonical
leap adapter and deployment policy unchanged. Checkpoints and full results
belong under `outputs/neural-pilots-v1/`. You need fresh human games to test
human win-rate gains.

We fitted the predictors on 16,067 decisions and selected epochs on 5,135
validation decisions. On the 608-game league holdout, the selector won 555,
the transformer blend won 553, and the GRU blend won 552. Frozen translated
Hal won 541. The selector's paired gain was 2.30 percentage points with an
unadjusted 95% interval of [0.33, 4.28]. Its raw prediction loss fell from
3.095 to 2.822 nats. On the four reserved behavior families, it won 124/128
against translated Hal's 123/128. We treat these pilot comparisons as
exploratory and retain the deployment candidate.

The adversary won 90/256 against translated Hal; its untrained control won
95/256. The session-reward actor won 462/608 league games against the
immediate-payoff control's 545/608. Neither experiment supports its proposed
advantage under this training budget.

We repaired the first shuffle control because it consumed the action RNG.
We retained the original report and runner source under the output directory.
The correction changes two runner lines and leaves checkpoint bytes fixed.
A fresh 1,824-game audit found 550/608 wins for full transformer history,
550/608 for shuffled history, and 552/608 for cleared history. We found no
neural-history benefit. You can run this separate audit once per experiment:

```bash
uv run python -m arena.policies.audit_neural_pilot_memory --output outputs/neural-pilots-v1-reproduction
```

`config/neural_pilots_v1_results.json` binds the checkpoints and evidence,
including the superseded shuffle comparison and its corrected audit.

### Neural selector architecture study

You can reproduce the selector study with a new output directory:

```bash
uv run python -m arena.policies.run_selector_study train --output outputs/selector-study-v1-reproduction
uv run python -m arena.policies.run_selector_study evaluate --output outputs/selector-study-v1-reproduction
uv run --with matplotlib python -m arena.policies.plot_selector_study --output outputs/selector-study-v1-reproduction
```

We compare a 7,256-parameter MLP, a 161,560-parameter residual MLP, and a
24,001-parameter attention model. The attention model reads one token per
expert, including its forecast over 60 actions. Each model learns a bounded
correction to translated Hal's 24 expert weights. We retain the exact DTH
matrix for the response calculation.

We train each architecture with three seeds on 43,845 public decisions from
fresh synthetic identities. We select each seed's epoch on 9,483 validation
decisions, then compare three-seed ensembles on 624 validation games. We
select the architecture by game score before training its feature ablations.
We retrain without recent-error inputs, without public-context inputs, and
without inherited expert weights. If attention wins validation, we also
retrain without forecast-shape inputs. A fitted static correction and an
untrained network serve as controls. Removing context or error inputs leaves
indirect information through the statistical experts.

We freeze sources, checkpoints, and data hashes before test. The test uses
2,432 fresh games per variant, grouped into four-game opponent sessions. We
report paired bootstrap intervals over identities. Architecture and ablation
contrasts use unadjusted intervals. Four families remain outside this study's
training and selection; prior experiments included those family definitions.
We make no human win-rate claim. You can inspect the full protocol and frozen
artifacts under `outputs/selector-study-v1/`.

We selected attention after 600/624 validation wins, compared with 592 for
the residual MLP and 589 for the small MLP. We then ran 31,616 holdout games:

| Variant | Parameters per member | Members | Wins / 2,432 |
| --- | ---: | ---: | ---: |
| Old Hal | Statistical | 1 | 2,123 |
| Frozen translated Hal | Statistical | 1 | 2,198 |
| Prior neural pilot | 7,256 | 1 | 2,225 |
| Retrained small MLP | 7,256 | 3 | 2,232 |
| Residual MLP | 161,560 | 3 | 2,241 |
| Selected attention | 24,001 | 3 | 2,235 |

Attention gained 1.52 percentage points over translated Hal, with a paired
95% interval of [0.49, 2.59]. Its prediction loss fell from 3.164 to 2.863
nats per action. We found no resolved win-rate gain over the prior neural
pilot or the small MLP. The residual MLP's gain over the small MLP was 0.37
points, with an interval of [-0.37, 1.11]. We kept the validation selection.

The retrained no-errors control won 2,247 games. The no-context, no-prior,
and no-shapes controls won 2,234, 2,236, and 2,232. Their paired contrasts
with full attention include zero. The static correction won 2,218; one
attention seed won 2,235. These comparisons do not establish a need for
attention, each input group, or the three-member ensemble. The untrained
control matched translated Hal's game records, including its 2,198 wins.

We measured a mean session p95 of 0.507 ms for attention's decision code
on the local CPU, versus 0.229 ms for translated Hal. This timer excludes
environment transitions and hosted request overhead. No games reached the
half-round cap. We retain these networks as research artifacts and preserve
the deployment candidate. We have not integrated neural memory recovery or
canonical leap handling into a hosted provider.

`config/selector_study_v1_results.json` records compact evidence and artifact
hashes. The output directory contains the ablation plot, learning curves,
and expert-weight heatmap in PNG and SVG formats.

### Bayesian Perfect Hal evaluation

We follow the separation of opponent inference and response in
[Bayesian Opponent Exploitation](https://arxiv.org/abs/1603.03491).
We adapt [Bayesian change-point filtering](https://arxiv.org/abs/0710.3742)
with a hazard grid and posterior-mass pruning. Recent
[opponent modelling and planning work](https://proceedings.mlr.press/v235/huang24p.html)
supports treating those as separate components; we make no claim to reproduce
its planner.

You can run the frozen evaluator with an anonymous ledger export:

```sh
uv run python -m arena.policies.evaluate_bayesian_hal --human-data outputs/perfect-hal-bayes-v2/human-games.json --artifact outputs/perfect-hal-bayes-v2/tablebase --split test --output outputs/perfect-hal-bayes-v2/test-v2.json
```

You can play the Bayesian candidate with:

```sh
uv run python -m arena.web --hal-agent perfect-hal --perfect-hal-model bayesian-v2 --pure-dth --dth-complete-tablebase outputs/perfect-hal-bayes-v2/tablebase
```

You must choose a fresh output path. The test command checks source and input
hashes against `config/perfect_hal_bayes_v2_selection.json`. You need a new
protocol and fresh test seeds to select another model after this test.

We split each player's complete games into chronological training, validation,
and test partitions. A population prior excludes the target player's identity
and uses other players' training partitions. We retain incomplete games as
prediction evidence and exclude leap turns from pure-DTH scoring. We clear
pattern history after an excluded turn. Logged-state scores measure a one-step
deviation followed by pure-DTH equilibrium, not a human full-game win rate.

We measure full-game wins in paired-seat simulations against 14 reactive
families plus uniform and equilibrium opponents. We compare v1, equilibrium,
and a memory-reset control. Human-fitted categorical and response emulators
use training games. We group their uncertainty by source browser identity.
Those emulators omit unobserved human responses, and several browser identities
can belong to one person. A live randomized comparison is required to establish
a human win-rate uplift. No production policy or database schema changes form
part of this experiment.

The September 21 v2 test used 98 recorded games and 1,489 public moves from ten
browser identities. Eight identities had enough games for a held-out partition.
Bayesian Hal won 677/768 synthetic games; equilibrium won 393/768 and v1 won
675/768. The paired score gain over v1 was 0.26 percentage points with a 95%
identity-bootstrap interval of [-1.56, 2.08]. Against human-fitted emulators,
Bayesian Hal won 114/128, equilibrium won 63/128, and v1 won 120/128. We retain
v1 as the play default because v2 did not establish an improvement over it.
The memory-reset control stopped 40 synthetic games; reports distinguish raw
wins from scores that assign half a point to a stopped game.

The `PaleRider` test partition contained eleven ordinary-turn decisions.
Bayesian Hal's mean one-step gain over equilibrium was 3.64 percentage points
under the pure-DTH projection. This is a small logged-state estimate, not an
observed human win-rate increase. The generated report lives at
`outputs/perfect-hal-bayes-v2/test-v2.json`; `config/perfect_hal_bayes_v2_results.json`
records its hash and compact results.

### Policy ensemble experiment

You can select the Old/Bayesian policy ensemble with
`--perfect-hal-model ensemble` on the terminal or local browser in pure DTH.
Both constituent providers propose distributions from the same public history.
We fetch one certified stage and mix the proposed policies before the reveal.
We retain separate Old/Bayesian weights for each of Hal's roles across games.

After the reveal, we score each frozen proposal against the opponent's action
with `M` for Dropper or `-M.T` for Checker. We map utility to `[0, 1]` and apply
`posterior = normalize(weight * exp(2 * reward))`, followed by
`weight = .96 * posterior + .02`. Each model retains at least 2% weight.
Both models learn from the reveal, regardless of Hal's sampled action.
Session reset clears both models and restores equal weights.

We froze the learning rate and sharing mass before the experiment in
`config/perfect_hal_ensemble_v1.json`. The evaluator compares exact, Old,
Bayesian, fixed 50/50, and adaptive mixtures on fresh paired-seat simulations:

```sh
uv run python -m arena.policies.evaluate_ensemble_hal --artifact outputs/perfect-hal-bayes-v2/tablebase --human-data outputs/perfect-hal-bayes-v2/human-games.json --output outputs/perfect-hal-ensemble-v1/test-v1.json
```

You must choose a new output path for each run. Reports bind source and input
hashes and group uncertainty by opponent identity. Human emulator variants and
replicates share their source browser identity for uncertainty estimates. We
reuse training prefixes from the prior experiment; fresh simulation seeds do
not create a new human holdout. We count stopped games as non-wins and report
them apart from losses. Certified stage rewards assume equilibrium continuation;
they do not certify the ensemble's full-game win rate or resistance to exploitation.

The frozen experiment completed 10,240 games with no stopped games. Old won
1,370/1,536 synthetic games (89.19%); the ensemble won 1,355 (88.22%), the equal
mixture won 1,362 (88.67%), and Bayesian won 1,349 (87.83%). Against human-fitted
emulators, Old won 473/512 (92.38%); both mixtures won 465 (90.82%), and Bayesian
won 462 (90.23%). The ensemble-minus-Old 95% identity-bootstrap intervals were
[-2.02, 0.07] percentage points for synthetic opponents and [-3.32, 0.20] for
human-fitted opponents. We retain Old as the play default. These results do not
establish an ensemble improvement or a difference in real human win rates.

**Perfect Mode (PM) Hal** is the synthesis layer for repeated pure-DTH play.
It keeps Adaptive Hal's role-separated Dirichlet evidence and independently
checked epsilon frontier, Perfect Hal's interpretable pattern forecast, a
categorical Bayesian online change-point model, an outcome-conditioned expert,
and—when a compatible checkpoint is supplied—Aggro Hal's GRU forecast and
direct residual policy. The sources earn role-separated authority by true
Fixed Share updates under reveal-time prequential log loss. Uniform and the
opponent's exact equilibrium policy remain explicit recovery sources.

PM Hal turns evidence into four modes. `shield` admits only the exact face at a
cold start or after a change shock; `probe` spends at most `.05` local epsilon;
`press` admits up to `.50`; and sustained high-confidence agreement can enter
`dominate` at up to `2.0` local epsilon within a `12.0` per-game budget. These
values are deliberately overbearing and are registered in
`pm_hal_controller_v3.json`. Every chosen direct or frontier policy has its
actual local worst-case loss recomputed from the fresh exact matrix. Frontier
candidates are charged their declared epsilon; direct candidates are charged
their measured loss. The provider records both and refuses any policy that
would exceed the active mode or game budget.

The confidence product—effective evidence, low change probability, forecast
agreement, and prequential skill—is a PM-specific controller heuristic. Fixed
Share has a switching-expert interpretation, the evidence fraction is inspired
by data-biased robust response, and the local epsilon checks are exact for the
current matrix. Their product is not a new whole-game safety theorem. The name
describes Hal's intended character: fast pressure against stable habits,
immediate retreat after a surprise, and renewed pressure when the new regime is
supported. It does not claim mind reading, human psychological validity, or
unbeatable play.

The optional Aggro checkpoint is explicit. Omitting it leaves a complete,
checkpoint-free PM controller and reports `aggro_enabled: false`; PM never
pretends an absent or incompatible recurrent model contributed evidence. A
compatible network is used as one forecast and one independently risk-measured
direct-policy candidate, never as continuation-value authority.

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

# Checkpoint-free hard-best-response match. Canonical STL play omits Perfect Hal.
uv run python -m arena match --candidate perfect-hal --opponent dth --pure-dth --games 50 --output outputs/perfect-hal-v1/vs-exact.json

# Build the current schema-v2 exact artifact at the Arena default path.
uv run python -m dth complete output_dir=src/dth/artifacts/complete_full_v1 report_path=outputs/pm-hal/dth-complete-v2-report.json backend=rust lp_workers=4 progress_every=50

# Frozen bounded training run for PM's compatible recurrent component candidate.
uv run python -m arena.policies.train_aggro_hal train --config src/arena/config/pm_hal_aggro_component_v1.yaml --output-dir outputs/pm-hal/aggro-component-v1

# Full PM Hal match with the compatible recurrent component.
uv run python -m arena match --candidate pm-hal --opponent dth --pure-dth --games 50 --dth-complete-tablebase src/dth/artifacts/complete_full_v1 --pm-hal-aggro-checkpoint outputs/pm-hal/aggro-component-v1/checkpoint.pt --output outputs/pm-hal/v3/vs-exact.json

# Human play on the permanent 1..60 pure-DTH surface.
uv run python -m arena play --hal-agent pm-hal --pure-dth --dth-complete-tablebase src/dth/artifacts/complete_full_v1 --pm-hal-aggro-checkpoint outputs/pm-hal/aggro-component-v1/checkpoint.pt

# Git-registered v3 confirmation: 56 identities, every family, common sessions,
# no-Aggro ablation, and independent matrix-based risk audit.
uv run python -m arena.policies.evaluate_pm_hal --config src/arena/config/pm_hal_evaluation_v3.json --pm-config src/arena/config/pm_hal_controller_v3.json --artifact-dir src/dth/artifacts/complete_full_v1 --aggro-checkpoint outputs/pm-hal/aggro-component-v1/checkpoint.pt --output outputs/pm-hal/v3/evaluation-report.json
```

The artifact directory is generated and gitignored despite its historical
`complete_full_v1` profile name; PM requires manifest schema
`dth.complete-tablebase.v2` and fails with a rebuild command when an older
artifact is opened. The recurrent checkpoint is also generated. The v3
protocol binds the controller bytes, checkpoint, and table digest by SHA-256,
requires a clean Git worktree, and records the executing commit before opening
any registered identity.

The v1 and v2 configurations and reports are retained as development history.
They were not durably registered in Git before execution, so their status text
does not establish preregistration and their promotion gates carry no release
authority. The v2 development run produced 44 wins and 4 losses for PM Hal in 48
games (`.9167` all-game score), compared with Perfect `.8958`, Adaptive
`.6563`, Aggro `.6458`, and Exact `.5417`. The opponent-identity-clustered
paired difference was `+.3750 [.2083, .5417]` against Exact and
`+.0208 [-.0417, .0833]` against Perfect. These descriptive results motivated
the repaired v3 confirmation; they do not promote the controller. The v2
evaluator also trusted controller-supplied risk diagnostics, reused one random
stream across identities, and used a permissive change-alarm metric. V3 closes
those proof gaps with independent stage-matrix recomputation, identity-seeded
randomness, alarm-onset accounting, a no-Aggro ablation, and seat/family gates.

The Git-registered v3 protocol ran once from commit
`b76147fc3718492353b24131ce756c558c4c43c8`. The immutable compact evidence
record is `config/pm_hal_confirmation_v3.json`; the full generated report stays
gitignored and is bound there by SHA-256. PM scored 97 wins, 14 losses, and one
stopped game in 112 games (`.8705`), behind Perfect's 107-5 (`.9554`) but ahead
of PM without Aggro `.8259`, Aggro `.7232`, Adaptive `.7188`, and Exact
`.4955`. Paired PM-minus-baseline differences were
`-.0848 [-.1518,-.0223]` against Perfect, `+.0446 [.0089,.0893]` against no-Aggro,
and `+.3750 [.2723,.4777]` against Exact. The promotion gate failed because
PM was inferior to Perfect in aggregate and in both seats, and at least one
family slice was negative. V3 therefore freezes a failed promotion result,
not a release claim.

The fused forecast also improved opponent-identity-clustered expected log loss
in that development run against every individual source. Its difference from the Perfect source was
`-.2520 [-.3409, -.1622]` nats and from uniform was
`-.6137 [-1.1842, -.1187]`; negative means the fused forecast was better.
These are synthetic seed-holdout measurements. They establish neither human
psychological validity, the Aggro component's causal contribution, promotion,
nor a whole-game safety theorem. In registered v3, the independent audit found
zero violations across 788 decisions and 112 games; maximum measured local
worst-case loss was `1.5373`. The enabled Aggro component improved aggregate
score over the no-Aggro ablation, while that comparison does not isolate a
causal recurrent-memory advantage. Fused expected NLL was `3.5524` versus
uniform `4.0943`; the change detector found only 35 of 168 truth shifts and
produced 23 false-alarm onsets. These mixed diagnostics remain synthetic.

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

## Browser deployment

See [web/DEPLOYMENT.md](web/DEPLOYMENT.md) for the certified recurrence build,
local launch commands, and the per-player session changes needed for Vercel.
The complete DTH reader accepts its source-bound v3 artifact. The browser
keeps the existing commit-before-sampling and reveal contracts.
