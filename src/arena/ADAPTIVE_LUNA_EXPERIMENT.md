# Adaptive Hal Luna experiment

Status: preregistered before the discovery cohort.

Analysis amendment, fixed after all transcripts were collected but before any
study action distribution or outcome was inspected: cell means use the
hierarchical backoff and action-61 treatment specified below. Live collection
checks had been limited to completed-game and half-round counts.

## Decision

Choose the smallest opponent model and safest exploitation schedule that make
Adaptive Hal predict a previously unseen player's revealed actions better than
a role-only population baseline without evidence of worse play than complete
DTH. Match wins are a policy guardrail, not the criterion used to choose the
opponent-model structure.

## Information boundary

Every player is a fresh `gpt-5.6-luna` Codex agent at `xhigh` reasoning. The
agent receives the complete public L2 rules and an opaque play command. It may
choose every Baku action freely, learn during a session, and change strategy.
It may not inspect files, source, transcripts, tablebases, policy diagnostics,
or use any command other than the supplied play command and terminal input.
Codex JSON events are retained to audit command compliance.

One `arena play --games N` process is one repeated-opponent session. Hal keeps
the opponent posterior across games; each game gets a fresh canonical referee,
player state, safety budget, and seed. The transcript contains only public
pre-decision state, revealed actions, and outcomes.

## Stage 1: structure discovery

- 12 independent Luna contexts.
- 8 complete-DTH games per context, for 96 games total.
- Four games start at the canonical 8:12 clock (`720`) and four use the
  leap-focused 8:57 clock (`3420`). Start order is counterbalanced across
  contexts while the public rules and agent prompt remain identical.
- Each context receives a unique eight-seed block. Discovery seeds do not
  appear in confirmation.
- A Luna context, not an action or game, is the independent holdout unit.

Candidate predictors are fitted only to Baku's revealed actions:

1. role only;
2. role plus Baku load band;
3. role plus Hal load band;
4. role plus previous half-round outcome;
5. role plus leap phase;
6. role plus Baku load band and previous outcome;
7. the full coarse combination of both load bands, previous outcome, and leap
   phase;
8. a two-archetype role model, if it has enough training contexts to fit both
   components.

Load bands are fixed before inspection as `0`, `1..60`, `61..120`, `121..180`,
`181..239`, and `240+`. Previous outcome is `opening`, `check success`,
`failed check survived`, `failed check died`, `overflow survived`, or
`overflow died`. Leap phase is `before`, `window`, or `after`.

For every candidate, predictive probabilities use Dirichlet smoothing and are
evaluated by leave-one-Luna-context-out negative log likelihood. Prior strength
and exponential evidence decay are selected within each training fold from
fixed grids:

```text
prior strength: 0.25, 0.5, 1, 2, 4, 8, 16
decay:          0.5, 0.7, 0.9, 1.0
```

Hyperparameters are selected by an inner leave-one-context-out loop inside
each outer holdout. Training role means use a symmetric `0.5` pseudocount per
ordinary action. A conditioning cell backs off to its training role mean with
eight pseudo-observations:

```text
mu_cell = (cell_action_counts + 8 * mu_role) / (cell_total + 8)
```

The held-out player's online evidence starts at zero. Before each same-role
update, all evidence cells for that role are multiplied by the candidate
decay; the revealed ordinary action is then added to its current cell. This
matches the live model's interpretation that recency applies to an opponent's
role behavior rather than only when an identical state recurs.

Primary predictive loss covers ordinary actions `1..60`, the domain Adaptive
DTH can exploit. Baku's leap-only Dropper action `61` is retained in the corpus
and reported separately as an unsupported-action rate and fallback diagnostic;
it cannot select a phase-conditioned predictor merely by being outside DTH's
action contract.

The selected structure is the least complex candidate within one standard
error of the best held-out log loss. A more complex structure must improve
mean held-out loss by at least `0.02` nats per action to displace a simpler
candidate. Archetype mixtures face the same threshold. Role-specific empirical
Bayes population means are then refitted using all 12 discovery contexts.

Secondary discovery measures are top-k action coverage, calibration by
predictive-probability decile, performance in early versus late session games,
and the proportion of actions in unseen conditioning cells. Wins are reported
but cannot select the model.

## Stage 2: held-out policy confirmation

Operational amendment, fixed before any confirmation context was launched:
use four paired seeds per arm rather than eight, and add the discovery-best
two-archetype model as an explicitly exploratory fifth arm. This keeps 12
independent held-out contexts while allocating the manual xhigh play budget to
the unresolved structural question. Policy win-rate comparisons remain
directional; predictive and safety diagnostics remain the main evidence.

- 12 new Luna contexts that never appear in discovery.
- Five blinded Hal conditions per context and four paired seeds per condition:
  two canonical starts and two leap-focused starts.
- 240 confirmation games. Together with discovery, the experiment contains
  336 games across 24 independent Luna contexts.
- Condition order follows a balanced Latin square. Start-clock order is
  counterbalanced inside each four-game arm. The same seed and start-clock
  schedule is used for all five conditions within a Luna context.
- Hal's opponent posterior resets between conditions and persists across the
  four games within a condition. Luna's context persists across conditions;
  the balanced order controls first-order learning and fatigue effects.

The primary four conditions are fixed by purpose; Stage 1 supplies only the
selected predictor and fitted population prior. Arm E is a labeled exploratory
addition based on the discovery ranking:

| Opaque arm | Hal policy | Purpose |
|---|---|---|
| A | complete DTH | non-adaptive control |
| B | selected predictor, epsilon `0` | exploitation available inside the exact minimax polytope |
| C | selected predictor, confidence `0.975`, epsilon grid `0,.0025,.005,.01`, match budget `.02` | conservative safety spending |
| D | selected predictor, confidence `0.95`, epsilon grid `0,.005,.01,.02`, match budget `.05` | responsive safety spending |
| E | discovery-best two-archetype role mixture, confidence `0.975`, epsilon grid `0,.0025,.005,.01`, match budget `.02` | exploratory test of whether its raw predictive gain improves decisions |

Arm implementations are fixed for analysis, while their order follows the
balanced Latin square. The player sees neither arm letters nor implementations,
only the opaque play launcher and ordinary public game output.

## Outcomes and inference

Primary structure outcome:

- held-out negative log likelihood per revealed Baku action, clustered by Luna
  context.

Primary policy comparison:

- Hal win-rate difference, arm C minus complete DTH, paired by Luna context,
  seed, and start-clock stratum.

Secondary policy outcomes:

- arms B, D, and E versus complete DTH;
- early games 1..2 versus late games 3..4;
- canonical versus leap-focused starts;
- half-round count;
- exploit-selection rate and declared epsilon spend;
- posterior retreat after a within-session change in Baku's empirical action
  distribution;
- DTH saddle-gap gate and unsupported action-61 fallbacks.

Report paired risk differences with cluster-bootstrap 95% intervals using Luna
context as the resampling unit. Also report raw paired discordances and
context-level effects; do not treat actions or games from the same Luna context
as independent. The confirmatory comparison is arm C versus complete DTH.
Other arm and subgroup comparisons are exploratory and are labeled as such.

This design is powered primarily for opponent-model structure because each
game yields multiple action observations. Forty-eight paired games per policy
comparison cannot reliably resolve a roughly five-percentage-point win-rate
effect; a null win result therefore means inconclusive, not equivalent.

## Stopping and exclusions

- Complete all preregistered contexts unless the public CLI, model, or rules
  change during collection.
- Replay only an interrupted game, from its original seed, and retain neither
  partial transcript nor partial outcome.
- Exclude a Luna context if its audit log uses a command other than the opaque
  play launcher or terminal input. Replace it with the next numbered fresh
  context without inspecting its outcomes.
- Do not weaken a safety budget, posterior confidence gate, saddle-gap gate, or
  action-legality rule to obtain a favorable result.
- Generated prompts, audits, transcripts, fitted artifacts, and reports remain
  under gitignored `outputs/luna-adaptive/`.
