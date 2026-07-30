# DTH workflows

All commands run from the repository root. Generated databases and JSON reports
remain under ignored `src/dth/artifacts/`.

## Exact census and solve

There is one current Hydra entry point and one generic config:

```powershell
uv run python -m dth exact
```

Override roots, the shared database, report path, census bounds, solve bounds,
batches, and workers through Hydra:

```powershell
uv run python -m dth exact `
  database_path=src/dth/artifacts/exact.sqlite `
  report_path=src/dth/artifacts/exact_report.json `
  'roots=[{state:[240,0,240,0]}]' `
  census.max_expansions=5000 census.max_states=10000 census.max_seconds=30 `
  solve.max_new_solutions=10000 solve.max_seconds=120 `
  solve.batch_size=64 solve.workers=1
```

At least one census bound is mandatory. Enabling solve also requires a state or
time bound. The report embeds the fully resolved Hydra configuration. Census
and exact values use the same database so queue completion and value commit are
atomic. Policies are reconstructed and optionally cached only for queried
roots.

For a census-only run:

```powershell
uv run python -m dth exact solve.enabled=false
```

## Current exact results

| Root | Status | Reachable identity count | Evidence |
|---|---:|---:|---|
| `(239,241,299,300)` | exact closure | 61 | exhausted persisted frontier |
| `(240,0,240,0)` | exact quotient closure | 3,541 | checker-turn bitset and persisted census |
| `(0,0,0,0)` | bounded census | 757,337 persisted states; 183,647 expanded | `exact_opening_consolidated_v1_report.json`, frozen schema and frontier index |
| boundary roots | not rerun on current schema | — | no projection |
| frozen development roots | not rerun on current schema | — | no projection |
| `(0,0,0,0)` | not run on current schema | — | no complete-game claim |

Full rank, branching, deduplication, timing, memory, storage, profiling, and
interval detail belongs in the generated JSON report.

## Existing finite data and training workflows

```powershell
uv run python -m dth dataset
uv run python -m dth train
uv run python -m dth self-play
uv run python -m dth mcts-audit
```

Those workflows remain separate from exact authority. Neural, RL, self-play,
CFR, or MCTS outputs may order future selective expansion, but they cannot
provide leaf values or certification bounds.

## Depth-ladder baseline and the G1 gate

The [`AGENT_GOAL.md`](AGENT_GOAL.md) M1 sequence, in order. Dataset
regeneration uses only versioned configs; the two `emission: merge` configs
reconstruct reference artifacts whose original producers were never
versioned.

```powershell
# 1. Exact corpora (strategic chain, development and readiness closures).
uv run python -m dth dataset --config-name strategic_exact_v1   # ... v2..v6
uv run python -m dth dataset --config-name readiness_development_h3_v2
uv run python -m dth dataset --config-name readiness_development_h4a_v2
uv run python -m dth dataset --config-name readiness_development_h4_v2
uv run python -m dth dataset --config-name readiness_exact_h3_v1
uv run python -m dth dataset --config-name readiness_exact_h4_v1
uv run python -m dth dataset --config-name readiness_training_reference_v2
uv run python -m dth dataset --config-name self_play_readiness_reference_v1

# 2. Baseline checkpoint: rows from scratch, then the v19 decision recipe.
uv run python -m dth train --config-name train_depth_baseline_rows_v1
uv run python -m dth train --config-name train_depth_baseline_v1

# 3. Full-width resolve ladder at depths 1..3 (see config comment for the
#    exact override lines), then the fail-closed G1 verdict.
uv run python -m dth mcts-audit --config-name mcts_depth_ladder_v1
uv run python src/dth/readiness.py depth-gate `
  --reports src/dth/artifacts/mcts_depth_ladder_v1_d1.json `
            src/dth/artifacts/mcts_depth_ladder_v1_d2.json `
            src/dth/artifacts/mcts_depth_ladder_v1_d3.json `
  --output src/dth/artifacts/depth_gate_v1.json
```

Every ladder record carries `lp_fallbacks`; reports without the count are
rejected by the gate, because an uncounted silent LP fallback could fake a
healthy search. The promotion pipeline is now `python src/dth/readiness.py
promotion ...` with the original flags, and the G2 orientation balance gate
is `python src/dth/readiness.py orientation-gate --report <ladder> --output
<verdict>`.

## The flagship agent and its exact anchor

```powershell
# The complete failure-dead quotient band (3,541 classes) as one durable
# artifact; the play-time agent reads it and never expands it.
uv run python -m dth exact --config-name exact_band_v1

# Play against the bounded-resolve agent through the neutral arena.
uv run python -m arena play --hal-agent dth
```

`dth.agent.BoundedResolveAgent` owns the per-move deadline ladder: certified
tablebase hit, then full-width class resolve with network leaves, then a
finite-horizon certificate. Durable closure deposits stay in the offline
`dth exact` workflow because one artifact binds one root manifest; the agent
may only cache finite-horizon certificates.

## Promotion and league

```powershell
# Candidate-versus-baseline resolve ladders, then the fail-closed pipeline.
uv run python -m dth mcts-audit --config-name mcts_depth_ladder_v1 `
  mcts.max_depth=2 'evaluators=[network]' 'seeds=[0,1,2]' `
  checkpoint=<candidate>/best.pt output=src/dth/artifacts/<candidate>_d2.json
uv run python src/dth/readiness.py promotion `
  --baseline src/dth/artifacts/promo_baseline_d2.json `
  --candidate src/dth/artifacts/<candidate>_d2.json `
  --checkpoint <candidate>/best.pt `
  --exact-targets src/dth/artifacts/self_play_readiness_reference_v1.npz `
  --replay-a src/dth/artifacts/promo_replay_a.json `
  --replay-b src/dth/artifacts/promo_replay_b.json `
  --output src/dth/artifacts/promotion_<candidate>.json

# Paired-seat strength series against the abstract tablebases.
uv run python -m arena match --candidate dth --opponent abstract --buckets 5 `
  --games 30 --max-half-rounds 120 `
  --output src/dth/artifacts/arena_league_bucket12_v2.json
```

Gate verdicts and their artifacts are recorded in
[`AGENT_GOAL.md`](AGENT_GOAL.md).

## Expert iteration

One generation of the coverage loop: self-play under the current agent,
depth-amplified labels from the resolve's interior solutions, exact rows
winning every merge collision, then the unchanged candidate recipe and the
same gates.

```powershell
uv run python -m dth dataset --config-name resolve_labeled_gen1
uv run python -m dth dataset --config-name class_head_gen1_corpus
uv run python -m dth train --config-name train_class_head_gen1
```

Resolve-labeled rows are training data, never certification input: they
carry the agent's query horizon and the resolve's measured gap, and their
values are depth-amplified play estimates.

From generation four onward the two value meanings are separated end to end:
every row carries `value_semantics` (0 finite-horizon-exact, 1 resolve-play),
play rows supervise the network's dedicated play head, `ExactTargetStore`
refuses play rows as exact authority, and the agent's scalar resolve leaves
answer from the play head.

```powershell
uv run python -m dth dataset --config-name class_head_gen4_corpus
uv run python -m dth train --config-name train_class_head_gen4
```

Generation five adds the exact anchor closure (which proved the anchor-leaf
rows already existed — see AGENT_GOAL.md) and replays the two leaf layers
the depth-two anchor resolves evaluate:

```powershell
uv run python -m dth dataset --config-name anchor_leaf_closure_v1
uv run python -m dth dataset --config-name class_head_gen5_corpus
uv run python -m dth train --config-name train_class_head_gen5
```

## Validation

```powershell
uv run python -m pytest src/dth/tests -q
uv run python -m pytest --collect-only -q
uv run python -m pytest -q
cargo test --workspace
graphify update .
```
