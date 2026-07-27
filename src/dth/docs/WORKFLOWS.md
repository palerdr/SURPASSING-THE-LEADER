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

## Validation

```powershell
uv run python -m pytest src/dth/tests -q
uv run python -m pytest --collect-only -q
uv run python -m pytest -q
cargo test --workspace
graphify update .
```
