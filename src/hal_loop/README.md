# Hal Loop Instructions

`src/hal_loop/` is the live opponent-modelling loop. It fits a versioned
population prior for the adaptive Hal from the hosted ledger's games. A Dagster
code location runs it, a dbt project on DuckDB builds the warehouse, and a
local MLflow registry records each fit.

## Asset graph

1. `bronze_games` copies one UTC day of ledger rows to
   `data/bronze/games/<day>.jsonl` and writes a manifest beside it.
2. The dbt models `stg_games`, `stg_moves`, `player_splits`, `move_context`,
   and `role_histograms` build the warehouse `data/warehouse.duckdb`.
3. `adaptive_prior` fits one Dirichlet per human role on the training split
   and picks each role's strength on the validation split. It writes
   `data/models/<version>/prior.json` and `card.json`, and it registers an
   MLflow model version once per prior version.
4. `offline_evaluation` scores the candidate on the test split. The asset
   checks `likelihood_gate` and `exploitation_gate` judge the result.
5. The `propose_prior` job writes a passing prior as the next versioned
   config. Here "passing" means that `likelihood_gate` passed. The
   `exploitation_gate` result never controls promotion. The
   `evaluation_passed_sensor` always starts the job as a dry run, which
   writes beside the model. A human runs the job with `dry_run: false` to
   open the pull request.

The `daily_refresh` job runs steps 1 to 4. Its schedule runs at 03:00 UTC on
the previous day's partition.

## Boundaries

- [`docs/PROJECTS.toml`](../../docs/PROJECTS.toml) lets hal_loop import
  `arena` and `dth.agent`, and nothing else. No project imports hal_loop.
- hal_loop imports no `torch`, `gymnasium`, `stable_baselines3`, or
  `sb3_contrib`.
- The loop is its own uv project with its own `uv.lock`. Run every command
  with `uv run --project src/hal_loop`. A dependency change here leaves the
  root lock, the DTH digest, and the leap builder hash unchanged.
- Generated data stays under the ignored `src/hal_loop/data/`.

## Environment

- `uv sync --project src/hal_loop` installs the ignored `src/hal_loop/.venv/`.
- The root package depends on hydra-core, and hydra-core pins
  `antlr4-python3-runtime` 4.9. Dagster's asset selection parser needs 4.13.
  hal_loop never imports hydra, so `[tool.uv] override-dependencies` installs
  `antlr4-python3-runtime>=4.13` here. The pytest options disable hydra's
  pytest plugin, because that plugin needs antlr 4.9.
- The loop reads the ledger with `SUPABASE_URL` and `SUPABASE_SECRET_KEY`.
  The hosted browser app reads the same two names, and also accepts the
  older `SUPABASE_SERVICE_ROLE_KEY`, which the loop does not. The root
  `.env` uses the two names above. Never print or commit these values.
- `DAGSTER_HOME` must name `src/hal_loop/data/dagster`. Without it,
  `dagster dev` writes a `.tmp_dagster_home_*` instance into the working
  directory.
- All generated data lives under `src/hal_loop/data/`: `bronze/`,
  `warehouse.duckdb`, `models/`, `dagster/`, and `dbt/`. MLflow keeps its
  SQLite store in `mlruns/mlflow.db` and its artifacts in `mlruns/artifacts/`.
- `HAL_LOOP_DATA_DIR` moves the data directory, including the dbt manifest
  under `dbt/target`; `HAL_LOOP_DBT_TARGET` moves the manifest alone.
  `HAL_LOOP_BRONZE_DIR` points dbt at another bronze directory, such as the
  test fixture.

## Commands

Run each command from the repository root. Set up the shell first:

```sh
set -a; source .env; set +a
export DAGSTER_HOME="$PWD/src/hal_loop/data/dagster"
mkdir -p "$DAGSTER_HOME"
```

Serve the Dagster UI and the daemon:

```sh
uv run --project src/hal_loop dagster dev -m hal_loop.definitions
```

Parse the dbt project once. The code location reads this manifest. Delete
`src/hal_loop/data/dbt/target` or parse again after you change the dbt project.

```sh
uv run --project src/hal_loop dbt parse --project-dir src/hal_loop/dbt --profiles-dir src/hal_loop/dbt --target-path "$PWD/src/hal_loop/data/dbt/target" --log-path "$PWD/src/hal_loop/data/dbt/logs"
```

Backfill the bronze extracts in one run, from the first ledger day to
yesterday. Use `--partition <day>` for one day.

```sh
uv run --project src/hal_loop dagster asset materialize -m hal_loop.definitions --select bronze_games --partition-range 2026-09-18...2026-10-16
```

Build the warehouse, fit the prior, and evaluate it. This command skips the
exploitation gate:

```sh
uv run --project src/hal_loop dagster asset materialize -m hal_loop.definitions --select 'group:warehouse or group:model'
```

Add `--config-json '{"ops": {"exploitation_gate": {"config": {"with_simulation": true}}}}'`
to play the exploitation gate's series against `src/dth/artifacts/complete_full_v1`.

Build the dbt project on the test fixture, and generate its documentation.
Each command sets `HAL_LOOP_WAREHOUSE` for itself alone. An exported value
would also redirect the Dagster commands above to the fixture warehouse.

```sh
DBT_ARGS=(--project-dir src/hal_loop/dbt --profiles-dir src/hal_loop/dbt --target-path "$PWD/src/hal_loop/data/dbt/fixture" --log-path "$PWD/src/hal_loop/data/dbt/logs" --vars "{\"bronze_dir\": \"$PWD/src/hal_loop/tests/fixtures/bronze\"}")
HAL_LOOP_WAREHOUSE="$PWD/src/hal_loop/data/fixture.duckdb" uv run --project src/hal_loop dbt build "${DBT_ARGS[@]}"
HAL_LOOP_WAREHOUSE="$PWD/src/hal_loop/data/fixture.duckdb" uv run --project src/hal_loop dbt docs generate "${DBT_ARGS[@]}"
```

## Invariants

- `bronze_games` asks the ledger for the exact row count of each day and
  pages until it holds that many rows, or until an empty page when the
  gateway omits the count. A page shorter than the requested size does not
  end the read, because the gateway can cap pages below the requested size.
- `bronze_games` is partitioned by the UTC day of `updated_at`, and a run
  adds or rewrites the files of its own days alone, so earlier days stay as
  extracted. A
  rerun on unchanged rows writes the same JSONL bytes. A day without games
  gets an empty JSONL file and a manifest with `row_count` 0.
- The ledger rewrites a game row while play continues, so one game can appear
  in two partitions. `base_games` keeps the latest write of each game.
- The loop assumes that the ledger's `abandon_series` RPC updates
  `updated_at`, so an abandoned game reaches the partition of its last
  change; verify this assumption against the ledger's SQL.
- The ledger marks a game `finished`, `stopped`, `abandoned`, or `active`.
  The moves of finished, stopped, and abandoned games enter the warehouse.
  Active games keep their `stg_games` row, and their moves stay out. A game
  without moves takes no split slot.
- Each player's games split by start time into about 60% train, 20%
  validation, and 20% test. All games of a player with fewer than 3 games go
  to train.
- The dbt tests gate the downstream assets. A failed test fails the dbt step,
  and Dagster then skips `adaptive_prior` and `offline_evaluation`.
- The card's verdict is `fitted` when the training split holds 50 or more
  finished games. Stopped and abandoned games add moves to the fit and do not
  count toward the 50.
- `arena.policies.adaptive.load_opponent_model` loads every fitted prior
  before the asset writes it, so the live loader validates the schema.
- The version hashes the protocol constants with the sorted
  `(player_id, human_role, split, second, count)` rows of `role_histograms`
  in the train and validation splits. A rewritten game or a dbt change that
  moves one of those counts moves the version. The same counts and the same
  protocol give the same version, and MLflow then keeps its existing model
  version for it.
- `offline_evaluation` scores each player's test-split moves after the prior
  absorbs the player's training and validation moves. The validation split
  chose the strength, so it never enters the score.
- `likelihood_gate` passes when four conditions hold. The card verdict is
  `fitted`. The test split holds at least 5 scored players and 100 scored
  moves. The pooled candidate-minus-production log-likelihood per move is
  positive. The lower bound of its 95% interval is positive. `gates.py`
  holds these minimums. The interval comes from 2,000 player-clustered
  bootstrap resamples with seed 20261017. Production is
  `src/arena/config/adaptive_prior_current.json`, or the uniform prior when
  that file is absent.
- `exploitation_gate` plays 200 paired-seat games of the adaptive provider and
  200 of the equilibrium provider against a histogram emulator of the test
  split. It passes when the adaptive mean score is at most 0.01 below the
  equilibrium mean score. It runs only when `with_simulation` is true.
- `evaluation_passed_sensor` reads finished `likelihood_gate` records alone.
  A run in progress or a failed run leaves a PLANNED record, and the sensor
  skips it and keeps its cursor on the last finished record. The run key is
  the model version, and the sensor skips a version that already has a
  proposal under `data/models/<version>/proposal/`.
- `propose_prior` refuses a version whose card verdict is not `fitted` or
  whose evaluation fails `likelihood_gate`. It runs as a dry run by default.
  Every run writes `adaptive_prior_v<n>.json` and
  `adaptive_prior_current.json` into the model directory's `proposal/`
  folder and `card.md` into the model directory. A dry run changes nothing
  under `src/arena/config/`. With `dry_run: false`, the op fetches
  `origin/main` and adds a temporary worktree on the new branch
  `hal-loop/prior-v<n>`. It copies the two files into the worktree's
  `src/arena/config/`, commits them, pushes the branch, and opens the pull
  request from the worktree. Then it removes the worktree. The live
  checkout keeps its branch and its files. A human reviews and merges the
  pull request.
- The hosted runtime serves no adaptive policy. Until a hosted `adaptive-v1`
  policy exists, the prior feeds local play with `--hal-agent adaptive-dth
  --adaptive-prior-json <path>` and the `python -m hal_lab match` series.

## Operations

- The schedule and the sensor start stopped. Turn them on in the Dagster UI.
- The daemon runs the schedule and the sensor. For unattended nights, run
  `dagster dev` or `dagster-daemon run` under a launchd agent, with
  `DAGSTER_HOME` and the two ledger settings in its environment.
- Open the MLflow UI with
  `uv run --project src/hal_loop mlflow ui --backend-store-uri "sqlite:///$PWD/src/hal_loop/data/mlruns/mlflow.db"`.

Run `uv run --project src/hal_loop python -m pytest src/hal_loop/tests -q`.
