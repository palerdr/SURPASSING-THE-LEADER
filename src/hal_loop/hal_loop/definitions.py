"""The hal_loop Dagster code location.

Load it with:

    uv run --project src/hal_loop dagster dev -m hal_loop.definitions

SUPABASE_URL and SUPABASE_SECRET_KEY name the hosted ledger, as in the browser app.
HAL_LOOP_DATA_DIR moves the data directory, and HAL_LOOP_BRONZE_DIR points dbt at
another bronze directory, such as the test fixture. The daily schedule runs at
03:00 UTC on yesterday's partition, so bronze_games writes a file before dbt reads the glob.
"""

import os

from dagster import (
    AssetSelection,
    Definitions,
    EnvVar,
    build_schedule_from_partitioned_job,
    define_asset_job,
)

from hal_loop.assets.bronze import bronze_games
from hal_loop.assets.checks import exploitation_gate, likelihood_gate
from hal_loop.assets.dbt import hal_loop_dbt_assets
from hal_loop.assets.evaluation import offline_evaluation
from hal_loop.assets.model import adaptive_prior
from hal_loop.jobs.propose import evaluation_passed_sensor, propose_prior
from hal_loop.resources import DEFAULT_DATA_DIR, PathsResource, SupabaseResource, make_dbt_resource

daily_refresh = define_asset_job(
    "daily_refresh",
    selection=AssetSelection.assets(bronze_games, hal_loop_dbt_assets, adaptive_prior, offline_evaluation),
    description="Extract yesterday's ledger rows, rebuild the marts, refit the prior, and evaluate it.",
)

daily_refresh_schedule = build_schedule_from_partitioned_job(
    daily_refresh,
    name="daily_refresh_schedule",
    hour_of_day=3,
    minute_of_hour=0,
)

defs = Definitions(
    assets=[bronze_games, hal_loop_dbt_assets, adaptive_prior, offline_evaluation],
    asset_checks=[likelihood_gate, exploitation_gate],
    jobs=[daily_refresh, propose_prior],
    schedules=[daily_refresh_schedule],
    sensors=[evaluation_passed_sensor],
    resources={
        # The hosted browser app reads the same two settings in src/browser/deploy/production.py.
        "supabase": SupabaseResource(url=EnvVar("SUPABASE_URL"), service_key=EnvVar("SUPABASE_SECRET_KEY")),
        "paths": PathsResource(
            data_dir=os.environ.get("HAL_LOOP_DATA_DIR", str(DEFAULT_DATA_DIR)),
            bronze_root=os.environ.get("HAL_LOOP_BRONZE_DIR"),
        ),
        "dbt": make_dbt_resource(),
    },
)
