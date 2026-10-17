"""The code location loads and exposes the expected assets, checks, jobs, schedule, and sensor."""

import importlib
import os
from datetime import datetime, timezone

import pytest
from dagster import AssetKey, Definitions, build_schedule_context

EXPECTED_ASSETS = {
    "bronze_games",
    "stg_games",
    "stg_moves",
    "player_splits",
    "move_context",
    "role_histograms",
    "adaptive_prior",
    "offline_evaluation",
}


@pytest.fixture(scope="module")
def definitions(tmp_path_factory):
    # A fresh target directory forces the manifest parse and keeps output out of src/hal_loop/dbt.
    previous = os.environ.get("HAL_LOOP_DBT_TARGET")
    os.environ["HAL_LOOP_DBT_TARGET"] = str(tmp_path_factory.mktemp("dbt") / "target")
    try:
        module = importlib.import_module("hal_loop.definitions")
        yield module
    finally:
        if previous is None:
            os.environ.pop("HAL_LOOP_DBT_TARGET", None)
        else:
            os.environ["HAL_LOOP_DBT_TARGET"] = previous


def test_definitions_validate(definitions):
    Definitions.validate_loadable(definitions.defs)


def test_asset_keys(definitions):
    graph = definitions.defs.resolve_asset_graph()
    assert {key.to_user_string() for key in graph.get_all_asset_keys()} == EXPECTED_ASSETS


def test_dbt_models_depend_on_bronze(definitions):
    graph = definitions.defs.resolve_asset_graph()
    assert graph.get(AssetKey("stg_games")).parent_keys == {AssetKey("bronze_games")}
    assert graph.get(AssetKey("adaptive_prior")).parent_keys == {AssetKey("role_histograms"), AssetKey("move_context")}
    assert graph.get(AssetKey("offline_evaluation")).parent_keys == {AssetKey("adaptive_prior")}


def test_gate_checks(definitions):
    graph = definitions.defs.resolve_asset_graph()
    names = {key.name for key in graph.asset_check_keys if key.asset_key == AssetKey("offline_evaluation")}
    assert names == {"likelihood_gate", "exploitation_gate"}


def test_jobs_schedule_and_sensor(definitions):
    defs = definitions.defs
    assert defs.resolve_job_def("propose_prior").name == "propose_prior"
    schedule = defs.resolve_schedule_def("daily_refresh_schedule")
    assert (schedule.cron_schedule, schedule.execution_timezone) == ("0 3 * * *", "UTC")
    assert schedule.job_name == "daily_refresh"
    assert defs.resolve_sensor_def("evaluation_passed_sensor").job_name == "propose_prior"


def test_schedule_targets_yesterdays_partition(definitions):
    defs = definitions.defs
    schedule = defs.resolve_schedule_def("daily_refresh_schedule")
    context = build_schedule_context(
        scheduled_execution_time=datetime(2026, 10, 17, 3, 0, tzinfo=timezone.utc),
        repository_def=defs.get_repository_def(),
    )
    requests = schedule.evaluate_tick(context).run_requests
    assert [request.partition_key for request in requests] == ["2026-10-16"]
