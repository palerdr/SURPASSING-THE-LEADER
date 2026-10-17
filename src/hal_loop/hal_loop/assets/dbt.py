"""The dbt models of src/hal_loop/dbt as Dagster assets.

The translator maps the dbt source `bronze.games` to the `bronze_games` asset.
It also gives each model the plain model name as its key, such as `move_context`.
Every dbt run receives the absolute bronze directory and warehouse path.

Dagster reads the dbt manifest at data/dbt/target/manifest.json when this module loads.
`dagster dev` re-parses the project on each load. Other entry points, such as
`dagster asset materialize`, parse it only when the manifest is missing, so delete
data/dbt/target after a change to the dbt project.

HAL_LOOP_DBT_TARGET moves the dbt target directory, for example into a test's tmp_path.
"""


import json
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from dagster import AssetExecutionContext, AssetKey, Failure
from dagster_dbt import DagsterDbtTranslator, DbtCliResource, DbtProject, dbt_assets

from hal_loop.resources import DBT_DIR, DEFAULT_DATA_DIR, PathsResource


class HalLoopTranslator(DagsterDbtTranslator):
    """Key the bronze source by its Dagster asset and each model by its name."""

    def get_asset_key(self, dbt_resource_props: Mapping[str, Any]) -> AssetKey:
        if dbt_resource_props["resource_type"] == "source":
            if (dbt_resource_props["source_name"], dbt_resource_props["name"]) == ("bronze", "games"):
                return AssetKey("bronze_games")
            return super().get_asset_key(dbt_resource_props)
        return AssetKey(dbt_resource_props["name"])

    def get_group_name(self, dbt_resource_props: Mapping[str, Any]) -> str | None:
        return "warehouse"


def load_dbt_project() -> DbtProject:
    """Build the DbtProject and make sure its manifest exists."""
    project = DbtProject(
        project_dir=DBT_DIR,
        profiles_dir=DBT_DIR,
        target_path=os.environ.get(
            "HAL_LOOP_DBT_TARGET",
            str(Path(os.environ.get("HAL_LOOP_DATA_DIR", str(DEFAULT_DATA_DIR))) / "dbt" / "target"),
        ),
    )
    project.prepare_if_dev()
    if not project.manifest_path.exists():
        project.preparer.prepare(project)
    return project


dbt_project = load_dbt_project()


def dbt_vars(paths: PathsResource) -> str:
    """Return the --vars JSON that every dbt invocation receives."""
    return json.dumps({"bronze_dir": str(paths.bronze_dir)}, sort_keys=True)


def require_bronze(paths: PathsResource) -> None:
    """Fail before dbt runs when no bronze extract holds a row.

    DuckDB's JSON reader accepts empty files beside nonempty ones. It cannot infer
    the ledger's columns from an empty glob or from empty files alone.
    """
    games = paths.bronze_dir / "games"
    if not any(path.stat().st_size for path in games.glob("*.jsonl")):
        raise Failure(f"no bronze rows under {games}; materialize bronze_games for a partition with games first")


@dbt_assets(
    manifest=dbt_project.manifest_path,
    project=dbt_project,
    dagster_dbt_translator=HalLoopTranslator(),
)
def hal_loop_dbt_assets(context: AssetExecutionContext, dbt: DbtCliResource, paths: PathsResource):
    """Build the selected dbt models and run their tests."""
    require_bronze(paths)
    # dbt resolves relative paths against the process directory, so pass absolute ones.
    os.environ["HAL_LOOP_WAREHOUSE"] = str(paths.warehouse)
    paths.warehouse.parent.mkdir(parents=True, exist_ok=True)
    # Run artifacts and logs go under the data directory, not the dbt project.
    target = paths.root / "dbt" / context.run_id
    yield from dbt.cli(["build", "--vars", dbt_vars(paths)], context=context, target_path=target).stream()
