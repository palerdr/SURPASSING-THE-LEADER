"""End to end: the fixture warehouse, the fit and the evaluation in process, and a short simulated series."""

import json

import numpy as np
import pytest
from dagster import materialize

from arena.policies.adaptive import RoleDirichletOpponent, load_opponent_model
from hal_loop.assets.evaluation import offline_evaluation
from hal_loop.assets.model import adaptive_prior
from hal_loop.prior import ACTIONS, HistogramEmulator
from hal_loop.resources import DEFAULT_DTH_ARTIFACT_DIR, PathsResource

from .test_dbt import FIXTURE, dbt_build

EVALUATION_KEYS = {
    "bootstrap", "directory", "history_splits", "players", "pooled", "production_source",
    "roles", "split", "verdict", "version",
}


def test_fixture_warehouse_fits_and_evaluates_in_process(tmp_path, monkeypatch):
    result, warehouse = dbt_build(tmp_path, FIXTURE)
    assert result.returncode == 0, result.stdout + result.stderr
    assert warehouse == tmp_path / "warehouse.duckdb"
    monkeypatch.delenv("HAL_LOOP_WAREHOUSE", raising=False)
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    paths = PathsResource(data_dir=str(tmp_path), arena_config_dir=str(config_dir))

    run = materialize([adaptive_prior, offline_evaluation], resources={"paths": paths})
    assert run.success
    output = run.output_for_node("adaptive_prior")
    directory = tmp_path / "models" / output["version"]
    assert output["directory"] == str(directory)

    model = load_opponent_model(directory / "prior.json")
    assert isinstance(model, RoleDirichletOpponent)
    card = json.loads((directory / "card.json").read_text())
    # The fixture holds 13 finished games, below the 50 that the verdict needs.
    assert card["verdict"] == "insufficient data" and card["registered_model_version"] == "1"

    evaluation = json.loads((directory / "evaluation.json").read_text())
    assert set(evaluation) == EVALUATION_KEYS
    assert (evaluation["split"], evaluation["verdict"], evaluation["production_source"]) == ("test", "insufficient data", "uniform")
    assert set(evaluation["pooled"]) == {"moves", "log_likelihood_per_move", "difference", "interval"}
    assert evaluation["players"] >= 1 and evaluation["pooled"]["moves"] > 0

    # A second run on the same warehouse keeps the version and its MLflow model version.
    again = materialize([adaptive_prior], resources={"paths": paths})
    assert again.output_for_node("adaptive_prior")["version"] == output["version"]
    metadata = again.asset_materializations_for_node("adaptive_prior")[0].metadata
    assert (metadata["registered_model_version"].value, metadata["newly_registered"].value) == ("1", False)


@pytest.mark.skipif(not (DEFAULT_DTH_ARTIFACT_DIR / "tablebase.json").is_file(), reason="needs src/dth/artifacts/complete_full_v1")
def test_paired_series_smoke_against_the_dth_artifact():
    from hal_loop.simulation import build_provider, paired_series

    counts = {"dropper": np.ones(ACTIONS), "checker": np.ones(ACTIONS)}
    provider = build_provider("dth", DEFAULT_DTH_ARTIFACT_DIR)
    series = paired_series(provider, lambda: HistogramEmulator(counts), games=2, max_half_rounds=40)
    assert series["games"] == 2
    assert series["wins"] + series["losses"] + series["capped"] == 2
    assert 0.0 <= series["mean_score"] <= 1.0
