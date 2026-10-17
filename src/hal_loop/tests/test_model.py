"""The training-game count behind the verdict, and the SQLite MLflow store."""

import json
import os

import duckdb

from hal_loop.assets.model import FIT_SPLITS, prior_version
from hal_loop.tracking import REGISTERED_MODEL, log_prior_run, tracking_uri
from hal_loop.warehouse import histogram_rows, training_games


def write_histograms(warehouse, rows):
    with duckdb.connect(str(warehouse)) as connection:
        connection.execute("create or replace table role_histograms (player_id varchar, human_role varchar, split varchar, second integer, count integer)")
        connection.executemany("insert into role_histograms values (?, ?, ?, ?, ?)", rows)


def test_training_games_count_finished_games_and_keep_stopped_moves(tmp_path):
    warehouse = tmp_path / "warehouse.duckdb"
    with duckdb.connect(str(warehouse)) as connection:
        connection.execute("create table stg_games as select * from (values ('a:0', 'finished'), ('b:0', 'stopped'), ('c:0', 'finished')) t(game_key, status)")
        connection.execute(
            "create table move_context as select * from (values "
            "('a:0', 'p1', 'train'), ('a:0', 'p1', 'train'), ('b:0', 'p1', 'train'), ('c:0', 'p2', 'validation')"
            ") t(game_key, player_id, split)"
        )
    games = training_games(warehouse)
    assert games.game_keys == ["a:0", "b:0"]
    assert (games.finished, games.moves, games.players) == (1, 3, 1)


def test_log_prior_run_uses_a_sqlite_store(tmp_path):
    assert "MLFLOW_ALLOW_FILE_STORE" not in os.environ
    mlruns = tmp_path / "mlruns"
    artifact = tmp_path / "prior.json"
    artifact.write_text(json.dumps({"schema_version": "adaptive-dth-role-prior-v1"}))
    first = log_prior_run(mlruns, version="v1", params={"weighting": "none"}, metrics={"m": 1.0}, artifacts=[artifact])
    second = log_prior_run(mlruns, version="v2", params={"weighting": "none"}, metrics={"m": 2.0}, artifacts=[artifact])
    assert (first[1], second[1]) == ("1", "2")
    assert tracking_uri(mlruns) == f"sqlite:///{(mlruns / 'mlflow.db').resolve()}"
    assert (mlruns / "mlflow.db").is_file()
    stored = list((mlruns / "artifacts").glob("*/artifacts/prior.json"))
    assert {path.parent.parent.name for path in stored} == {first[0], second[0]}

    from mlflow import MlflowClient

    client = MlflowClient(tracking_uri=tracking_uri(mlruns))
    run = client.get_run(first[0])
    assert run.data.params == {"weighting": "none"} and run.info.status == "FINISHED"


HISTOGRAM_ROWS = [
    ("p1", "dropper", "train", 44, 3),
    ("p1", "dropper", "validation", 44, 1),
    ("p2", "checker", "train", 12, 2),
    ("p2", "checker", "test", 12, 5),
]


def test_version_follows_the_fit_counts(tmp_path):
    warehouse = tmp_path / "warehouse.duckdb"
    write_histograms(warehouse, HISTOGRAM_ROWS)
    rows = histogram_rows(warehouse, FIT_SPLITS)
    assert rows == sorted(row for row in HISTOGRAM_ROWS if row[2] != "test")
    version = prior_version(rows)
    # Row order and test-split counts leave the version alone.
    write_histograms(warehouse, [HISTOGRAM_ROWS[3], *reversed(HISTOGRAM_ROWS[:3])])
    assert prior_version(histogram_rows(warehouse, FIT_SPLITS)) == version
    write_histograms(warehouse, [*HISTOGRAM_ROWS[:3], ("p2", "checker", "test", 12, 9)])
    assert prior_version(histogram_rows(warehouse, FIT_SPLITS)) == version
    # A rewritten training game that moves one count moves the version.
    write_histograms(warehouse, [("p1", "dropper", "train", 44, 4), *HISTOGRAM_ROWS[1:]])
    assert prior_version(histogram_rows(warehouse, FIT_SPLITS)) != version
    # So does a validation count, because the validation split picks the strength.
    write_histograms(warehouse, [HISTOGRAM_ROWS[0], ("p1", "dropper", "validation", 43, 1), *HISTOGRAM_ROWS[2:]])
    assert prior_version(histogram_rows(warehouse, FIT_SPLITS)) != version


def test_same_version_registers_once(tmp_path):
    mlruns = tmp_path / "mlruns"
    artifact = tmp_path / "prior.json"
    artifact.write_text(json.dumps({"schema_version": "adaptive-dth-role-prior-v1"}))
    first = log_prior_run(mlruns, version="v1", params={}, metrics={"m": 1.0}, artifacts=[artifact])
    again = log_prior_run(mlruns, version="v1", params={}, metrics={"m": 1.0}, artifacts=[artifact])
    other = log_prior_run(mlruns, version="v2", params={}, metrics={"m": 1.0}, artifacts=[artifact])
    assert first == (first[0], "1", True)
    assert again == (first[0], "1", False)
    assert other[1:] == ("2", True)

    from mlflow import MlflowClient

    client = MlflowClient(tracking_uri=tracking_uri(mlruns), registry_uri=tracking_uri(mlruns))
    versions = client.search_model_versions(f"name = '{REGISTERED_MODEL}'")
    assert sorted((str(v.version), v.tags["prior_version"]) for v in versions) == [("1", "v1"), ("2", "v2")]
    experiment = client.get_experiment_by_name("adaptive_prior")
    assert len(client.search_runs([experiment.experiment_id])) == 2
