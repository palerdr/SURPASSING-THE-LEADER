"""Build the dbt project on the bronze fixture and check the warehouse it writes."""
import json
import os
import subprocess
import sys
from pathlib import Path

import duckdb
import pytest

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent / "dbt"
FIXTURE = HERE / "fixtures" / "bronze"
FIXTURE_GAMES = FIXTURE / "games" / "2026-10-13.jsonl"
DBT = Path(sys.executable).with_name("dbt")


def dbt_build(tmp_path, bronze_dir):
    warehouse = tmp_path / "warehouse.duckdb"
    command = [
        str(DBT), "build",
        "--profiles-dir", str(PROJECT),
        "--project-dir", str(PROJECT),
        "--target-path", str(tmp_path / "target"),
        "--log-path", str(tmp_path / "logs"),
        "--vars", json.dumps({"bronze_dir": str(bronze_dir)}),
    ]
    env = {**os.environ, "HAL_LOOP_WAREHOUSE": str(warehouse)}
    result = subprocess.run(command, env=env, capture_output=True, text=True)
    return result, warehouse


def write_bronze(tmp_path, rows):
    games = tmp_path / "bronze" / "games"
    games.mkdir(parents=True)
    (games / "2026-10-13.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    return games.parent


def fixture_rows():
    return [json.loads(line) for line in FIXTURE_GAMES.read_text().splitlines()]


@pytest.fixture(scope="module")
def warehouse(tmp_path_factory):
    tmp_path = tmp_path_factory.mktemp("dbt")
    result, path = dbt_build(tmp_path, FIXTURE)
    assert result.returncode == 0, result.stdout + result.stderr
    with duckdb.connect(str(path), read_only=True) as connection:
        yield connection


def scalar(connection, sql):
    return connection.sql(sql).fetchone()[0]


def test_stg_games_keeps_every_game(warehouse):
    assert scalar(warehouse, "select count(*) from stg_games") == 15


def test_stg_moves_skips_the_active_game(warehouse):
    assert scalar(warehouse, "select count(distinct game_key) from stg_moves") == 14
    active = "select game_key from stg_games where status = 'active'"
    assert scalar(warehouse, f"select count(*) from stg_moves where game_key in ({active})") == 0


def test_player_splits_cover_finished_and_stopped_games(warehouse):
    assert scalar(warehouse, "select count(*) from player_splits") == 14


def test_leap_window_move_stays_out_of_move_context(warehouse):
    leap = warehouse.sql(
        "select human_role, human_second, hal_second from stg_moves where turn_duration = 61"
    ).fetchall()
    assert leap == [("dropper", 61, 60)]
    assert scalar(warehouse, "select count(*) from stg_moves where turn_duration = 60") == scalar(
        warehouse, "select count(*) from move_context"
    )


def test_role_histograms_hold_60_seconds_per_group(warehouse):
    sizes = warehouse.sql(
        "select count(*) from role_histograms group by player_id, human_role, split"
    ).fetchall()
    assert sizes and {size for (size,) in sizes} == {60}


def test_move_context_seconds_are_ordinary(warehouse):
    low, high = warehouse.sql(
        "select min(least(human_second, hal_second)), max(greatest(human_second, hal_second)) from move_context"
    ).fetchone()
    assert 1 <= low and high <= 60


def test_stg_games_keeps_the_latest_write(tmp_path):
    rows = fixture_rows()
    stale = {**rows[0], "status": "active", "updated_at": "2026-10-13 09:00:00.000000+00"}
    result, path = dbt_build(tmp_path, write_bronze(tmp_path, [stale, *rows]))
    assert result.returncode == 0, result.stdout + result.stderr
    key = f"{rows[0]['series_id']}:{rows[0]['game_index']}"
    with duckdb.connect(str(path), read_only=True) as connection:
        assert scalar(connection, "select count(*) from stg_games") == 15
        assert scalar(connection, f"select status from stg_games where game_key = '{key}'") == "finished"


def test_build_fails_on_a_zero_second(tmp_path):
    rows = fixture_rows()
    rows[0]["public_history"][0]["drop_second"] = 0
    result, _ = dbt_build(tmp_path, write_bronze(tmp_path, rows))
    assert result.returncode != 0
    assert "FAIL 1 assert_move_seconds_in_range" in result.stdout


def test_build_fails_on_an_unknown_turn_duration(tmp_path):
    rows = fixture_rows()
    rows[0]["public_history"][0]["public_state_before"]["turn_duration"] = 62
    result, _ = dbt_build(tmp_path, write_bronze(tmp_path, rows))
    assert result.returncode != 0
    assert "FAIL 1 accepted_values_stg_moves_turn_duration" in result.stdout


def test_abandoned_games_keep_their_moves(tmp_path):
    rows = fixture_rows()
    rows[0]["status"] = "abandoned"
    result, path = dbt_build(tmp_path, write_bronze(tmp_path, rows))
    assert result.returncode == 0, result.stdout + result.stderr
    key = f"{rows[0]['series_id']}:{rows[0]['game_index']}"
    with duckdb.connect(str(path), read_only=True) as connection:
        assert scalar(connection, f"select count(*) from stg_moves where game_key = '{key}'") == len(rows[0]["public_history"])
        assert scalar(connection, f"select count(*) from player_splits where game_key = '{key}'") == 1
