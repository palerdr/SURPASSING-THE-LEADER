"""Protocol layout, an end-to-end smoke run, and the exact-policy control."""

from __future__ import annotations

import json
import math
import shutil

import pytest

from hal_lab.experiments.hal_predictability_v1 import run

SMOKE = {**run.PROTOCOL, "games": 6, "k": 2, "lbfgs_max_iter": 20, "resamples": 20,
         "exploit": {**run.PROTOCOL["exploit"], "games": 4}}
# The curves read only the translated-v1 table, so their smoke generates that pair alone.
TABLE_SMOKE = {**SMOKE, "policies": {"translated-v1": run.PROTOCOL["policies"]["translated-v1"]}, "opponents": ["exact"]}
NO_CURVE = {**run.CURVES, "games": 4, "block": 2, "departure_games": 4, "mix": [], "online": False, "temperatures": []}
REQUIRES_ARTIFACT = pytest.mark.skipif(not (run.ARTIFACT / "tablebase.json").is_file(),
                                       reason="complete DTH artifact is absent")


def test_seeds_and_split_are_deterministic():
    seeds = run.game_seeds(run.PROTOCOL)
    assert seeds == run.game_seeds(dict(run.PROTOCOL))
    assert len(seeds) == len(set(seeds)) == run.PROTOCOL["games"]
    test = run.test_games(run.PROTOCOL)
    assert test == run.test_games(dict(run.PROTOCOL))
    assert len(test) == 200 and len(set(test)) == 200
    assert all(0 <= g < run.PROTOCOL["games"] for g in test)
    assert run.test_games(SMOKE) == run.test_games(dict(SMOKE))
    assert len(run.test_games(SMOKE)) == 1
    assert run.game_seeds(SMOKE, stream=1) == run.game_seeds(dict(SMOKE), stream=1)
    assert len(run.game_seeds(SMOKE, stream=1)) == 4
    assert not set(run.game_seeds(SMOKE, stream=1)) & set(run.game_seeds(SMOKE))


@pytest.mark.skipif(
    not (run.ARTIFACT / "tablebase.json").is_file(),
    reason="complete DTH artifact is absent",
)
def test_smoke_run_writes_every_phase(tmp_path, monkeypatch):
    # One test game gives a zero-width interval; the control has its own test.
    monkeypatch.setattr(run, "check_control", lambda results: None)
    run.generate(SMOKE, tmp_path)
    run.fit(SMOKE, tmp_path)
    played = run.exploit(SMOKE, tmp_path)
    assert all(row["games"] == 4 for row in played["rows"].values())
    table = run.report(SMOKE, tmp_path)
    for name in ("manifest.json", "fit.json", "exploit.json", "results.json", "results.md"):
        assert (tmp_path / name).is_file()
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert len(manifest["pairs"]) == len(SMOKE["policies"]) * len(SMOKE["opponents"])
    for pair in manifest["pairs"]:
        rows = [json.loads(line) for line in (tmp_path / "moves" / f"{pair}.jsonl").read_text().splitlines()]
        assert rows
        for row in rows:
            assert 1 <= row["second"] <= 60
            assert len(row["floor"]) == 60
            assert abs(sum(row["floor"]) - 1.0) < 1e-6
            assert len(row["own"]) == len(row["opp"]) == SMOKE["k"]
            assert len(row["opp_counts"]) == 60
    results = json.loads((tmp_path / "results.json").read_text())
    assert results["protocol_hash"] == run.protocol_hash(SMOKE)
    assert "dth" in table and "pooled" in table


def test_control_fires_on_a_fabricated_exact_departure():
    fabricated = {"rows": {"dth__uniform": {"policy": "dth", "departure": 0.1, "departure_ci": [0.08, 0.12]},
                           "translated-v1__uniform": {"policy": "translated-v1", "departure": 2.0,
                                                      "departure_ci": [1.5, 2.5]}}}
    with pytest.raises(RuntimeError, match="0.1000"):
        run.check_control(fabricated)
    fabricated["rows"]["dth__uniform"].update(departure=0.0, departure_ci=[-0.01, 0.01])
    run.check_control(fabricated)


def test_wilson_interval_brackets_the_rate():
    lo, hi = run._wilson(250, 500)
    assert lo < 0.5 < hi and abs((lo + hi) / 2 - 0.5) < 1e-12
    assert 0.0 <= run._wilson(0, 500)[0] and run._wilson(500, 500)[1] <= 1.0 + 1e-12


@pytest.fixture(scope="module")
def table_moves(tmp_path_factory):
    out = tmp_path_factory.mktemp("table")
    run.generate(TABLE_SMOKE, out)
    return out


def _curve(table_moves, tmp_path, **spec):
    shutil.copytree(table_moves, tmp_path, dirs_exist_ok=True)
    result = run.curves(TABLE_SMOKE, tmp_path, {**NO_CURVE, **spec})
    assert json.loads((tmp_path / "curves.json").read_text()) == result
    return result


def _rate_ok(row, games):
    return row["games"] == games and 0 <= row["wins"] <= games and 0.0 <= row["win_rate"] <= 1.0 \
        and 0.0 <= row["wilson_95"][0] <= row["wilson_95"][1] <= 1.0 + 1e-12


@REQUIRES_ARTIFACT
def test_mixed_curve_smoke(table_moves, tmp_path):
    result = _curve(table_moves, tmp_path, mix=run.CURVES["mix"])
    assert [row["p"] for row in result["mixed"]] == run.CURVES["mix"]
    assert all(_rate_ok(row, 4) for row in result["mixed"])
    assert result["online"] is None and result["temperature"] == []


@REQUIRES_ARTIFACT
def test_online_curve_smoke(table_moves, tmp_path):
    online = _curve(table_moves, tmp_path, online=True)["online"]
    assert _rate_ok(online, 4)
    assert [block["block"] for block in online["blocks"]] == ["1-2", "3-4"]
    assert all(_rate_ok(block, 2) for block in online["blocks"])
    assert sum(block["wins"] for block in online["blocks"]) == online["wins"]


@REQUIRES_ARTIFACT
def test_temperature_curve_smoke(table_moves, tmp_path):
    rows = _curve(table_moves, tmp_path, temperatures=run.CURVES["temperatures"])["temperature"]
    assert [row["temperature"] for row in rows] == run.CURVES["temperatures"]
    for row in rows:
        assert _rate_ok(row, 4)
        departure = row["departure_vs_exact"]
        assert departure["moves"] > 0 and math.isfinite(departure["departure"])
        assert departure["departure_ci"][0] <= departure["departure_ci"][1]
    # At the selected temperature the constructor path plays the registry's translated-v1 move for move.
    agent, seeds, moves = run.dth.agent.CompleteDTHAgent(run.ARTIFACT), run.game_seeds(SMOKE, stream=1)[:2], []
    for hal in (run._make_hal(SMOKE, SMOKE["policies"]["translated-v1"]), run.translated(0.0, agent)):
        recorder = run.Recorder(hal, agent, SMOKE["k"], {})
        run._play(recorder, run.ExactProvider(agent), seeds, SMOKE)
        moves.append([(r["game"], r["move"], r["second"]) for r in recorder.rows])
    assert moves[0] and moves[0] == moves[1]
