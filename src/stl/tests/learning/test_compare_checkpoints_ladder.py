import os
import sys
import math

sys.path.insert(0, os.getcwd())

from stl.commands.compare_ladder import _aggregate_reports, _json_safe


def _counts(wins: int, games: int = 4):
    return {
        "games": games,
        "wins": wins,
        "draws": 0,
        "losses": games - wins,
        "win_rate": wins / games,
        "score_rate": wins / games,
    }


def _report(seed: int, champion_wins: int, candidate_wins: int):
    return {
        "seed": seed,
        "champion": {
            "overall": _counts(champion_wins),
            "opponents": {"safe": _counts(champion_wins)},
        },
        "candidate": {
            "overall": _counts(candidate_wins),
            "opponents": {"safe": _counts(candidate_wins)},
        },
    }


def test_aggregate_reports_sums_checkpoint_deltas():
    aggregate = _aggregate_reports(
        [
            _report(0, champion_wins=2, candidate_wins=3),
            _report(1, champion_wins=1, candidate_wins=2),
        ],
        ["safe"],
    )

    assert aggregate["champion"]["games"] == 8
    assert aggregate["champion"]["wins"] == 3
    assert aggregate["candidate"]["wins"] == 5
    assert aggregate["delta"]["wins_delta"] == 2
    assert aggregate["delta"]["win_rate_delta"] == 0.25
    assert aggregate["opponents"]["safe"]["delta"]["wins_delta"] == 2


def test_json_safe_converts_non_finite_floats():
    assert _json_safe({"llr": math.inf, "nested": [-math.inf]}) == {
        "llr": "inf",
        "nested": ["-inf"],
    }


