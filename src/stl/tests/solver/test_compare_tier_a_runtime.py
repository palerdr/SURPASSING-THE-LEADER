import os
import sys
import math
import json

sys.path.insert(0, os.getcwd())

from stl.commands.compare_tier_a_runtime import _aggregate_reports, _json_safe, _tier_a_stats


def _report(seed: int, base_wins: int, tier_wins: int):
    games = 4
    return {
        "seed": seed,
        "baseline": {
            "overall": {
                "games": games,
                "wins": base_wins,
                "draws": 0,
                "losses": games - base_wins,
            },
            "opponents": {
                "safe": {
                    "games": games,
                    "wins": base_wins,
                    "draws": 0,
                    "losses": games - base_wins,
                }
            },
        },
        "tier_a": {
            "overall": {
                "games": games,
                "wins": tier_wins,
                "draws": 0,
                "losses": games - tier_wins,
            },
            "opponents": {
                "safe": {
                    "games": games,
                    "wins": tier_wins,
                    "draws": 0,
                    "losses": games - tier_wins,
                }
            },
        },
        "tier_a_evaluator_stats": {
            "hits": seed + 1,
            "wide_hits": seed,
            "misses": {"not_post_leap": seed + 2},
        },
    }


def test_aggregate_reports_sums_seeds_and_reports_delta():
    aggregate = _aggregate_reports(
        [_report(0, base_wins=2, tier_wins=3), _report(1, base_wins=1, tier_wins=2)],
        ["safe"],
    )

    assert aggregate["baseline"]["games"] == 8
    assert aggregate["baseline"]["wins"] == 3
    assert aggregate["tier_a"]["wins"] == 5
    assert aggregate["delta"]["wins_delta"] == 2
    assert aggregate["delta"]["win_rate_delta"] == 0.25
    assert aggregate["opponents"]["safe"]["delta"]["wins_delta"] == 2
    assert aggregate["tier_a_evaluator_stats"] == {
        "hits": 3,
        "wide_hits": 1,
        "misses": {"not_post_leap": 5},
    }


def test_tier_a_stats_defaults_for_non_tier_a_agent():
    class DummyAgent:
        evaluator = object()

    assert _tier_a_stats(DummyAgent()) == {"hits": 0, "wide_hits": 0, "misses": {}}


def test_json_safe_output_can_be_dumped_with_strict_json():
    safe = _json_safe({"llr": math.inf, "nested": [-math.inf]})

    assert safe == {"llr": "inf", "nested": ["-inf"]}
    json.dumps(safe, allow_nan=False)
