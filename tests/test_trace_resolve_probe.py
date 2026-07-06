import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from environment.cfr.evaluator import TerminalOnlyEvaluator
from training.tournament import _default_starting_game
from training.trace_resolve_probe import ResolveProbeSpec, probe_trace_resolves


def test_probe_trace_resolves_reports_bounded_resolve_for_target_hint():
    game = _default_starting_game(123)
    trace = {
        "games": {
            "champion": [],
            "candidate": [
                {
                    "seed": 0,
                    "game_index": 0,
                    "game_seed": 123,
                    "winner": None,
                    "cause": None,
                    "history": [],
                }
            ],
        }
    }
    audit = {
        "target_hints": [
            {
                "reason": "unit",
                "trajectory": "candidate",
                "seed": 0,
                "game_index": 0,
                "history_index": 0,
                "hal_role": "dropper",
            }
        ]
    }

    report = probe_trace_resolves(
        trace_report=trace,
        audit_report=audit,
        evaluators=[ResolveProbeSpec("terminal", TerminalOnlyEvaluator())],
        horizons=(1,),
        top_k=3,
    )

    assert report["summary"]["states"] == 1
    assert report["summary"]["resolves"] == 1
    row = report["rows"][0]
    assert row["state"]["round_num"] == game.round_num
    assert row["candidate_actions"]["joint_count"] > 0
    resolve = row["resolves"][0]
    assert resolve["label"] == "terminal"
    assert resolve["horizon"] == 1
    assert resolve["candidate_count"] == row["candidate_actions"]["joint_count"]
    assert sum(resolve["dropper"]["probabilities"]) == 1.0
    assert sum(resolve["checker"]["probabilities"]) == 1.0
    assert resolve["dropper"]["top"]


def test_probe_trace_resolves_counts_missing_games():
    report = probe_trace_resolves(
        trace_report={"games": {"champion": [], "candidate": []}},
        audit_report={
            "target_hints": [
                {
                    "trajectory": "candidate",
                    "seed": 0,
                    "game_index": 0,
                    "history_index": 0,
                }
            ]
        },
        evaluators=[ResolveProbeSpec("terminal", TerminalOnlyEvaluator())],
        horizons=(1,),
    )

    assert report["summary"]["states"] == 0
    assert report["summary"]["resolves"] == 0
    assert report["summary"]["skipped_missing_games"] == 1
