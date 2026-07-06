import os
import sys
from dataclasses import asdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training.trace_divergence import analyze_trace_divergences
from training.trace_net_probe import probe_trace_net_outputs
from training.trace_policy_probe import CheckpointProbeSpec
from training.tournament import _default_starting_game


def _record_to_json(record):
    row = asdict(record)
    row["result"] = record.result.value
    return row


def _trace_report():
    champion_game = _default_starting_game(123)
    champion_first = champion_game.resolve_half_round(4, 60, survived_outcome=None)
    champion_second = champion_game.resolve_half_round(30, 2, survived_outcome=True)

    candidate_game = _default_starting_game(123)
    candidate_first = candidate_game.resolve_half_round(4, 60, survived_outcome=None)
    candidate_second = candidate_game.resolve_half_round(30, 60, survived_outcome=None)
    return {
        "games": {
            "champion": [
                {
                    "seed": 0,
                    "game_index": 0,
                    "game_seed": 123,
                    "winner": "Hal",
                    "history": [_record_to_json(champion_first), _record_to_json(champion_second)],
                }
            ],
            "candidate": [
                {
                    "seed": 0,
                    "game_index": 0,
                    "game_seed": 123,
                    "winner": "Baku",
                    "history": [_record_to_json(candidate_first), _record_to_json(candidate_second)],
                }
            ],
        }
    }


def _predict(value, second):
    def predict(_game):
        dropper = np.zeros(61, dtype=np.float64)
        checker = np.zeros(61, dtype=np.float64)
        checker[second - 1] = 1.0
        dropper[0] = 1.0
        return value, dropper, checker

    return predict


def test_probe_trace_net_outputs_reports_trace_action_probabilities():
    trace = _trace_report()
    audit = analyze_trace_divergences(trace)
    specs = [
        CheckpointProbeSpec("base", "base.pt"),
        CheckpointProbeSpec("candidate", "candidate.pt"),
    ]

    report = probe_trace_net_outputs(
        trace_report=trace,
        audit_report=audit,
        checkpoints=specs,
        roles=("trace",),
        predict_fns=[
            (specs[0], _predict(0.1, 2)),
            (specs[1], _predict(0.4, 60)),
        ],
    )

    assert report["summary"]["states"] == 1
    assert report["summary"]["max_abs_value_delta"] == 0.30000000000000004
    assert report["summary"]["max_policy_tv"] == 1.0
    row = report["rows"][0]
    assert row["roles"] == ["checker"]
    assert row["checkpoints"][0]["policies"]["checker"]["trace_actions"]["champion"] == {
        "second": 2,
        "probability": 1.0,
    }
    assert row["checkpoints"][1]["policies"]["checker"]["trace_actions"]["candidate"] == {
        "second": 60,
        "probability": 1.0,
    }
