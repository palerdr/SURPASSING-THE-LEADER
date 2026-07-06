import os
import sys
from dataclasses import asdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from environment.cfr.mcts import MCTSResult
from training.trace_divergence import analyze_trace_divergences
from training.trace_policy_probe import (
    CheckpointProbeSpec,
    parse_checkpoint_spec,
    probe_trace_policies,
)
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
                    "cause": "baku_failed_check",
                    "history": [_record_to_json(champion_first), _record_to_json(champion_second)],
                }
            ],
            "candidate": [
                {
                    "seed": 0,
                    "game_index": 0,
                    "game_seed": 123,
                    "winner": "Baku",
                    "cause": "hal_failed_check",
                    "history": [_record_to_json(candidate_first), _record_to_json(candidate_second)],
                }
            ],
        }
    }


class _FakeAgent:
    def __init__(self, checkpoint, **_kwargs):
        self.checkpoint = checkpoint

    def search(self, game):
        value = 0.1 if self.checkpoint == "base.pt" else 0.4
        return MCTSResult(
            root_strategy_dropper=np.array([1.0, 0.0]),
            root_strategy_checker=np.array([1.0, 0.0]),
            root_value_for_hal=value,
            root_visits=3,
            principal_line=[],
            cells_used=3,
            root_drop_seconds=(1, 2),
            root_check_seconds=(1, 2),
            root_strategy_dropper_avg=np.array([0.8, 0.2]),
            root_strategy_checker_avg=np.array([0.7, 0.3]),
        )

    def policy(self, game, role):
        if self.checkpoint == "base.pt":
            return (1, 2), np.array([0.8, 0.2])
        return (1, 2), np.array([0.25, 0.75])


def test_parse_checkpoint_spec_requires_label_and_path():
    assert parse_checkpoint_spec("base:foo.pt") == CheckpointProbeSpec(
        label="base",
        checkpoint="foo.pt",
    )


def test_probe_trace_policies_reports_value_and_policy_deltas():
    trace = _trace_report()
    audit = analyze_trace_divergences(trace)

    report = probe_trace_policies(
        trace_report=trace,
        audit_report=audit,
        checkpoints=[
            CheckpointProbeSpec("base", "base.pt"),
            CheckpointProbeSpec("candidate", "candidate.pt"),
        ],
        iterations=8,
        roles=("trace",),
        top_k=2,
        agent_factory=_FakeAgent,
    )

    assert report["summary"]["states"] == 1
    assert report["summary"]["max_abs_value_delta"] == 0.30000000000000004
    assert report["summary"]["max_policy_tv"] == 0.55
    row = report["rows"][0]
    assert row["roles"] == ["checker"]
    assert row["checkpoints"][0]["policies"]["checker"]["top"][0]["second"] == 1
    assert row["checkpoints"][0]["policies"]["checker"]["trace_actions"]["champion"] == {
        "second": 2,
        "probability": 0.2,
    }
    assert row["checkpoints"][0]["policies"]["checker"]["trace_actions"]["candidate"] == {
        "second": 60,
        "probability": 0.0,
    }
    assert row["deltas_vs_first"][0]["value_delta"] == 0.30000000000000004
    assert row["deltas_vs_first"][0]["policies"]["checker"]["tv"] == 0.55
