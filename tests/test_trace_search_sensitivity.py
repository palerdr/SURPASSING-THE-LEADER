import os
import sys
from dataclasses import asdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from environment.cfr.mcts import MCTSResult
from training.trace_divergence import analyze_trace_divergences
from training.trace_policy_probe import CheckpointProbeSpec
from training.trace_search_sensitivity import probe_trace_search_sensitivity
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
                    "history": [
                        _record_to_json(champion_first),
                        _record_to_json(champion_second),
                    ],
                }
            ],
            "candidate": [
                {
                    "seed": 0,
                    "game_index": 0,
                    "game_seed": 123,
                    "winner": "Baku",
                    "history": [
                        _record_to_json(candidate_first),
                        _record_to_json(candidate_second),
                    ],
                }
            ],
        }
    }


def _dist(second):
    vector = np.zeros(61, dtype=np.float64)
    vector[second - 1] = 1.0
    return vector


def _predict(value, drop_second, check_second):
    def predict(_game):
        return float(value), _dist(drop_second), _dist(check_second)

    return predict


class _FakeAgent:
    def __init__(self, checkpoint_path, *, evaluator, **_kwargs):
        self.checkpoint_path = checkpoint_path
        self.evaluator = evaluator

    @staticmethod
    def _project(vector):
        projected = np.asarray([vector[0], vector[1]], dtype=np.float64)
        total = float(projected.sum())
        if total <= 0.0:
            return np.array([0.5, 0.5], dtype=np.float64)
        return projected / total

    def search(self, game):
        value, dropper, checker = self.evaluator(game)
        return MCTSResult(
            root_strategy_dropper=self._project(dropper),
            root_strategy_checker=self._project(checker),
            root_value_for_hal=float(value),
            root_visits=3,
            principal_line=[],
            cells_used=3,
            root_drop_seconds=(1, 2),
            root_check_seconds=(1, 2),
            root_strategy_dropper_avg=self._project(dropper),
            root_strategy_checker_avg=self._project(checker),
        )

    def policy(self, game, role):
        _, dropper, checker = self.evaluator(game)
        if role == "dropper":
            return (1, 2), self._project(dropper)
        return (1, 2), self._project(checker)


def test_probe_trace_search_sensitivity_cross_wires_value_and_policy_sources():
    trace = _trace_report()
    audit = analyze_trace_divergences(trace)
    baseline = CheckpointProbeSpec("base", "base.pt")
    candidate = CheckpointProbeSpec("candidate", "candidate.pt")

    report = probe_trace_search_sensitivity(
        trace_report=trace,
        audit_report=audit,
        baseline=baseline,
        candidate=candidate,
        iterations=8,
        roles=("trace",),
        top_k=2,
        use_tablebase=False,
        predict_fns={
            "base": _predict(0.1, 1, 1),
            "candidate": _predict(0.4, 2, 2),
        },
        agent_factory=_FakeAgent,
    )

    assert report["summary"]["states"] == 1
    assert report["summary"]["max_abs_value_delta_by_variant"] == {
        "candidate_full": 0.30000000000000004,
        "candidate_value_baseline_policy": 0.30000000000000004,
        "baseline_value_candidate_policy": 0.0,
    }
    assert report["summary"]["max_policy_tv_by_variant"] == {
        "candidate_full": 1.0,
        "candidate_value_baseline_policy": 0.0,
        "baseline_value_candidate_policy": 1.0,
    }
    row = report["rows"][0]
    assert row["roles"] == ["checker"]
    variants = {variant["label"]: variant for variant in row["variants"]}
    assert variants["candidate_value_baseline_policy"]["value_source"] == "candidate"
    assert variants["candidate_value_baseline_policy"]["policy_source"] == "base"
    assert row["deltas_vs_baseline"][1]["policies"]["checker"]["tv"] == 0.0
    assert row["deltas_vs_baseline"][2]["policies"]["checker"]["tv"] == 1.0
