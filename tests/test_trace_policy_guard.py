import os
import sys
from dataclasses import asdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from environment.cfr.mcts import MCTSResult
from training.trace_divergence import analyze_trace_divergences
from training.trace_policy_guard import generate_trace_policy_guard_records
from training.tournament import _default_starting_game
from training.value_targets import SOURCE_OPPONENT_TRACE_GUARD, SOURCE_POLICY_GUARD


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


class _LabelAgent:
    iterations = 8
    policy_ensemble_size = 1
    policy_uniform_mix = 0.0
    resolve_at_critical = False
    resolve_horizon = 3

    def search(self, game):
        return MCTSResult(
            root_strategy_dropper=np.array([1.0]),
            root_strategy_checker=np.array([1.0]),
            root_value_for_hal=0.25,
            root_visits=1,
            principal_line=[],
            cells_used=1,
            root_drop_seconds=(1,),
            root_check_seconds=(2,),
            root_strategy_dropper_avg=np.array([1.0]),
            root_strategy_checker_avg=np.array([1.0]),
        )

    def policy(self, game, role):
        return ((1,), np.array([1.0])) if role == "dropper" else ((2,), np.array([1.0]))


def test_generate_trace_policy_guard_records_labels_audit_hints():
    trace = _trace_report()
    audit = analyze_trace_divergences(trace)

    records, summary = generate_trace_policy_guard_records(
        trace_report=trace,
        audit_report=audit,
        label_agent=_LabelAgent(),
        label_checkpoint="champion.pt",
    )

    assert summary.records == len(records) == 1
    assert summary.states_considered == 1
    assert records[0].target.source == SOURCE_POLICY_GUARD
    assert records[0].target.value == 0.25
    assert records[0].target.checker_dist[1] == 1.0
    assert records[0].metadata.scenario.startswith("trace_policy_guard_candidate_s0_g0_h1")
    assert records[0].metadata.checkpoint == "champion.pt"


def test_generate_trace_policy_guard_records_accepts_custom_source():
    trace = _trace_report()
    audit = analyze_trace_divergences(trace)

    records, _summary = generate_trace_policy_guard_records(
        trace_report=trace,
        audit_report=audit,
        label_agent=_LabelAgent(),
        label_checkpoint="champion.pt",
        source=SOURCE_OPPONENT_TRACE_GUARD,
    )

    assert records[0].target.source == SOURCE_OPPONENT_TRACE_GUARD
