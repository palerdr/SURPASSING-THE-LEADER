import os
import sys
from dataclasses import asdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from environment.cfr.mcts import MCTSResult
from training.tournament import _default_starting_game
from training.trace_bootstrap import (
    generate_trace_bootstrap_records,
    generate_trace_bootstrap_records_from_hints,
    replay_trace_state,
    save_trace_bootstrap_records,
)
from training.value_targets import SOURCE_MCTS_BOOTSTRAP


def _record_to_json(record):
    row = asdict(record)
    row["result"] = record.result.value
    return row


def _trace_row(*, winner: str, game_seed: int = 123):
    game = _default_starting_game(game_seed)
    first = game.resolve_half_round(4, 60, survived_outcome=None)
    second = game.resolve_half_round(30, 1, survived_outcome=True)
    return {
        "seed": 0,
        "game_index": 0,
        "game_seed": game_seed,
        "winner": winner,
        "history": [_record_to_json(first), _record_to_json(second)],
    }


def _trace_report():
    return {
        "games": {
            "champion": [_trace_row(winner="Hal")],
            "candidate": [_trace_row(winner="Baku")],
        }
    }


def _fake_mcts_search(*_args, **_kwargs):
    return MCTSResult(
        root_strategy_dropper=np.array([1.0, 0.0]),
        root_strategy_checker=np.array([1.0, 0.0]),
        root_value_for_hal=0.375,
        root_visits=4,
        principal_line=[],
        cells_used=4,
        root_drop_seconds=(1, 2),
        root_check_seconds=(1, 2),
        root_strategy_dropper_avg=np.array([0.25, 0.75]),
        root_strategy_checker_avg=np.array([0.6, 0.4]),
    )


def test_replay_trace_state_restores_pre_decision_state():
    row = _trace_row(winner="Baku")
    game = replay_trace_state(row, 1)

    assert len(game.history) == 1
    assert game.current_half == 2
    assert game.round_num == 0


def test_generate_trace_bootstrap_records_uses_average_policy(monkeypatch):
    monkeypatch.setattr("environment.cfr.mcts.mcts_search", _fake_mcts_search)

    records, summary = generate_trace_bootstrap_records(
        trace_report=_trace_report(),
        predict_fn=lambda _game: (0.0, np.zeros(61), np.zeros(61)),
        checkpoint="champion.pt",
        iterations=8,
        state_window=1,
        critical_only=False,
    )

    assert summary.matched_games == 1
    assert summary.records == len(records) == 1
    [record] = records
    assert record.target.source == SOURCE_MCTS_BOOTSTRAP
    assert record.target.value == 0.375
    assert record.target.horizon == 8
    assert record.target.dropper_dist[0] == 0.25
    assert record.target.dropper_dist[1] == 0.75
    assert record.target.checker_dist[0] == 0.6
    assert record.target.checker_dist[1] == 0.4
    assert record.metadata.trajectory == "candidate"
    assert record.metadata.history_index == 1
    assert record.metadata.subgame_resolve_horizon == 3


def test_generate_trace_bootstrap_records_from_hints_labels_exact_state(monkeypatch):
    monkeypatch.setattr("environment.cfr.mcts.mcts_search", _fake_mcts_search)

    records, summary = generate_trace_bootstrap_records_from_hints(
        trace_report=_trace_report(),
        target_hints=[
            {
                "reason": "candidate_regression_first_divergence",
                "trajectory": "candidate",
                "seed": 0,
                "game_index": 0,
                "history_index": 1,
            }
        ],
        predict_fn=lambda _game: (0.0, np.zeros(61), np.zeros(61)),
        checkpoint="champion.pt",
        iterations=8,
        critical_only=False,
    )

    assert summary.records == len(records) == 1
    assert summary.matched_games == 1
    assert summary.states_considered == 1
    assert summary.skipped_missing_games == 0
    assert records[0].target.dropper_dist[1] == 0.75
    assert records[0].metadata.scenario == (
        "candidate_regression_first_divergence_candidate_s0_g0_h1"
    )
    assert records[0].metadata.history_index == 1


def test_save_trace_bootstrap_records_writes_metadata(monkeypatch, tmp_path):
    monkeypatch.setattr("environment.cfr.mcts.mcts_search", _fake_mcts_search)
    records, _summary = generate_trace_bootstrap_records(
        trace_report=_trace_report(),
        predict_fn=lambda _game: (0.0, np.zeros(61), np.zeros(61)),
        checkpoint="champion.pt",
        iterations=8,
        state_window=1,
        critical_only=False,
    )

    path = tmp_path / "trace_targets.npz"
    save_trace_bootstrap_records(records, path)

    with np.load(path, allow_pickle=True) as data:
        assert data["sources"][0] == SOURCE_MCTS_BOOTSTRAP
        assert data["trace_bootstrap_scenarios"][0].startswith(
            "champion_win_candidate_loss_candidate_s0_g0_h"
        )
        assert data["trace_bootstrap_trajectories"][0] == "candidate"
        assert data["trace_bootstrap_checkpoints"][0] == "champion.pt"
        assert data["trace_bootstrap_iterations"][0] == 8
