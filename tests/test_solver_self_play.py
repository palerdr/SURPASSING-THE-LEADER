import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from environment.cfr.evaluator import TerminalOnlyEvaluator
from hal.agent import SolverAgent
from training.solver_self_play import play_solver_self_play_game, save_self_play_records
from training.value_targets import (
    SOURCE_SELF_PLAY_MCTS,
    load_targets_as_records,
)


def _agent(player_name: str, *, seed: int) -> SolverAgent:
    return SolverAgent(
        "unused",
        player_name=player_name,
        iterations=8,
        seed=seed,
        evaluator=TerminalOnlyEvaluator(),
    )


def test_solver_self_play_emits_mcts_policy_targets_with_metadata():
    records = play_solver_self_play_game(
        hal_agent=_agent("Hal", seed=0),
        baku_controller=_agent("Baku", seed=1),
        seed=7,
        hal_checkpoint_id="hal.pt",
        opponent_checkpoint_id="baku.pt",
        opponent_source="checkpoint",
        max_half_rounds=2,
    )

    assert 1 <= len(records) <= 2
    record = records[0]
    target = record.target
    assert target.source == SOURCE_SELF_PLAY_MCTS
    assert target.horizon == 8
    assert target.dropper_dist.shape == (61,)
    assert target.checker_dist.shape == (61,)
    assert target.dropper_dist.sum() == pytest.approx(1.0)
    assert target.checker_dist.sum() == pytest.approx(1.0)
    assert record.root_value == pytest.approx(target.value)
    assert record.final_outcome in (-1.0, 0.0, 1.0)
    assert record.seed == 7
    assert record.hal_checkpoint_id == "hal.pt"
    assert record.opponent_checkpoint_id == "baku.pt"
    assert record.opponent_source == "checkpoint"


def test_solver_self_play_npz_is_training_compatible_and_keeps_audit_fields(tmp_path):
    records = play_solver_self_play_game(
        hal_agent=_agent("Hal", seed=2),
        baku_controller=_agent("Baku", seed=3),
        seed=9,
        hal_checkpoint_id="hal.pt",
        opponent_checkpoint_id="baku.pt",
        opponent_source="checkpoint",
        max_half_rounds=1,
    )
    path = tmp_path / "self_play.npz"
    save_self_play_records(records, path)

    loaded = load_targets_as_records(path)
    assert [target.source for target in loaded] == [SOURCE_SELF_PLAY_MCTS] * len(records)

    data = np.load(path, allow_pickle=True)
    assert "final_outcomes" in data
    assert "root_values" in data
    assert "seeds" in data
    assert "hal_checkpoint_ids" in data
    assert "opponent_sources" in data
    assert data["opponent_sources"][0] == "checkpoint"
