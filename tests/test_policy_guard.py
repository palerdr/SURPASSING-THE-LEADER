import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from environment.cfr.evaluator import TerminalOnlyEvaluator
from environment.cfr.mcts import MCTSResult
from hal.agent import SolverAgent
from tests.test_hal_agent import make_game
from training.policy_guard import (
    policy_guard_records_for_game,
    policy_guard_target_for_game,
    save_policy_guard_records,
)
from training.value_targets import SOURCE_POLICY_GUARD, VALID_SOURCES, load_targets_as_records


def _agent() -> SolverAgent:
    return SolverAgent(
        "unused",
        player_name="Hal",
        iterations=8,
        seed=0,
        evaluator=TerminalOnlyEvaluator(),
    )


class _StubAgent:
    iterations = 8
    policy_ensemble_size = 1
    policy_uniform_mix = 0.0

    def search(self, game):
        if game.current_half == 1:
            return MCTSResult(
                root_strategy_dropper=np.array([1.0]),
                root_strategy_checker=np.array([1.0]),
                root_value_for_hal=0.25,
                root_visits=1,
                principal_line=[],
                cells_used=1,
                root_drop_seconds=(1,),
                root_check_seconds=(1,),
                root_strategy_dropper_avg=np.array([1.0]),
                root_strategy_checker_avg=np.array([1.0]),
            )
        return MCTSResult(
            root_strategy_dropper=np.array([1.0]),
            root_strategy_checker=np.array([1.0]),
            root_value_for_hal=0.5,
            root_visits=1,
            principal_line=[],
            cells_used=1,
            root_drop_seconds=(2,),
            root_check_seconds=(2,),
            root_strategy_dropper_avg=np.array([1.0]),
            root_strategy_checker_avg=np.array([1.0]),
        )

    def policy(self, game, role):
        if game.current_half == 1:
            return (1,), np.array([1.0])
        return (2,), np.array([1.0])


class _MixedPolicyAgent(_StubAgent):
    policy_ensemble_size = 3
    policy_uniform_mix = 0.25
    resolve_at_critical = True
    resolve_horizon = 3

    def policy(self, game, role):
        return (1, 2), np.array([0.875, 0.125])


def test_policy_guard_source_is_registered():
    assert SOURCE_POLICY_GUARD in VALID_SOURCES


def test_policy_guard_target_uses_deployed_average_policy_shape():
    game = make_game(clock=720.0, half=1)
    record = policy_guard_target_for_game(
        game=game,
        agent=_agent(),
        scenario="opening",
        seed=0,
        checkpoint="champion.pt",
    )

    target = record.target
    assert target.source == SOURCE_POLICY_GUARD
    assert target.horizon == 8
    assert target.dropper_dist.shape == (61,)
    assert target.checker_dist.shape == (61,)
    assert target.dropper_dist.sum() == pytest.approx(1.0)
    assert target.checker_dist.sum() == pytest.approx(1.0)
    assert record.metadata.scenario == "opening"
    assert record.metadata.checkpoint == "champion.pt"
    assert record.metadata.depth == 0
    assert record.metadata.survival_outcome == "root"


def test_policy_guard_target_uses_deployed_policy_semantics():
    record = policy_guard_target_for_game(
        game=make_game(clock=720.0, half=1),
        agent=_MixedPolicyAgent(),
        scenario="opening",
        seed=0,
        checkpoint="champion.pt",
    )

    assert record.target.dropper_dist[0] == pytest.approx(0.875)
    assert record.target.dropper_dist[1] == pytest.approx(0.125)
    assert record.target.checker_dist[0] == pytest.approx(0.875)
    assert record.target.checker_dist[1] == pytest.approx(0.125)
    assert record.metadata.policy_ensemble_size == 3
    assert record.metadata.policy_uniform_mix == pytest.approx(0.25)
    assert record.metadata.resolve_at_critical is True
    assert record.metadata.resolve_horizon == 3


def test_policy_guard_can_include_top_child_frontier_targets():
    game = make_game(clock=720.0, half=1)
    records = policy_guard_records_for_game(
        game=game,
        agent=_StubAgent(),
        scenario="opening",
        seed=0,
        checkpoint="champion.pt",
        include_children=True,
        child_top_k=1,
    )

    assert len(records) == 2
    root, child = records
    assert root.metadata.depth == 0
    assert child.metadata.depth == 1
    assert child.metadata.drop_time == 1
    assert child.metadata.check_time == 1
    assert child.metadata.survival_outcome == "none"
    assert child.metadata.branch_probability == pytest.approx(1.0)
    assert child.target.value == pytest.approx(0.5)


def test_policy_guard_npz_round_trip_keeps_metadata(tmp_path):
    record = policy_guard_target_for_game(
        game=make_game(clock=720.0, half=1),
        agent=_agent(),
        scenario="opening",
        seed=3,
        checkpoint="champion.pt",
    )
    path = tmp_path / "guard.npz"
    save_policy_guard_records([record], path)

    [loaded] = load_targets_as_records(path)
    assert loaded.source == SOURCE_POLICY_GUARD

    data = np.load(path, allow_pickle=True)
    assert data["policy_guard_scenarios"][0] == "opening"
    assert data["policy_guard_seeds"][0] == 3
    assert data["policy_guard_checkpoints"][0] == "champion.pt"
    assert data["policy_guard_iterations"][0] == 8
    assert data["policy_guard_policy_ensemble_sizes"][0] == 1
    assert data["policy_guard_policy_uniform_mixes"][0] == pytest.approx(0.0)
    assert data["policy_guard_resolve_at_critical"][0] == np.False_
    assert data["policy_guard_resolve_horizons"][0] == 3
    assert data["policy_guard_depths"][0] == 0
    assert data["policy_guard_drop_times"][0] == -1
    assert data["policy_guard_check_times"][0] == -1
    assert data["policy_guard_survival_outcomes"][0] == "root"
    assert data["policy_guard_branch_probabilities"][0] == pytest.approx(1.0)
