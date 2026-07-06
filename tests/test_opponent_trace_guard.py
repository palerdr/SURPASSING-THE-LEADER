import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from environment.cfr.mcts import MCTSResult
from environment.opponents.base import Opponent
from training.opponent_trace_guard import generate_opponent_trace_guard_records
from training.value_targets import SOURCE_OPPONENT_TRACE_GUARD


class _FixedProbeAgent:
    def choose_action(self, game, role, turn_duration):
        return 4 if role == "dropper" else min(60, turn_duration)


class _FixedOpponent(Opponent):
    def __init__(self):
        self.resets = 0

    def reset(self):
        self.resets += 1

    def choose_action(self, game, role, turn_duration):
        return 30 if role == "dropper" else min(60, turn_duration)


class _LabelAgent:
    iterations = 8
    policy_ensemble_size = 1
    policy_uniform_mix = 0.0

    def search(self, game):
        return MCTSResult(
            root_strategy_dropper=np.array([1.0]),
            root_strategy_checker=np.array([1.0]),
            root_value_for_hal=0.125,
            root_visits=1,
            principal_line=[],
            cells_used=1,
            root_drop_seconds=(1,),
            root_check_seconds=(1,),
            root_strategy_dropper_avg=np.array([1.0]),
            root_strategy_checker_avg=np.array([1.0]),
        )

    def policy(self, game, role):
        return (1,), np.array([1.0])


def test_generate_opponent_trace_guard_records_labels_trace_states():
    records, summary = generate_opponent_trace_guard_records(
        probe_agent=_FixedProbeAgent(),
        label_agent=_LabelAgent(),
        label_checkpoint="champion.pt",
        opponent_name="fixed",
        opponent_factory=lambda seed: _FixedOpponent(),
        seeds=[0],
        games_per_seed=1,
        max_states=2,
    )

    assert summary.opponent == "fixed"
    assert summary.records == len(records) == 2
    assert summary.games == 1
    assert all(record.target.source == SOURCE_OPPONENT_TRACE_GUARD for record in records)
    assert records[0].metadata.scenario.startswith("fixed_trace_s0_g0_h")


def test_generate_opponent_trace_guard_records_reuses_ladder_seeded_opponent():
    calls = []
    opponents = []

    def factory(seed):
        opponent = _FixedOpponent()
        calls.append(seed)
        opponents.append(opponent)
        return opponent

    records, summary = generate_opponent_trace_guard_records(
        probe_agent=_FixedProbeAgent(),
        label_agent=_LabelAgent(),
        label_checkpoint="champion.pt",
        opponent_name="fixed",
        opponent_factory=factory,
        seeds=[3],
        games_per_seed=2,
        max_states=100,
    )

    assert calls == [3]
    assert len(opponents) == 1
    assert opponents[0].resets == 2
    assert summary.games == 2
    assert len(records) > 2
