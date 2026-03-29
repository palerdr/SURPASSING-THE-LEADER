import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from environment.dth_env import DTHEnv
from environment.opponents.baku_teachers import BakuTeacher
from environment.opponents.league import WeightedOpponentLeague
from environment.opponents.safe_bot import BridgePressureBot, LeapAwareSafeBot, SafeBot
from scripts.evaluate import (
    GameMetrics,
    build_parser as build_eval_parser,
    make_scenario_sampler,
    play_game,
    summarize_games,
)
from training.train_ppo import (
    build_or_load_model,
    build_parser as build_train_ppo_parser,
    build_training_opponent,
)
from training.train_self_play import build_parser as build_train_self_play_parser


class MaxMaskModel:
    def predict(self, obs, action_masks, deterministic=True):
        del obs, deterministic
        return int(np.flatnonzero(action_masks)[-1]), None


def test_train_and_eval_parsers_default_max_steps_to_none():
    train_ppo_args = build_train_ppo_parser().parse_args([])
    train_self_play_args = build_train_self_play_parser().parse_args([])
    eval_args = build_eval_parser().parse_args(["--model", "dummy.zip"])

    assert train_ppo_args.max_steps is None
    assert train_self_play_args.max_steps is None
    assert eval_args.max_steps is None
    assert train_ppo_args.init_model is None
    assert train_ppo_args.opponent_model == []
    assert train_ppo_args.teacher_demo_file == []
    assert train_ppo_args.shaping_preset == "light"


def test_eval_can_build_fixed_scenario_sampler():
    sampler = make_scenario_sampler(
        scenario_name="round9_leap_deduced",
        curriculum_name=None,
        seed=123,
    )

    assert sampler is not None
    assert sampler(None)["name"] == "round9_leap_deduced"


def test_eval_rejects_scenario_and_curriculum_together():
    try:
        make_scenario_sampler(
            scenario_name="round9_leap_deduced",
            curriculum_name="critical",
            seed=123,
        )
    except ValueError as exc:
        assert "either --scenario or --curriculum" in str(exc)
        return

    raise AssertionError("Expected make_scenario_sampler to reject overlapping options")


def test_evaluation_counts_dropper_61_on_leap_turn():
    env = DTHEnv(
        opponent=SafeBot(),
        agent_role="hal",
        seed=123,
        max_steps=1,
        scenario_sampler=lambda _rng: {
            "game_clock": 3540.0,
            "current_half": 1,
            "first_dropper": "hal",
            "awareness": "unaware",
        },
    )

    stats = play_game(MaxMaskModel(), env, verbose=False)

    assert stats.reached_leap_turn is True
    assert stats.dropper_61_count == 1
    assert stats.checker_61_count == 0


def test_evaluation_counts_checker_61_when_awareness_allows_it():
    env = DTHEnv(
        opponent=SafeBot(),
        agent_role="hal",
        seed=123,
        max_steps=1,
        scenario_sampler=lambda _rng: {
            "game_clock": 3540.0,
            "current_half": 2,
            "first_dropper": "hal",
            "awareness": "deduced",
        },
    )

    stats = play_game(MaxMaskModel(), env, verbose=False)

    assert stats.reached_leap_turn is True
    assert stats.checker_61_count == 1
    assert stats.dropper_61_count == 0


def test_evaluation_counts_awareness_transitions_only_on_change():
    env = DTHEnv(
        opponent=LeapAwareSafeBot(),
        agent_role="hal",
        seed=123,
        max_steps=1,
        scenario_sampler=lambda _rng: {
            "game_clock": 3540.0,
            "current_half": 2,
            "first_dropper": "hal",
            "awareness": "unaware",
        },
    )

    stats = play_game(MaxMaskModel(), env, verbose=False)

    assert stats.awareness_transitions == 1
    assert stats.checker_61_count == 0


def test_truncation_summary_stays_zero_without_truncated_games():
    summary = summarize_games([
        GameMetrics(
            won=True,
            half_rounds=3,
            reached_round7_pressure=True,
            reached_round8_bridge=True,
            reached_round9_pre_leap=True,
            reached_leap_window=True,
            reached_leap_turn=True,
            started_before_round7_pressure=True,
            started_before_round8_bridge=True,
            started_before_round9_pre_leap=True,
            started_before_leap_window=True,
            started_before_leap_turn=True,
            awareness_transitions=0,
            checker_61_count=0,
            checker_actions=1,
            dropper_61_count=1,
            dropper_actions=2,
            truncated=False,
        )
    ])

    assert summary["truncation_count"] == 0
    assert summary["truncation_rate"] == 0.0


def test_stage_summary_counts_conditional_route_progression():
    summary = summarize_games([
        GameMetrics(
            won=True,
            half_rounds=10,
            reached_round7_pressure=True,
            reached_round8_bridge=True,
            reached_round9_pre_leap=True,
            reached_leap_window=True,
            reached_leap_turn=True,
            started_before_round7_pressure=True,
            started_before_round8_bridge=True,
            started_before_round9_pre_leap=True,
            started_before_leap_window=True,
            started_before_leap_turn=True,
            awareness_transitions=0,
            checker_61_count=1,
            checker_actions=1,
            dropper_61_count=0,
            dropper_actions=0,
            truncated=False,
        ),
        GameMetrics(
            won=False,
            half_rounds=6,
            reached_round7_pressure=True,
            reached_round8_bridge=False,
            reached_round9_pre_leap=False,
            reached_leap_window=False,
            reached_leap_turn=False,
            started_before_round7_pressure=True,
            started_before_round8_bridge=True,
            started_before_round9_pre_leap=True,
            started_before_leap_window=True,
            started_before_leap_turn=True,
            awareness_transitions=0,
            checker_61_count=0,
            checker_actions=1,
            dropper_61_count=0,
            dropper_actions=0,
            truncated=False,
        ),
    ])

    assert summary["opening_to_round7_pressure_count"] == 2
    assert summary["opening_to_round7_pressure_reach_rate"] == 1.0
    assert summary["opening_to_round8_bridge_count"] == 1
    assert summary["opening_to_round8_bridge_reach_rate"] == 0.5
    assert summary["opening_to_round9_pre_leap_count"] == 1
    assert summary["opening_to_leap_window_count"] == 1
    assert summary["leap_window_to_leap_turn_count"] == 1
    assert summary["leap_window_to_leap_turn_reach_rate"] == 1.0
    assert summary["leap_turn_to_win_count"] == 1
    assert summary["leap_turn_to_win_rate"] == 1.0


def test_build_or_load_model_uses_init_checkpoint(monkeypatch):
    calls = []

    class FakeMaskablePPO:
        @staticmethod
        def load(path, env=None):
            calls.append((path, env))
            return "loaded-model"

    monkeypatch.setattr("training.train_ppo.MaskablePPO", FakeMaskablePPO)

    env = object()
    model = build_or_load_model(env, "models/checkpoints/baku_seed.zip", seed=123)

    assert model == "loaded-model"
    assert calls == [("models/checkpoints/baku_seed.zip", env)]


def test_build_training_opponent_creates_weighted_league_with_model_paths(monkeypatch):
    class FakeModelOpponent:
        def __init__(self, model_path, role):
            self.model_path = model_path
            self.role = role

        def choose_action(self, game, role, turn_duration):
            raise AssertionError("Not needed for this test")

        def reset(self):
            pass

    monkeypatch.setattr("environment.opponents.factory.ModelOpponent", FakeModelOpponent)

    opponent = build_training_opponent(
        agent_role="hal",
        opponent_name="leap_safe",
        opponent_weight=0.10,
        opponent_model_specs=[
            "models/checkpoints/baku_vs_safe_critical_100000.zip:0.45",
            "models/checkpoints/baku_vs_leap_safe_critical_100000.zip:0.45",
        ],
        seed=123,
    )

    entries = list(opponent.entries)

    assert len(entries) == 3
    assert entries[0].label == "leap_safe"
    assert entries[0].weight == 0.10
    assert entries[1].opponent.model_path == "models/checkpoints/baku_vs_safe_critical_100000.zip"
    assert entries[1].opponent.role == "baku"
    assert entries[1].weight == 0.45
    assert entries[2].opponent.model_path == "models/checkpoints/baku_vs_leap_safe_critical_100000.zip"
    assert entries[2].opponent.role == "baku"
    assert entries[2].weight == 0.45


def test_build_training_opponent_supports_bridge_pressure_bot():
    opponent = build_training_opponent(
        agent_role="baku",
        opponent_name="bridge_pressure",
        opponent_weight=1.0,
        opponent_model_specs=[],
        seed=123,
    )

    assert isinstance(opponent, BridgePressureBot)


def test_build_training_opponent_supports_teacher_bots():
    opponent = build_training_opponent(
        agent_role="hal",
        opponent_name="baku_teacher",
        opponent_weight=1.0,
        opponent_model_specs=[],
        seed=123,
    )

    assert isinstance(opponent, BakuTeacher)


def test_build_training_opponent_supports_scripted_opening_league():
    opponent = build_training_opponent(
        agent_role="baku",
        opponent_name="opening_league",
        opponent_weight=1.0,
        opponent_model_specs=[],
        seed=123,
    )

    assert isinstance(opponent, WeightedOpponentLeague)
    assert [entry.label for entry in opponent.entries] == [
        "bridge_pressure",
        "hal_death_trade",
        "hal_pressure",
    ]
