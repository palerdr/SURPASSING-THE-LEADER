"""Tests for modular opening-head infrastructure.

Covers:
- ModularBakuSelector routing logic
- _game_turn_from_obs correctness
- OpeningAutoPlayEnv reset/step behavior
- LateCurriculumEnv mixing behavior
"""

from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest
from unittest.mock import MagicMock, patch
import gymnasium as gym

from training.modular_policy import ActionClampEnv, DenseRewardWrapper, ModularBakuSelector, RecurrentModularBakuSelector


class TestGameTurnFromObs:
    """Test observation-based game turn extraction."""

    def test_opening_turn_0(self):
        obs = np.zeros(20, dtype=np.float32)
        obs[8] = 0.0  # round 0
        obs[9] = 0.0  # half 1
        assert ModularBakuSelector._game_turn_from_obs(obs) == 0

    def test_opening_turn_1(self):
        obs = np.zeros(20, dtype=np.float32)
        obs[8] = 0.0  # round 0
        obs[9] = 1.0  # half 2
        assert ModularBakuSelector._game_turn_from_obs(obs) == 1

    def test_turn_4(self):
        obs = np.zeros(20, dtype=np.float32)
        obs[8] = 0.2  # round 2
        obs[9] = 0.0  # half 1
        assert ModularBakuSelector._game_turn_from_obs(obs) == 4

    def test_round7_turn(self):
        obs = np.zeros(20, dtype=np.float32)
        obs[8] = 0.6  # round 6
        obs[9] = 0.0  # half 1
        assert ModularBakuSelector._game_turn_from_obs(obs) == 12

    def test_round9_turn(self):
        obs = np.zeros(20, dtype=np.float32)
        obs[8] = 0.8  # round 8
        obs[9] = 1.0  # half 2
        assert ModularBakuSelector._game_turn_from_obs(obs) == 17

    def test_2d_obs(self):
        obs = np.zeros((1, 20), dtype=np.float32)
        obs[0, 8] = 0.3  # round 3
        obs[0, 9] = 1.0  # half 2
        assert ModularBakuSelector._game_turn_from_obs(obs) == 7


class TestModularBakuSelectorRouting:
    """Test that the selector routes to the correct model."""

    def _make_mock_model(self, action_value: int):
        m = MagicMock()
        m.predict.return_value = (np.array(action_value), None)
        return m

    def _make_selector(self, opening_horizon=8):
        bp = self._make_mock_model(action_value=50)
        opening = self._make_mock_model(action_value=9)
        late = self._make_mock_model(action_value=30)
        clf = MagicMock()
        clf.predict.return_value = np.array([1])  # 1 = hal by default
        return ModularBakuSelector(
            bp_specialist=bp,
            opening_model=opening,
            late_model=late,
            classifier=clf,
            opening_horizon=opening_horizon,
            classify_turn=2,
        )

    def test_opening_turn_uses_opening_model(self):
        sel = self._make_selector()
        obs = np.zeros(20, dtype=np.float32)
        obs[8] = 0.0  # round 0, half 1 = game_turn 0
        obs[9] = 0.0
        action, _ = sel.predict(obs, action_masks=np.ones(61, dtype=bool))
        assert action == 9  # opening model's action

    def test_late_turn_uses_late_model(self):
        sel = self._make_selector(opening_horizon=8)
        obs = np.zeros(20, dtype=np.float32)
        obs[8] = 0.5  # round 5 = game_turn 10
        obs[9] = 0.0
        # Need to advance past classify_turn
        sel._turn = 3
        sel._is_bp = False  # already classified as hal
        action, _ = sel.predict(obs, action_masks=np.ones(61, dtype=bool))
        assert action == 30  # late model's action

    def test_bp_classification_routes_to_bp_specialist(self):
        sel = self._make_selector()
        sel.classifier.predict = MagicMock(return_value=np.array([0]))  # 0 = bp
        obs = np.zeros(20, dtype=np.float32)
        obs[8] = 0.1  # round 1
        obs[9] = 0.0
        # Advance to classify_turn
        sel._turn = 2
        action, _ = sel.predict(obs, action_masks=np.ones(61, dtype=bool))
        assert action == 50  # bp specialist's action

    def test_r9_scenario_uses_late_model(self):
        """r9 scenario (game_turn >> opening_horizon) should use late model."""
        sel = self._make_selector(opening_horizon=8)
        obs = np.zeros(20, dtype=np.float32)
        obs[8] = 0.8  # round 8 = game_turn 16
        obs[9] = 0.0
        action, _ = sel.predict(obs, action_masks=np.ones(61, dtype=bool))
        assert action == 30  # late model (not opening model)

    def test_reset_clears_state(self):
        sel = self._make_selector()
        sel._turn = 5
        sel._is_bp = True
        sel.reset()
        assert sel._turn == 0
        assert sel._is_bp is None


class TestRecurrentModularBakuSelector:
    def _make_mock_model(self, action_value: int):
        m = MagicMock()
        m.predict.return_value = (np.array(action_value), "next_state")
        return m

    def test_late_model_receives_episode_start_once(self):
        bp = MagicMock()
        opening = MagicMock()
        late = MagicMock()
        late.predict.side_effect = [
            (np.array(12), "state1"),
            (np.array(13), "state2"),
        ]
        clf = MagicMock()
        clf.predict = MagicMock(return_value=np.array([1]))
        sel = RecurrentModularBakuSelector(
            bp_specialist=bp,
            opening_model=opening,
            late_model=late,
            classifier=clf,
            opening_horizon=8,
            classify_turn=0,
        )

        obs = np.zeros(20, dtype=np.float32)
        obs[8] = 0.5  # round 5 -> game turn 10
        obs[9] = 0.0

        action1, _ = sel.predict(obs, action_masks=np.ones(61, dtype=bool))
        action2, _ = sel.predict(obs, action_masks=np.ones(61, dtype=bool))

        assert action1 == 12
        assert action2 == 13
        first_call = late.predict.call_args_list[0].kwargs
        second_call = late.predict.call_args_list[1].kwargs
        assert first_call["episode_start"].tolist() == [True]
        assert second_call["episode_start"].tolist() == [False]

    def test_reset_clears_lstm_state(self):
        sel = RecurrentModularBakuSelector(
            bp_specialist=MagicMock(),
            opening_model=MagicMock(),
            late_model=MagicMock(),
            classifier=MagicMock(),
        )
        sel._turn = 4
        sel._is_bp = False
        sel._lstm_state = "state"
        sel._lstm_started = True
        sel.reset()
        assert sel._turn == 0
        assert sel._is_bp is None
        assert sel._lstm_state is None
        assert sel._lstm_started is False


class TestActionClampEnv:
    class DummyEnv(gym.Env):
        metadata = {"render_modes": []}

        def __init__(self, mask):
            super().__init__()
            self._mask = mask
            self.last_action = None

        def action_masks(self):
            return self._mask

        def step(self, action):
            self.last_action = action
            return ("obs", 0.0, False, False, {})

        def reset(self, *, seed=None, options=None):
            return "obs", {}

    def test_illegal_action_is_clamped(self):
        inner = self.DummyEnv(np.array([True] * 60 + [False], dtype=bool))
        env = ActionClampEnv(inner)
        env.step(60)
        assert inner.last_action == 59

    def test_legal_action_passes_through(self):
        inner = self.DummyEnv(np.array([True] * 61, dtype=bool))
        env = ActionClampEnv(inner)
        env.step(17)
        assert inner.last_action == 17


class TestDenseRewardWrapper:
    class DummyGame:
        def __init__(self):
            self.history = [object()]

    class DummyLeafEnv(gym.Env):
        metadata = {"render_modes": []}

        def __init__(self):
            super().__init__()
            self.game = TestDenseRewardWrapper.DummyGame()
            self.agent = object()
            self.opp_player = object()

        def reset(self, *, seed=None, options=None):
            return np.zeros(1, dtype=np.float32), {}

        def step(self, action):
            return np.zeros(1, dtype=np.float32), 1.0, False, False, {}

        def action_masks(self):
            return np.array([True])

    def test_dense_reward_added(self):
        env = DenseRewardWrapper(self.DummyLeafEnv(), reward_fn=lambda g, a, o: 2.5, reward_scale=2.0)
        _, reward, _, _, _ = env.step(0)
        assert reward == 6.0

    def test_dense_reward_walks_wrapped_env_chain(self):
        base = self.DummyLeafEnv()
        wrapped = ActionClampEnv(base)
        env = DenseRewardWrapper(wrapped, reward_fn=lambda g, a, o: 1.0, reward_scale=1.0)
        _, reward, _, _, _ = env.step(0)
        assert reward == 2.0
