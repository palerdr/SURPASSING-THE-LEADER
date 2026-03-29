"""Tests for opening-delivery routing helpers introduced for native-Hal work."""

from __future__ import annotations

import os
import sys
from unittest.mock import MagicMock

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training.expert_selector import FeatureRuleController, MLPOpeningController
from scripts.run_opening_delivery_native_hal_sprint import (
    ControllerAsModel,
    MLPControllerAsModel,
    OpeningPrioritySelector,
)


def _obs(game_round: float, game_half: float, *, role: float = 0.0) -> np.ndarray:
    obs = np.zeros(20, dtype=np.float32)
    obs[7] = role
    obs[8] = game_round
    obs[9] = game_half
    return obs


class TestFeatureRuleController:
    def test_extract_key_matches_game_structure(self):
        obs = _obs(0.2, 1.0, role=1.0)
        assert FeatureRuleController._extract_key(obs) == (1, 2, 1)

    def test_query_returns_mapped_action_only_in_turn_window(self):
        controller = FeatureRuleController({(0, 0, 0): 10}, start_turn=0, end_turn=2)
        assert controller.query(_obs(0.0, 0.0, role=0.0), 0) == 10
        assert controller.query(_obs(0.0, 0.0, role=0.0), 3) is None


class TestMLPOpeningController:
    def test_query_respects_turn_window(self):
        w1 = np.zeros((20, 2), dtype=np.float32)
        b1 = np.zeros(2, dtype=np.float32)
        w2 = np.zeros((2, 61), dtype=np.float32)
        b2 = np.zeros(61, dtype=np.float32)
        b2[7] = 5.0  # argmax -> action_second 8
        controller = MLPOpeningController([w1, w2], [b1, b2], start_turn=1, end_turn=2)
        assert controller.query(_obs(0.0, 0.0), 0) is None
        assert controller.query(_obs(0.0, 0.0), 1) == 8


class TestControllerWrappers:
    def test_controller_as_model_uses_controller_action(self):
        controller = MagicMock()
        controller.query.return_value = 15
        wrapper = ControllerAsModel(controller)
        action, _ = wrapper.predict(_obs(0.0, 0.0), action_masks=np.ones(61, dtype=bool))
        assert int(action) == 14

    def test_controller_as_model_falls_back_when_controller_returns_none(self):
        controller = MagicMock()
        controller.query.return_value = None
        wrapper = ControllerAsModel(controller, fallback_action=9)
        action, _ = wrapper.predict(_obs(0.0, 0.0), action_masks=np.ones(61, dtype=bool))
        assert int(action) == 9

    def test_mlp_controller_as_model_uses_controller_action(self):
        controller = MagicMock()
        controller.query.return_value = 21
        wrapper = MLPControllerAsModel(controller)
        action, _ = wrapper.predict(_obs(0.1, 1.0), action_masks=np.ones(61, dtype=bool))
        assert int(action) == 20


class TestOpeningPrioritySelector:
    def _make_model(self, action_value: int):
        model = MagicMock()
        model.predict.return_value = (np.array(action_value), None)
        return model

    def test_opening_turn_ignores_classifier_and_uses_opening_model(self):
        bp = self._make_model(50)
        opening = self._make_model(9)
        late = self._make_model(30)
        clf = MagicMock()
        clf.predict.return_value = np.array([0])
        selector = OpeningPrioritySelector(
            bp_specialist=bp,
            opening_model=opening,
            late_model=late,
            classifier=clf,
            opening_horizon=8,
            classify_turn=2,
        )
        action, _ = selector.predict(_obs(0.0, 0.0), action_masks=np.ones(61, dtype=bool))
        assert int(action) == 9
        bp.predict.assert_not_called()

    def test_late_turn_routes_to_bp_specialist_after_bp_classification(self):
        bp = self._make_model(50)
        opening = self._make_model(9)
        late = self._make_model(30)
        clf = MagicMock()
        clf.predict.return_value = np.array([0])
        selector = OpeningPrioritySelector(
            bp_specialist=bp,
            opening_model=opening,
            late_model=late,
            classifier=clf,
            opening_horizon=8,
            classify_turn=2,
        )
        selector._turn = 2
        action, _ = selector.predict(_obs(0.5, 0.0), action_masks=np.ones(61, dtype=bool))
        assert int(action) == 50

    def test_late_turn_routes_to_late_model_after_hal_classification(self):
        bp = self._make_model(50)
        opening = self._make_model(9)
        late = self._make_model(30)
        clf = MagicMock()
        clf.predict.return_value = np.array([1])
        selector = OpeningPrioritySelector(
            bp_specialist=bp,
            opening_model=opening,
            late_model=late,
            classifier=clf,
            opening_horizon=8,
            classify_turn=2,
        )
        selector._turn = 2
        action, _ = selector.predict(_obs(0.5, 0.0), action_masks=np.ones(61, dtype=bool))
        assert int(action) == 30

    def test_reset_clears_turn_and_locked_route(self):
        selector = OpeningPrioritySelector(
            bp_specialist=self._make_model(50),
            opening_model=self._make_model(9),
            late_model=self._make_model(30),
            classifier=MagicMock(),
            opening_horizon=8,
            classify_turn=2,
        )
        selector._turn = 7
        selector._is_bp = True
        selector.reset()
        assert selector._turn == 0
        assert selector._is_bp is None
