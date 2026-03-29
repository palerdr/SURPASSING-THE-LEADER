"""Tests for dual-head opening routing infrastructure."""

from __future__ import annotations

import os
import sys
from unittest.mock import MagicMock

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.run_dual_head_opening_sprint import (
    ConfidenceRoutingSelector,
    ControllerAsModel,
    DualHeadOpeningSelector,
    StackedRoutingSelector,
)
from training.expert_selector import FeatureRuleController


def _obs(game_round: float, game_half: float, *, role: float = 0.0) -> np.ndarray:
    obs = np.zeros(20, dtype=np.float32)
    obs[7] = role
    obs[8] = game_round
    obs[9] = game_half
    return obs


def _make_model(action_value: int):
    model = MagicMock()
    model.predict.return_value = (np.array(action_value), None)
    return model


def _make_clf(label: int):
    clf = MagicMock()
    clf.predict.return_value = np.array([label])
    clf.predict_proba.return_value = np.array(
        [[0.9, 0.1]] if label == 0 else [[0.1, 0.9]]
    )
    return clf


def _make_ambiguous_clf():
    """Classifier that returns low-confidence predictions."""
    clf = MagicMock()
    clf.predict.return_value = np.array([0])
    clf.predict_proba.return_value = np.array([[0.55, 0.45]])
    return clf


# ═════════════════════════════════════════════════════════════════════════
# DualHeadOpeningSelector tests
# ═════════════════════════════════════════════════════════════════════════
class TestDualHeadOpeningSelector:
    def test_bp_classified_uses_bp_opening_head_during_opening(self):
        bp_opening = _make_model(10)
        hal_opening = _make_model(20)
        bp_specialist = _make_model(50)
        late = _make_model(30)
        clf = _make_clf(0)  # bp
        sel = DualHeadOpeningSelector(
            bp_opening_head=bp_opening, hal_opening_head=hal_opening,
            bp_specialist=bp_specialist, late_model=late,
            classifier=clf, opening_horizon=8, classify_turn=2,
        )
        sel._turn = 2  # at classify turn
        action, _ = sel.predict(_obs(0.1, 0.0), action_masks=np.ones(61, dtype=bool))
        assert int(action) == 10  # bp_opening_head
        hal_opening.predict.assert_not_called()

    def test_hal_classified_uses_hal_opening_head_during_opening(self):
        bp_opening = _make_model(10)
        hal_opening = _make_model(20)
        bp_specialist = _make_model(50)
        late = _make_model(30)
        clf = _make_clf(1)  # hal
        sel = DualHeadOpeningSelector(
            bp_opening_head=bp_opening, hal_opening_head=hal_opening,
            bp_specialist=bp_specialist, late_model=late,
            classifier=clf, opening_horizon=8, classify_turn=2,
        )
        sel._turn = 2
        action, _ = sel.predict(_obs(0.1, 0.0), action_masks=np.ones(61, dtype=bool))
        assert int(action) == 20  # hal_opening_head
        bp_opening.predict.assert_not_called()

    def test_bp_classified_uses_bp_specialist_after_opening(self):
        bp_opening = _make_model(10)
        hal_opening = _make_model(20)
        bp_specialist = _make_model(50)
        late = _make_model(30)
        clf = _make_clf(0)
        sel = DualHeadOpeningSelector(
            bp_opening_head=bp_opening, hal_opening_head=hal_opening,
            bp_specialist=bp_specialist, late_model=late,
            classifier=clf, opening_horizon=8, classify_turn=2,
        )
        sel._turn = 2
        sel._is_bp = True
        action, _ = sel.predict(_obs(0.5, 0.0), action_masks=np.ones(61, dtype=bool))
        # game_turn = 5*2+0 = 10 >= 8 → bp_specialist
        assert int(action) == 50

    def test_hal_classified_uses_late_model_after_opening(self):
        bp_opening = _make_model(10)
        hal_opening = _make_model(20)
        bp_specialist = _make_model(50)
        late = _make_model(30)
        clf = _make_clf(1)
        sel = DualHeadOpeningSelector(
            bp_opening_head=bp_opening, hal_opening_head=hal_opening,
            bp_specialist=bp_specialist, late_model=late,
            classifier=clf, opening_horizon=8, classify_turn=2,
        )
        sel._turn = 2
        sel._is_bp = False
        action, _ = sel.predict(_obs(0.5, 0.0), action_masks=np.ones(61, dtype=bool))
        assert int(action) == 30  # late_model

    def test_pre_classification_defaults_to_bp_opening(self):
        bp_opening = _make_model(10)
        hal_opening = _make_model(20)
        sel = DualHeadOpeningSelector(
            bp_opening_head=bp_opening, hal_opening_head=hal_opening,
            bp_specialist=_make_model(50), late_model=_make_model(30),
            classifier=_make_clf(0), opening_horizon=8, classify_turn=2,
        )
        # Turn 0 (before classify_turn)
        action, _ = sel.predict(_obs(0.0, 0.0), action_masks=np.ones(61, dtype=bool))
        assert int(action) == 10  # defaults to bp_opening

    def test_reset_clears_state(self):
        sel = DualHeadOpeningSelector(
            bp_opening_head=_make_model(10), hal_opening_head=_make_model(20),
            bp_specialist=_make_model(50), late_model=_make_model(30),
            classifier=_make_clf(0), opening_horizon=8, classify_turn=2,
        )
        sel._turn = 5
        sel._is_bp = True
        sel.reset()
        assert sel._turn == 0
        assert sel._is_bp is None


# ═════════════════════════════════════════════════════════════════════════
# StackedRoutingSelector tests
# ═════════════════════════════════════════════════════════════════════════
class TestStackedRoutingSelector:
    def test_universal_opening_mode(self):
        """When opening_classifier is None and universal_opening is set, all
        opponents get the universal opening during opening phase."""
        universal = _make_model(15)
        sel = StackedRoutingSelector(
            bp_opening_head=_make_model(10), hal_opening_head=_make_model(20),
            bp_specialist=_make_model(50), late_model=_make_model(30),
            opening_classifier=None, late_classifier=_make_clf(0),
            opening_horizon=8, universal_opening=universal,
        )
        action, _ = sel.predict(_obs(0.0, 0.0), action_masks=np.ones(61, dtype=bool))
        assert int(action) == 15  # universal opening

    def test_separate_opening_and_late_classifiers(self):
        """Opening classifier and late classifier make independent decisions."""
        bp_opening = _make_model(10)
        bp_specialist = _make_model(50)
        late = _make_model(30)
        # Opening classifier says bp (0), late classifier says hal (1)
        open_clf = _make_clf(0)
        late_clf = _make_clf(1)
        sel = StackedRoutingSelector(
            bp_opening_head=bp_opening, hal_opening_head=_make_model(20),
            bp_specialist=bp_specialist, late_model=late,
            opening_classifier=open_clf, late_classifier=late_clf,
            opening_horizon=8, opening_classify_turn=2, late_classify_turn=2,
        )
        # During opening (turn 2, game_turn=0.1*10*2+0=2 < 8)
        sel._turn = 2
        action, _ = sel.predict(_obs(0.1, 0.0), action_masks=np.ones(61, dtype=bool))
        assert int(action) == 10  # bp_opening (opening clf says bp)

        # After opening (game_turn=10 >= 8)
        sel._turn = 2
        action, _ = sel.predict(_obs(0.5, 0.0), action_masks=np.ones(61, dtype=bool))
        assert int(action) == 30  # late_model (late clf says hal)

    def test_late_phase_bp_routing(self):
        """Late classifier routing bp to bp_specialist."""
        sel = StackedRoutingSelector(
            bp_opening_head=_make_model(10), hal_opening_head=_make_model(20),
            bp_specialist=_make_model(50), late_model=_make_model(30),
            opening_classifier=None, late_classifier=_make_clf(0),
            opening_horizon=8, universal_opening=_make_model(15),
        )
        sel._turn = 2
        action, _ = sel.predict(_obs(0.5, 0.0), action_masks=np.ones(61, dtype=bool))
        assert int(action) == 50  # bp_specialist

    def test_reset_clears_both_labels(self):
        sel = StackedRoutingSelector(
            bp_opening_head=_make_model(10), hal_opening_head=_make_model(20),
            bp_specialist=_make_model(50), late_model=_make_model(30),
            opening_classifier=_make_clf(0), late_classifier=_make_clf(1),
            opening_horizon=8,
        )
        sel._opening_label = 0
        sel._late_label = 1
        sel._turn = 5
        sel.reset()
        assert sel._turn == 0
        assert sel._opening_label is None
        assert sel._late_label is None


# ═════════════════════════════════════════════════════════════════════════
# ConfidenceRoutingSelector tests
# ═════════════════════════════════════════════════════════════════════════
class TestConfidenceRoutingSelector:
    def test_high_confidence_bp_routes_to_bp_path(self):
        clf = MagicMock()
        clf.predict_proba.return_value = np.array([[0.95, 0.05]])
        sel = ConfidenceRoutingSelector(
            bp_opening_head=_make_model(10), hal_opening_head=_make_model(20),
            fallback_opening_head=_make_model(15),
            bp_specialist=_make_model(50), late_model=_make_model(30),
            classifier=clf, bp_threshold=0.7, hal_threshold=0.7,
            opening_horizon=8, classify_turn=2,
        )
        sel._turn = 2
        action, _ = sel.predict(_obs(0.1, 0.0), action_masks=np.ones(61, dtype=bool))
        assert int(action) == 10  # bp_opening
        assert sel._route == "bp"

    def test_high_confidence_hal_routes_to_hal_path(self):
        clf = MagicMock()
        clf.predict_proba.return_value = np.array([[0.05, 0.95]])
        sel = ConfidenceRoutingSelector(
            bp_opening_head=_make_model(10), hal_opening_head=_make_model(20),
            fallback_opening_head=_make_model(15),
            bp_specialist=_make_model(50), late_model=_make_model(30),
            classifier=clf, bp_threshold=0.7, hal_threshold=0.7,
            opening_horizon=8, classify_turn=2,
        )
        sel._turn = 2
        action, _ = sel.predict(_obs(0.1, 0.0), action_masks=np.ones(61, dtype=bool))
        assert int(action) == 20  # hal_opening
        assert sel._route == "hal"

    def test_ambiguous_confidence_routes_to_fallback(self):
        clf = _make_ambiguous_clf()  # P(bp)=0.55, P(hal)=0.45
        sel = ConfidenceRoutingSelector(
            bp_opening_head=_make_model(10), hal_opening_head=_make_model(20),
            fallback_opening_head=_make_model(15),
            bp_specialist=_make_model(50), late_model=_make_model(30),
            classifier=clf, bp_threshold=0.7, hal_threshold=0.7,
            opening_horizon=8, classify_turn=2,
        )
        sel._turn = 2
        action, _ = sel.predict(_obs(0.1, 0.0), action_masks=np.ones(61, dtype=bool))
        assert int(action) == 15  # fallback
        assert sel._route == "fallback"

    def test_fallback_uses_late_model_after_opening(self):
        clf = _make_ambiguous_clf()
        sel = ConfidenceRoutingSelector(
            bp_opening_head=_make_model(10), hal_opening_head=_make_model(20),
            fallback_opening_head=_make_model(15),
            bp_specialist=_make_model(50), late_model=_make_model(30),
            classifier=clf, bp_threshold=0.7, hal_threshold=0.7,
            opening_horizon=8, classify_turn=2,
        )
        sel._turn = 2
        sel._route = "fallback"
        action, _ = sel.predict(_obs(0.5, 0.0), action_masks=np.ones(61, dtype=bool))
        assert int(action) == 30  # late_model (not bp_specialist)

    def test_bp_route_uses_bp_specialist_after_opening(self):
        sel = ConfidenceRoutingSelector(
            bp_opening_head=_make_model(10), hal_opening_head=_make_model(20),
            fallback_opening_head=_make_model(15),
            bp_specialist=_make_model(50), late_model=_make_model(30),
            classifier=_make_clf(0), bp_threshold=0.7, hal_threshold=0.7,
            opening_horizon=8, classify_turn=2,
        )
        sel._turn = 2
        sel._route = "bp"
        action, _ = sel.predict(_obs(0.5, 0.0), action_masks=np.ones(61, dtype=bool))
        assert int(action) == 50  # bp_specialist

    def test_asymmetric_thresholds(self):
        """With bp_threshold=0.8, hal_threshold=0.5:
        P(bp)=0.6 → not bp (0.6 < 0.8), but P(hal)=0.4 → not hal (0.4 < 0.5)
        → fallback."""
        clf = MagicMock()
        clf.predict_proba.return_value = np.array([[0.6, 0.4]])
        sel = ConfidenceRoutingSelector(
            bp_opening_head=_make_model(10), hal_opening_head=_make_model(20),
            fallback_opening_head=_make_model(15),
            bp_specialist=_make_model(50), late_model=_make_model(30),
            classifier=clf, bp_threshold=0.8, hal_threshold=0.5,
            opening_horizon=8, classify_turn=2,
        )
        sel._turn = 2
        sel.predict(_obs(0.1, 0.0), action_masks=np.ones(61, dtype=bool))
        assert sel._route == "fallback"

    def test_reset_clears_route(self):
        sel = ConfidenceRoutingSelector(
            bp_opening_head=_make_model(10), hal_opening_head=_make_model(20),
            fallback_opening_head=_make_model(15),
            bp_specialist=_make_model(50), late_model=_make_model(30),
            classifier=_make_clf(0), bp_threshold=0.7, hal_threshold=0.7,
            opening_horizon=8, classify_turn=2,
        )
        sel._route = "bp"
        sel._turn = 5
        sel.reset()
        assert sel._turn == 0
        assert sel._route is None


# ═════════════════════════════════════════════════════════════════════════
# ControllerAsModel wrapper tests
# ═════════════════════════════════════════════════════════════════════════
class TestControllerAsModel:
    def test_returns_controller_action(self):
        frc = FeatureRuleController({(0, 0, 0): 10}, start_turn=0, end_turn=7)
        wrapper = ControllerAsModel(frc)
        action, _ = wrapper.predict(_obs(0.0, 0.0, role=0.0),
                                     action_masks=np.ones(61, dtype=bool))
        assert int(action) == 9  # action_second 10 → 0-indexed 9

    def test_returns_fallback_when_no_match(self):
        frc = FeatureRuleController({}, start_turn=0, end_turn=7)
        wrapper = ControllerAsModel(frc, fallback_action=5)
        action, _ = wrapper.predict(_obs(0.0, 0.0, role=0.0),
                                     action_masks=np.ones(61, dtype=bool))
        assert int(action) == 5

    def test_game_turn_extraction(self):
        """Verify game_turn computed correctly from obs."""
        # round=0.3, half=1 → game_turn = 3*2+1 = 7
        frc = FeatureRuleController({(0, 3, 1): 42}, start_turn=0, end_turn=7)
        wrapper = ControllerAsModel(frc)
        action, _ = wrapper.predict(_obs(0.3, 1.0, role=0.0),
                                     action_masks=np.ones(61, dtype=bool))
        assert int(action) == 41  # action_second 42 → 0-indexed 41
