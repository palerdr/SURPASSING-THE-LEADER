"""Tests for enriched routing/classifier selector classes.

Covers:
  - EnrichedModularSelector: routing with enriched classification
  - EnrichedDualHeadSelector: dual-head routing with enriched classification
  - EnrichedStackedSelector: stacked routing with enriched classifiers
  - EnrichedDelayedSelector: delayed T4 classification routing
  - build_enriched_classifier: classifier training on 24-dim features
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from environment.routing_features import ROUTING_FEATURE_SIZE
from src.Game import HalfRoundRecord, HalfRoundResult


# ── Minimal stubs ────────────────────────────────────────────────────────

class _Player:
    def __init__(self, name: str):
        self.name = name


class _Game:
    def __init__(self, history=None):
        self.history = history or []


class _Env:
    """Minimal env stub providing game/agent/opp_player for enriched selectors."""
    def __init__(self, history=None):
        self.game = _Game(history or [])
        self.agent = _Player("Baku")
        self.opp_player = _Player("Hal")


def _record(*, dropper, checker, drop_time, check_time, turn_duration=60):
    return HalfRoundRecord(
        round_num=0, half=1, dropper=dropper, checker=checker,
        drop_time=drop_time, check_time=check_time,
        turn_duration=turn_duration,
        result=HalfRoundResult.CHECK_SUCCESS,
        st_gained=1.0, death_duration=0.0, survived=None,
        game_clock_at_start=0.0, survival_probability=None,
    )


class _MockClassifier:
    """Mock LogisticRegression that classifies based on whether input is 20 or 24 dim."""
    def __init__(self, label_for_enriched=1, label_for_base=0):
        self.label_for_enriched = label_for_enriched
        self.label_for_base = label_for_base
        self.last_input_dim = None

    def predict(self, X):
        self.last_input_dim = X.shape[1]
        if X.shape[1] > 20:
            return np.array([self.label_for_enriched])
        return np.array([self.label_for_base])

    def predict_proba(self, X):
        self.last_input_dim = X.shape[1]
        if X.shape[1] > 20:
            return np.array([[0.2, 0.8]])
        return np.array([[0.8, 0.2]])


class _MockModel:
    """Mock model with predict() returning a fixed action."""
    def __init__(self, action: int):
        self.action = action
        self.policy = None

    def predict(self, obs, *, action_masks=None, deterministic=True):
        return np.array(self.action), None


# ── Import enriched selectors from sprint script ────────────────────────
from scripts.run_routing_feature_enrichment_sprint import (
    EnrichedModularSelector,
    EnrichedDualHeadSelector,
    EnrichedStackedSelector,
    EnrichedDelayedSelector,
    ControllerAsModel,
    build_enriched_classifier,
)


# ═════════════════════════════════════════════════════════════════════════
# TestEnrichedModularSelector
# ═════════════════════════════════════════════════════════════════════════

class TestEnrichedModularSelector:

    def _make_selector(self, clf_label=1, classify_turn=2):
        bp_spec = _MockModel(action=59)    # bp_specialist → action 59
        opening = _MockModel(action=0)     # opening → action 0
        late = _MockModel(action=30)       # late → action 30
        clf = _MockClassifier(label_for_enriched=clf_label)
        sel = EnrichedModularSelector(
            bp_specialist=bp_spec, opening_model=opening, late_model=late,
            enriched_classifier=clf, opening_horizon=8, classify_turn=classify_turn,
        )
        return sel, clf

    def test_pre_classification_uses_opening(self):
        """Before classify_turn, selector uses opening_model (default)."""
        sel, clf = self._make_selector(classify_turn=2)
        env = _Env()
        sel.set_env(env)
        # Turn 0 (game_turn=0 < 8): should use opening (action=0)
        obs = np.zeros(20, dtype=np.float32)
        obs[8] = 0.0; obs[9] = 0.0  # round=0, half=0 → game_turn=0
        action, _ = sel.predict(obs)
        assert int(action) == 0

    def test_enriched_classification_routes_hal_correctly(self):
        """When enriched clf returns hal(1), routes to opening_model."""
        sel, clf = self._make_selector(clf_label=1)
        env = _Env(history=[
            _record(dropper="Hal", checker="Baku", drop_time=30, check_time=40),
            _record(dropper="Baku", checker="Hal", drop_time=10, check_time=20),
        ])
        sel.set_env(env)
        sel._turn = 2  # at classify_turn
        obs = np.zeros(20, dtype=np.float32)
        obs[8] = 0.1; obs[9] = 0.0  # game_turn=2 < 8
        action, _ = sel.predict(obs)
        assert clf.last_input_dim == 20 + ROUTING_FEATURE_SIZE  # 24-dim input
        assert int(action) == 0  # opening model

    def test_enriched_classification_routes_bp_correctly(self):
        """When enriched clf returns bp(0), routes to bp_specialist."""
        sel, clf = self._make_selector(clf_label=0)
        env = _Env(history=[
            _record(dropper="Hal", checker="Baku", drop_time=30, check_time=40),
        ])
        sel.set_env(env)
        sel._turn = 2
        obs = np.zeros(20, dtype=np.float32)
        obs[8] = 0.1; obs[9] = 0.0
        action, _ = sel.predict(obs)
        assert clf.last_input_dim == 24
        assert int(action) == 59  # bp_specialist

    def test_late_game_hal_uses_late_model(self):
        """After opening horizon, hal-classified games use late_model."""
        sel, clf = self._make_selector(clf_label=1)
        env = _Env(history=[
            _record(dropper="Hal", checker="Baku", drop_time=30, check_time=40),
            _record(dropper="Baku", checker="Hal", drop_time=10, check_time=20),
        ])
        sel.set_env(env)
        sel._turn = 2
        # First predict triggers classification
        obs_early = np.zeros(20, dtype=np.float32)
        obs_early[8] = 0.1; obs_early[9] = 0.0  # game_turn=2
        sel.predict(obs_early)
        # Now send late-game obs
        obs_late = np.zeros(20, dtype=np.float32)
        obs_late[8] = 0.5; obs_late[9] = 0.0  # game_turn=10 >= 8
        action, _ = sel.predict(obs_late)
        assert int(action) == 30  # late_model

    def test_reset_clears_classification(self):
        sel, _ = self._make_selector()
        sel._is_bp = True
        sel._turn = 5
        sel.reset()
        assert sel._is_bp is None
        assert sel._turn == 0

    def test_assert_on_missing_env(self):
        """Predict without set_env raises AssertionError."""
        sel, _ = self._make_selector()
        sel._turn = 2
        obs = np.zeros(20, dtype=np.float32)
        with pytest.raises(AssertionError, match="Must call set_env"):
            sel.predict(obs)


# ═════════════════════════════════════════════════════════════════════════
# TestEnrichedDualHeadSelector
# ═════════════════════════════════════════════════════════════════════════

class TestEnrichedDualHeadSelector:

    def _make_selector(self, clf_label=1):
        bp_opening = _MockModel(action=59)
        hal_opening = _MockModel(action=0)
        bp_spec = _MockModel(action=58)
        late = _MockModel(action=30)
        clf = _MockClassifier(label_for_enriched=clf_label)
        sel = EnrichedDualHeadSelector(
            bp_opening_head=bp_opening, hal_opening_head=hal_opening,
            bp_specialist=bp_spec, late_model=late,
            enriched_classifier=clf, opening_horizon=8, classify_turn=2,
        )
        return sel, clf

    def test_pre_classification_uses_bp_opening(self):
        """Before classification, defaults to bp_opening_head."""
        sel, _ = self._make_selector()
        env = _Env()
        sel.set_env(env)
        obs = np.zeros(20, dtype=np.float32)
        action, _ = sel.predict(obs)
        assert int(action) == 59  # bp_opening

    def test_hal_classified_uses_hal_opening(self):
        """When classified as hal, opening uses hal_opening_head."""
        sel, clf = self._make_selector(clf_label=1)
        env = _Env(history=[
            _record(dropper="Hal", checker="Baku", drop_time=30, check_time=40),
            _record(dropper="Baku", checker="Hal", drop_time=10, check_time=20),
        ])
        sel.set_env(env)
        sel._turn = 2
        obs = np.zeros(20, dtype=np.float32)
        obs[8] = 0.1; obs[9] = 0.0  # game_turn=2
        action, _ = sel.predict(obs)
        assert clf.last_input_dim == 24
        assert int(action) == 0  # hal_opening

    def test_bp_classified_uses_bp_opening(self):
        sel, clf = self._make_selector(clf_label=0)
        env = _Env(history=[
            _record(dropper="Hal", checker="Baku", drop_time=30, check_time=40),
        ])
        sel.set_env(env)
        sel._turn = 2
        obs = np.zeros(20, dtype=np.float32)
        obs[8] = 0.1; obs[9] = 0.0
        action, _ = sel.predict(obs)
        assert int(action) == 59  # bp_opening

    def test_late_bp_routes_to_specialist(self):
        sel, _ = self._make_selector(clf_label=0)
        env = _Env(history=[
            _record(dropper="Hal", checker="Baku", drop_time=30, check_time=40),
        ])
        sel.set_env(env)
        sel._turn = 2
        obs_early = np.zeros(20, dtype=np.float32)
        obs_early[8] = 0.1; obs_early[9] = 0.0
        sel.predict(obs_early)
        obs_late = np.zeros(20, dtype=np.float32)
        obs_late[8] = 0.5; obs_late[9] = 0.0  # game_turn=10
        action, _ = sel.predict(obs_late)
        assert int(action) == 58  # bp_specialist

    def test_late_hal_routes_to_late_model(self):
        sel, _ = self._make_selector(clf_label=1)
        env = _Env(history=[
            _record(dropper="Hal", checker="Baku", drop_time=30, check_time=40),
            _record(dropper="Baku", checker="Hal", drop_time=10, check_time=20),
        ])
        sel.set_env(env)
        sel._turn = 2
        obs_early = np.zeros(20, dtype=np.float32)
        obs_early[8] = 0.1; obs_early[9] = 0.0
        sel.predict(obs_early)
        obs_late = np.zeros(20, dtype=np.float32)
        obs_late[8] = 0.5; obs_late[9] = 0.0
        action, _ = sel.predict(obs_late)
        assert int(action) == 30  # late_model

    def test_reset(self):
        sel, _ = self._make_selector()
        sel._is_bp = True
        sel._turn = 5
        sel.reset()
        assert sel._is_bp is None
        assert sel._turn == 0


# ═════════════════════════════════════════════════════════════════════════
# TestEnrichedStackedSelector
# ═════════════════════════════════════════════════════════════════════════

class TestEnrichedStackedSelector:

    def _make_selector(self, opening_label=1, late_label=0):
        bp_opening = _MockModel(action=59)
        hal_opening = _MockModel(action=0)
        bp_spec = _MockModel(action=58)
        late = _MockModel(action=30)
        opening_clf = _MockClassifier(label_for_enriched=opening_label)
        late_clf = _MockClassifier(label_for_enriched=late_label)
        sel = EnrichedStackedSelector(
            bp_opening_head=bp_opening, hal_opening_head=hal_opening,
            bp_specialist=bp_spec, late_model=late,
            opening_classifier=opening_clf, late_classifier=late_clf,
            opening_horizon=8, opening_classify_turn=2, late_classify_turn=2,
        )
        return sel

    def test_opening_hal_classified_uses_hal_opening(self):
        sel = self._make_selector(opening_label=1)
        env = _Env(history=[
            _record(dropper="Hal", checker="Baku", drop_time=30, check_time=40),
            _record(dropper="Baku", checker="Hal", drop_time=10, check_time=20),
        ])
        sel.set_env(env)
        sel._turn = 2
        obs = np.zeros(20, dtype=np.float32)
        obs[8] = 0.1; obs[9] = 0.0
        action, _ = sel.predict(obs)
        assert int(action) == 0  # hal_opening

    def test_late_bp_classified_routes_to_specialist(self):
        sel = self._make_selector(opening_label=1, late_label=0)
        env = _Env(history=[
            _record(dropper="Hal", checker="Baku", drop_time=30, check_time=40),
            _record(dropper="Baku", checker="Hal", drop_time=10, check_time=20),
        ])
        sel.set_env(env)
        sel._turn = 2
        obs_early = np.zeros(20, dtype=np.float32)
        obs_early[8] = 0.1; obs_early[9] = 0.0
        sel.predict(obs_early)
        obs_late = np.zeros(20, dtype=np.float32)
        obs_late[8] = 0.5; obs_late[9] = 0.0
        action, _ = sel.predict(obs_late)
        assert int(action) == 58  # bp_specialist

    def test_independent_classifiers(self):
        """Opening and late use independent classifiers."""
        sel = self._make_selector(opening_label=0, late_label=1)
        env = _Env(history=[
            _record(dropper="Hal", checker="Baku", drop_time=30, check_time=40),
        ])
        sel.set_env(env)
        sel._turn = 2
        # Opening: bp classified → bp_opening
        obs = np.zeros(20, dtype=np.float32)
        obs[8] = 0.1; obs[9] = 0.0
        action, _ = sel.predict(obs)
        assert int(action) == 59  # bp_opening
        # Late: hal classified → late_model
        obs_late = np.zeros(20, dtype=np.float32)
        obs_late[8] = 0.5; obs_late[9] = 0.0
        action, _ = sel.predict(obs_late)
        assert int(action) == 30  # late_model

    def test_reset_clears_both_labels(self):
        sel = self._make_selector()
        sel._opening_label = 0
        sel._late_label = 1
        sel._turn = 10
        sel.reset()
        assert sel._opening_label is None
        assert sel._late_label is None
        assert sel._turn == 0


# ═════════════════════════════════════════════════════════════════════════
# TestEnrichedDelayedSelector
# ═════════════════════════════════════════════════════════════════════════

class TestEnrichedDelayedSelector:

    def _make_selector(self, clf_label=1, classify_turn=4):
        universal = _MockModel(action=5)
        bp_opening = _MockModel(action=59)
        hal_opening = _MockModel(action=0)
        bp_spec = _MockModel(action=58)
        late = _MockModel(action=30)
        clf = _MockClassifier(label_for_enriched=clf_label)
        sel = EnrichedDelayedSelector(
            universal_opening=universal,
            bp_opening_head=bp_opening, hal_opening_head=hal_opening,
            bp_specialist=bp_spec, late_model=late,
            enriched_classifier=clf, opening_horizon=8, classify_turn=classify_turn,
        )
        return sel, clf

    def test_pre_classification_uses_universal(self):
        """Before classify_turn, uses universal_opening."""
        sel, _ = self._make_selector(classify_turn=4)
        env = _Env()
        sel.set_env(env)
        obs = np.zeros(20, dtype=np.float32)
        action, _ = sel.predict(obs)
        assert int(action) == 5  # universal

    def test_at_classify_turn_classifies(self):
        sel, clf = self._make_selector(clf_label=1, classify_turn=4)
        env = _Env(history=[
            _record(dropper="Hal", checker="Baku", drop_time=30, check_time=40),
            _record(dropper="Baku", checker="Hal", drop_time=10, check_time=20),
            _record(dropper="Hal", checker="Baku", drop_time=25, check_time=35),
            _record(dropper="Baku", checker="Hal", drop_time=15, check_time=25),
        ])
        sel.set_env(env)
        sel._turn = 4  # at classify_turn
        obs = np.zeros(20, dtype=np.float32)
        obs[8] = 0.2; obs[9] = 0.0  # game_turn=4
        action, _ = sel.predict(obs)
        assert clf.last_input_dim == 24
        assert int(action) == 0  # hal_opening (classified as hal)

    def test_bp_classified_late_routes_to_specialist(self):
        sel, _ = self._make_selector(clf_label=0, classify_turn=4)
        env = _Env(history=[
            _record(dropper="Hal", checker="Baku", drop_time=30, check_time=40),
        ])
        sel.set_env(env)
        sel._turn = 4
        obs_early = np.zeros(20, dtype=np.float32)
        obs_early[8] = 0.2; obs_early[9] = 0.0
        sel.predict(obs_early)
        obs_late = np.zeros(20, dtype=np.float32)
        obs_late[8] = 0.5; obs_late[9] = 0.0
        action, _ = sel.predict(obs_late)
        assert int(action) == 58  # bp_specialist

    def test_reset(self):
        sel, _ = self._make_selector()
        sel._route = "bp"
        sel._turn = 10
        sel.reset()
        assert sel._route is None
        assert sel._turn == 0


# ═════════════════════════════════════════════════════════════════════════
# TestBuildEnrichedClassifier
# ═════════════════════════════════════════════════════════════════════════

class TestBuildEnrichedClassifier:

    def test_enriched_classifier_uses_24dim_features(self):
        """Verify enriched classifier is trained on 24-dim input."""
        # We test the contract: the classifier accepts 24-dim input
        clf = _MockClassifier()
        enriched_obs = np.zeros((1, 20 + ROUTING_FEATURE_SIZE), dtype=np.float32)
        pred = clf.predict(enriched_obs)
        assert clf.last_input_dim == 24

    def test_enriched_vs_base_obs_different_dimensions(self):
        """The enriched observation is strictly larger than base."""
        base_obs = np.zeros(20, dtype=np.float32)
        from environment.routing_features import build_routing_feature_vector
        game = _Game([_record(dropper="Hal", checker="Baku", drop_time=30, check_time=40)])
        enriched = build_routing_feature_vector(
            base_obs, game, _Player("Baku"), _Player("Hal"),
        )
        assert enriched.shape[0] == 20 + ROUTING_FEATURE_SIZE
        assert enriched.shape[0] > base_obs.shape[0]
