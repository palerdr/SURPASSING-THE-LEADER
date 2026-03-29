from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from environment.routing_features import (
    ROUTING_FEATURE_SIZE,
    build_public_opponent_history_features,
    build_routing_feature_vector,
)
from src.Game import HalfRoundRecord, HalfRoundResult


class _Player:
    def __init__(self, name: str):
        self.name = name


class _Game:
    def __init__(self, history):
        self.history = history


def _record(*, dropper, checker, drop_time, check_time, turn_duration=60):
    return HalfRoundRecord(
        round_num=0,
        half=1,
        dropper=dropper,
        checker=checker,
        drop_time=drop_time,
        check_time=check_time,
        turn_duration=turn_duration,
        result=HalfRoundResult.CHECK_SUCCESS,
        st_gained=1.0,
        death_duration=0.0,
        survived=None,
        game_clock_at_start=0.0,
        survival_probability=None,
    )


def test_empty_history_zero_padded():
    game = _Game([])
    me = _Player("Baku")
    opp = _Player("Hal")
    feats = build_public_opponent_history_features(game, me, opp)
    assert feats.shape == (ROUTING_FEATURE_SIZE,)
    assert np.all(feats == 0)


def test_extracts_recent_opponent_actions_and_roles():
    game = _Game([
        _record(dropper="Hal", checker="Baku", drop_time=30, check_time=40),
        _record(dropper="Baku", checker="Hal", drop_time=10, check_time=20),
    ])
    me = _Player("Baku")
    opp = _Player("Hal")
    feats = build_public_opponent_history_features(game, me, opp)
    # Most recent record first: Hal acted as checker at 20/60, then dropper at 30/60.
    assert np.allclose(feats, np.array([20 / 60.0, 0.0, 30 / 60.0, 1.0], dtype=np.float32))


def test_build_routing_feature_vector_appends_history():
    base = np.array([0.1, 0.2], dtype=np.float32)
    game = _Game([_record(dropper="Hal", checker="Baku", drop_time=6, check_time=10)])
    me = _Player("Baku")
    opp = _Player("Hal")
    full = build_routing_feature_vector(base, game, me, opp)
    assert full.shape == (2 + ROUTING_FEATURE_SIZE,)
    assert np.allclose(full[:2], base)
    assert np.isclose(full[2], 6 / 60.0)
    assert np.isclose(full[3], 1.0)
