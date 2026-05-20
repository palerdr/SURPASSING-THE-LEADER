import os
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hal.value_net import FEATURE_DIM, REFEREE_CPR_FEATURE_SCALE, ValueNet, extract_features
from src.Constants import PHYSICALITY_BAKU, PHYSICALITY_HAL
from src.Game import Game
from src.Player import Player
from src.Referee import Referee
from training.train_value_net import make_predict_fn


def _game_at_leap_with_baku_dropper() -> Game:
    hal = Player(name="Hal", physicality=PHYSICALITY_HAL)
    baku = Player(name="Baku", physicality=PHYSICALITY_BAKU)
    game = Game(player1=hal, player2=baku, referee=Referee(), first_dropper=hal)
    game.seed(0)
    game.game_clock = 3540.0
    game.current_half = 2
    return game


def test_value_net_returns_value_dropper_policy_checker_policy_triple():
    model = ValueNet()
    x = torch.zeros(4, FEATURE_DIM)
    value, dropper_logits, checker_logits = model(x)

    assert value.shape == (4, 1)
    assert dropper_logits.shape == (4, 61)
    assert checker_logits.shape == (4, 61)
    assert torch.all(value <= 1.0)
    assert torch.all(value >= -1.0)


def test_legality_mask_zeroes_hal_checker_second_61():
    model = ValueNet()
    predict = make_predict_fn(model)
    value, dropper_dist, checker_dist = predict(_game_at_leap_with_baku_dropper())

    assert isinstance(value, float)
    assert checker_dist[60] == pytest.approx(0.0)
    assert checker_dist.sum() == pytest.approx(1.0)


def test_baku_dropper_in_leap_window_keeps_second_61_mass():
    model = ValueNet()
    _value, dropper_dist, _checker_dist = make_predict_fn(model)(
        _game_at_leap_with_baku_dropper()
    )

    assert dropper_dist[60] > 0.0
    assert np.count_nonzero(dropper_dist) == 61
    assert dropper_dist.sum() == pytest.approx(1.0)


def test_cpr_feature_distinguishes_fatigue_values_above_six():
    hal = Player(name="Hal", physicality=PHYSICALITY_HAL)
    baku = Player(name="Baku", physicality=PHYSICALITY_BAKU)
    game = Game(player1=hal, player2=baku, referee=Referee(), first_dropper=hal)

    game.referee.cprs_performed = 8
    cpr_at_8 = extract_features(game)[20]
    game.referee.cprs_performed = 10
    cpr_at_10 = extract_features(game)[20]
    game.referee.cprs_performed = int(REFEREE_CPR_FEATURE_SCALE)
    cpr_at_scale = extract_features(game)[20]

    assert cpr_at_8 < cpr_at_10 < cpr_at_scale
    assert cpr_at_scale == pytest.approx(1.0)
