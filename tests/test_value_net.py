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


# ── Phase I-1: hidden_dim expansion ───────────────────────────────────────


def test_value_net_default_hidden_dim_unchanged():
    """Backward compat: default ValueNet still has 13.7K params + identical
    forward shapes. Phase I-1 only adds a kwarg; it doesn't change defaults.
    """
    model = ValueNet()
    total = sum(p.numel() for p in model.parameters())
    assert 13_000 < total < 14_500, f"default arch param count regressed: {total}"


def test_value_net_hidden_dim_128_has_expected_param_count_and_shapes():
    """Phase I-1: hidden_dim=128 expands the trunk + policy/value heads.

    Param count stays under the 50K guard (35.5K at hidden=128).
    Forward output shapes are identical to hidden=64 — only the internal
    capacity grows.
    """
    model = ValueNet(hidden_dim=128)
    total = sum(p.numel() for p in model.parameters())
    assert 30_000 < total < 50_000, (
        f"hidden=128 should give ~35.5K params (well under 50K guard); got {total}"
    )

    x = torch.zeros(2, FEATURE_DIM)
    value, dropper_logits, checker_logits = model(x)
    assert value.shape == (2, 1)
    assert dropper_logits.shape == (2, 61)
    assert checker_logits.shape == (2, 61)


def test_value_net_checkpoint_round_trip_at_hidden_128(tmp_path):
    """Saving + loading a hidden=128 checkpoint must round-trip via
    ``load_checkpoint``'s auto-inferred hidden_dim. Guards the inference
    logic that reads the state_dict's trunk.0.weight shape — if it
    misreads, the load either fails or produces wrong-capacity weights.
    """
    from training.train_value_net import load_checkpoint

    model = ValueNet(hidden_dim=128)
    ckpt_path = tmp_path / "phase_i_test.pt"
    torch.save(model.state_dict(), ckpt_path)

    loaded = load_checkpoint(str(ckpt_path), device="cpu")
    assert sum(p.numel() for p in loaded.parameters()) == sum(
        p.numel() for p in model.parameters()
    )

    x = torch.zeros(3, FEATURE_DIM)
    with torch.no_grad():
        out_orig = model(x)
        out_loaded = loaded(x)
    for a, b in zip(out_orig, out_loaded):
        assert torch.allclose(a, b)
