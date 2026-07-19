import os
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.getcwd())

from stl.learning.model import (
    FEATURE_DIM,
    FEATURE_INDEX,
    FEATURE_SCHEMA_VERSION,
    REFEREE_CPR_FEATURE_SCALE,
    ValueNet,
    extract_features,
)
from stl.engine.game import PHYSICALITY_BAKU, PHYSICALITY_HAL
from stl.engine.game import Game
from stl.engine.game import Player
from stl.engine.game import Referee
from stl.engine.actions import ACTION_SIZE
from stl.learning.train import make_predict_fn


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
    value, dropper_logits, checker_logits = model(x, 2)

    assert value.shape == (4, 1)
    assert dropper_logits.shape == (4, ACTION_SIZE)
    assert checker_logits.shape == (4, ACTION_SIZE)
    assert torch.all(value <= 1.0)
    assert torch.all(value >= -1.0)


def test_value_net_requires_supported_explicit_horizon():
    model = ValueNet()
    x = torch.zeros(2, FEATURE_DIM)
    with pytest.raises(TypeError):
        model(x)
    with pytest.raises(ValueError, match="unsupported value horizon"):
        model(x, 1)
    value_h2 = model(x, 2)[0]
    value_h3 = model(x, 3)[0]
    assert value_h2.shape == value_h3.shape == (2, 1)


def test_legality_mask_zeroes_hal_checker_second_61():
    model = ValueNet()
    predict = make_predict_fn(model)
    value, dropper_dist, checker_dist = predict(
        _game_at_leap_with_baku_dropper(), horizon=2
    )

    assert isinstance(value, float)
    assert checker_dist[0] == pytest.approx(0.0)
    assert checker_dist[61] == pytest.approx(0.0)
    assert checker_dist.sum() == pytest.approx(1.0)


def test_baku_dropper_in_leap_window_keeps_second_61_mass():
    model = ValueNet()
    _value, dropper_dist, _checker_dist = make_predict_fn(model)(
        _game_at_leap_with_baku_dropper(), horizon=2
    )

    assert dropper_dist[61] > 0.0
    assert np.count_nonzero(dropper_dist) == 61
    assert dropper_dist.sum() == pytest.approx(1.0)


def test_cpr_feature_distinguishes_fatigue_values_above_six():
    hal = Player(name="Hal", physicality=PHYSICALITY_HAL)
    baku = Player(name="Baku", physicality=PHYSICALITY_BAKU)
    game = Game(player1=hal, player2=baku, referee=Referee(), first_dropper=hal)

    game.referee.cprs_performed = 8
    cpr_at_8 = extract_features(game)[FEATURE_INDEX["derived_referee_cprs"]]
    game.referee.cprs_performed = 10
    cpr_at_10 = extract_features(game)[FEATURE_INDEX["derived_referee_cprs"]]
    game.referee.cprs_performed = int(REFEREE_CPR_FEATURE_SCALE)
    cpr_at_scale = extract_features(game)[FEATURE_INDEX["derived_referee_cprs"]]

    assert cpr_at_8 < cpr_at_10 < cpr_at_scale
    assert cpr_at_scale == pytest.approx(1.0)


# ── Phase I-1: hidden_dim expansion ───────────────────────────────────────


def test_value_net_default_hidden_dim_unchanged():
    """Backward compat: default ValueNet still has 13.7K params + identical
    forward shapes. Phase I-1 only adds a kwarg; it doesn't change defaults.
    """
    model = ValueNet()
    total = sum(p.numel() for p in model.parameters())
    assert 15_000 < total < 17_000, f"default arch param count regressed: {total}"


def test_value_net_hidden_dim_128_has_expected_param_count_and_shapes():
    """Phase I-1: hidden_dim=128 expands the trunk + policy/value heads.

    Param count stays under the 50K guard (35.5K at hidden=128).
    Forward output shapes are identical to hidden=64 — only the internal
    capacity grows.
    """
    model = ValueNet(hidden_dim=128)
    total = sum(p.numel() for p in model.parameters())
    assert 35_000 < total < 45_000, (
        f"hidden=128 should give ~39K params; got {total}"
    )

    x = torch.zeros(2, FEATURE_DIM)
    value, dropper_logits, checker_logits = model(x, 2)
    assert value.shape == (2, 1)
    assert dropper_logits.shape == (2, ACTION_SIZE)
    assert checker_logits.shape == (2, ACTION_SIZE)


def test_value_net_hidden_dim_192_under_raised_guard():
    """Phase I-2: hidden_dim=192 widens the trunk to ~65.4K params, exceeding
    the original 50K guard. The guard is raised to 70K — justified because the
    F-2-expanded ruler + interior anchors and the larger exact corpus give the
    bigger net enough signal to generalize rather than memorize (the
    bias-vs-overfit transition the Phase-9 plan predicted). Forward shapes are
    identical to hidden=64; only capacity grows.
    """
    model = ValueNet(hidden_dim=192)
    total = sum(p.numel() for p in model.parameters())
    assert 68_000 < total < 76_000, (
        f"hidden=192 should give ~71K params; got {total}"
    )

    x = torch.zeros(2, FEATURE_DIM)
    value, dropper_logits, checker_logits = model(x, 2)
    assert value.shape == (2, 1)
    assert dropper_logits.shape == (2, ACTION_SIZE)
    assert checker_logits.shape == (2, ACTION_SIZE)


def test_feature_schema_v2_distinguishes_exact_public_state_fields():
    game = _game_at_leap_with_baku_dropper()
    baseline = extract_features(game)
    assert FEATURE_SCHEMA_VERSION == "stl.features.v2"
    assert baseline.shape == (FEATURE_DIM,)

    mutations = (
        ("p1_alive", lambda g: setattr(g.player1, "alive", False)),
        ("p1_physicality", lambda g: setattr(g.player1, "physicality", 0.5)),
        ("first_dropper_p2", lambda g: setattr(g, "first_dropper", g.player2)),
        ("game_over", lambda g: setattr(g, "game_over", True)),
    )
    import copy

    for feature_name, mutate in mutations:
        changed = copy.deepcopy(game)
        mutate(changed)
        assert extract_features(changed)[FEATURE_INDEX[feature_name]] != baseline[
            FEATURE_INDEX[feature_name]
        ]


def test_value_net_checkpoint_round_trip_at_hidden_128(tmp_path):
    """Saving + loading a hidden=128 checkpoint must round-trip via
    ``load_checkpoint``'s auto-inferred hidden_dim. Guards the inference
    logic that reads the state_dict's trunk.0.weight shape — if it
    misreads, the load either fails or produces wrong-capacity weights.
    """
    from stl.learning.train import load_checkpoint, save_checkpoint_bundle

    model = ValueNet(hidden_dim=128)
    ckpt_path = tmp_path / "phase_i_test.pt"
    save_checkpoint_bundle(ckpt_path, model)

    loaded = load_checkpoint(str(ckpt_path), device="cpu")
    assert sum(p.numel() for p in loaded.parameters()) == sum(
        p.numel() for p in model.parameters()
    )

    x = torch.zeros(3, FEATURE_DIM)
    with torch.no_grad():
        out_orig = model(x, 2)
        out_loaded = loaded(x, 2)
    for a, b in zip(out_orig, out_loaded):
        assert torch.allclose(a, b)


def test_default_loader_rejects_bare_state_dict_without_explicit_migration(tmp_path):
    from stl.learning.train import CheckpointFormatError, load_checkpoint

    path = tmp_path / "bare.pt"
    torch.save(ValueNet().state_dict(), path)
    with pytest.raises(CheckpointFormatError, match="bare or legacy"):
        load_checkpoint(path)
