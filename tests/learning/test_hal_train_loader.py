"""Loud-fail checkpoint loading (plan ticket 3 / bug B3).

The legacy ``hal.train.load_checkpoint`` used to construct a hidden=64 net,
catch the RuntimeError from a modern hidden=192 ``trunk.*`` checkpoint,
migrate zero keys, and ``load_state_dict({}, strict=False)`` — silently
returning a RANDOM-weight net to the play path. These tests pin the fixed
behavior: shape inference for modern checkpoints, clean migration for
legacy ones, and a hard error for anything else.
"""

import os
import sys

import pytest
import torch

sys.path.insert(0, os.getcwd())

from stl.learning.legacy_train import load_checkpoint, save_checkpoint
from stl.learning.model import FEATURE_DIM, ValueNet

REAL_CHECKPOINT = os.path.join(
    os.getcwd(),
    "checkpoints",
    "gen_ceiling_tbw15_wd1e-4",
    "best.pt",
)


def test_modern_trunk_checkpoint_loads_with_inferred_hidden_dim(tmp_path):
    source = ValueNet(FEATURE_DIM, hidden_dim=192)
    path = tmp_path / "modern.pt"
    save_checkpoint(source, str(path))

    loaded = load_checkpoint(str(path))

    assert loaded.trunk[0].out_features == 192
    for p_src, p_loaded in zip(source.parameters(), loaded.parameters()):
        assert torch.equal(p_src, p_loaded)


@pytest.mark.skipif(
    not os.path.exists(REAL_CHECKPOINT), reason="headline checkpoint not pulled"
)
def test_real_headline_checkpoint_is_stale_under_action_core_reset():
    with pytest.raises(RuntimeError, match="policy_head"):
        load_checkpoint(REAL_CHECKPOINT)


def test_legacy_layers_checkpoint_migrates_cleanly(tmp_path):
    hidden = 64
    legacy = {
        "layers.0.weight": torch.randn(hidden, FEATURE_DIM),
        "layers.0.bias": torch.randn(hidden),
        "layers.2.weight": torch.randn(hidden, hidden),
        "layers.2.bias": torch.randn(hidden),
        "layers.4.weight": torch.randn(1, hidden),
        "layers.4.bias": torch.randn(1),
    }
    path = tmp_path / "legacy.pt"
    torch.save(legacy, str(path))

    loaded = load_checkpoint(str(path))

    assert loaded.trunk[0].out_features == hidden
    assert torch.equal(loaded.trunk[0].weight, legacy["layers.0.weight"])
    assert torch.equal(loaded.value_head[0].weight, legacy["layers.4.weight"])


def test_unrecognized_checkpoint_raises_instead_of_random_net(tmp_path):
    path = tmp_path / "garbage.pt"
    torch.save({"encoder.weight": torch.randn(4, 4)}, str(path))

    with pytest.raises(RuntimeError, match="Unrecognized checkpoint format"):
        load_checkpoint(str(path))


def test_truncated_legacy_checkpoint_raises(tmp_path):
    path = tmp_path / "truncated.pt"
    torch.save({"layers.2.weight": torch.randn(64, 64)}, str(path))

    with pytest.raises(RuntimeError):
        load_checkpoint(str(path))
