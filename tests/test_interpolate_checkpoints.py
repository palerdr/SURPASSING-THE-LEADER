import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hal.value_net import ValueNet
from scripts.interpolate_checkpoints import interpolate_checkpoints


def _save_constant_checkpoint(path, value: float) -> None:
    model = ValueNet()
    state = model.state_dict()
    for key, tensor in state.items():
        if torch.is_floating_point(tensor):
            state[key] = torch.full_like(tensor, value)
    torch.save(state, path)


def test_interpolate_checkpoints_writes_weighted_average(tmp_path):
    base = tmp_path / "base.pt"
    target = tmp_path / "target.pt"
    out = tmp_path / "mixed.pt"
    _save_constant_checkpoint(base, 2.0)
    _save_constant_checkpoint(target, 6.0)

    interpolate_checkpoints(base, target, 0.25, out)

    state = torch.load(out, map_location="cpu")
    assert torch.allclose(state["value_head.0.bias"], torch.full_like(state["value_head.0.bias"], 3.0))


def test_interpolate_checkpoints_rejects_invalid_alpha(tmp_path):
    base = tmp_path / "base.pt"
    target = tmp_path / "target.pt"
    _save_constant_checkpoint(base, 2.0)
    _save_constant_checkpoint(target, 6.0)

    with pytest.raises(ValueError, match="alpha"):
        interpolate_checkpoints(base, target, 1.5, tmp_path / "mixed.pt")
