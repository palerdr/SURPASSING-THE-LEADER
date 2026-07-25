from pathlib import Path

import pytest
import torch

from dth.network import DTHNetworkConfig, DTHPolicyValueNet
from dth.widen_checkpoint import widen_checkpoint, widen_state_dict


def test_widen_state_dict_preserves_all_outputs_and_unlocks_capacity():
    torch.manual_seed(7)
    source_config = DTHNetworkConfig(hidden_width=4, hidden_layers=2)
    source = DTHPolicyValueNet(source_config).eval()
    widened_state, widened_config = widen_state_dict(
        source.state_dict(), source_config, target_width=8
    )
    widened = DTHPolicyValueNet(widened_config).eval()
    widened.load_state_dict(widened_state)

    features = torch.randn(37, 5)
    with torch.no_grad():
        source_outputs = source(features)
        widened_outputs = widened(features)

    for source_output, widened_output in zip(
        source_outputs, widened_outputs, strict=True
    ):
        torch.testing.assert_close(
            source_output, widened_output, rtol=1e-6, atol=1e-7
        )

    assert torch.equal(widened.trunk[0].weight[:4], widened.trunk[0].weight[4:])
    optimizer = torch.optim.SGD(widened.parameters(), lr=0.01)
    optimizer.zero_grad(set_to_none=True)
    value, drop_logits, check_logits = widened(features)
    (
        value.square().mean()
        + drop_logits.square().mean()
        + check_logits.square().mean()
    ).backward()
    optimizer.step()
    assert not torch.equal(widened.trunk[0].weight[:4], widened.trunk[0].weight[4:])


def test_widen_checkpoint_records_provenance_and_loads(tmp_path: Path):
    source_config = DTHNetworkConfig(hidden_width=4, hidden_layers=2)
    source_model = DTHPolicyValueNet(source_config)
    source_path = tmp_path / "source.pt"
    destination = tmp_path / "widened.pt"
    torch.save(
        {
            "state_dict": source_model.state_dict(),
            "model_config": source_config.to_dict(),
            "epoch": 12,
        },
        source_path,
    )

    payload = widen_checkpoint(source_path, destination, target_width=8)
    loaded = torch.load(destination, map_location="cpu", weights_only=False)

    assert payload["model_config"]["hidden_width"] == 8
    assert loaded["epoch"] == 12
    assert loaded["widening"]["source_hidden_width"] == 4
    assert loaded["widening"]["target_hidden_width"] == 8
    assert loaded["widening"]["method"] == "asymmetric_neuron_duplication_v1"
    widened = DTHPolicyValueNet(DTHNetworkConfig(**loaded["model_config"]))
    widened.load_state_dict(loaded["state_dict"], strict=True)


def test_widen_rejects_non_increasing_width():
    config = DTHNetworkConfig(hidden_width=4, hidden_layers=2)
    model = DTHPolicyValueNet(config)

    with pytest.raises(ValueError, match="greater than source"):
        widen_state_dict(model.state_dict(), config, target_width=4)
