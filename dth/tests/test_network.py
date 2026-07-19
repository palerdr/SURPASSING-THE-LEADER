import pytest
import torch

from dth.network import DTHNetworkConfig, DTHPolicyValueNet, encode_features


def test_feature_encoding_and_network_shapes():
    states = torch.tensor([[0, 0, 299, 300], [150, 240, 1, 60]])
    horizons = torch.tensor([1, 3])
    features = encode_features(states, horizons, horizon_scale=3.0)

    assert features.shape == (2, 5)
    assert features[0].tolist() == pytest.approx([0, 0, 299 / 300, 1, 1 / 3])

    model = DTHPolicyValueNet(DTHNetworkConfig(hidden_width=8))
    value, drop_logits, check_logits = model(features)
    assert value.shape == (2,)
    assert drop_logits.shape == (2, 60)
    assert check_logits.shape == (2, 60)
    assert torch.all(value >= -1.0)
    assert torch.all(value <= 1.0)
