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


def test_identity_lift_preserves_existing_behavior_and_width():
    torch.manual_seed(11)
    features = torch.randn(5, 5)
    default_model = DTHPolicyValueNet(DTHNetworkConfig(hidden_width=8)).eval()
    identity_model = DTHPolicyValueNet(
        DTHNetworkConfig(hidden_width=8, feature_lift="identity")
    ).eval()
    identity_model.load_state_dict(default_model.state_dict())

    assert default_model.apply_feature_lift(features).shape == (5, 5)
    assert default_model.trunk[0].in_features == 5
    for expected, actual in zip(
        default_model(features), identity_model(features), strict=True
    ):
        torch.testing.assert_close(expected, actual)


def test_boundary_v1_features_at_and_around_both_240_boundaries():
    states = torch.tensor(
        [
            [239, 0, 239, 0],
            [240, 0, 240, 0],
            [241, 0, 241, 0],
            [200, 40, 200, 40],
            [200, 41, 200, 41],
        ],
        dtype=torch.float32,
    )
    features = encode_features(states, torch.ones(5), horizon_scale=3.0)
    model = DTHPolicyValueNet(
        DTHNetworkConfig(hidden_width=8, feature_lift="boundary_v1")
    )
    lifted = model.apply_feature_lift(features)

    assert lifted.shape == (5, 9)
    torch.testing.assert_close(lifted[0, 5:], torch.zeros(4))
    torch.testing.assert_close(lifted[1, 5:], torch.zeros(4))
    torch.testing.assert_close(
        lifted[2, 5:], torch.tensor([1 / 60, 1 / 300, 1 / 60, 1 / 300])
    )
    torch.testing.assert_close(lifted[3, 5:], torch.zeros(4))
    torch.testing.assert_close(lifted[4, 5:], torch.tensor([0.0, 1 / 300, 0.0, 1 / 300]))
    assert model.trunk[0].in_features == 9


def test_boundary_v2_features_encode_inclusive_dose_and_strict_ttd_boundaries():
    states = torch.tensor(
        [
            [239, 0, 239, 0],
            [240, 0, 240, 0],
            [241, 0, 241, 0],
            [200, 40, 200, 40],
            [200, 41, 200, 41],
        ],
        dtype=torch.float32,
    )
    features = encode_features(states, torch.ones(5), horizon_scale=3.0)
    model = DTHPolicyValueNet(
        DTHNetworkConfig(hidden_width=8, feature_lift="boundary_v2")
    )
    lifted = model.apply_feature_lift(features)

    assert lifted.shape == (5, 9)
    torch.testing.assert_close(lifted[0, 5:], torch.zeros(4))
    torch.testing.assert_close(lifted[1, 5:], torch.tensor([1.0, 0.0, 1.0, 0.0]))
    torch.testing.assert_close(lifted[2, 5:], torch.ones(4))
    torch.testing.assert_close(lifted[3, 5:], torch.zeros(4))
    torch.testing.assert_close(lifted[4, 5:], torch.tensor([0.0, 1.0, 0.0, 1.0]))
    assert model.trunk[0].in_features == 9


def test_boundary_columns_receive_finite_nonzero_gradients():
    model = DTHPolicyValueNet(
        DTHNetworkConfig(hidden_width=4, hidden_layers=1, feature_lift="boundary_v1")
    )
    with torch.no_grad():
        model.trunk[0].weight.zero_()
        model.trunk[0].bias.fill_(1.0)
        model.drop_head.weight.fill_(0.1)
        model.drop_head.bias.zero_()
    states = torch.tensor([[241, 0, 241, 0], [200, 41, 200, 41]], dtype=torch.float32)
    features = encode_features(states, torch.ones(2), horizon_scale=3.0)

    model.zero_grad(set_to_none=True)
    _, drop_logits, _ = model(features)
    drop_logits.sum().backward()
    gradient = model.trunk[0].weight.grad[:, 5:]

    assert gradient is not None
    assert torch.isfinite(gradient).all()
    assert torch.any(gradient.abs() > 0)


def test_old_checkpoint_without_feature_lift_field_loads_as_identity():
    config = DTHNetworkConfig(hidden_width=4, hidden_layers=1)
    model = DTHPolicyValueNet(config)
    legacy_config = config.to_dict()
    legacy_config.pop("feature_lift")

    loaded_config = DTHNetworkConfig(**legacy_config)
    loaded = DTHPolicyValueNet(loaded_config)
    loaded.load_state_dict(model.state_dict(), strict=True)

    assert loaded.config.feature_lift == "identity"
    assert loaded.trunk[0].in_features == 5


def test_unknown_feature_lift_fails_closed():
    with pytest.raises(ValueError, match="unknown feature lift"):
        DTHPolicyValueNet(DTHNetworkConfig(feature_lift="unknown"))


def test_continuation_residual_starts_as_an_exact_zero_matrix():
    model = DTHPolicyValueNet(
        DTHNetworkConfig(
            hidden_width=4,
            hidden_layers=1,
            continuation_residual=True,
        )
    )
    features = torch.rand((3, 5), dtype=torch.float32)

    residual = model.continuation_residual_matrix(features)

    assert residual.shape == (3, 60, 60)
    torch.testing.assert_close(residual, torch.zeros_like(residual))


def test_action_mlp_residual_receives_the_literal_action_pair():
    model = DTHPolicyValueNet(
        DTHNetworkConfig(
            hidden_width=4,
            hidden_layers=1,
            continuation_residual=True,
            continuation_residual_mode="action_mlp",
        )
    )
    with torch.no_grad():
        model.continuation_action_hidden.weight.zero_()
        model.continuation_action_hidden.bias.zero_()
        model.continuation_action_hidden.weight[0, -2] = 1.0
        model.continuation_action_out.weight.zero_()
        model.continuation_action_out.bias.zero_()
        model.continuation_action_out.weight[0, 0] = 1.0

    residual = model.continuation_residual_matrix(torch.zeros((1, 5)))

    assert residual.shape == (1, 60, 60)
    assert residual[0, 0, 0].item() == pytest.approx(1 / 60)
    assert residual[0, 59, 0].item() == pytest.approx(1.0)
    assert residual[0, 0, 0].item() == residual[0, 0, 59].item()


def test_unknown_continuation_residual_mode_fails_closed():
    with pytest.raises(ValueError, match="unknown continuation residual mode"):
        DTHPolicyValueNet(
            DTHNetworkConfig(continuation_residual_mode="unknown")
        )
