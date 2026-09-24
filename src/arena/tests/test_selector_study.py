import pytest
import torch

from arena.policies.neural_pilots import ExpertSelector, EXPERTS, GATE_DIM, TOKEN_DIM
from arena.policies.selector_study import Selector, SelectorEnsemble


def inputs():
    torch.manual_seed(77)
    features = torch.randn(3, GATE_DIM) * .1
    experts = torch.softmax(torch.randn(3, EXPERTS, 60), dim=-1)
    return features, experts


@pytest.mark.parametrize("architecture", ["small", "wide", "attention", "static"])
def test_initial_forecast_preserves_baseline_and_training_survives_reload(architecture, tmp_path):
    features, experts = inputs()
    saved = features.clone()
    network = Selector(architecture)
    torch.testing.assert_close(network(features, experts), ExpertSelector()(features, experts))
    optimizer = torch.optim.AdamW(network.parameters(), lr=.01)
    for _ in range(2):
        optimizer.zero_grad()
        prediction = network(features, experts)
        (-prediction[:, 0].log().mean()).backward()
        assert all(p.grad is None or torch.isfinite(p.grad).all() for p in network.parameters())
        optimizer.step()
    assert not torch.allclose(network(features, experts), ExpertSelector()(features, experts))
    torch.testing.assert_close(features, saved, rtol=0, atol=0)
    torch.testing.assert_close(network(features, experts).sum(-1), torch.ones(3))
    path = tmp_path / "selector.pt"
    torch.save(network.state_dict(), path)
    restored = Selector(architecture)
    restored.load_state_dict(torch.load(path, weights_only=True), strict=True)
    torch.testing.assert_close(network(features, experts), restored(features, experts), rtol=0, atol=0)


@pytest.mark.parametrize("architecture", ["small", "wide", "attention"])
@pytest.mark.parametrize("ablation,group", [
    ("no_context", slice(0, TOKEN_DIM)),
    ("no_prior", slice(TOKEN_DIM, TOKEN_DIM + EXPERTS)),
    ("no_errors", slice(TOKEN_DIM + EXPERTS, TOKEN_DIM + 2 * EXPERTS)),
])
def test_retrained_ablation_cannot_read_masked_features(architecture, ablation, group):
    features, experts = inputs()
    network = Selector(architecture, ablation).eval()
    torch.nn.init.normal_(network.head.weight, std=.2)
    changed = features.clone()
    changed[:, group] += torch.randn_like(changed[:, group]) * 20
    torch.testing.assert_close(network(features, experts), network(changed, experts), rtol=0, atol=0)


def test_no_prior_removes_base_weight_path():
    features, experts = inputs()
    network = Selector("wide", "no_prior")
    torch.testing.assert_close(network(features, experts), experts.mean(1))


def test_no_shapes_masks_attention_input_but_keeps_actual_expert_forecasts():
    features, experts = inputs()
    network = Selector("attention", "no_shapes").eval()
    torch.nn.init.normal_(network.head.weight, std=.2)
    changed = experts.roll(1, dims=-1)
    torch.testing.assert_close(network.weights(features, experts), network.weights(features, changed), rtol=0, atol=0)
    torch.testing.assert_close(network(features, changed), network(features, experts).roll(1, dims=-1))
    assert not torch.allclose(network(features, changed), network(features, experts))


def test_static_control_uses_prior_but_cannot_read_context_or_error_inputs():
    features, experts = inputs()
    network = Selector("static")
    with torch.no_grad():
        network.bias.copy_(torch.linspace(-.3, .3, EXPERTS))
    changed = torch.randn_like(features)
    changed[:, TOKEN_DIM:TOKEN_DIM + EXPERTS] = features[:, TOKEN_DIM:TOKEN_DIM + EXPERTS]
    torch.testing.assert_close(network(features, experts), network(changed, experts), rtol=0, atol=0)
    changed[:, TOKEN_DIM:TOKEN_DIM + EXPERTS] += torch.randn(3, EXPERTS)
    assert not torch.allclose(network(features, experts), network(changed, experts))


def test_ensemble_averages_forecasts_instead_of_selecting_seed():
    features, experts = inputs()
    members = [Selector("small"), Selector("wide")]
    for member in members:
        torch.nn.init.normal_(member.head.weight, std=.2)
    ensemble = SelectorEnsemble(members)
    torch.testing.assert_close(ensemble(features, experts), (members[0](features, experts) + members[1](features, experts)) / 2)
