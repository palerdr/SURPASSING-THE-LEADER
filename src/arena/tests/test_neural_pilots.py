from types import SimpleNamespace

import numpy as np
import pytest
import torch

from arena.policies.neural_pilots import ExpertSelector, PublicMemory, SequencePredictor, SessionActor, TOKEN_DIM, WINDOW
from arena.policies.translated_hal import TranslatedHalConfig
from arena.testing import make_decision as _decision


def decision(role="dropper", game=0, index=0):
    return SimpleNamespace(canonical_decision=_decision(role=role), game_index=game,
        half_round_index=index, opponent_role="checker" if role == "dropper" else "dropper")


def test_inputs_precede_reveal_and_independent_memories_do_not_share_state():
    first, second = PublicMemory(TranslatedHalConfig()), PublicMemory(TranslatedHalConfig())
    f1, w1, g1 = first.prepare(decision())
    f2, w2, g2 = second.prepare(decision())
    np.testing.assert_array_equal(w1, w2)
    np.testing.assert_array_equal(g1, g2)
    first.observe(17, 20)
    second.observe(40, 20)
    assert w1[-1, 11] == 0
    _, after_first, _ = first.prepare(decision(index=1))
    _, after_second, _ = second.prepare(decision(index=1))
    assert after_first[-1, 7] == pytest.approx(17 / 60)
    assert after_second[-1, 7] == pytest.approx(40 / 60)


def test_pending_and_action_61_guards_preserve_memory():
    memory = PublicMemory(TranslatedHalConfig())
    memory.prepare(decision())
    with pytest.raises(RuntimeError, match="pending"):
        memory.prepare(decision())
    with pytest.raises(ValueError, match="1..60"):
        memory.observe(61, 20)
    assert memory.base.total_observations == 0
    assert memory.last is None
    memory.observe(60, 20)
    with pytest.raises(RuntimeError, match="no prediction"):
        memory.observe(60, 20)
    leap = decision()
    leap.canonical_decision = _decision(legal_seconds=tuple(range(1, 62)))
    with pytest.raises(ValueError, match="pure DTH"):
        memory.prepare(leap)


def test_game_boundary_retains_opponent_history_and_native_state_is_unused():
    memory = PublicMemory(TranslatedHalConfig())
    memory.prepare(decision())
    memory.observe(17, 20)
    forecast, window, _ = memory.prepare(decision(game=1))
    assert forecast.context.observation_index == 1
    assert window[-1, 5] == 1
    assert window[-1, 11] == 1
    assert window[-1, 13] == pytest.approx(20 / 60)


@pytest.mark.parametrize("kind", ["transformer", "gru"])
def test_predictors_train_with_finite_gradients_and_reload(kind, tmp_path):
    torch.manual_seed(51)
    network = SequencePredictor(kind)
    x = torch.randn(3, WINDOW, TOKEN_DIM)
    logits = network(x)
    loss = torch.nn.functional.cross_entropy(logits, torch.tensor([0, 20, 59]))
    loss.backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in network.parameters())
    assert network.head.weight.grad.abs().sum() > 0
    path = tmp_path / "weights.pt"
    torch.save(network.state_dict(), path)
    restored = SequencePredictor(kind)
    restored.load_state_dict(torch.load(path, weights_only=True))
    torch.testing.assert_close(logits, restored(x), rtol=0, atol=0)


def test_selector_starts_at_frozen_weights_and_can_learn():
    memory = PublicMemory(TranslatedHalConfig())
    forecast, _, gate = memory.prepare(decision())
    network = ExpertSelector()
    experts = torch.tensor(forecast.expert_policies[None], dtype=torch.float32)
    prediction = network(torch.tensor(gate[None]), experts)
    np.testing.assert_allclose(prediction.detach()[0], forecast.policy, atol=1e-7)
    # Nonidentical experts produce a gradient for the selector head.
    experts = torch.softmax(torch.randn(2, 24, 60), dim=-1)
    prediction = network(torch.randn(2, len(gate)), experts)
    (-prediction[:, 0].log().mean()).backward()
    assert network.net[-1].weight.grad.abs().sum() > 0
    torch.testing.assert_close(prediction.sum(-1), torch.ones(2))


def test_score_weighting_updates_after_reveal_and_reset_ablation_preserves_current_token():
    memory = PublicMemory(TranslatedHalConfig())
    network = SequencePredictor("gru").eval()
    forecast, window, gate = memory.prepare(decision())
    initial = memory.neural_log_odds.copy()
    prediction, weight = memory.predict(network, "gru", forecast, window, gate, ablation="reset")
    assert memory.neural_log_odds == initial
    assert prediction.sum() == pytest.approx(1)
    assert .002 < weight < .998
    original = window.copy()
    memory.observe(20, 30)
    assert memory.neural_log_odds["checker"] != initial["checker"]
    np.testing.assert_array_equal(window, original)
    assert memory.neural_log_odds["dropper"] == initial["dropper"]


def test_actor_initial_policy_uses_tactical_values_and_hidden_state_can_change_logits():
    actor = SessionActor()
    token, forecast, values = torch.zeros(1, TOKEN_DIM), torch.full((1, 60), 1 / 60), torch.linspace(-1, 1, 60)[None]
    logits, _, hidden = actor(token, forecast, values)
    torch.testing.assert_close(logits, 12 * values)
    assert hidden.shape == (1, 64)
    torch.nn.init.normal_(actor.actor.weight, std=.1)
    full, _, _ = actor(token, forecast, values, hidden)
    reset, _, _ = actor(token, forecast, values)
    assert not torch.allclose(full, reset)


def test_shuffle_randomness_cannot_change_actions_if_predictions_match(monkeypatch, tmp_path):
    from arena.policies.run_neural_pilots import Experiment
    from arena.testing import StageAgent as _StageAgent

    experiment = Experiment.__new__(Experiment)
    experiment.agent = _StageAgent()
    experiment.config = TranslatedHalConfig()
    experiment.args = SimpleNamespace(artifact=tmp_path)
    consumed = []

    def fixed_prediction(self, network, kind, forecast, window, gate, *, ablation=None, rng=None):
        if ablation == "shuffle":
            consumed.append(rng.permutation(15))
        return forecast.policy, 0.

    monkeypatch.setattr(PublicMemory, "predict", fixed_prediction)
    reference = experiment.session("copy_recent", 598, "transformer")
    shuffled = experiment.session("copy_recent", 598, "transformer", ablation="shuffle")
    assert consumed
    assert reference["games"] == shuffled["games"]
    assert reference["nll"] == shuffled["nll"]
