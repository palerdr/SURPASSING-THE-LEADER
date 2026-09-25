import numpy as np
import pytest

from arena.policies.perfect_hal import PerfectHalOpponentModel
from arena.policies.translated_hal import TranslatedHalConfig, TranslatedHalOpponentModel
from arena.testing import make_forecast as _forecast


@pytest.mark.parametrize("role", ["dropper", "checker"])
def test_offsets_transfer_to_an_unseen_hal_action(role):
    model = TranslatedHalOpponentModel()
    last = 10
    for own in (13, 22, 8, 37, 15, 28, 42, 19, 31, 47, 25, 51):
        forecast = _forecast(model, role)
        model.observe(forecast, opponent_action=last + 2, self_action=own)
        last = own
    forecast = _forecast(model, role)
    index = forecast.expert_names.index("offset_last_copy_global")
    assert forecast.expert_policies[index, 52] > 0.95
    assert forecast.policy[52] > 0.8
    assert model.observations("checker" if role == "dropper" else "dropper") == 0


def test_offsets_use_the_last_reveal_across_roles_and_track_a_switch():
    model = TranslatedHalOpponentModel(TranslatedHalConfig(offset_retention=0.7))
    rng = np.random.default_rng(917)
    last = 20
    for index in range(90):
        role = "dropper" if index % 2 else "checker"
        forecast = _forecast(model, role)
        action = last + (3 if index < 40 else -3)
        own = int(rng.integers(5, 55))
        model.observe(forecast, opponent_action=action, self_action=own)
        last = own
    forecast = _forecast(model, "checker")
    recent = forecast.expert_names.index("offset_last_copy_recent")
    assert forecast.expert_policies[recent, last - 4] > 0.95
    assert forecast.policy[last - 4] > 0.8


def test_no_offset_ablation_matches_old_bit_for_bit():
    old = PerfectHalOpponentModel()
    candidate = TranslatedHalOpponentModel(TranslatedHalConfig(translated=False))
    rng = np.random.default_rng(81)
    for index in range(30):
        role = "dropper" if index % 2 else "checker"
        before = [_forecast(model, role) for model in (old, candidate)]
        np.testing.assert_array_equal(before[0].policy, before[1].policy)
        own, action = rng.integers(1, 61, 2)
        for model, forecast in zip((old, candidate), before):
            model.observe(forecast, opponent_action=int(action), self_action=int(own))


def test_reset_and_excluded_turn_clear_translation_references():
    model = TranslatedHalOpponentModel()
    forecast = _forecast(model, "dropper")
    model.observe(forecast, opponent_action=12, self_action=10)
    with pytest.raises(RuntimeError, match="stale"):
        model.observe(forecast, opponent_action=13, self_action=11)
    model.break_sequence()
    forecast = _forecast(model, "dropper")
    assert forecast.context.last_self_action is None
    assert forecast.context.previous_self_action is None
    assert model.total_observations == 1
    model.reset()
    assert model.total_observations == 0
    np.testing.assert_allclose(_forecast(model, "dropper").policy, np.ones(60) / 60)


def test_offset_projection_clips_and_preserves_probability_mass():
    model = TranslatedHalOpponentModel()
    for own in (10, 20, 30, 40, 60):
        before = _forecast(model, "dropper")
        last = before.context.last_self_action
        model.observe(before, opponent_action=20 if last is None else last + 4, self_action=own)
    forecast = _forecast(model, "dropper")
    index = forecast.expert_names.index("offset_last_copy_global")
    assert forecast.expert_policies[index, 59] > 0.95
    np.testing.assert_allclose(forecast.expert_policies.sum(axis=1), 1)
