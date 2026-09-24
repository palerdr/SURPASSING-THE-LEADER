import numpy as np
import pytest

from arena.policies.evaluate_reward_prior import external_scores, participant_split, transitions
from arena.policies.reward_prior_hal import RewardPriorConfig, RewardPriorHal, transfer_prior
from arena.tests.test_perfect_hal import _forecast


def make_model():
    return RewardPriorHal(RewardPriorConfig(expert_weight_retention=.97, offset_retention=.97), [[1, 3], [3, 1]])


def test_transfer_preserves_chance_relative_odds_and_strength():
    binary = np.array([[1., 3.], [3., 1.]])
    target = transfer_prior(binary)
    np.testing.assert_allclose(target.sum(axis=1), binary.sum(axis=1))
    np.testing.assert_allclose(target[:, 0] / target[:, 1] * 59, binary[:, 0] / binary[:, 1])
    np.testing.assert_allclose(transfer_prior(np.ones((2, 2)))[:, 0] / 2, 1 / 60)
    np.testing.assert_allclose(transfer_prior(binary, 2), binary)


@pytest.mark.parametrize("prior", [[[0, 1], [1, 1]], [[np.nan, 1], [1, 1]], [1, 1]])
def test_invalid_prior_is_rejected(prior):
    with pytest.raises(ValueError, match="prior"):
        transfer_prior(prior)


@pytest.mark.parametrize("role,favorable", [("checker", 1), ("dropper", 0)])
def test_reward_uses_previous_reveal_and_inclusive_check(role, favorable):
    model = make_model()
    first = _forecast(model, role)
    np.testing.assert_allclose(first.policy, 1 / 60)
    model.observe(first, opponent_action=20, self_action=20)
    prediction = _forecast(model, role)
    expected = model.repeat_prior[favorable, 0] / model.repeat_prior[favorable].sum()
    assert prediction.expert_policies[-1, 19] == pytest.approx(expected)
    model.observe(prediction, opponent_action=20, self_action=10)
    assert model._repeat_counts[role][favorable, 0] == 1
    assert model._repeat_counts[role].sum() == 1
    np.testing.assert_allclose(prediction.expert_policies.sum(axis=1), 1)


def test_roles_instances_and_sequence_boundaries_are_isolated():
    model, untouched = make_model(), make_model()
    for _ in range(4):
        model.observe(_forecast(model, "checker"), opponent_action=30, self_action=15)
    np.testing.assert_allclose(_forecast(model, "dropper").policy, _forecast(untouched, "dropper").policy)
    assert untouched.total_observations == 0
    counts = model._repeat_counts["checker"].copy()
    model.break_sequence()
    forecast = _forecast(model, "checker")
    np.testing.assert_array_equal(forecast.expert_policies[-1], forecast.expert_policies[1])
    model.observe(forecast, opponent_action=10, self_action=15)
    np.testing.assert_array_equal(model._repeat_counts["checker"], counts)
    model.reset()
    assert model.total_observations == 0
    np.testing.assert_array_equal(_forecast(model, "checker").policy, _forecast(untouched, "checker").policy)


def test_invalid_and_duplicate_reveals_do_not_change_reward_memory():
    model = make_model()
    forecast = _forecast(model, "checker")
    with pytest.raises(ValueError, match="1..60"):
        model.observe(forecast, opponent_action=61, self_action=15)
    assert model._outcomes["checker"] is None
    assert model.total_observations == 0
    model.observe(forecast, opponent_action=30, self_action=15)
    with pytest.raises(RuntimeError, match="stale"):
        model.observe(forecast, opponent_action=10, self_action=15)
    assert model._outcomes["checker"] == 1
    assert model._repeat_counts["checker"].sum() == 0


def test_external_score_precedes_update_and_resets_between_games():
    games = {"PD": [(0, 0), (0, 1), (1, 1)], "BoS": [(1, 0), (1, 1)]}
    assert list(transitions(games)) == [(0, 1), (1, 0), (0, 1)]
    result = external_scores({"person": games}, [[1, 3], [3, 1]])[0]
    assert result["transitions"] == 3
    assert result["nll"] == pytest.approx(-np.log(.25))
    assert {participant_split(str(i)) for i in range(195)} == {"train", "validation", "test"}


def test_public_outcome_changes_forecast_for_the_same_opponent_action():
    first, second = make_model(), make_model()
    first.observe(_forecast(first, "checker"), opponent_action=20, self_action=10)
    second.observe(_forecast(second, "checker"), opponent_action=20, self_action=30)
    positive, zero = _forecast(first, "checker"), _forecast(second, "checker")
    assert positive.expert_policies[-1, 19] > zero.expert_policies[-1, 19]
    assert not positive.expert_policies.flags.writeable
