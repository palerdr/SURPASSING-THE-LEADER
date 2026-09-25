from __future__ import annotations

import itertools
import json

import numpy as np
import pytest

from arena.policies.bayesian_hal import BayesianHalConfig, BayesianHalOpponentModel, RunLengthFilter


def predict(model, role="checker"):
    return model.predict(role, state_regime=(0, 0, 0, 0), game_index=0, game_decision_index=model.total_observations)


def test_run_filter_matches_enumeration_of_change_histories():
    prior = np.full(60, 1 / 60)
    config = BayesianHalConfig(hazards=(0.03, 0.4))
    model = RunLengthFilter(prior, config)
    history = [3, 3, 40, 40]
    for end in range(1, len(history) + 1):
        model.observe(history[end - 1])
        evidence = []
        prediction = np.zeros(60)
        change_mass = 0.0
        for hazard in config.hazards:
            hazard_evidence = 0.0
            for changes in itertools.product((False, True), repeat=end):
                counts = np.zeros(60)
                mass = 1 / len(config.hazards)
                for action, change in zip(history[:end], changes):
                    mass *= hazard if change else 1 - hazard
                    if change:
                        counts[:] = 0
                    mass *= (counts[action] + config.concentration * prior[action]) / (counts.sum() + config.concentration)
                    counts[action] += 1
                hazard_evidence += mass
                prediction += mass * ((1 - hazard) * (counts + config.concentration * prior) / (counts.sum() + config.concentration) + hazard * prior)
                change_mass += mass * changes[-1]
            evidence.append(hazard_evidence)
        total = sum(evidence)
        assert model.predict() == pytest.approx(prediction / total, abs=1e-13)
        assert model.hazard_weights == pytest.approx(np.array(evidence) / total)
        assert model.last_change_probability == pytest.approx(change_mass / total)


def test_surprise_raises_change_posterior_and_retargets():
    model = RunLengthFilter(np.full(60, 1 / 60), BayesianHalConfig())
    for _ in range(20):
        model.observe(6)
    before = model.last_change_probability
    model.observe(48)
    assert model.last_change_probability > before * 10
    for _ in range(6):
        model.observe(48)
    assert model.predict()[48] > model.predict()[6]


def test_model_weights_follow_bayes_rule_with_switch_prior():
    model = BayesianHalOpponentModel()
    for action in [7, 7, 7, 22]:
        forecast = predict(model)
        posterior = forecast.expert_weights * forecast.expert_policies[:, action - 1]
        posterior /= posterior.sum()
        model.observe(forecast, opponent_action=action, self_action=12)
        next_forecast = predict(model)
        expected = (1 - model.bayes.switch_probability) * posterior + model.bayes.switch_probability / len(posterior)
        assert next_forecast.expert_weights == pytest.approx(expected)


def test_role_isolation_reveal_order_and_frozen_forecast():
    model = BayesianHalOpponentModel()
    initial = predict(model)
    before = initial.policy.copy()
    model.observe(initial, opponent_action=7, self_action=12)
    assert initial.policy == pytest.approx(before)
    assert predict(model, "dropper").policy == pytest.approx(np.full(60, 1 / 60))
    with pytest.raises(RuntimeError, match="stale"):
        model.observe(initial, opponent_action=7, self_action=12)
    forecast = predict(model)
    with pytest.raises(ValueError, match="integer"):
        model.observe(forecast, opponent_action=True, self_action=12)
    assert model.total_observations == 1
    model.reset()
    assert model.total_observations == 0


def test_bounded_run_filter_preserves_mass_after_pruning():
    model = BayesianHalOpponentModel(bayes=BayesianHalConfig(max_run_lengths=4))
    for index in range(100):
        forecast = predict(model)
        assert np.isfinite(forecast.policy).all()
        assert forecast.policy.sum() == pytest.approx(1)
        assert np.all(forecast.policy > 0)
        model.observe(forecast, opponent_action=1 + index % 13, self_action=8)
    run = model._roles["checker"].run
    assert run.counts.shape[0] == 4
    assert run.weights.sum(axis=1) == pytest.approx(np.ones(3))
    assert run.discarded_mass > 0
    json.dumps(model.diagnostics())


def test_population_prior_and_empty_roles():
    priors = {"checker": np.arange(1, 61), "dropper": np.arange(60, 0, -1)}
    model = BayesianHalOpponentModel(role_priors=priors)
    assert predict(model).policy == pytest.approx(priors["checker"] / sum(priors["checker"]))
    with pytest.raises(ValueError, match="positive support"):
        BayesianHalOpponentModel(role_priors={r: np.eye(60)[0] for r in priors})


def test_offset_predictor_learns_translation_without_absolute_action_repeats():
    model = BayesianHalOpponentModel()
    for action in range(5, 40):
        f = predict(model, "dropper")
        model.observe(f, opponent_action=action, self_action=action + 3)
    f = predict(model, "dropper")
    assert f.expert_policies[f.expert_names.index("delta"), 39] > 0.99
    assert f.policy[39] > 0.8
