from __future__ import annotations

import numpy as np
import pytest

from arena.policies.perfect_hal.model import (
    ACTION_COUNT,
    PerfectHalConfig,
    PerfectHalOpponentModel,
)
from arena.testing import make_forecast as _forecast


def test_config_fixes_pure_dth_and_hard_response_defaults() -> None:
    config = PerfectHalConfig()
    assert config.action_count == 60
    assert config.response_temperature == 0.0
    assert config.expert_names[:5] == (
        "uniform",
        "global",
        "slow_recency",
        "fast_recency",
        "flash_recency",
    )

    with pytest.raises(ValueError, match="60"):
        PerfectHalConfig(action_count=61)
    with pytest.raises(ValueError, match="slow to flash"):
        PerfectHalConfig(recency_retentions=(0.3, 0.8, 0.9))
    with pytest.raises(ValueError, match="nonnegative"):
        PerfectHalConfig(response_temperature=-0.1)


def test_online_model_learns_a_deterministic_role_without_cross_role_leakage() -> None:
    model = PerfectHalOpponentModel()
    initial_checker = _forecast(model, "checker")
    initial_dropper = _forecast(model, "dropper")
    assert initial_checker.policy == pytest.approx(
        np.full(ACTION_COUNT, 1.0 / ACTION_COUNT)
    )
    assert initial_dropper.policy == pytest.approx(initial_checker.policy)

    for index in range(8):
        prediction = _forecast(model, "checker", decision_index=index)
        model.observe(prediction, opponent_action=7, self_action=3)

    learned = _forecast(model, "checker", decision_index=8)
    untouched = _forecast(model, "dropper", decision_index=8)
    assert learned.policy[6] > 0.90
    assert learned.confidence > 0.70
    assert untouched.policy == pytest.approx(initial_dropper.policy)
    assert model.observations("checker") == 8
    assert model.observations("dropper") == 0


def test_online_model_retargets_after_a_public_change_point() -> None:
    model = PerfectHalOpponentModel()
    decision_index = 0
    for action, repeats in ((7, 20), (49, 8)):
        for _ in range(repeats):
            prediction = _forecast(
                model,
                "checker",
                decision_index=decision_index,
            )
            model.observe(prediction, opponent_action=action, self_action=11)
            decision_index += 1

    switched = _forecast(model, "checker", decision_index=decision_index)
    weights = dict(zip(switched.expert_names, switched.expert_weights, strict=True))
    assert switched.policy[48] > switched.policy[6]
    assert weights["flash_recency"] > weights["global"]


def test_self_response_expert_uses_only_the_previous_revealed_self_action() -> None:
    model = PerfectHalOpponentModel()
    previous_self = 10
    for index in range(20):
        prediction = _forecast(model, "dropper", decision_index=index)
        current_self = 20 if previous_self == 10 else 10
        model.observe(
            prediction,
            opponent_action=previous_self,
            self_action=current_self,
        )
        previous_self = current_self

    prediction = _forecast(model, "dropper", decision_index=20)
    response_index = prediction.expert_names.index("self_response")
    expected_action = previous_self
    assert prediction.expert_policies[response_index, expected_action - 1] > 0.95
    assert prediction.policy[expected_action - 1] > 0.70


def test_forecast_tokens_are_single_use_and_causally_ordered() -> None:
    model = PerfectHalOpponentModel()
    prediction = _forecast(model, "checker")
    model.observe(prediction, opponent_action=5, self_action=9)
    with pytest.raises(RuntimeError, match="stale or already observed"):
        model.observe(prediction, opponent_action=5, self_action=9)


def test_noninteger_actions_fail_without_consuming_the_forecast() -> None:
    model = PerfectHalOpponentModel()
    prediction = _forecast(model, "checker")
    with pytest.raises(ValueError, match="integer second"):
        model.observe(prediction, opponent_action=True, self_action=9)
    with pytest.raises(ValueError, match="integer second"):
        model.observe(prediction, opponent_action=5, self_action=9.0)

    model.observe(prediction, opponent_action=5, self_action=9)
    assert model.observations("checker") == 1
