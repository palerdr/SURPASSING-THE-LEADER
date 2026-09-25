"""Tests of the tracked PM Hal v3 study configs."""

from __future__ import annotations

import pytest

from arena.policies.pm_hal import PMHalConfig, load_pm_hal_config
from hal_lab.training.train_aggro_hal import load_training_config


def test_tracked_pm_aggro_component_config_targets_the_current_artifact() -> None:
    model, trainer = load_training_config(
        "src/hal_lab/experiments/pm_hal_v3/pm_hal_aggro_component_v1.yaml"
    )
    assert model.hidden_size == 128
    assert trainer.dth_artifact == "src/dth/artifacts/complete_full_v1"
    assert trainer.warmstart_updates == 16
    assert trainer.ppo_updates == 4


def test_tracked_overbearing_controller_matches_code_defaults() -> None:
    tracked = load_pm_hal_config()
    assert tracked == PMHalConfig()
    assert tracked.minimum_role_observations == 1
    assert tracked.press_confidence_threshold == pytest.approx(0.15)
    assert tracked.game_epsilon_budget == pytest.approx(12.0)
    assert tracked.dominate_epsilon_cap == pytest.approx(2.0)
    assert load_pm_hal_config("src/hal_lab/experiments/pm_hal_v3/pm_hal_controller_v2.json") == tracked
