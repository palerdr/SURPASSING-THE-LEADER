import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from environment.dth_env import DTHEnv
from environment.opponents.random_bot import RandomBot
from environment.opponents.safe_bot import SafeBot, LeapAwareSafeBot
from training.curriculum import get_scenario


def make_env(role: str = "hal", **kwargs) -> DTHEnv:
    return DTHEnv(opponent=RandomBot(), agent_role=role, seed=123, **kwargs)


def test_unaware_checker_cannot_use_61_on_leap():
    env = make_env("hal")
    env.reset(options={
        "scenario": {
            "game_clock": 3540.0,
            "current_half": 2,
            "first_dropper": "hal",
            "awareness": "unaware",
        }
    })

    assert env.action_masks()[60] == False


def test_deduced_checker_can_use_61_on_leap():
    env = make_env("hal")
    env.reset(options={
        "scenario": {
            "game_clock": 3540.0,
            "current_half": 2,
            "first_dropper": "hal",
            "awareness": "deduced",
        }
    })

    assert env.action_masks()[60] == True


def test_dropper_can_use_61_on_leap_even_if_unaware():
    env = make_env("hal")
    env.reset(options={
        "scenario": {
            "game_clock": 3540.0,
            "current_half": 1,
            "first_dropper": "hal",
            "awareness": "unaware",
        }
    })

    assert env.action_masks()[60] == True


def test_reset_applies_scenario_overrides():
    env = make_env("hal")
    obs, info = env.reset(options={
        "scenario": {
            "name": "test_state",
            "game_clock": 3420.0,
            "round_num": 8,
            "current_half": 1,
            "first_dropper": "baku",
            "hal": {"cylinder": 10.0, "ttd": 84.0, "deaths": 1, "alive": True},
            "baku": {"cylinder": 25.0, "ttd": 60.0, "deaths": 1, "alive": True},
            "referee_cprs": 2,
            "awareness": "deduced",
        }
    })

    assert env.game is not None
    assert env.game.first_dropper is not None
    assert env.game.game_clock == 3420.0
    assert env.game.round_num == 8
    assert env.game.current_half == 1
    assert env.game.first_dropper.name == "Baku"
    assert env.game.player1.cylinder == 10.0
    assert env.game.player2.cylinder == 25.0
    assert env.awareness.value == "deduced"
    assert info["scenario_name"] == "test_state"
    assert obs.shape == (20,)


def test_scenario_sampler_is_used_when_no_options_passed():
    def sampler(_rng):
        return {
            "name": "sampled",
            "game_clock": 3540.0,
            "current_half": 2,
            "first_dropper": "hal",
            "awareness": "deduced",
        }

    env = make_env("hal", scenario_sampler=sampler)
    _obs, info = env.reset()

    assert env.game is not None
    assert env.game.game_clock == 3540.0
    assert info["scenario_name"] == "sampled"


def test_reset_makes_opening_first_dropper_seed_deterministic():
    env_a = DTHEnv(opponent=RandomBot(), agent_role="hal", seed=1990)
    env_b = DTHEnv(opponent=RandomBot(), agent_role="hal", seed=1990)

    env_a.reset()
    env_b.reset()

    assert env_a.game is not None
    assert env_b.game is not None
    assert env_a.game.first_dropper is not None
    assert env_b.game.first_dropper is not None
    assert env_a.game.first_dropper.name == "Hal"
    assert env_b.game.first_dropper.name == "Hal"


def test_max_steps_truncates_nonterminal_episode():
    env = DTHEnv(opponent=SafeBot(), agent_role="hal", seed=123, max_steps=1)
    env.reset()
    _obs, reward, terminated, truncated, info = env.step(59)

    assert terminated == False
    assert truncated == True
    assert reward == 0.0
    assert info["truncated_reason"] == "max_steps"


def test_max_steps_none_does_not_truncate_by_default():
    env = DTHEnv(opponent=SafeBot(), agent_role="hal", seed=123, max_steps=None)
    env.reset()
    _obs, _reward, terminated, truncated, info = env.step(59)

    assert terminated == False
    assert truncated == False
    assert "truncated_reason" not in info


def test_hal_awareness_updates_after_death_evidence():
    env = DTHEnv(opponent=LeapAwareSafeBot(), agent_role="hal", seed=123)
    env.reset(options={
        "scenario": {
            "game_clock": 3540.0,
            "current_half": 2,
            "first_dropper": "hal",
            "awareness": "unaware",
        }
    })

    assert env.awareness.value == "unaware"
    _obs, _reward, _terminated, _truncated, info = env.step(0)
    assert info["awareness"] == "deduced"


@pytest.mark.parametrize(
    "scenario_name",
    [
        "round7_pressure",
        "round8_bridge",
        "round9_pre_leap",
        "round9_leap_deduced",
        "round9_leap_impaired",
    ],
)
def test_named_curriculum_scenarios_pass_runtime_validation(scenario_name: str):
    env = make_env("hal")
    _obs, info = env.reset(options={"scenario": get_scenario(scenario_name)})

    assert info["scenario_name"] == scenario_name


def test_invalid_named_scenario_semantics_raise_clear_error():
    env = make_env("hal")

    with pytest.raises(ValueError, match="round9_leap_deduced requires awareness=deduced"):
        env.reset(options={
            "scenario": {
                **get_scenario("round9_leap_deduced"),
                "awareness": "unaware",
            }
        })
