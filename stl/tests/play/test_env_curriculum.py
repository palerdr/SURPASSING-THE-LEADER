import os
import sys

import pytest

sys.path.insert(0, os.getcwd())

from stl.play.env import STLEnv
from stl.play.opponents.base import Opponent
from stl.play.opponents.random_bot import RandomBot
from stl.play.opponents.safe_bot import SafeBot
from stl.learning.curriculum import get_scenario


def make_env(role: str = "hal", **kwargs) -> STLEnv:
    return STLEnv(opponent=RandomBot(), agent_role=role, seed=123, **kwargs)


def test_hal_checker_cannot_use_61_on_leap():
    env = make_env("hal")
    env.reset(options={
        "scenario": {
            "game_clock": 3540.0,
            "current_half": 2,
            "first_dropper": "hal",
        }
    })

    assert env.action_masks()[61] == False


def test_hal_checker_can_never_use_61_on_leap():
    # Per the Stockfish-style design (HAL.md), Hal can never check at 61
    # Leap knowledge does not change structural action legality.
    env = make_env("hal")
    env.reset(options={
        "scenario": {
            "game_clock": 3540.0,
            "current_half": 2,
            "first_dropper": "hal",
        }
    })

    assert env.action_masks()[61] == False


def test_baku_checker_cannot_use_61_on_leap():
    env = make_env("baku")
    env.reset(options={
        "scenario": {
            "game_clock": 3540.0,
            "current_half": 1,
            "first_dropper": "hal",
        }
    })

    assert env.action_masks()[61] == False


def test_hal_dropper_cannot_use_61_on_leap():
    env = make_env("hal")
    env.reset(options={
        "scenario": {
            "game_clock": 3540.0,
            "current_half": 1,
            "first_dropper": "hal",
        }
    })

    assert env.action_masks()[61] == False


def test_baku_dropper_can_use_61_on_leap():
    env = make_env("baku")
    env.reset(options={
        "scenario": {
            "game_clock": 3540.0,
            "current_half": 1,
            "first_dropper": "baku",
        }
    })

    assert env.action_masks()[61] == True


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
    assert info["leap_aware"] is True
    assert info["scenario_name"] == "test_state"
    assert obs.shape == (19,)


def test_scenario_sampler_is_used_when_no_options_passed():
    def sampler(_rng):
        return {
            "name": "sampled",
            "game_clock": 3540.0,
            "current_half": 2,
            "first_dropper": "hal",
        }

    env = make_env("hal", scenario_sampler=sampler)
    _obs, info = env.reset()

    assert env.game is not None
    assert env.game.game_clock == 3540.0
    assert info["scenario_name"] == "sampled"


def test_reset_makes_opening_first_dropper_seed_deterministic():
    env_a = STLEnv(opponent=RandomBot(), agent_role="hal", seed=1990)
    env_b = STLEnv(opponent=RandomBot(), agent_role="hal", seed=1990)

    env_a.reset()
    env_b.reset()

    assert env_a.game is not None
    assert env_b.game is not None
    assert env_a.game.first_dropper is not None
    assert env_b.game.first_dropper is not None
    assert env_a.game.first_dropper.name == "Hal"
    assert env_b.game.first_dropper.name == "Hal"


def test_max_steps_truncates_nonterminal_episode():
    env = STLEnv(opponent=SafeBot(), agent_role="hal", seed=123, max_steps=1)
    env.reset()
    _obs, reward, terminated, truncated, info = env.step(59)

    assert terminated == False
    assert truncated == True
    assert reward == 0.0
    assert info["truncated_reason"] == "max_steps"


def test_max_steps_none_does_not_truncate_by_default():
    env = STLEnv(opponent=SafeBot(), agent_role="hal", seed=123, max_steps=None)
    env.reset()
    _obs, _reward, terminated, truncated, info = env.step(59)

    assert terminated == False
    assert truncated == False
    assert "truncated_reason" not in info


@pytest.mark.parametrize(
    "scenario_name",
    [
        "round7_pressure",
        "round8_bridge",
        "round9_pre_leap",
        "round9_leap",
    ],
)
def test_named_curriculum_scenarios_pass_runtime_validation(scenario_name: str):
    env = make_env("hal")
    _obs, info = env.reset(options={"scenario": get_scenario(scenario_name)})

    assert info["scenario_name"] == scenario_name


def test_invalid_named_scenario_semantics_raise_clear_error():
    env = make_env("hal")

    with pytest.raises(ValueError, match="round9_leap requires current_half=2"):
        env.reset(options={
            "scenario": {
                **get_scenario("round9_leap"),
                "current_half": 1,
            }
        })


# ── runtime legality enforcement at step() boundary ───────────────────────


class _FixedSecondOpponent(Opponent):
    """Opponent that always returns ``second`` regardless of state."""

    def __init__(self, second: int):
        self._second = second

    def reset(self):
        pass

    def choose_action(self, game, role, turn_duration):
        del game, role, turn_duration
        return self._second


def test_step_rejects_illegal_agent_action_outside_mask():
    """Agent passing an illegal action_index raises rather than slipping through.

    Hal-as-dropper outside leap window: action=61 (second=61) is illegal for
    Hal-dropper at any time. ``step(61)`` must raise ValueError.
    """
    env = STLEnv(opponent=_FixedSecondOpponent(30), agent_role="hal", seed=0)
    env.reset(seed=0)
    # First half-round of the canonical opening: Hal is dropper, no leap window.
    assert env.action_masks()[61] == False
    with pytest.raises(ValueError, match="action=61.*illegal"):
        env.step(61)


def test_step_rejects_illegal_hal_dropper_61_when_opponent_controls_hal():
    """Reviewer's exact Phase 1 leak: agent_role='baku', leap window, opponent
    (= Hal) returns 61 for Hal-dropper. The env must reject it at step()."""
    env = STLEnv(opponent=_FixedSecondOpponent(61), agent_role="baku", seed=0)
    env.reset(options={
        "scenario": {
            "game_clock": 3540.0,
            "current_half": 1,
            "first_dropper": "hal",
        }
    })
    # Sanity: Hal is dropper this half-round (first_dropper=hal, current_half=1).
    D, _C = env.game.get_roles_for_half(env.game.current_half)
    assert D.name == "Hal"
    assert env.game.is_leap_second_turn()

    # Agent (Baku) plays a legal checker action; opponent (Hal) returns 61, illegal.
    with pytest.raises(ValueError, match="actor='hal'.*role='dropper'"):
        env.step(60)


def test_step_rejects_illegal_hal_checker_61_when_opponent_controls_hal():
    """Symmetric case: leap window, Hal as checker — Hal-checker can never use
    second 61 regardless of leap window. Opponent returning 61 must be rejected.
    """
    env = STLEnv(opponent=_FixedSecondOpponent(61), agent_role="baku", seed=0)
    env.reset(options={
        "scenario": {
            "game_clock": 3540.0,
            "current_half": 2,
            "first_dropper": "hal",
        }
    })
    # current_half=2 with first_dropper=hal: Baku=dropper, Hal=checker.
    D, C = env.game.get_roles_for_half(env.game.current_half)
    assert D.name == "Baku" and C.name == "Hal"

    with pytest.raises(ValueError, match="actor='hal'.*role='checker'"):
        env.step(60)


def test_step_accepts_legal_baku_dropper_61_in_leap_window():
    """Sanity: the Phase 1 asymmetry must let Baku-as-dropper play 61 inside
    the leap window. With agent_role='hal' and Baku as opponent dropping at 61,
    step() must succeed without raising."""
    env = STLEnv(opponent=_FixedSecondOpponent(61), agent_role="hal", seed=0)
    env.reset(options={
        "scenario": {
            "game_clock": 3540.0,
            "current_half": 2,
            "first_dropper": "hal",
        }
    })
    # current_half=2 with first_dropper=hal: Baku=dropper, Hal=checker.
    D, C = env.game.get_roles_for_half(env.game.current_half)
    assert D.name == "Baku" and C.name == "Hal"
    # Hal-checker plays second 30 (action=30). Opponent Baku-dropper returns 61 (legal).
    obs, rew, term, trunc, info = env.step(30)
    assert isinstance(obs, type(obs))  # no exception, environment advanced
