import os
import sys

sys.path.insert(0, os.getcwd())

from stl.play.env import STLEnv
from stl.play.opponents.baku_teachers import BakuLeapExecutorTeacher, BakuRouteBuilderTeacher
from stl.play.opponents.hal_teachers import HalDeathTradeTeacher, HalMemoryLossTeacher, HalPressureTeacher
from stl.play.opponents.random_bot import RandomBot
from stl.learning.curriculum import get_scenario


def make_env(role: str = "hal", **kwargs) -> STLEnv:
    return STLEnv(opponent=RandomBot(), agent_role=role, seed=123, **kwargs)


def test_baku_route_builder_uses_opening_risky_check():
    env = make_env("baku")
    env.reset(options={
        "scenario": {
            "name": "opening_hal_first",
            "game_clock": 0.0,
            "round_num": 0,
            "current_half": 1,
            "first_dropper": "hal",
        }
    })

    teacher = BakuRouteBuilderTeacher()

    assert teacher.choose_action(env.game, "checker", env.game.get_turn_duration()) == 30


def test_baku_leap_executor_uses_sixty_one_as_leap_dropper():
    env = make_env("baku")
    env.reset(options={"scenario": get_scenario("round9_leap")})

    teacher = BakuLeapExecutorTeacher()

    assert teacher.choose_action(env.game, "dropper", env.game.get_turn_duration()) == 61


def test_hal_death_trade_teacher_probes_failed_check_window():
    env = make_env("hal")
    env.reset(options={
        "scenario": {
            "name": "hal_trade_window",
            "game_clock": 1020.0,
            "round_num": 1,
            "current_half": 2,
            "first_dropper": "hal",
            "hal": {"cylinder": 24.0, "ttd": 0.0, "deaths": 0, "alive": True},
            "baku": {"cylinder": 25.0, "ttd": 60.0, "deaths": 1, "alive": True},
            "referee_cprs": 1,
        }
    })

    teacher = HalDeathTradeTeacher()

    assert teacher.choose_action(env.game, "checker", env.game.get_turn_duration()) == 5


def test_hal_pressure_teacher_uses_instant_drop_in_round7_pressure():
    env = make_env("hal")
    env.reset(options={"scenario": get_scenario("round7_pressure")})

    teacher = HalPressureTeacher()

    assert teacher.choose_action(env.game, "dropper", env.game.get_turn_duration()) == 1


def test_hal_memory_teacher_falls_for_leap_at_sixty():
    env = make_env("hal")
    env.reset(options={"scenario": get_scenario("round9_leap")})

    teacher = HalMemoryLossTeacher()

    assert teacher.choose_action(env.game, "checker", env.game.get_turn_duration()) == 60


def test_hal_teachers_never_drop_61_on_leap_turn():
    """Item 23: Hal teachers must never drop 61, even on leap turn.

    Baku may drop 61 (as the canonical leap-second exploit) but Hal cannot —
    Structural actor/role rules forbid Hal from using the extra
    second. This is enforced via the actor-aware clamp_second contract.
    """
    from stl.play.opponents.hal_teachers import (
        HalTeacher,
        HalLeapInferenceTeacher,
        HalDeathTradeTeacher,
        HalPressureTeacher,
        HalDeviationTeacher,
        HalEcholocationTeacher,
        HalMemoryLossTeacher,
        HalResilienceTeacher,
    )

    env = make_env("hal")
    env.reset(options={"scenario": get_scenario("round9_leap")})
    td = env.game.get_turn_duration()
    assert td == 61

    teachers = [
        HalTeacher(),
        HalLeapInferenceTeacher(),
        HalDeathTradeTeacher(),
        HalPressureTeacher(),
        HalDeviationTeacher(),
        HalEcholocationTeacher(),
        HalMemoryLossTeacher(),
        HalResilienceTeacher(),
    ]

    for teacher in teachers:
        action = teacher.choose_action(env.game, "dropper", td)
        assert 1 <= action <= 60, f"{type(teacher).__name__} dropped {action} on leap turn"


def test_baku_leap_executor_still_drops_61():
    env = make_env("baku")
    env.reset(options={"scenario": get_scenario("round9_leap")})

    teacher = BakuLeapExecutorTeacher()
    action = teacher.choose_action(env.game, "dropper", env.game.get_turn_duration())
    assert action == 61
