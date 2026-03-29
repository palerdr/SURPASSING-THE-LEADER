import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from environment.dth_env import DTHEnv
from environment.opponents.random_bot import RandomBot
from environment.reward import ROUTE_SHAPING_PRESETS, compute_route_shaping_bonus
from environment.route_stages import current_route_stage_flags
from training.curriculum import CURRICULA, get_scenario


def make_env(**kwargs) -> DTHEnv:
    return DTHEnv(opponent=RandomBot(), agent_role="hal", seed=123, **kwargs)


def test_bridge_curriculum_weights_match_contract():
    assert CURRICULA["bridge"] == [
        (0.20, None),
        (0.20, "round7_pressure"),
        (0.30, "round8_bridge"),
        (0.20, "round9_pre_leap"),
        (0.10, "round9_leap_deduced"),
    ]


def test_segmented_curricula_start_only_from_their_source_stage():
    assert CURRICULA["opening_to_round7"] == [(1.00, None)]
    assert CURRICULA["round7_to_round8"] == [(1.00, "round7_pressure")]
    assert CURRICULA["round8_to_round9"] == [(1.00, "round8_bridge")]


def test_round8_bridge_stage_flags_match_named_scenario():
    env = make_env()
    env.reset(options={"scenario": get_scenario("round8_bridge")})

    flags = current_route_stage_flags(env.game)

    assert flags["round7_pressure"] is False
    assert flags["round8_bridge"] is True
    assert flags["round9_pre_leap"] is False
    assert flags["leap_window"] is False
    assert flags["leap_turn"] is False


def test_route_shaping_pays_each_stage_once():
    env = make_env()
    env.reset(options={"scenario": get_scenario("round8_bridge")})

    bonus, awarded = compute_route_shaping_bonus(env.game, set())
    repeat_bonus, repeat_awarded = compute_route_shaping_bonus(env.game, awarded)

    assert bonus == 0.03
    assert awarded == {"round8_bridge"}
    assert repeat_bonus == 0.0
    assert repeat_awarded == awarded


def test_leap_turn_shaping_does_not_repeat_with_existing_awards():
    env = make_env()
    env.reset(options={"scenario": get_scenario("round9_leap_deduced")})

    bonus, awarded = compute_route_shaping_bonus(env.game, {"round7_pressure", "round8_bridge", "round9_pre_leap"})
    repeat_bonus, repeat_awarded = compute_route_shaping_bonus(env.game, awarded)

    assert bonus == 0.05
    assert awarded == {"round7_pressure", "round8_bridge", "round9_pre_leap", "leap_turn"}
    assert repeat_bonus == 0.0
    assert repeat_awarded == awarded


def test_bridge_shaping_preset_makes_milestones_dominate_terminal_reward():
    bridge = ROUTE_SHAPING_PRESETS["bridge"]

    assert bridge["round7_pressure"] + bridge["round8_bridge"] + bridge["round9_pre_leap"] > 1.0


def test_bridge_shaping_preset_uses_larger_bonus_than_light():
    env = make_env()
    env.reset(options={"scenario": get_scenario("round8_bridge")})

    light_bonus, _ = compute_route_shaping_bonus(env.game, set(), preset="light")
    bridge_bonus, _ = compute_route_shaping_bonus(env.game, set(), preset="bridge")

    assert bridge_bonus > light_bonus


def test_exact_bridge_shaping_zeroes_leap_turn_bonus():
    env = make_env()
    env.reset(options={"scenario": get_scenario("round9_leap_deduced")})

    bonus, awarded = compute_route_shaping_bonus(
        env.game,
        {"round7_pressure", "round8_bridge", "round9_pre_leap"},
        preset="exact_bridge",
    )

    assert bonus == 0.0
    assert awarded == {"round7_pressure", "round8_bridge", "round9_pre_leap", "leap_turn"}


def test_exact_bridge_shaping_milestones_dominate_terminal_reward():
    exact_bridge = ROUTE_SHAPING_PRESETS["exact_bridge"]

    assert exact_bridge["round7_pressure"] > 1.0
    assert exact_bridge["round8_bridge"] > 1.0
    assert exact_bridge["round9_pre_leap"] > 1.0
