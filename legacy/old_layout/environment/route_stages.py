"""Named bridge and leap stages used by shaping, eval, and audits.

These stage markers are intentionally simple and inspectable so training
metrics can be phrased in terms of route progression rather than win rate.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.Constants import LS_WINDOW_START


@dataclass(frozen=True)
class RouteStageSpec:
    name: str
    game_clock: float
    round_num: int
    current_half: int
    is_leap_turn: bool


ROUND7_PRESSURE_STAGE = RouteStageSpec(
    name="round7_pressure",
    game_clock=2700.0,
    round_num=6,
    current_half=1,
    is_leap_turn=False,
)

ROUND8_BRIDGE_STAGE = RouteStageSpec(
    name="round8_bridge",
    game_clock=3060.0,
    round_num=7,
    current_half=1,
    is_leap_turn=False,
)

ROUND9_PRE_LEAP_STAGE = RouteStageSpec(
    name="round9_pre_leap",
    game_clock=3420.0,
    round_num=8,
    current_half=1,
    is_leap_turn=False,
)


ROUTE_STAGE_SPECS = {
    ROUND7_PRESSURE_STAGE.name: ROUND7_PRESSURE_STAGE,
    ROUND8_BRIDGE_STAGE.name: ROUND8_BRIDGE_STAGE,
    ROUND9_PRE_LEAP_STAGE.name: ROUND9_PRE_LEAP_STAGE,
}

ROUTE_STAGE_REWARD_ORDER = (
    "round7_pressure",
    "round8_bridge",
    "round9_pre_leap",
    "leap_turn",
)

ROUTE_STAGE_BONUSES = {
    "round7_pressure": 0.02,
    "round8_bridge": 0.03,
    "round9_pre_leap": 0.05,
    "leap_turn": 0.05,
}


def is_named_route_stage(game, stage_name: str) -> bool:
    stage = ROUTE_STAGE_SPECS[stage_name]
    return (
        game.round_num == stage.round_num
        and game.current_half == stage.current_half
        and game.is_leap_second_turn() == stage.is_leap_turn
        and game.game_clock >= stage.game_clock
    )


def current_route_stage_flags(game) -> dict[str, bool]:
    flags = {
        stage_name: is_named_route_stage(game, stage_name)
        for stage_name in ROUTE_STAGE_SPECS
    }
    flags["leap_window"] = game.game_clock >= LS_WINDOW_START
    flags["leap_turn"] = game.is_leap_second_turn()
    return flags


def stage_is_eligible_from_start(game, stage_name: str) -> bool:
    if stage_name == "leap_window":
        return game.game_clock < LS_WINDOW_START
    if stage_name == "leap_turn":
        return not game.is_leap_second_turn()

    stage = ROUTE_STAGE_SPECS[stage_name]
    return (game.round_num, game.current_half) < (stage.round_num, stage.current_half)
