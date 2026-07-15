"""High-level strategic snapshots built from engine state.

The teacher bots use this module to turn raw game state into a compact
view of route parity, role alignment, budgets, and current named stage.
"""

from __future__ import annotations

from dataclasses import dataclass

from .route_math import (
    current_dropper_checker,
    current_lsr_variation,
    get_named_players,
    is_active_lsr,
    player_budget,
    rounds_until_leap_window,
)
from .route_stages import current_route_stage_flags


@dataclass(frozen=True)
class StrategySnapshot:
    round_num: int
    current_half: int
    game_clock: float
    lsr_variation: int
    active_lsr: bool
    rounds_until_leap: int
    is_leap_turn: bool
    current_dropper_name: str
    current_checker_name: str
    hal_budget: object
    baku_budget: object
    referee_cprs: int
    route_flags: dict[str, bool]


def build_strategy_snapshot(game) -> StrategySnapshot:
    hal, baku = get_named_players(game)
    dropper, checker = current_dropper_checker(game)
    return StrategySnapshot(
        round_num=game.round_num,
        current_half=game.current_half,
        game_clock=game.game_clock,
        lsr_variation=current_lsr_variation(game),
        active_lsr=is_active_lsr(game),
        rounds_until_leap=rounds_until_leap_window(game),
        is_leap_turn=game.is_leap_second_turn(),
        current_dropper_name=dropper.name.lower(),
        current_checker_name=checker.name.lower(),
        hal_budget=player_budget(hal),
        baku_budget=player_budget(baku),
        referee_cprs=game.referee.cprs_performed,
        route_flags=current_route_stage_flags(game),
    )
