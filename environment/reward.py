"""Reward helpers for sparse and route-shaped training.

This module is intentionally small: the real strategic structure should
live in scenarios, metrics, and teachers rather than hidden reward hacks.
"""

from __future__ import annotations

from .route_stages import (
    ROUTE_STAGE_REWARD_ORDER,
    current_route_stage_flags,
)

ROUTE_SHAPING_PRESETS = {
    "light": {
        "round7_pressure": 0.02,
        "round8_bridge": 0.03,
        "round9_pre_leap": 0.05,
        "leap_turn": 0.05,
    },
    "bridge": {
        "round7_pressure": 0.30,
        "round8_bridge": 0.45,
        "round9_pre_leap": 0.75,
        "leap_turn": 0.50,
    },
    "exact_bridge": {
        "round7_pressure": 1.25,
        "round8_bridge": 1.75,
        "round9_pre_leap": 2.50,
        "leap_turn": 0.00,
    },
}


def sparse_reward(game_over: bool, agent_won: bool) -> float:
    """Terminal-only reward."""
    if not game_over:
        return 0.0
    return 1.0 if agent_won else -1.0


def compute_route_shaping_bonus(
    game,
    awarded_stages: set[str],
    preset: str = "light",
) -> tuple[float, set[str]]:
    if preset not in ROUTE_SHAPING_PRESETS:
        raise ValueError(f"Unknown route shaping preset: {preset}")

    bonuses = ROUTE_SHAPING_PRESETS[preset]
    current_flags = current_route_stage_flags(game)
    newly_awarded = {
        stage_name
        for stage_name in ROUTE_STAGE_REWARD_ORDER
        if stage_name not in awarded_stages and current_flags.get(stage_name, False)
    }
    bonus = sum(bonuses[stage_name] for stage_name in newly_awarded)
    return bonus, awarded_stages | newly_awarded


def shaped_reward(
    game_over: bool,
    agent_won: bool,
    route_bonus: float,
) -> float:
    return sparse_reward(game_over, agent_won) + route_bonus
