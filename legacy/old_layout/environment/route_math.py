"""Route and budget math for teachers, audits, evaluation, and tests.

The pure engine-derived subset (clock geometry, LSR variation, projected
gaps, survival-budget math) lives in ``environment.cfr.timing_features``
so rigorous CFR can use it without crossing the shaping firewall. This
module re-exports those symbols and keeps category-label helpers like
``classify_round_gap_seconds`` that are forbidden inside CFR.
"""

from __future__ import annotations

from environment.cfr.timing_features import (
    PlayerBudget,
    current_checker_fail_would_activate_lsr,
    current_dropper_checker,
    current_lsr_variation,
    get_named_players,
    is_active_lsr,
    is_leap_window,
    lsr_variation_from_clock,
    player_budget,
    player_named,
    projected_failed_check_death_duration,
    projected_post_fail_ttd,
    projected_round_gap_for_death_duration,
    projected_round_gap_for_two_deaths,
    projected_variation_after_current_checker_fail,
    projected_variation_after_gap,
    role_for_player,
    rounds_until_leap_window,
    safe_strategy_budget,
    strict_next_minute,
)


__all__ = [
    "PlayerBudget",
    "classify_round_gap_seconds",
    "current_checker_fail_would_activate_lsr",
    "current_dropper_checker",
    "current_lsr_variation",
    "get_named_players",
    "is_active_lsr",
    "is_leap_window",
    "lsr_variation_from_clock",
    "player_budget",
    "player_named",
    "projected_failed_check_death_duration",
    "projected_post_fail_ttd",
    "projected_round_gap_for_death_duration",
    "projected_round_gap_for_two_deaths",
    "projected_variation_after_current_checker_fail",
    "projected_variation_after_gap",
    "role_for_player",
    "rounds_until_leap_window",
    "safe_strategy_budget",
    "strict_next_minute",
]


def classify_round_gap_seconds(gap_seconds: int) -> str:
    if gap_seconds <= 240:
        return "clean_4m"
    if gap_seconds <= 420:
        return "death_7m"
    if gap_seconds <= 480:
        return "heavy_8m"
    return "double_or_larger"
