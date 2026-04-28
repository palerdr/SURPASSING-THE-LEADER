"""Engine-derived timing and route arithmetic for rigorous CFR.

Pure functions over the engine state: clock geometry, LSR variation,
projected gaps, and survival-budget math. No reward shaping, no stage
labels, no curriculum milestones, no "good for Hal" categories. Anything
that asserts progress without proving it through terminal utility lives
in ``environment/route_math.py`` or further out.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.Constants import (
    CYLINDER_MAX,
    FAILED_CHECK_PENALTY,
    LS_WINDOW_END,
    LS_WINDOW_START,
    TURN_DURATION_NORMAL,
)


__all__ = [
    "PlayerBudget",
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


def strict_next_minute(seconds: float) -> int:
    whole_seconds = int(seconds)
    return ((whole_seconds // 60) + 1) * 60


def lsr_variation_from_clock(game_clock: float) -> int:
    minutes_since_start = int(game_clock) // 60
    return (minutes_since_start % 4) + 1


def current_lsr_variation(game) -> int:
    return lsr_variation_from_clock(game.game_clock)


def is_active_lsr(game) -> bool:
    return current_lsr_variation(game) == 2


def is_leap_window(game_clock: float) -> bool:
    return LS_WINDOW_START <= game_clock < LS_WINDOW_END


def rounds_until_leap_window(game) -> int:
    remaining = max(0.0, LS_WINDOW_START - game.game_clock)
    return int(remaining // (TURN_DURATION_NORMAL * 4))


def get_named_players(game):
    hal = game.player1 if game.player1.name.lower() == "hal" else game.player2
    baku = game.player1 if game.player1.name.lower() == "baku" else game.player2
    return hal, baku


def player_named(game, name: str):
    lowered = name.lower()
    hal, baku = get_named_players(game)
    if lowered == "hal":
        return hal
    if lowered == "baku":
        return baku
    raise ValueError(f"Unknown player name: {name}")


def current_dropper_checker(game):
    return game.get_roles_for_half(game.current_half)


def role_for_player(game, player) -> str:
    dropper, checker = current_dropper_checker(game)
    if player is dropper:
        return "dropper"
    if player is checker:
        return "checker"
    raise ValueError("player is not part of this game")


def projected_failed_check_death_duration(player) -> float:
    return player.cylinder + FAILED_CHECK_PENALTY


def projected_post_fail_ttd(player) -> float:
    return player.ttd + projected_failed_check_death_duration(player)


def safe_strategy_budget(player) -> int:
    return max(0, int((CYLINDER_MAX - 1 - player.cylinder) // TURN_DURATION_NORMAL))


def projected_round_gap_for_death_duration(death_duration: float) -> int:
    return strict_next_minute(death_duration + 300.0)


def projected_round_gap_for_two_deaths(first_death_duration: float, second_death_duration: float) -> int:
    return strict_next_minute(first_death_duration + second_death_duration + 420.0)


def projected_variation_after_gap(round_start_clock: float, gap_seconds: int) -> int:
    return lsr_variation_from_clock(round_start_clock + gap_seconds)


def projected_variation_after_current_checker_fail(game) -> int:
    round_start_clock = game.game_clock if game.current_half == 1 else game.game_clock - 120.0
    checker = current_dropper_checker(game)[1]
    gap_seconds = projected_round_gap_for_death_duration(projected_failed_check_death_duration(checker))
    return projected_variation_after_gap(round_start_clock, gap_seconds)


def current_checker_fail_would_activate_lsr(game) -> bool:
    return projected_variation_after_current_checker_fail(game) == 2


@dataclass(frozen=True)
class PlayerBudget:
    cylinder: float
    ttd: float
    deaths: int
    safe_budget: int
    fail_death_duration: float
    fail_post_ttd: float


def player_budget(player) -> PlayerBudget:
    fail_death_duration = projected_failed_check_death_duration(player)
    return PlayerBudget(
        cylinder=player.cylinder,
        ttd=player.ttd,
        deaths=player.deaths,
        safe_budget=safe_strategy_budget(player),
        fail_death_duration=fail_death_duration,
        fail_post_ttd=player.ttd + fail_death_duration,
    )
