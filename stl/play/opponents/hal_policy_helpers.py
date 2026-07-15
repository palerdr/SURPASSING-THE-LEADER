"""Heuristic second-choice helpers for Hal teacher policies.

These rules encode the broad doc-derived timing skeleton that concrete
Hal teachers refine for death trade, pressure, deviation, and memory risk.
"""

from __future__ import annotations

from .teacher_helpers import death_probe_check, instant_check, instant_drop, late_drop, safe_check, target_st_drop


def choose_hal_checker_second(snapshot, turn_duration: int) -> int:
    if snapshot.is_leap_turn:
        return 60

    if snapshot.round_num == 1 and snapshot.current_half == 2:
        return death_probe_check(turn_duration)

    if snapshot.round_num == 3 and snapshot.current_half == 2:
        return 34

    if snapshot.round_num == 4 and snapshot.current_half == 2:
        return 15

    if snapshot.round_num == 5 and snapshot.current_half == 2:
        return 8

    if snapshot.round_num == 7 and snapshot.current_half == 2:
        return instant_check(turn_duration)

    if snapshot.active_lsr and snapshot.round_num >= 6 and snapshot.current_half == 2:
        return instant_check(turn_duration)

    return safe_check(turn_duration)


def choose_hal_dropper_second(snapshot, turn_duration: int) -> int:
    if snapshot.is_leap_turn:
        return instant_drop(turn_duration, actor="hal")

    if snapshot.round_num == 1 and snapshot.current_half == 1:
        return target_st_drop(25, turn_duration, actor="hal")

    if snapshot.round_num == 2 and snapshot.current_half == 1:
        return target_st_drop(4, turn_duration, actor="hal")

    if snapshot.round_num == 3 and snapshot.current_half == 1:
        return target_st_drop(3, turn_duration, actor="hal")

    if snapshot.round_num == 5 and snapshot.current_half == 1:
        return late_drop(turn_duration, second=60, actor="hal")

    if snapshot.round_num == 6 and snapshot.current_half == 1:
        return instant_drop(turn_duration, actor="hal")

    if snapshot.round_num == 7 and snapshot.current_half == 1:
        return target_st_drop(55, turn_duration, actor="hal")

    if snapshot.round_num == 8 and snapshot.current_half == 1:
        return instant_drop(turn_duration, actor="hal")

    if snapshot.active_lsr and snapshot.current_half == 1:
        return instant_drop(turn_duration, actor="hal")

    return target_st_drop(24, turn_duration, actor="hal")
