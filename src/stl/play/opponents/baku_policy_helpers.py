"""Heuristic second-choice helpers for Baku teacher policies.

This module holds the route-aware timing rules that the concrete Baku
teachers reuse and override in narrow situations.
"""

from __future__ import annotations

from .teacher_helpers import clamp_second, early_check, instant_check, instant_drop, late_drop, safe_check, target_st_drop


def choose_baku_checker_second(snapshot, turn_duration: int) -> int:
    if snapshot.route_flags["round9_pre_leap"] or snapshot.is_leap_turn:
        return safe_check(turn_duration)

    if snapshot.active_lsr:
        if snapshot.current_half == 1:
            return safe_check(turn_duration)
        return safe_check(turn_duration)

    if snapshot.lsr_variation == 3 and snapshot.baku_budget.fail_post_ttd < 300.0:
        return instant_check(turn_duration)

    if snapshot.baku_budget.cylinder >= 90.0:
        return instant_check(turn_duration)

    if snapshot.round_num == 0:
        return early_check(turn_duration, second=30)

    if snapshot.round_num in (1, 2):
        return safe_check(turn_duration)

    return safe_check(turn_duration)


def choose_baku_dropper_second(snapshot, turn_duration: int) -> int:
    if snapshot.is_leap_turn:
        return turn_duration

    if snapshot.round_num == 2 and snapshot.current_half == 2:
        return target_st_drop(36, turn_duration)

    if snapshot.round_num == 4 and snapshot.current_half == 2:
        return instant_drop(turn_duration)

    if snapshot.round_num == 6 and snapshot.current_half == 2:
        return clamp_second(10, role="dropper", turn_duration=turn_duration)

    if snapshot.round_num == 7 and snapshot.current_half == 2:
        return late_drop(turn_duration, second=60)

    if snapshot.active_lsr:
        return instant_drop(turn_duration)

    return instant_drop(turn_duration)
