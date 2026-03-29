"""Small action-selection helpers shared by scripted teacher bots.

These helpers keep second-picking conventions consistent across Baku and
Hal teachers so the strategic modules can stay focused on intent.
"""

from __future__ import annotations

from src.Constants import TURN_DURATION_LEAP, TURN_DURATION_NORMAL


def clamp_second(second: int, *, role: str, turn_duration: int) -> int:
    max_second = turn_duration if role == "dropper" else TURN_DURATION_NORMAL
    return max(1, min(max_second, int(second)))


def safe_check(turn_duration: int) -> int:
    del turn_duration
    return TURN_DURATION_NORMAL


def instant_check(turn_duration: int) -> int:
    del turn_duration
    return 1


def instant_drop(turn_duration: int) -> int:
    return TURN_DURATION_LEAP if turn_duration == TURN_DURATION_LEAP else 1


def target_st_drop(target_st: int, turn_duration: int) -> int:
    if turn_duration == TURN_DURATION_LEAP and target_st <= 0:
        return TURN_DURATION_LEAP
    target = TURN_DURATION_NORMAL - int(target_st)
    return clamp_second(target, role="dropper", turn_duration=turn_duration)


def death_probe_check(turn_duration: int) -> int:
    del turn_duration
    return 5


def medium_check(turn_duration: int) -> int:
    del turn_duration
    return 30


def early_check(turn_duration: int, second: int = 10) -> int:
    return clamp_second(second, role="checker", turn_duration=turn_duration)


def late_drop(turn_duration: int, second: int = 60) -> int:
    return clamp_second(second, role="dropper", turn_duration=turn_duration)
