"""Small action-selection helpers shared by scripted teacher bots.

These helpers keep second-picking conventions consistent across Baku and
Hal teachers so the strategic modules can stay focused on intent.

All helpers are actor-aware via the legality contract in
environment.legal_actions so Hal teachers can never accidentally drop 61
and Baku teachers can never accidentally check 61.
"""

from __future__ import annotations

from src.Constants import TURN_DURATION_LEAP, TURN_DURATION_NORMAL
from environment.legal_actions import clamp_action


def clamp_second(second: int, *, role: str, turn_duration: int, actor: str = "baku") -> int:
    return clamp_action(second, actor=actor, role=role, turn_duration=turn_duration)


def safe_check(turn_duration: int) -> int:
    del turn_duration
    return TURN_DURATION_NORMAL


def instant_check(turn_duration: int) -> int:
    del turn_duration
    return 1


def instant_drop(turn_duration: int, actor: str = "baku") -> int:
    if turn_duration == TURN_DURATION_LEAP and actor.lower() != "hal":
        return TURN_DURATION_LEAP
    return 1


def target_st_drop(target_st: int, turn_duration: int, actor: str = "baku") -> int:
    if turn_duration == TURN_DURATION_LEAP and target_st <= 0 and actor.lower() != "hal":
        return TURN_DURATION_LEAP
    target = TURN_DURATION_NORMAL - int(target_st)
    return clamp_second(target, role="dropper", turn_duration=turn_duration, actor=actor)


def death_probe_check(turn_duration: int) -> int:
    del turn_duration
    return 5


def medium_check(turn_duration: int) -> int:
    del turn_duration
    return 30


def early_check(turn_duration: int, second: int = 10) -> int:
    return clamp_second(second, role="checker", turn_duration=turn_duration)


def late_drop(turn_duration: int, second: int = 60, actor: str = "baku") -> int:
    return clamp_second(second, role="dropper", turn_duration=turn_duration, actor=actor)
