"""Single source of truth for actor-aware action legality.

Engine stays objective. This module encodes the policy-layer legality
asymmetries between Hal and Baku for the leap second:

  - Hal may CHECK 61 only when leap_deduced and memory is not AMNESIA.
  - Hal may NEVER DROP 61.
  - Baku may DROP 61 (always allowed when turn_duration == 61).
  - Baku may NEVER CHECK 61.

All teacher helpers, env action masks, and the Hal search/sampling layer
read legality from here so they can never disagree.
"""

from __future__ import annotations

from src.Constants import TURN_DURATION_LEAP, TURN_DURATION_NORMAL


def legal_max_second(
    actor: str,
    role: str,
    turn_duration: int,
    *,
    hal_leap_deduced: bool = False,
    hal_memory_impaired: bool = False,
) -> int:
    actor_l = actor.lower()
    role_l = role.lower()

    if turn_duration < TURN_DURATION_LEAP:
        if role_l == "dropper":
            return turn_duration
        return min(turn_duration, TURN_DURATION_NORMAL)

    if role_l == "dropper":
        if actor_l == "hal":
            return TURN_DURATION_NORMAL
        return TURN_DURATION_LEAP

    if actor_l == "baku":
        return TURN_DURATION_NORMAL

    if hal_leap_deduced and not hal_memory_impaired:
        return TURN_DURATION_LEAP
    return TURN_DURATION_NORMAL


def can_use_leap_second(
    actor: str,
    role: str,
    *,
    hal_leap_deduced: bool = False,
    hal_memory_impaired: bool = False,
) -> bool:
    actor_l = actor.lower()
    role_l = role.lower()

    if actor_l == "hal":
        if role_l == "dropper":
            return False
        return hal_leap_deduced and not hal_memory_impaired

    return role_l == "dropper"


def clamp_action(
    second: int,
    *,
    actor: str,
    role: str,
    turn_duration: int,
    hal_leap_deduced: bool = False,
    hal_memory_impaired: bool = False,
) -> int:
    max_sec = legal_max_second(
        actor, role, turn_duration,
        hal_leap_deduced=hal_leap_deduced,
        hal_memory_impaired=hal_memory_impaired,
    )
    return max(1, min(int(second), max_sec))
