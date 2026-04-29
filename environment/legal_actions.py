"""Single source of truth for actor-aware action legality.

Engine stays objective. Per the Stockfish-style design (see HAL.md /
plan), Hal can never play check=61. This module hard-codes that:

  - Hal may NEVER CHECK 61.
  - Hal may NEVER DROP 61.
  - Baku may DROP 61 (always allowed when turn_duration == 61).
  - Baku may NEVER CHECK 61.

The asymmetry — Baku-dropper at second 61 inside the leap window vs.
Hal-checker capped at 60 — is the structural constraint that forces
Hal to play around the leap second via deviation rather than through
it. All teacher helpers, env action masks, and the rigorous CFR layer
read legality from here so they can never disagree.
"""

from __future__ import annotations

from src.Constants import TURN_DURATION_LEAP, TURN_DURATION_NORMAL


def legal_max_second(actor: str, role: str, turn_duration: int) -> int:
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

    return TURN_DURATION_NORMAL


def can_use_leap_second(actor: str, role: str) -> bool:
    actor_l = actor.lower()
    role_l = role.lower()

    if actor_l == "hal":
        return False

    return role_l == "dropper"


def clamp_action(second: int, *, actor: str, role: str, turn_duration: int) -> int:
    max_sec = legal_max_second(actor, role, turn_duration)
    return max(1, min(int(second), max_sec))
