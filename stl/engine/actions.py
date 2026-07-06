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

from stl.engine.game import TURN_DURATION_LEAP, TURN_DURATION_NORMAL


class IllegalActionError(ValueError):
    """Raised when an action second violates actor/role legality."""


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


def validate_action(second: int, *, actor: str, role: str, turn_duration: int) -> None:
    """Validate a one-indexed second against the actor-aware legality rule."""
    max_sec = legal_max_second(actor, role, turn_duration)
    try:
        value = int(second)
    except (TypeError, ValueError) as exc:
        raise IllegalActionError(
            f"Illegal action second={second!r} for actor={actor!r} role={role!r}; "
            "seconds must be integers."
        ) from exc

    if isinstance(second, bool) or value != second:
        raise IllegalActionError(
            f"Illegal action second={second!r} for actor={actor!r} role={role!r}; "
            "seconds must be integral."
        )
    if not (1 <= value <= max_sec):
        raise IllegalActionError(
            f"Illegal action second={second!r} for actor={actor!r} role={role!r} "
            f"turn_duration={turn_duration}; legal range is [1, {max_sec}]."
        )
