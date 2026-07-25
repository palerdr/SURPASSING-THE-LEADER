"""Single source of truth for actor-aware action legality.

Action index equals action second throughout the active codebase.

Normal action seconds are 1..60. During a leap-window half-round only a
non-Hal dropper (Baku in canonical play) may also use second 61. Checkers
are capped at 60, Hal can never use 61 in either role, and index/second 0
is permanently illegal padding in dense policy vectors. These are ordinal
literal seconds: action 1 is an immediate first-second action. Successful ST
is counted inclusively by the engine as ``check - drop + 1``. See
``docs/ACTION_TIMING.md``.
"""

from __future__ import annotations

import numpy as np

from stl.engine.game import TURN_DURATION_LEAP, TURN_DURATION_NORMAL


NORMAL_MAX_SECOND = TURN_DURATION_NORMAL
LEAP_MAX_SECOND = TURN_DURATION_LEAP
ACTION_SIZE = LEAP_MAX_SECOND + 1
TIMING_CONVENTION_ID = "ordinal-seconds-inclusive-st-v1"
NORMAL_SECONDS = tuple(range(1, NORMAL_MAX_SECOND + 1))
LEAP_SECONDS = tuple(range(1, LEAP_MAX_SECOND + 1))


class IllegalActionError(ValueError):
    """Raised when an action second violates actor/role legality."""


def legal_max_second(actor: str, role: str, turn_duration: int) -> int:
    actor_l = actor.lower()
    role_l = role.lower()

    if turn_duration < TURN_DURATION_LEAP:
        return min(int(turn_duration), TURN_DURATION_NORMAL)

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


def legal_seconds(actor: str, role: str, turn_duration: int) -> tuple[int, ...]:
    """Return literal legal action seconds for the actor and role."""
    return tuple(range(1, legal_max_second(actor, role, turn_duration) + 1))


def legal_mask(actor: str, role: str, turn_duration: int) -> np.ndarray:
    """Return a length-62 boolean mask indexed by action second."""
    mask = np.zeros(ACTION_SIZE, dtype=bool)
    max_sec = legal_max_second(actor, role, turn_duration)
    if max_sec >= 1:
        mask[1 : max_sec + 1] = True
    return mask


def clamp_action(second: int, *, actor: str, role: str, turn_duration: int) -> int:
    max_sec = legal_max_second(actor, role, turn_duration)
    return max(1, min(int(second), max_sec))


def validate_action(second: int, *, actor: str, role: str, turn_duration: int) -> None:
    """Validate a literal second against the actor-aware legality rule."""
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
