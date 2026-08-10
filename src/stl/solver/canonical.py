"""Small pure helpers for the canonical STL state and leap route."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, TypeAlias

from stl.engine.game import (
    LS_WINDOW_END,
    LS_WINDOW_START,
    OPENING_START_CLOCK,
    TURN_DURATION_LEAP,
    TURN_DURATION_NORMAL,
)

PlayerIdentity: TypeAlias = Literal["Hal", "Baku"]


@dataclass(frozen=True, slots=True)
class WorldState:
    baku_load: int
    baku_ttd: int
    hal_load: int
    hal_ttd: int
    half: int
    clock: int
    hal_leap_memory: bool


@dataclass(frozen=True, slots=True)
class PublicHalfRound:
    """Minimal revealed half-round record needed to replay canonical play."""

    drop_second: int
    check_second: int
    survived: bool | None


@dataclass(frozen=True, slots=True)
class GameState:
    """Immutable canonical node, including revealed history and terminal identity."""

    world: WorldState
    public_history: tuple[PublicHalfRound, ...]
    winner: PlayerIdentity | None


def is_leap_window(clock: int | float) -> bool:
    """Return whether a half-round starting at ``clock`` spans second 61."""

    return LS_WINDOW_START <= clock <= LS_WINDOW_END


def turn_duration(clock: int | float) -> int:
    """Return the number of literal action seconds at ``clock``."""

    return TURN_DURATION_LEAP if is_leap_window(clock) else TURN_DURATION_NORMAL


def lsr_variation(round_start_minute: int) -> int:
    """Return V1..V4 from the round-start minute after 8:00."""

    return 1 + ((round_start_minute - 12) % 4)


def is_active_lsr(round_start_minute: int) -> bool:
    """Return whether the round-start minute is on active route V2."""

    return lsr_variation(round_start_minute) == 2


def leap_drop_available(state: WorldState) -> bool:
    """Return whether canonical Baku may drop on literal second 61."""

    return state.half == 2 and is_leap_window(state.clock)


def root_node() -> WorldState:
    """Return the exact canonical opening state at 8:12 AM."""

    return WorldState(
        baku_load=0,
        baku_ttd=0,
        hal_load=0,
        hal_ttd=0,
        half=1,
        clock=OPENING_START_CLOCK,
        hal_leap_memory=False,
    )
