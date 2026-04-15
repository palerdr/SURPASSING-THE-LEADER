from __future__ import annotations

from src.Game import Game
from .types import MemoryMode


def hal_can_check_leap(memory: MemoryMode, leap_deduced: bool) -> bool:
    """Hal may check second 61 only when both:
      - leap_deduced is True (Hal knows the leap second exists), AND
      - memory != AMNESIA (Hal's recall isn't actively impaired).

    PRE_AMNESIA and RECOVERED states still allow it; only AMNESIA blocks.
    Hal as dropper can never use second 61 regardless of state.
    """
    return leap_deduced and memory != MemoryMode.AMNESIA


def update_memory(memory: MemoryMode, game: Game, death_occurred: bool) -> MemoryMode:
    if memory == MemoryMode.NORMAL:
        if game.game_clock >= 3300 and not game.is_leap_second_turn():
            return MemoryMode.PRE_AMNESIA
        return MemoryMode.NORMAL

    if memory == MemoryMode.PRE_AMNESIA:
        if game.is_leap_second_turn():
            return MemoryMode.AMNESIA
        return MemoryMode.PRE_AMNESIA

    if memory == MemoryMode.AMNESIA:
        if death_occurred:
            return MemoryMode.RECOVERED
        return MemoryMode.AMNESIA

    return MemoryMode.RECOVERED
