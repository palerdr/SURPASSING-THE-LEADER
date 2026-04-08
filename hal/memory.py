from __future__ import annotations

from src.Game import Game
from .types import MemoryMode


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
