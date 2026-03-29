"""Random opponent — uniform random timing.

Useful as a baseline: any trained agent should crush this easily.
"""

from __future__ import annotations
import random

from src.Game import Game
from src.Constants import TURN_DURATION_NORMAL
from .base import Opponent


class RandomBot(Opponent):
    """Picks a uniformly random second each turn."""

    def choose_action(self, game: Game, role: str, turn_duration: int) -> int:
        if role == "dropper":
            return random.randint(1, turn_duration)
        else:
            # Checker doesn't know about leap seconds — max is always 60
            return random.randint(1, TURN_DURATION_NORMAL)
