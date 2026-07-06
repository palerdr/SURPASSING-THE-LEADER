"""Weighted opponent leagues for mixed scripted and learned training.

One opponent is sampled per episode, which lets PPO train against a small
population instead of overfitting to a single fixed adversary.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from src.Game import Game

from .base import Opponent


@dataclass(frozen=True)
class LeagueEntry:
    label: str
    weight: float
    opponent: Opponent


class WeightedOpponentLeague(Opponent):
    def __init__(self, entries: list[LeagueEntry], seed: int | None = None):
        if not entries:
            raise ValueError("WeightedOpponentLeague requires at least one entry")

        for entry in entries:
            if entry.weight <= 0:
                raise ValueError(f"League entry weight must be > 0, got {entry.weight} for {entry.label}")

        self.entries = tuple(entries)
        self._weights = [entry.weight for entry in self.entries]
        self._rng = random.Random(seed)
        self.active_entry = self.entries[0]

    def choose_action(self, game: Game, role: str, turn_duration: int) -> int:
        return self.active_entry.opponent.choose_action(game, role, turn_duration)

    def reset(self) -> None:
        self.active_entry = self._rng.choices(self.entries, weights=self._weights, k=1)[0]
        self.active_entry.opponent.reset()
