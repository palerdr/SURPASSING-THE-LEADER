"""Base opponent interface.

Every opponent bot implements this interface so the environment can
query any opponent the same way. This is the Strategy pattern — the
env doesn't care whether it's facing a random bot or a neural network.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from src.Game import Game


class Opponent(ABC):
    """Abstract base for all opponent policies."""

    @abstractmethod
    def choose_action(self, game: Game, role: str, turn_duration: int) -> int:
        """Pick a second [1, turn_duration] for dropping or [1, 60] for checking.

        Args:
            game: Current game state (full access — opponents are privileged).
            role: Either "dropper" or "checker".
            turn_duration: 60 normally, 61 during leap second turn.

        Returns:
            The chosen second (integer).
        """
        ...

    def reset(self) -> None:
        """Called at the start of each episode. Override if your bot has state."""
        pass
