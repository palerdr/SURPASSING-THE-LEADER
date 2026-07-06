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
        """Pick a second within the actor's legal range for this role/state.

        The legal max depends on actor identity, role, and leap awareness:
          - Hal dropper: never 61.
          - Hal checker: may use 61 only when leap_deduced and not in AMNESIA.
          - Baku dropper: may use 61 when turn_duration == 61.
          - Baku checker: never 61.

        See environment.legal_actions for the canonical legality rules.

        Args:
            game: Current game state (full access — opponents are privileged).
            role: Either "dropper" or "checker".
            turn_duration: 60 normally, 61 during leap second turn (engine-level).

        Returns:
            The chosen second (integer) within the actor's legal range.
        """
        ...

    def reset(self) -> None:
        """Called at the start of each episode. Override if your bot has state."""
        pass
