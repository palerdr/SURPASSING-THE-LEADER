"""Safe strategy bots — the baselines that the leap second is designed to beat.

SafeBot: Always checks at 60, drops at 1. This is the "rational" strategy
that minimizes risk — you can never be caught if you check at the last second.
It's also the strategy the leap second explicitly punishes.

LeapAwareSafeBot: Same safe strategy, but knows about the leap second and
exploits it when dropping. This is the "kill shot" — drop at 61 during a
leap turn, and the opponent checking at 60 will fail.
"""

from __future__ import annotations

from src.Game import Game
from .base import Opponent
from src.Constants import TURN_DURATION_LEAP


class SafeBot(Opponent):
    """

    As checker: always check at second 60 (the last normal second).
    As dropper: always drop at second 1 (earliest possible).

    This is the "rational" strategy from game theory — checking at 60
    means the handkerchief is guaranteed to be on the ground (since D
    must drop by 60), so you never fail a check. The cost is maximum
    squandered time (ST = 60 - drop_time).

    As dropper, dropping at 1 maximizes the opponent's ST if they use
    safe strategy (ST = 59).
    """

    def choose_action(self, game: Game, role: str, turn_duration: int) -> int:
        if role == "checker":
          return 60
        else:
          return 1


class LeapAwareSafeBot(Opponent):
    """

    Same as SafeBot, except:
    As dropper during a leap second turn: drop at 61 instead of 1.

    This is the "kill shot" — the whole point of the leap second mechanic.
    If the opponent checks at 60 (safe strategy), and you drop at 61,
    they check BEFORE the handkerchief is on the ground → failed check
    → 60s penalty → full cylinder injection.

    Hint: How do you know if it's a leap second turn? Look at turn_duration.
    """

    def choose_action(self, game: Game, role: str, turn_duration: int) -> int:
        if role == "dropper":
          if turn_duration == TURN_DURATION_LEAP:
            return 61
          else:
            return 1
        else:
          return 60


class BridgePressureBot(Opponent):
    """Opening-pressure bot that explicitly breaks the 2/1 farming shortcut.

    Policy:
    - As checker: check at 2, so a drop at 2 yields only minimum ST instead of 58.
    - As dropper: drop at 2, so a checker fixed at 1 fails immediately.
    - During leap turns as dropper: still use 61.

    This is not intended as a final strong opponent. It exists to remove the
    easy scripted-bot exploit from the opening training distribution so the
    bridge route is no longer dominated by a short accumulation win.
    """

    def choose_action(self, game: Game, role: str, turn_duration: int) -> int:
        del game
        if role == "dropper":
            if turn_duration == TURN_DURATION_LEAP:
                return 61
            return 2
        return 2
