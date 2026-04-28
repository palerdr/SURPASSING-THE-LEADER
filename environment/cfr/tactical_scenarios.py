"""Small exact tactical positions for CFR/search audits.

These are tablebase-style fixtures, not training reward shortcuts. They exist
to provide known exact targets for diagnostics, MCTS, MCCFR, and value nets.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.Constants import PHYSICALITY_BAKU, PHYSICALITY_HAL
from src.Game import Game
from src.Player import Player
from src.Referee import Referee

from .exact_transition import ExactSearchConfig


@dataclass(frozen=True)
class TacticalScenario:
    name: str
    game: Game
    config: ExactSearchConfig
    half_round_horizon: int
    expected_note: str


def _base_game(*, clock: float = 720.0, current_half: int = 1) -> Game:
    hal = Player(name="Hal", physicality=PHYSICALITY_HAL)
    baku = Player(name="Baku", physicality=PHYSICALITY_BAKU)
    game = Game(player1=hal, player2=baku, referee=Referee(), first_dropper=hal)
    game.seed(0)
    game.game_clock = clock
    game.current_half = current_half
    return game


def forced_baku_overflow_death() -> TacticalScenario:
    """Hal drops, Baku checks with cylinder already at 299.

    Every legal exact action either succeeds with at least ST=1 or fails and
    injects the full cylinder. Both paths are terminal for Baku because
    death_duration reaches 300 seconds.
    """
    game = _base_game()
    game.player2.cylinder = 299.0
    return TacticalScenario(
        name="forced_baku_overflow_death",
        game=game,
        config=ExactSearchConfig(),
        half_round_horizon=1,
        expected_note="All joint actions are terminal Hal wins.",
    )


def leap_second_check_61_probe() -> TacticalScenario:
    """Baku drops on leap turn while Hal is checker and has deduced the leap."""
    game = _base_game(clock=3540.0, current_half=2)
    return TacticalScenario(
        name="leap_second_check_61_probe",
        game=game,
        config=ExactSearchConfig(hal_leap_deduced=True),
        half_round_horizon=1,
        expected_note="Against drop=61, check=61 avoids the failed-check death branch that check=60 triggers.",
    )

