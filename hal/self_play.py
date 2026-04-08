from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from environment.opponents.base import Opponent
from environment.opponents.factory import create_scripted_opponent
from src.Constants import PHYSICALITY_BAKU, PHYSICALITY_HAL
from src.Game import Game
from src.Player import Player
from src.Referee import Referee

from .hal_opponent import CanonicalHal
from .value_net import extract_features

#outcomes from Hal's perspective
@dataclass(slots=True)
class Experience:
    features: np.ndarray  # shape (23,)
    outcome: float  # +1.0 or -1.0


def play_one_game(
    hal_ai: CanonicalHal,
    opponent: Opponent,
    seed: int,
) -> list[Experience]:
    """Play a full game, return experiences from Hal's perspective.

    At each half-round (before Hal acts), extract features.
    After game ends, label all positions with the outcome.
    """
    hal = Player(name="Hal", physicality=PHYSICALITY_HAL)
    baku = Player(name="Baku", physicality=PHYSICALITY_BAKU)
    game = Game(player1=hal, player2=baku, referee=Referee(), first_dropper=hal)
    game.seed(seed)

    hal_ai.reset()
    opponent.reset()

    features: list[np.ndarray] = []
    while not game.game_over:
        features.append(extract_features(game))

        dropper, _ = game.get_roles_for_half(game.current_half)
        turn_duration = game.get_turn_duration()

        if dropper is hal:
            drop_time = hal_ai.choose_action(game, "dropper", turn_duration)
            check_time = opponent.choose_action(game, "checker", turn_duration)
        else:
            drop_time = opponent.choose_action(game, "dropper", turn_duration)
            check_time = hal_ai.choose_action(game, "checker", turn_duration)

        game.play_half_round(drop_time, check_time)

    outcome = 1.0 if game.winner is hal else -1.0
    return [Experience(state_features, outcome) for state_features in features]


def generate_dataset(
    n_games: int,
    opponent_names: list[str],
    search_depth: int = 1,
    base_seed: int = 0,
) -> list[Experience]:
    """Play n_games across the opponent pool, return all experiences."""
    
    dataset = []
    for i in range(n_games):
        seed = base_seed + i
        opp_name = opponent_names[i % len(opponent_names)]
        opponent = create_scripted_opponent(opp_name, seed=seed)
        hal_ai = CanonicalHal(seed=seed, depth=search_depth)
        dataset.extend(play_one_game(hal_ai=hal_ai, opponent=opponent, seed=seed))
    return dataset
        