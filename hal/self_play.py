from __future__ import annotations

from dataclasses import dataclass
from multiprocessing import Pool, cpu_count

import numpy as np

from environment.opponents.base import Opponent
from environment.opponents.factory import create_scripted_opponent
from src.Constants import PHYSICALITY_BAKU, PHYSICALITY_HAL
from src.Game import Game
from src.Player import Player
from src.Referee import Referee

from .hal_opponent import CanonicalHal
from .value_net import extract_features

@dataclass(slots=True)
class Experience:
    features: np.ndarray
    outcome: float


def play_one_game(
    hal_ai: CanonicalHal,
    opponent: Opponent,
    seed: int,
) -> list[Experience]:
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


def _play_one_game_worker(args: tuple) -> list[Experience]:
    seed, opp_name, search_depth = args
    opponent = create_scripted_opponent(opp_name, seed=seed)
    hal_ai = CanonicalHal(seed=seed, depth=search_depth)
    return play_one_game(hal_ai=hal_ai, opponent=opponent, seed=seed)


def generate_dataset(
    n_games: int,
    opponent_names: list[str],
    search_depth: int = 1,
    base_seed: int = 0,
    workers: int | None = None,
) -> list[Experience]:
    tasks = [
        (base_seed + i, opponent_names[i % len(opponent_names)], search_depth)
        for i in range(n_games)
    ]

    n_workers = workers or max(1, cpu_count() - 1)

    if n_workers <= 1:
        dataset = []
        for t in tasks:
            dataset.extend(_play_one_game_worker(t))
        return dataset

    dataset = []
    with Pool(n_workers) as pool:
        for result in pool.imap_unordered(_play_one_game_worker, tasks):
            dataset.extend(result)
    return dataset
        