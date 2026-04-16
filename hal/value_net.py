from __future__ import annotations

import numpy as np
import torch
from torch import nn

from environment.cfr.half_round import survival_probability
from environment.route_math import (
    get_named_players,
    lsr_variation_from_clock,
    rounds_until_leap_window,
    safe_strategy_budget,
)
from src.Constants import CYLINDER_MAX, FAILED_CHECK_PENALTY, LS_WINDOW_START, OPENING_START_CLOCK
from src.Game import Game

HIDDEN_DIM = 64
FEATURE_DIM = 23
DEVICE = torch.device("cpu")


def _clip01(value: float) -> float:
    return float(np.clip(value, 0.0, 1.0))


def _projected_fail_survival(game: Game, player) -> float:
    death_duration = min(player.cylinder + FAILED_CHECK_PENALTY, CYLINDER_MAX)
    return survival_probability(
        death_duration,
        player.ttd,
        game.referee.cprs_performed,
        player.physicality,
    )


def extract_features(game: Game) -> np.ndarray:
    hal, baku = get_named_players(game)
    dropper, _ = game.get_roles_for_half(game.current_half)
    hal_is_dropper = dropper.name.lower() == "hal"

    lsr_variation = lsr_variation_from_clock(game.game_clock)
    lsr_one_hot = [1.0 if lsr_variation == idx else 0.0 for idx in range(1, 5)]

    dist_to_leap = max(0.0, LS_WINDOW_START - game.game_clock)
    leap_span = max(1.0, LS_WINDOW_START - OPENING_START_CLOCK)
    proximity_to_leap = 1.0 - (dist_to_leap / leap_span)

    features = [
        _clip01(hal.cylinder / CYLINDER_MAX),
        _clip01(baku.cylinder / CYLINDER_MAX),
        _clip01(hal.ttd / CYLINDER_MAX),
        _clip01(baku.ttd / CYLINDER_MAX),
        _clip01(hal.deaths / 4.0),
        _clip01(baku.deaths / 4.0),
        _clip01(safe_strategy_budget(hal) / 5.0),
        _clip01(safe_strategy_budget(baku) / 5.0),
        _clip01(_projected_fail_survival(game, hal)),
        _clip01(_projected_fail_survival(game, baku)),
        float(hal_is_dropper),
        _clip01(game.game_clock / 3600.0),
        _clip01(game.round_num / 10.0),
        float(game.current_half == 2),
        *lsr_one_hot,
        float(game.is_leap_second_turn()),
        _clip01(rounds_until_leap_window(game) / 10.0),
        _clip01(game.referee.cprs_performed / 6.0),
        float(lsr_variation == 2),
        _clip01(proximity_to_leap),
    ]

    assert len(features) == FEATURE_DIM
    return np.asarray(features, dtype=np.float32)


class ValueNet(nn.Module):
    def __init__(self, input_dim: int = FEATURE_DIM, hidden_dim: int = HIDDEN_DIM):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Tanh(),
        )

    def forward(self, x):
        return self.layers(x)
