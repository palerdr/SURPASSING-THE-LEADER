from __future__ import annotations

import numpy as np
import torch
from torch import nn

from stl.solver.half_round import survival_probability
from stl.learning.route_math import (
    get_named_players,
    lsr_variation_from_clock,
    rounds_until_leap_window,
    safe_strategy_budget,
)
from stl.engine.game import CYLINDER_MAX, FAILED_CHECK_PENALTY, LS_WINDOW_START, OPENING_START_CLOCK
from stl.engine.game import Game

HIDDEN_DIM = 64
FEATURE_DIM = 23
REFEREE_CPR_FEATURE_SCALE = 12.0
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
        _clip01(game.referee.cprs_performed / REFEREE_CPR_FEATURE_SCALE),
        float(lsr_variation == 2),
        _clip01(proximity_to_leap),
    ]

    assert len(features) == FEATURE_DIM
    return np.asarray(features, dtype=np.float32)


class ValueNet(nn.Module):
    def __init__(self, input_dim: int = FEATURE_DIM, hidden_dim: int = HIDDEN_DIM):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, 1),
            nn.Tanh(),
        )
        self.policy_head = nn.Linear(hidden_dim, 122)

    def forward(self, x):
        hidden = self.trunk(x)
        value = self.value_head(hidden)
        policy_logits = self.policy_head(hidden)
        dropper_logits = policy_logits[..., :61]
        checker_logits = policy_logits[..., 61:]
        return value, dropper_logits, checker_logits


def value_output(output) -> torch.Tensor:
    """Return the scalar value tensor from a ValueNet-style output."""
    if isinstance(output, tuple):
        return output[0]
    return output
