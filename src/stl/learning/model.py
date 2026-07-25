from __future__ import annotations

import numpy as np
import torch
from torch import nn

from stl.solver.exact import survival_probability
from stl.learning.route_math import (
    get_named_players,
    lsr_variation_from_clock,
    rounds_until_leap_window,
    safe_strategy_budget,
)
from stl.engine.game import CYLINDER_MAX, FAILED_CHECK_PENALTY, LS_WINDOW_START, OPENING_START_CLOCK
from stl.engine.game import Game
from stl.engine.actions import ACTION_SIZE

HIDDEN_DIM = 64
LEGACY_FEATURE_SCHEMA_VERSION = "stl.features.v1"
FEATURE_SCHEMA_VERSION = "stl.features.v2"
LEGACY_FEATURE_DIM = 23
REFEREE_CPR_FEATURE_SCALE = 12.0
DEVICE = torch.device("cpu")

EXACT_FEATURE_NAMES = (
    "p1_is_hal",
    "p1_is_baku",
    "p1_physicality",
    "p1_cylinder",
    "p1_ttd",
    "p1_deaths",
    "p1_alive",
    "p2_is_hal",
    "p2_is_baku",
    "p2_physicality",
    "p2_cylinder",
    "p2_ttd",
    "p2_deaths",
    "p2_alive",
    "referee_cprs",
    "game_clock",
    "current_half_1",
    "current_half_2",
    "round_num",
    "first_dropper_p1",
    "first_dropper_p2",
    "first_dropper_none",
    "game_over",
    "winner_p1",
    "winner_p2",
    "winner_none",
    "loser_p1",
    "loser_p2",
    "loser_none",
)

DERIVED_FEATURE_NAMES = (
    "derived_hal_cylinder",
    "derived_baku_cylinder",
    "derived_hal_ttd",
    "derived_baku_ttd",
    "derived_hal_deaths",
    "derived_baku_deaths",
    "derived_hal_safe_budget",
    "derived_baku_safe_budget",
    "derived_hal_fail_survival",
    "derived_baku_fail_survival",
    "derived_hal_is_dropper",
    "derived_game_clock",
    "derived_round_num",
    "derived_half_2",
    "derived_lsr_variation_1",
    "derived_lsr_variation_2",
    "derived_lsr_variation_3",
    "derived_lsr_variation_4",
    "derived_is_leap_turn",
    "derived_rounds_until_leap",
    "derived_referee_cprs",
    "derived_lsr_variation_2_repeat",
    "derived_proximity_to_leap",
)

FEATURE_NAMES = EXACT_FEATURE_NAMES + DERIVED_FEATURE_NAMES
FEATURE_INDEX = {name: index for index, name in enumerate(FEATURE_NAMES)}
FEATURE_DIM = len(FEATURE_NAMES)
HORIZON_SCHEMA_VERSION = "stl.value-horizon.v1"
SUPPORTED_HORIZONS = (0, 2, 3)
HORIZON_DIM = len(SUPPORTED_HORIZONS)
MODEL_INPUT_DIM = FEATURE_DIM + HORIZON_DIM


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


def extract_features_v1(game: Game) -> np.ndarray:
    """Return the legacy lossy 23-float feature vector.

    V1 remains available only as an explicitly named adapter for artifact
    inspection.  New targets and checkpoints use :func:`extract_features`.
    """

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

    assert len(features) == LEGACY_FEATURE_DIM
    return np.asarray(features, dtype=np.float32)


def _identity_one_hot(name: str) -> tuple[float, float]:
    lowered = name.lower()
    if lowered == "hal":
        return 1.0, 0.0
    if lowered == "baku":
        return 0.0, 1.0
    raise ValueError(f"feature schema V2 requires canonical Hal/Baku names, got {name!r}")


def _player_reference_one_hot(game: Game, player) -> tuple[float, float, float]:
    if player is None:
        return 0.0, 0.0, 1.0
    if player is game.player1 or player.name == game.player1.name:
        return 1.0, 0.0, 0.0
    if player is game.player2 or player.name == game.player2.name:
        return 0.0, 1.0, 0.0
    raise ValueError(f"player reference {player.name!r} is not part of this game")


def extract_features_v2(game: Game) -> np.ndarray:
    """Encode exact public state fields followed by documented derivatives.

    The exact prefix avoids the clipping collisions in V1.  Scale factors are
    linear and deliberately unclipped; they improve numerical conditioning
    without identifying distinct reachable values.
    """

    p1_hal, p1_baku = _identity_one_hot(game.player1.name)
    p2_hal, p2_baku = _identity_one_hot(game.player2.name)
    first = _player_reference_one_hot(game, game.first_dropper)
    winner = _player_reference_one_hot(game, game.winner)
    loser = _player_reference_one_hot(game, game.loser)
    exact = [
        p1_hal,
        p1_baku,
        float(game.player1.physicality),
        float(game.player1.cylinder) / CYLINDER_MAX,
        float(game.player1.ttd) / CYLINDER_MAX,
        float(game.player1.deaths) / 4.0,
        float(game.player1.alive),
        p2_hal,
        p2_baku,
        float(game.player2.physicality),
        float(game.player2.cylinder) / CYLINDER_MAX,
        float(game.player2.ttd) / CYLINDER_MAX,
        float(game.player2.deaths) / 4.0,
        float(game.player2.alive),
        float(game.referee.cprs_performed) / REFEREE_CPR_FEATURE_SCALE,
        float(game.game_clock) / 3600.0,
        float(game.current_half == 1),
        float(game.current_half == 2),
        float(game.round_num) / 10.0,
        *first,
        float(game.game_over),
        *winner,
        *loser,
    ]
    features = np.concatenate(
        (
            np.asarray(exact, dtype=np.float32),
            extract_features_v1(game),
        )
    )
    assert features.shape == (FEATURE_DIM,)
    return features


def extract_features(game: Game) -> np.ndarray:
    """Return the active, versioned V2 feature vector."""

    return extract_features_v2(game)


class ValueNet(nn.Module):
    def __init__(self, input_dim: int = FEATURE_DIM, hidden_dim: int = HIDDEN_DIM):
        super().__init__()
        if input_dim != FEATURE_DIM:
            raise ValueError(
                f"ValueNet physical input_dim must be {FEATURE_DIM}, got {input_dim}"
            )
        self.trunk = nn.Sequential(
            nn.Linear(MODEL_INPUT_DIM, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, 1),
            nn.Tanh(),
        )
        self.policy_head = nn.Linear(hidden_dim, 2 * ACTION_SIZE)

    def forward(self, x: torch.Tensor, horizon: int | torch.Tensor):
        """Predict a value and role policies under an explicit value horizon.

        Horizon is evaluator context, not part of the physical Markov state.
        Requiring it at every call prevents the old mixed-horizon corpus from
        training one ambiguous scalar function over identical state features.
        """

        if x.shape[-1] != FEATURE_DIM:
            raise ValueError(
                f"ValueNet requires {FEATURE_DIM} physical features, got {x.shape[-1]}"
            )
        horizon_tensor = torch.as_tensor(horizon, device=x.device)
        if horizon_tensor.ndim == 0:
            horizon_tensor = horizon_tensor.expand(x.shape[:-1])
        if tuple(horizon_tensor.shape) != tuple(x.shape[:-1]):
            raise ValueError(
                "horizon shape must be scalar or match the feature batch dimensions"
            )
        valid = torch.zeros_like(horizon_tensor, dtype=torch.bool)
        horizon_parts = []
        for supported in SUPPORTED_HORIZONS:
            selected = horizon_tensor == supported
            valid |= selected
            horizon_parts.append(selected.to(dtype=x.dtype))
        if not bool(torch.all(valid)):
            invalid = torch.unique(horizon_tensor[~valid]).detach().cpu().tolist()
            raise ValueError(
                f"unsupported value horizon(s) {invalid}; "
                f"supported={SUPPORTED_HORIZONS}"
            )
        horizon_features = torch.stack(horizon_parts, dim=-1)
        hidden = self.trunk(torch.cat((x, horizon_features), dim=-1))
        value = self.value_head(hidden)
        policy_logits = self.policy_head(hidden)
        dropper_logits = policy_logits[..., :ACTION_SIZE]
        checker_logits = policy_logits[..., ACTION_SIZE:]
        return value, dropper_logits, checker_logits


def value_output(output) -> torch.Tensor:
    """Return the scalar value tensor from a ValueNet-style output."""
    if isinstance(output, tuple):
        return output[0]
    return output
