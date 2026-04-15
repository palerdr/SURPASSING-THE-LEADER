from __future__ import annotations

import numpy as np
import torch

from src.Player import Player
from src.Constants import (
    CYLINDER_MAX,
    FAILED_CHECK_PENALTY,
    LS_WINDOW_START,
    OPENING_START_CLOCK,
    PHYSICALITY_BAKU,
    CARDIAC_DECAY,
)
from src.Game import Game
from environment.cfr.half_round import survival_probability
from environment.route_math import get_named_players, lsr_variation_from_clock, safe_strategy_budget

_nn_model = None
_nn_device = None


def set_nn_evaluator(net) -> None:
    global _nn_model, _nn_device
    if net is not None:
        from .value_net import DEVICE
        net = net.to(DEVICE)
        net.eval()
        _nn_device = DEVICE
    _nn_model = net


def get_nn_model():
    return _nn_model


def get_nn_device():
    return _nn_device


def _hal_terminal_value(game: Game) -> float:
    if game.winner is None:
        return 0.0
    return 1.0 if game.winner.name.lower() == "hal" else -1.0


def evaluate(game: Game) -> float:
    if game.game_over:
        return _hal_terminal_value(game)

    if _nn_model is not None:
        from .value_net import extract_features
        features = extract_features(game)
        with torch.no_grad():
            tensor = torch.tensor(features, device=_nn_device).unsqueeze(0)
            value = _nn_model(tensor).item()
        return value

    return _handcrafted_evaluate(game)


def evaluate_batch(games: list[Game]) -> np.ndarray:
    terminal_mask = []
    terminal_vals = []
    non_terminal_indices = []
    non_terminal_features = []

    from .value_net import extract_features

    for i, game in enumerate(games):
        if game.game_over:
            terminal_mask.append(True)
            terminal_vals.append(_hal_terminal_value(game))
        else:
            terminal_mask.append(False)
            terminal_vals.append(0.0)
            non_terminal_indices.append(i)
            non_terminal_features.append(extract_features(game))

    values = np.array(terminal_vals, dtype=np.float64)

    if non_terminal_features and _nn_model is not None:
        batch = torch.tensor(np.stack(non_terminal_features), device=_nn_device)
        with torch.no_grad():
            preds = _nn_model(batch).squeeze(-1).cpu().numpy()
        for idx, pred in zip(non_terminal_indices, preds):
            values[idx] = float(pred)
    elif non_terminal_features:
        for idx in non_terminal_indices:
            values[idx] = _handcrafted_evaluate(games[idx])

    return values


def _handcrafted_evaluate(game: Game) -> float:
    hal, baku = get_named_players(game=game)
    ref = game.referee

    def fail_death_dur(player: Player) -> float:
        return min(player.cylinder + FAILED_CHECK_PENALTY, CYLINDER_MAX)

    hal_dd = fail_death_dur(hal)
    baku_dd = fail_death_dur(baku)
    hal_surv = survival_probability(hal_dd, hal.ttd, ref.cprs_performed, hal.physicality)
    baku_surv = survival_probability(baku_dd, baku.ttd, ref.cprs_performed, baku.physicality)

    hal_budget = safe_strategy_budget(hal)
    baku_budget = safe_strategy_budget(baku)

    hal_cardiac = CARDIAC_DECAY ** (hal.ttd / 60.0)
    baku_cardiac = CARDIAC_DECAY ** (baku.ttd / 60.0)

    var = lsr_variation_from_clock(game.game_clock)
    dropper, _ = game.get_roles_for_half(game.current_half)
    hal_is_dropper = dropper.name.lower() == "hal"

    if var == 2 and hal_is_dropper:
        lsr_favor = 0.8
    elif var == 2 and not hal_is_dropper:
        lsr_favor = -0.5
    else:
        lsr_favor = 0.0

    dist_to_leap = max(0.0, LS_WINDOW_START - game.game_clock)
    proximity = 1.0 - (dist_to_leap / (LS_WINDOW_START - OPENING_START_CLOCK))

    score = 0.0
    score += 0.4 * (baku.cylinder - hal.cylinder) / CYLINDER_MAX
    score += 0.5 * (hal_budget - baku_budget) / 5.0
    score += 0.6 * (hal_surv - baku_surv)
    score += 0.3 * (baku.deaths - hal.deaths) / 4.0
    score += 0.5 * (hal_cardiac - baku_cardiac)
    score += 0.4 * lsr_favor
    score += 0.1 * proximity
    if ref.cprs_performed > 0:
        score += 0.05 * ref.cprs_performed * (1.0 - PHYSICALITY_BAKU)

    return max(-1.0, min(1.0, score))
