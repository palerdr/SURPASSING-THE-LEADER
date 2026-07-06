"""Observation builder for DTH environment.

Translates engine game state into a fixed-size numpy array that the RL
agent receives as input. The observation is always from one player's
*perspective* — "my" cylinder, "my" TTD, etc.

Key design decisions:
    - All values normalized to roughly [0, 1] for neural network stability.
    - LSR features (variation, rounds_until_leap, is_leap_turn) are masked
      to zero for Hal until he has died at least once.
    - Baku always sees full LSR information.

Observation vector layout (20 features):
    [0]  my_cylinder          / 300
    [1]  opp_cylinder         / 300
    [2]  my_ttd               / 600
    [3]  opp_ttd              / 600
    [4]  my_deaths            / 10
    [5]  opp_deaths           / 10
    [6]  referee_cprs         / 10
    [7]  my_role              0=checker, 1=dropper
    [8]  round_index          / 15
    [9]  half_index           0=first, 1=second
    [10] my_safe_checks       / 5
    [11] opp_safe_checks      / 5
    [12] clock_seconds        / 4200
    [13-16] lsr_variation     one-hot V1/V2/V3/V4
    [17] rounds_until_leap    / 15
    [18] is_leap_turn         binary
    [19] leap_known           binary (has this player unlocked LS knowledge?)
"""

from __future__ import annotations

import numpy as np
from src.Game import Game
from src.Player import Player
from src.Constants import (
    CYLINDER_MAX, LS_WINDOW_START, TURN_DURATION_NORMAL,
)
from .routing_features import (
    build_public_opponent_history_features,
    build_full_public_action_history,
    ROUTING_FEATURE_SIZE,
    FULL_HISTORY_SIZE,
)

OBS_SIZE = 20
OBS_V2_SIZE = 24  # v1 (20) + public opponent history features (4)
OBS_V3_SIZE = OBS_V2_SIZE + FULL_HISTORY_SIZE  # v2 (24) + full padded history (60) = 84


def compute_lsr_variation(game: Game) -> int:
    """Compute current LSR variation (1-4) from round start time.

    The four variations cycle based on which minute-mod-4 slot the round
    starts in, relative to the base rhythm. Only V2 is "Active LSR"
    where Baku is D during the leap second turn.

    Args:
        game: Current game state.

    Returns:
        Integer 1, 2, 3, or 4.
    """
    minutes_since_start = int(game.game_clock) // 60

    match minutes_since_start % 4:
        case 0:
            return 1
        #active LSR
        case 1:
            return 2
        case 2:
            return 3
        case 3:
            return 4


def compute_rounds_until_leap(game: Game) -> int:
    """Estimate how many rounds remain before the leap second window.
    Args:
        game: Current game state.

    Returns:
        Non-negative integer. 0 means we are at or past the leap window.
    """

    return max(0, int ((LS_WINDOW_START - game.game_clock) // (TURN_DURATION_NORMAL * 4)))

def build_observation(
    game: Game,
    perspective: Player,
    opponent: Player,
    leap_known: bool,
) -> np.ndarray:
    """Build the full observation vector from one player's perspective.

    This is the function the environment calls every step. It must return
    a numpy array of shape (OBS_SIZE,) with float32 values.

    [0]  my_cylinder / 300
    [1]  opp_cylinder / 300
    [2]  my_ttd / 600
    [3]  opp_ttd / 600
    [4]  my_deaths / 10
    [5]  opp_deaths / 10
    [6]  referee_cprs / 10
    [7]  my_role (0=checker, 1=dropper)
    [8]  round_index / 15
    [9]  half_index (0=first, 1=second)
    [10]  my_safe_checks / 5
    [11]  opp_safe_checks / 5
    [12]  clock_seconds / 4200
    [13]  lsr_v1 (one-hot)
    [14]  lsr_v2 (one-hot)
    [15]  lsr_v3 (one-hot)
    [16]  lsr_v4 (one-hot)
    [17]  rounds_until_leap / 15
    [18]  is_leap_turn (binary)
    [19]  leap_known (binary)

    Args:
        game: Current game state.
        perspective: The player whose POV we're building for.
        opponent: The other player.
        leap_known: Whether this player has unlocked leap second knowledge.\

    
    Returns:
        np.ndarray of shape (20,), dtype float32.
    """
    
    obs = np.zeros(OBS_SIZE, dtype = np.float32)

    obs[0] = perspective.cylinder / 300.0
    
    obs[1] = opponent.cylinder / 300.0

    obs[2] = perspective.ttd / 600.0

    obs[3] = opponent.ttd / 600.0

    obs[4] = perspective.deaths / 10.0

    obs[5] = opponent.deaths / 10.0

    obs[6] = game.referee.cprs_performed / 10.0

    dropper, checker = game.get_roles_for_half(game.current_half)
    obs[7] = 1.0 if perspective is dropper else 0.0

    obs[8] = game.round_num / 10.0

    obs[9] = 0.0 if game.current_half == 1 else 1.0
    
    obs[10] = perspective.safe_strategies_remaining / 5.0

    obs[11] = opponent.safe_strategies_remaining / 5.0

    obs[12] = game.game_clock / 4200.0

    if leap_known:
        lsr_idx = compute_lsr_variation(game)
        obs[12 + lsr_idx] = 1.0
        
        obs[17] = compute_rounds_until_leap(game) / 15.0

        obs[18] = 1.0 if game.is_leap_second_turn() else 0.0

    obs[19] = 1.0 if leap_known else 0.0

    return obs


def build_observation_v2(
    game: Game,
    perspective: Player,
    opponent: Player,
    leap_known: bool,
) -> np.ndarray:
    """Build v2 observation: v1 (20 dims) + public opponent history (4 dims).

    Indices 0-19: identical to v1 (see build_observation).
    Indices 20-23: public opponent action history features from routing_features.

    Returns:
        np.ndarray of shape (24,), dtype float32.
    """
    base = build_observation(game, perspective, opponent, leap_known)
    history = build_public_opponent_history_features(game, perspective, opponent)
    return np.concatenate((base, history))


def build_observation_v3(
    game: Game,
    perspective: Player,
    opponent: Player,
    leap_known: bool,
) -> np.ndarray:
    """Build v3 observation: v2 (24 dims) + full padded public action history (60 dims).

    Indices 0-23:  identical to v2 (see build_observation_v2).
    Indices 24-83: full public action history, zero-padded to 30 half-rounds.

    This is infrastructure for a future observation-v3 retrain. Not used as
    the promoted policy observation this sprint.

    Returns:
        np.ndarray of shape (84,), dtype float32.
    """
    v2 = build_observation_v2(game, perspective, opponent, leap_known)
    full_history = build_full_public_action_history(game, perspective, opponent)
    return np.concatenate((v2, full_history))