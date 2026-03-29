"""Public-history routing features for opponent discrimination.

This module is deliberately separate from the main observation builder so we can:
1. enrich classifier / routing inputs immediately without invalidating policy shapes
2. later promote the same features into a full observation-v2 retrain

All features here are derived from public game history only.
"""

from __future__ import annotations

import numpy as np

from src.Game import Game
from src.Player import Player


ROUTING_HISTORY_WINDOW = 2
ROUTING_FEATURE_SIZE = ROUTING_HISTORY_WINDOW * 2


def _public_action_for_player(record, player: Player) -> tuple[float, float]:
    """Return normalized public action + role flag for ``player`` in one record.

    Role flag: 1.0 if the player was dropper, 0.0 if checker.
    Action is normalized by actual turn duration, preserving leap-turn semantics.
    """
    if record.dropper == player.name:
        return record.drop_time / float(record.turn_duration), 1.0
    if record.checker == player.name:
        return record.check_time / float(record.turn_duration), 0.0
    raise ValueError(f"Player {player.name} not present in record")


def build_public_opponent_history_features(
    game: Game,
    perspective: Player,
    opponent: Player,
    *,
    history_window: int = ROUTING_HISTORY_WINDOW,
) -> np.ndarray:
    """Build compact public-history features for routing/classification.

    Output layout for window=2:
    [0] last_opponent_action_norm
    [1] last_opponent_role_flag
    [2] prev_opponent_action_norm
    [3] prev_opponent_role_flag

    Missing history is zero-padded.
    """
    features = np.zeros(history_window * 2, dtype=np.float32)
    if not game.history:
        return features

    relevant = []
    for record in reversed(game.history):
        try:
            relevant.append(_public_action_for_player(record, opponent))
        except ValueError:
            continue
        if len(relevant) >= history_window:
            break

    for idx, (action_norm, role_flag) in enumerate(relevant):
        base = idx * 2
        features[base] = action_norm
        features[base + 1] = role_flag

    return features


def build_routing_feature_vector(
    base_observation: np.ndarray,
    game: Game,
    perspective: Player,
    opponent: Player,
    *,
    history_window: int = ROUTING_HISTORY_WINDOW,
) -> np.ndarray:
    """Append public-history routing features to an existing observation.

    This is the intended bridge to a future observation-v2 migration: callers can
    use the enriched vector today for routing/classification, while the policy can
    keep consuming the legacy observation shape.
    """
    history_features = build_public_opponent_history_features(
        game,
        perspective,
        opponent,
        history_window=history_window,
    )
    return np.concatenate((base_observation.astype(np.float32), history_features))
