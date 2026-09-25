"""Recorded human game data: player splits, Hal-side observations, and priors."""

from __future__ import annotations

from collections import defaultdict

import numpy as np

from arena.contracts import CanonicalDecision
from arena.policies.bayesian_hal import ROLES


def split_games(games, protocol):
    """Hold out complete later games within each anonymous player identity."""
    players = defaultdict(list)
    for game in sorted(games, key=lambda g: g["ordinal"]):
        players[game["player"]].append(game)
    result = {}
    for player, records in players.items():
        n = len(records)
        train_end = max(1, int(n * protocol["human_train_fraction"]))
        validation_end = train_end + max(1, int(n * protocol["human_validation_fraction"]))
        if n < protocol["minimum_player_games"]:
            result[player] = {"train": records, "validation": [], "test": []}
        else:
            result[player] = {"train": records[:train_end], "validation": records[train_end:validation_end], "test": records[validation_end:]}
    return result


def human_observation(move):
    """Build Hal's decision using pre-reveal public fields, with no native state."""
    before = move["public_state_before"]
    if before["turn_duration"] != 60:
        return None
    drop, check = move["drop_second"], move["check_second"]
    if any(isinstance(a, bool) or not isinstance(a, int) or not 1 <= a <= 60 for a in (drop, check)):
        raise ValueError("invalid recorded ordinary-turn action")
    players = {p["name"]: p for p in before["players"]}
    if set(players) != {"Hal", "Baku"} or {move["dropper"], move["checker"]} != set(players):
        raise ValueError("recorded player identities are malformed")
    d, c = players[move["dropper"]], players[move["checker"]]
    hal_role = "dropper" if move["dropper"] == "Hal" else "checker"
    decision = CanonicalDecision(hal_role, "Hal", 60, tuple(range(1, 61)),
        c["cylinder_seconds"], c["ttd_seconds"], d["cylinder_seconds"], d["ttd_seconds"], None)
    return decision, ("checker" if hal_role == "dropper" else "dropper"), (check if hal_role == "dropper" else drop), (drop if hal_role == "dropper" else check)


def fit_priors(partitions, pseudocount, excluded_player=None):
    """Use training partitions only; leave the target player's identity out."""
    counts = {role: np.full(60, pseudocount) for role in ROLES}
    for player, splits in partitions.items():
        if player == excluded_player:
            continue
        for game in splits["train"]:
            for move in game["public_history"]:
                observation = human_observation(move)
                if observation is not None:
                    counts[observation[1]][observation[2] - 1] += 1
    return {role: values / values.sum() for role, values in counts.items()}
