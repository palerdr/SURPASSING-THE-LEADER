"""Offline comparison of a candidate prior with production. This module uses no Dagster.

Each player's score is the Dirichlet-multinomial log-likelihood of the player's
test moves after the prior absorbs the player's training and validation moves.
The validation split chose the prior's strength, so the score reads the test split alone.
The headline metric is the mean log-likelihood per human move.
The bootstrap resamples players, because one player's moves are not independent.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from hal_loop.prior import ROLES, role_log_likelihood, uniform_prior
from hal_loop.warehouse import Histograms

BOOTSTRAP_RESAMPLES = 2000
BOOTSTRAP_SEED = 20261017
INTERVAL = (2.5, 97.5)
SCORED_SPLIT = "test"
HISTORY_SPLITS = ("train", "validation")


def score_matrix(histograms: Histograms, priors: Mapping[str, Mapping], players: list[str]) -> dict:
    """Return log-likelihood and move-count arrays indexed [role][prior][player]."""
    index = {player: i for i, player in enumerate(players)}
    result = {}
    for role in ROLES:
        moves = np.zeros(len(players))
        scores = {name: np.zeros(len(players)) for name in priors}
        for player, counts in histograms.player_counts(role, HISTORY_SPLITS[0], SCORED_SPLIT):
            i = index[player]
            history = sum(histograms.get(player, role, split) for split in HISTORY_SPLITS)
            moves[i] = counts.held_out.sum()
            for name, prior in priors.items():
                scores[name][i] = role_log_likelihood(prior, role, counts.held_out, history)
        result[role] = {"moves": moves, "scores": scores}
    return result


def _per_move(total: np.ndarray, moves: np.ndarray) -> np.ndarray:
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(moves > 0, total / np.where(moves > 0, moves, 1), np.nan)


def _difference(scores: dict, rows: np.ndarray, roles) -> np.ndarray:
    """Return the per-move candidate-minus-production difference for each resample row."""
    delta = sum(scores[r]["scores"]["candidate"][rows] - scores[r]["scores"]["production"][rows] for r in roles)
    moves = sum(scores[r]["moves"][rows] for r in roles)
    return _per_move(delta.sum(axis=-1), moves.sum(axis=-1))


def _summary(scores: dict, roles, draws: np.ndarray) -> dict:
    everyone = np.arange(draws.shape[1])[None, :]
    moves = float(sum(scores[r]["moves"].sum() for r in roles))
    means = {
        name: (float(sum(scores[r]["scores"][name].sum() for r in roles)) / moves if moves else None)
        for name in ("candidate", "uniform", "production")
    }
    boot = _difference(scores, draws, roles)
    finite = boot[np.isfinite(boot)]
    low, high = (float(v) for v in np.percentile(finite, INTERVAL)) if finite.size else (None, None)
    point = _difference(scores, everyone, roles)[0]
    return {
        "moves": moves,
        "log_likelihood_per_move": means,
        "difference": float(point) if np.isfinite(point) else None,
        "interval": [low, high],
    }


def evaluate_candidate(
    histograms: Histograms,
    candidate: Mapping,
    production: Mapping | None,
    *,
    resamples: int = BOOTSTRAP_RESAMPLES,
    seed: int = BOOTSTRAP_SEED,
) -> dict:
    """Score the candidate, uniform, and production priors on the test split."""
    players = histograms.players(SCORED_SPLIT)
    uniform = uniform_prior()
    priors = {"candidate": candidate, "uniform": uniform, "production": production or uniform}
    scores = score_matrix(histograms, priors, players)
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(players), size=(resamples, len(players))) if players else np.zeros((resamples, 0), int)
    return {
        "split": SCORED_SPLIT,
        "history_splits": list(HISTORY_SPLITS),
        "players": len(players),
        "production_source": "current" if production is not None else "uniform",
        "bootstrap": {"resamples": resamples, "seed": seed, "cluster": "player_id", "interval_percentiles": list(INTERVAL)},
        "roles": {role: _summary(scores, (role,), draws) for role in ROLES},
        "pooled": _summary(scores, ROLES, draws),
    }
