"""Fit and score the adaptive Hal's role prior. This module uses no Dagster.

The prior is one Dirichlet per human role over the seconds 1..60.
Its mean is the smoothed population histogram of the training split.
Its strength is the effective sample size that best predicts held-out play.
The selection scores each player with a leave-one-player-out prior.
That prior is updated with the player's training moves and scored on the player's validation moves.
The score is the exact Dirichlet-multinomial probability of those moves.
"""

from __future__ import annotations

import json
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.special import gammaln

SCHEMA = "adaptive-dth-role-prior-v1"
ROLES = ("dropper", "checker")
ACTIONS = 60
STRENGTH_GRID = (1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0)
DEFAULT_PSEUDOCOUNT = 0.5
# A role with no held-out moves takes the weakest grid strength.
FALLBACK_STRENGTH = STRENGTH_GRID[0]


@dataclass(frozen=True)
class PlayerCounts:
    """One player's per-second move counts in the training and held-out splits."""

    train: np.ndarray
    held_out: np.ndarray


def as_counts(raw) -> np.ndarray:
    """Return a float array of 60 nonnegative counts."""
    counts = np.asarray(raw, dtype=np.float64)
    if counts.shape != (ACTIONS,) or np.any(counts < 0) or not np.all(np.isfinite(counts)):
        raise ValueError("counts must be 60 finite nonnegative numbers")
    return counts


def seconds_to_counts(seconds: Sequence[int]) -> np.ndarray:
    """Count literal seconds 1..60 into a histogram."""
    values = np.asarray(list(seconds), dtype=np.int64)
    if values.size and (values.min() < 1 or values.max() > ACTIONS):
        raise ValueError("moves must be literal seconds in 1..60")
    return np.bincount(values - 1, minlength=ACTIONS).astype(np.float64)


def dirichlet_multinomial_log_likelihood(alpha: np.ndarray, counts: np.ndarray) -> float:
    """Return the log probability of one ordered move sequence with these counts."""
    total = float(alpha.sum())
    n = float(counts.sum())
    return float(gammaln(total) - gammaln(total + n) + np.sum(gammaln(alpha + counts) - gammaln(alpha)))


def smoothed_mean(counts: np.ndarray, pseudocount: float) -> np.ndarray:
    smoothed = counts + pseudocount
    return smoothed / smoothed.sum()


def strength_scores(
    total: np.ndarray,
    players: Sequence[PlayerCounts],
    pseudocount: float = DEFAULT_PSEUDOCOUNT,
    grid: Sequence[float] = STRENGTH_GRID,
) -> dict[float, float]:
    """Return the summed held-out log-likelihood of each grid strength."""
    scores = {float(strength): 0.0 for strength in grid}
    for player in players:
        if player.held_out.sum() == 0:
            continue
        mean = smoothed_mean(np.clip(total - player.train, 0.0, None), pseudocount)
        for strength in scores:
            alpha = strength * mean + player.train
            scores[strength] += dirichlet_multinomial_log_likelihood(alpha, player.held_out)
    return scores


def select_strength(
    total: np.ndarray,
    players: Sequence[PlayerCounts],
    pseudocount: float = DEFAULT_PSEUDOCOUNT,
    grid: Sequence[float] = STRENGTH_GRID,
) -> tuple[float, dict[float, float]]:
    """Pick the grid strength with the highest held-out score; ties go to the weaker prior."""
    if not any(player.held_out.sum() for player in players):
        return FALLBACK_STRENGTH, {}
    scores = strength_scores(total, players, pseudocount, grid)
    best = max(scores.values())
    return min(strength for strength, score in scores.items() if score == best), scores


def fit_role_prior(
    counts_by_role: Mapping[str, np.ndarray],
    pseudocount: float = DEFAULT_PSEUDOCOUNT,
    *,
    players_by_role: Mapping[str, Sequence[PlayerCounts]] | None = None,
    grid: Sequence[float] = STRENGTH_GRID,
) -> dict:
    """Return an "adaptive-dth-role-prior-v1" dict fitted on training counts."""
    prior: dict = {"schema_version": SCHEMA}
    for role in ROLES:
        total = as_counts(counts_by_role[role])
        players = (players_by_role or {}).get(role, ())
        strength, _ = select_strength(total, players, pseudocount, grid)
        prior[role] = {
            "mean": [float(value) for value in smoothed_mean(total, pseudocount)],
            "strength": float(strength),
        }
    return prior


def uniform_prior(strength: float = 1.0) -> dict:
    """Return the neutral prior that the adaptive provider uses without a prior file."""
    role = {"mean": [1.0 / ACTIONS] * ACTIONS, "strength": float(strength)}
    return {"schema_version": SCHEMA, "dropper": dict(role), "checker": dict(role)}


def role_alpha(prior: Mapping, role: str) -> np.ndarray:
    entry = prior[role]
    mean = np.asarray(entry["mean"], dtype=np.float64)
    return float(entry["strength"]) * mean / mean.sum()


def role_log_likelihood(prior: Mapping, role: str, counts, history=None) -> float:
    """Score held-out counts for one role after updating the prior with history counts."""
    alpha = role_alpha(prior, role)
    if history is not None:
        alpha = alpha + as_counts(history)
    return dirichlet_multinomial_log_likelihood(alpha, as_counts(counts))


def validate_prior(prior: Mapping):
    """Load the prior through the live loader and return the opponent model it builds."""
    from arena.policies.adaptive import load_opponent_model

    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "prior.json"
        path.write_text(json.dumps(prior, sort_keys=True), encoding="utf-8")
        return load_opponent_model(path)


class HistogramEmulator:
    """A stateless human emulator that plays each role's smoothed histogram."""

    def __init__(self, counts_by_role: Mapping[str, np.ndarray], pseudocount: float = DEFAULT_PSEUDOCOUNT):
        self.distributions = {
            role: smoothed_mean(as_counts(counts_by_role[role]), pseudocount) for role in ROLES
        }

    def policy(self, decision) -> dict[int, float]:
        """Return the role's distribution over seconds 1..60 in the provider format."""
        return {second + 1: float(p) for second, p in enumerate(self.distributions[decision.role])}
