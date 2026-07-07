"""Frozen-policy adapters for the best-response probe.

Every adapter returns a PolicyFn: (game, role) -> (seconds, probabilities),
a pure function of the public state. Adapters cache by public state so the
DP's repeated visits are free.
"""

from __future__ import annotations

import numpy as np

from stl.solver.exact import exact_public_state
from stl.engine.actions import legal_max_second
from stl.engine.game import Game


def _role_actor(game: Game, role: str):
    dropper, checker = game.get_roles_for_half(game.current_half)
    return dropper if role == "dropper" else checker


def uniform_policy():
    """Uniform over the actor's legal seconds — a sanity baseline."""

    def policy(game: Game, role: str):
        actor = _role_actor(game, role)
        max_sec = legal_max_second(actor.name, role, game.get_turn_duration())
        seconds = tuple(range(1, max_sec + 1))
        return seconds, np.full(len(seconds), 1.0 / len(seconds))

    return policy


def fixed_second_policy(second: int):
    """Always play the same second (clamped to legality) — the deterministic
    straw man the probe must expose as maximally exploitable."""

    def policy(game: Game, role: str):
        actor = _role_actor(game, role)
        max_sec = legal_max_second(actor.name, role, game.get_turn_duration())
        chosen = max(1, min(int(second), max_sec))
        return (chosen,), np.array([1.0])

    return policy


def net_policy(checkpoint_path: str):
    """The raw policy head of the trained net (legal-masked softmax).

    This is the search-free policy — its exploitability is the floor the
    MCTS agent must beat.
    """
    from stl.learning.train import load_checkpoint, make_predict_fn

    predict = make_predict_fn(load_checkpoint(checkpoint_path))
    cache: dict = {}

    def policy(game: Game, role: str):
        key = (exact_public_state(game), role)
        if key in cache:
            return cache[key]
        _, dropper_dist, checker_dist = predict(game)
        dist = dropper_dist if role == "dropper" else checker_dist
        dist = np.maximum(np.asarray(dist, dtype=np.float64), 0.0)
        support = np.nonzero(dist > 1e-12)[0]
        seconds = tuple(int(i) for i in support)
        probs = dist[support]
        probs = probs / probs.sum()
        cache[key] = (seconds, probs)
        return cache[key]

    return policy


def agent_policy(agent):
    """The deployed SolverAgent's mixture (state-seeded search; cached
    inside the agent). This measures the agent actually shipped."""

    def policy(game: Game, role: str):
        return agent.policy(game, role)

    return policy
