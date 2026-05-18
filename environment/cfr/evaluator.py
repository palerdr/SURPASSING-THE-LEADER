"""Leaf evaluators for MCTS.

When MCTS reaches a leaf, it needs a Hal-perspective value in [-1, 1].
This module defines the LeafEvaluator protocol plus concrete evaluators:

    - TerminalOnlyEvaluator: returns terminal_value or 0.0 for unresolved.
    - TablebaseEvaluator: short-circuits on positions matching pinned tablebase
        entries; falls back to a wrapped evaluator otherwise.
    - ValueNetEvaluator: wraps a trained value net behind the protocol.
"""

from typing import Callable, Protocol

import numpy as np

from environment.legal_actions import legal_max_second

from .exact import ExactPublicState, exact_public_state
from .tablebase import REGISTRY
from .exact import terminal_value
from src.Game import Game


LeafEvaluation = tuple[float, np.ndarray, np.ndarray]


def _uniform_over_legal(max_second: int) -> np.ndarray:
    dist = np.zeros(61, dtype=np.float64)
    if max_second <= 0:
        return dist
    dist[:max_second] = 1.0 / max_second
    return dist


def uniform_policy_for_current_roles(game: Game) -> tuple[np.ndarray, np.ndarray]:
    """Return length-61 uniform distributions over legal root seconds."""
    if game.game_over:
        return np.zeros(61, dtype=np.float64), np.zeros(61, dtype=np.float64)

    dropper, checker = game.get_roles_for_half(game.current_half)
    turn_duration = game.get_turn_duration()
    drop_max = legal_max_second(dropper.name, "dropper", turn_duration)
    check_max = legal_max_second(checker.name, "checker", turn_duration)
    return _uniform_over_legal(drop_max), _uniform_over_legal(check_max)


def normalize_policy_vector(policy: np.ndarray | list[float] | tuple[float, ...]) -> np.ndarray:
    arr = np.asarray(policy, dtype=np.float64).reshape(-1)
    if arr.shape[0] != 61:
        raise ValueError(f"policy vector must have length 61, got {arr.shape[0]}")
    arr = np.maximum(arr, 0.0)
    total = float(arr.sum())
    if total > 1e-12:
        arr = arr / total
    return arr


def normalize_leaf_evaluation(value, game: Game) -> LeafEvaluation:
    """Coerce scalar legacy outputs or explicit triples into a leaf triple."""
    if isinstance(value, tuple):
        if len(value) != 3:
            raise ValueError("leaf evaluator tuples must be (value, dropper_policy, checker_policy)")
        scalar, dropper_policy, checker_policy = value
        return (
            float(scalar),
            normalize_policy_vector(dropper_policy),
            normalize_policy_vector(checker_policy),
        )

    dropper_policy, checker_policy = uniform_policy_for_current_roles(game)
    return float(value), dropper_policy, checker_policy


class LeafEvaluator(Protocol):
    def __call__(self, game: Game) -> LeafEvaluation: ...


class TerminalOnlyEvaluator:
    """Returns terminal_value(game) for terminal positions, 0.0 otherwise."""

    def __init__(self, perspective_name: str = "Hal") -> None:
        self.perspective = perspective_name

    def __call__(self, game: Game) -> LeafEvaluation:
        tval = terminal_value(game, perspective_name=self.perspective)
        if tval is not None:
            value = tval
        else:
            value = 0.0
        dropper_policy, checker_policy = uniform_policy_for_current_roles(game)
        return float(value), dropper_policy, checker_policy


class TablebaseEvaluator:
    """Wraps another evaluator; short-circuits on tablebase hits."""

    def __init__(self, fallback: LeafEvaluator) -> None:
        self.fallback = fallback
        #evaluates against the table of E[public state]
        self._table: dict[ExactPublicState, float] = {}
        for factory in REGISTRY.values():
            scenario = factory()
            if scenario.expected_value is not None and not scenario.holdout:
                key = exact_public_state(scenario.game)
                self._table[key] = scenario.expected_value

    def __call__(self, game: Game) -> LeafEvaluation:
        key = exact_public_state(game)
        if key in self._table:
            dropper_policy, checker_policy = uniform_policy_for_current_roles(game)
            return float(self._table[key]), dropper_policy, checker_policy
        return normalize_leaf_evaluation(self.fallback(game), game)


class ValueNetEvaluator:
    """Wraps the trained value net behind a clean interface."""

    def __init__(self, model_fn: Callable[[Game], float]) -> None:
        self.model = model_fn

    def __call__(self, game: Game) -> LeafEvaluation:
        return normalize_leaf_evaluation(self.model(game), game)
