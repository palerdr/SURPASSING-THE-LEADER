"""Leaf evaluators for MCTS.

When MCTS reaches a leaf, it needs a Hal-perspective value in [-1, 1].
This module defines the LeafEvaluator protocol plus concrete evaluators:

    - TerminalOnlyEvaluator: returns terminal_value or 0.0 for unresolved.
    - TablebaseEvaluator: short-circuits on positions matching pinned tablebase
        entries; falls back to a wrapped evaluator otherwise.
    - ValueNetEvaluator: wraps a trained value net behind the protocol.
"""

from typing import Protocol, Callable

from .exact import ExactPublicState, exact_public_state
from .tablebase import REGISTRY
from .exact import terminal_value
from src.Game import Game


class LeafEvaluator(Protocol):
    def __call__(self, game: Game) -> float: ...


class TerminalOnlyEvaluator:
    """Returns terminal_value(game) for terminal positions, 0.0 otherwise."""

    def __init__(self, perspective_name: str = "Hal") -> None:
        self.perspective = perspective_name

    def __call__(self, game: Game) -> float:
        tval = terminal_value(game, perspective_name=self.perspective)
        if tval is not None:
            return tval
        return 0.0


class TablebaseEvaluator:
    """Wraps another evaluator; short-circuits on tablebase hits."""

    def __init__(self, fallback: LeafEvaluator) -> None:
        self.fallback = fallback
        #evaluates against the table of E[public state]
        self._table: dict[ExactPublicState, float] = {}
        for factory in REGISTRY.values():
            scenario = factory()
            if scenario.expected_value is not None:
                key = exact_public_state(scenario.game)
                self._table[key] = scenario.expected_value

    def __call__(self, game: Game) -> float:
        key = exact_public_state(game)
        if key in self._table:
            return self._table[key]
        return self.fallback(game)


class ValueNetEvaluator:
    """Wraps the trained value net behind a clean interface."""

    def __init__(self, model_fn: Callable[[Game], float]) -> None:
        self.model = model_fn

    def __call__(self, game: Game) -> float:
        return self.model(game)
