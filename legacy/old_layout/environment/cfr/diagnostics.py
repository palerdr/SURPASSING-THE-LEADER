"""Exploitability and best-response diagnostics for exact CFR results."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.Game import Game

from .exact import ExactSolveResult


@dataclass(frozen=True)
class ExactStrategyDiagnostics:
    """One-state best-response audit for a simultaneous exact matrix game."""

    expected_value: float
    dropper_best_response_value: float
    checker_best_response_value: float
    dropper_exploitability: float
    checker_exploitability: float
    nash_gap: float
    dropper_best_action: int | None
    checker_best_action: int | None


def _require_payoff(result: ExactSolveResult) -> np.ndarray:
    if result.payoff_for_hal is None:
        raise ValueError("ExactSolveResult has no payoff matrix; terminal states cannot be diagnosed")
    return result.payoff_for_hal


def diagnose_exact_strategy(
    game: Game,
    result: ExactSolveResult,
    *,
    perspective_name: str = "Hal",
) -> ExactStrategyDiagnostics:
    """Compute one-step exploitability from the exact Hal-perspective payoff matrix.

    The payoff matrix rows are dropper actions and columns are checker actions.
    If Hal is the dropper, rows maximize the payoff and columns minimize it.
    If Hal is the checker, rows minimize the payoff and columns maximize it.
    """
    payoff = _require_payoff(result)
    if payoff.shape != (len(result.dropper_strategy), len(result.checker_strategy)):
        raise ValueError("Strategy lengths do not match payoff matrix shape")

    dropper, _checker = game.get_roles_for_half(game.current_half)
    hal_is_dropper = dropper.name.lower() == perspective_name.lower()

    dropper_action_values = payoff @ result.checker_strategy
    checker_action_values = result.dropper_strategy @ payoff
    expected = float(result.dropper_strategy @ payoff @ result.checker_strategy)

    if hal_is_dropper:
        drop_idx = int(np.argmax(dropper_action_values))
        check_idx = int(np.argmin(checker_action_values))
        dropper_br = float(dropper_action_values[drop_idx])
        checker_br = float(checker_action_values[check_idx])
        dropper_exploitability = max(0.0, dropper_br - expected)
        checker_exploitability = max(0.0, expected - checker_br)
    else:
        drop_idx = int(np.argmin(dropper_action_values))
        check_idx = int(np.argmax(checker_action_values))
        dropper_br = float(dropper_action_values[drop_idx])
        checker_br = float(checker_action_values[check_idx])
        dropper_exploitability = max(0.0, expected - dropper_br)
        checker_exploitability = max(0.0, checker_br - expected)

    return ExactStrategyDiagnostics(
        expected_value=expected,
        dropper_best_response_value=dropper_br,
        checker_best_response_value=checker_br,
        dropper_exploitability=dropper_exploitability,
        checker_exploitability=checker_exploitability,
        nash_gap=dropper_exploitability + checker_exploitability,
        dropper_best_action=result.drop_actions[drop_idx] if result.drop_actions else None,
        checker_best_action=result.check_actions[check_idx] if result.check_actions else None,
    )

