"""Exact-second local solvers and finite-horizon evaluator."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.Constants import CYLINDER_MAX, FAILED_CHECK_PENALTY
from src.Game import Game

from .exact_transition import (
    ExactGameSnapshot,
    ExactJointAction,
    ExactSearchConfig,
    enumerate_joint_actions,
)
from .minimax import solve_minimax
from .utility import UtilityBreakdown, terminal_value


@dataclass(frozen=True)
class ExactMatrixGame:
    drop_actions: tuple[int, ...]
    check_actions: tuple[int, ...]
    payoff: np.ndarray


@dataclass(frozen=True)
class ExactSolveResult:
    dropper_strategy: np.ndarray
    checker_strategy: np.ndarray
    value_for_hal: float
    breakdown: UtilityBreakdown
    unresolved_probability: float
    half_round_horizon: int
    drop_actions: tuple[int, ...] = ()
    check_actions: tuple[int, ...] = ()
    payoff_for_hal: np.ndarray | None = None


def exact_immediate_checker_payoff_matrix(game: Game, config: ExactSearchConfig | None = None) -> ExactMatrixGame:
    """Build exact checker-perspective immediate payoff matrix for the current half-round."""
    config = config or ExactSearchConfig()
    dropper, checker = game.get_roles_for_half(game.current_half)
    actions = enumerate_joint_actions(game, config)
    drop_actions = tuple(sorted({a.drop_time for a in actions}))
    check_actions = tuple(sorted({a.check_time for a in actions}))
    d_index = {second: i for i, second in enumerate(drop_actions)}
    c_index = {second: i for i, second in enumerate(check_actions)}
    payoff = np.zeros((len(drop_actions), len(check_actions)), dtype=np.float64)

    for action in actions:
        if action.check_time >= action.drop_time:
            st = max(1, action.check_time - action.drop_time)
            payoff[d_index[action.drop_time], c_index[action.check_time]] = (
                -CYLINDER_MAX if checker.cylinder + st >= CYLINDER_MAX else -st
            )
        else:
            payoff[d_index[action.drop_time], c_index[action.check_time]] = -min(
                checker.cylinder + FAILED_CHECK_PENALTY,
                CYLINDER_MAX,
            )
    return ExactMatrixGame(drop_actions=drop_actions, check_actions=check_actions, payoff=payoff)


def _terminal_breakdown(value: float | None) -> UtilityBreakdown:
    if value is None:
        return UtilityBreakdown(0.0, 0.0, 0.0, 1.0)
    if value > 0.0:
        return UtilityBreakdown(value, 1.0, 0.0, 0.0)
    if value < 0.0:
        return UtilityBreakdown(value, 0.0, 1.0, 0.0)
    return UtilityBreakdown(value, 0.0, 0.0, 0.0)


def _weighted_breakdown(parts: list[tuple[float, UtilityBreakdown]]) -> UtilityBreakdown:
    value = sum(weight * part.value for weight, part in parts)
    hal = sum(weight * part.hal_win_probability for weight, part in parts)
    baku = sum(weight * part.baku_win_probability for weight, part in parts)
    unresolved = sum(weight * part.unresolved_probability for weight, part in parts)
    return UtilityBreakdown(value, hal, baku, unresolved)


def evaluate_joint_action(
    game: Game,
    action: ExactJointAction,
    half_round_horizon: int,
    config: ExactSearchConfig | None = None,
) -> UtilityBreakdown:
    config = config or ExactSearchConfig()
    if game.game_over:
        return _terminal_breakdown(terminal_value(game, perspective_name=config.perspective_name))
    if half_round_horizon <= 0:
        return _terminal_breakdown(None)

    snap = ExactGameSnapshot(game)
    probe = game.resolve_half_round(action.drop_time, action.check_time, survived_outcome=None)
    death_occurred = probe.survived is not None
    survival_probability = probe.survival_probability
    snap.restore(game)

    parts: list[tuple[float, UtilityBreakdown]] = []

    if not death_occurred:
        game.resolve_half_round(action.drop_time, action.check_time, survived_outcome=None)
        value = terminal_value(game, perspective_name=config.perspective_name)
        if value is not None or half_round_horizon == 1:
            part = _terminal_breakdown(value)
        else:
            part = solve_exact_finite_horizon(game, half_round_horizon - 1, config).breakdown
        snap.restore(game)
        return part

    assert survival_probability is not None
    for survived, probability in ((True, survival_probability), (False, 1.0 - survival_probability)):
        if probability <= 0.0:
            continue
        game.resolve_half_round(action.drop_time, action.check_time, survived_outcome=survived)
        value = terminal_value(game, perspective_name=config.perspective_name)
        if value is not None or half_round_horizon == 1:
            part = _terminal_breakdown(value)
        else:
            part = solve_exact_finite_horizon(game, half_round_horizon - 1, config).breakdown
        parts.append((probability, part))
        snap.restore(game)

    return _weighted_breakdown(parts)


def solve_exact_finite_horizon(
    game: Game,
    half_round_horizon: int,
    config: ExactSearchConfig | None = None,
) -> ExactSolveResult:
    """Solve exact-second zero-sum matrix games recursively to a finite horizon.

    Horizon cutoff is reported as unresolved probability; no heuristic frontier
    value is used.
    """
    config = config or ExactSearchConfig()
    terminal = terminal_value(game, perspective_name=config.perspective_name)
    if terminal is not None or half_round_horizon <= 0:
        breakdown = _terminal_breakdown(terminal)
        return ExactSolveResult(
            dropper_strategy=np.zeros(0),
            checker_strategy=np.zeros(0),
            value_for_hal=breakdown.value,
            breakdown=breakdown,
            unresolved_probability=breakdown.unresolved_probability,
            half_round_horizon=half_round_horizon,
            payoff_for_hal=None,
        )

    dropper, _checker = game.get_roles_for_half(game.current_half)
    hal_is_dropper = dropper.name.lower() == config.perspective_name.lower()
    actions = enumerate_joint_actions(game, config)
    drop_actions = tuple(sorted({a.drop_time for a in actions}))
    check_actions = tuple(sorted({a.check_time for a in actions}))
    d_index = {second: i for i, second in enumerate(drop_actions)}
    c_index = {second: i for i, second in enumerate(check_actions)}
    hal_payoff = np.zeros((len(drop_actions), len(check_actions)), dtype=np.float64)
    breakdowns: dict[tuple[int, int], UtilityBreakdown] = {}

    for action in actions:
        breakdown = evaluate_joint_action(game, action, half_round_horizon, config)
        i = d_index[action.drop_time]
        j = c_index[action.check_time]
        hal_payoff[i, j] = breakdown.value
        breakdowns[(i, j)] = breakdown

    row_payoff = hal_payoff if hal_is_dropper else -hal_payoff.T
    row_strategy, row_value = solve_minimax(row_payoff)
    if hal_is_dropper:
        dropper_strategy = row_strategy
        checker_strategy, _ = solve_minimax((-hal_payoff).T)
    else:
        checker_strategy = row_strategy
        dropper_strategy, _ = solve_minimax(hal_payoff)

    parts: list[tuple[float, UtilityBreakdown]] = []
    for i, dp in enumerate(dropper_strategy):
        if dp <= 0.0:
            continue
        for j, cp in enumerate(checker_strategy):
            weight = float(dp * cp)
            if weight > 0.0:
                parts.append((weight, breakdowns[(i, j)]))

    breakdown = _weighted_breakdown(parts)
    value = float(row_value if hal_is_dropper else -row_value)
    return ExactSolveResult(
        dropper_strategy=dropper_strategy,
        checker_strategy=checker_strategy,
        value_for_hal=value,
        breakdown=breakdown,
        unresolved_probability=breakdown.unresolved_probability,
        half_round_horizon=half_round_horizon,
        drop_actions=drop_actions,
        check_actions=check_actions,
        payoff_for_hal=hal_payoff,
    )
