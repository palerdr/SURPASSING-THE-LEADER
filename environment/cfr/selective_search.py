"""Selective exact-second minimax search over candidate action sets.

Mirrors solve_exact_finite_horizon but expands only the candidate seconds
returned by ``generate_candidates``. ``audit_against_full_width`` runs the
same state under both selective and full enumeration and reports the value
gap, so any insufficiency in the candidate set is observable.

No heuristic frontier, no value-net evaluator, no shaping. Frontier states
(horizon exhausted, no terminal) are reported as unresolved mass.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.Game import Game

from .candidates import CandidateActions, generate_candidates
from .exact_solver import solve_exact_finite_horizon
from .exact_transition import (
    ExactGameSnapshot,
    ExactJointAction,
    ExactSearchConfig,
)
from .minimax import solve_minimax
from .utility import UtilityBreakdown, terminal_value


@dataclass(frozen=True)
class SelectiveSearchResult:
    value_for_hal: float
    breakdown: UtilityBreakdown
    unresolved_probability: float
    half_round_horizon: int
    drop_seconds: tuple[int, ...]
    check_seconds: tuple[int, ...]
    dropper_strategy: np.ndarray
    checker_strategy: np.ndarray
    payoff_for_hal: np.ndarray | None
    candidate_count: int


@dataclass(frozen=True)
class SelectiveAuditResult:
    selective: SelectiveSearchResult
    full_width_value: float
    value_gap: float
    candidate_joint_count: int
    full_width_joint_count: int


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


def _evaluate_joint_action_selective(
    game: Game,
    action: ExactJointAction,
    half_round_horizon: int,
    config: ExactSearchConfig,
) -> UtilityBreakdown:
    if game.game_over:
        return _terminal_breakdown(terminal_value(game, perspective_name=config.perspective_name))
    if half_round_horizon <= 0:
        return _terminal_breakdown(None)

    snap = ExactGameSnapshot(game)
    probe = game.resolve_half_round(action.drop_time, action.check_time, survived_outcome=None)
    death_occurred = probe.survived is not None
    survival_probability = probe.survival_probability
    snap.restore(game)

    if not death_occurred:
        game.resolve_half_round(action.drop_time, action.check_time, survived_outcome=None)
        value = terminal_value(game, perspective_name=config.perspective_name)
        if value is not None or half_round_horizon == 1:
            part = _terminal_breakdown(value)
        else:
            part = selective_solve(game, half_round_horizon - 1, config).breakdown
        snap.restore(game)
        return part

    assert survival_probability is not None
    parts: list[tuple[float, UtilityBreakdown]] = []
    for survived, probability in ((True, survival_probability), (False, 1.0 - survival_probability)):
        if probability <= 0.0:
            continue
        game.resolve_half_round(action.drop_time, action.check_time, survived_outcome=survived)
        value = terminal_value(game, perspective_name=config.perspective_name)
        if value is not None or half_round_horizon == 1:
            part = _terminal_breakdown(value)
        else:
            part = selective_solve(game, half_round_horizon - 1, config).breakdown
        parts.append((probability, part))
        snap.restore(game)
    return _weighted_breakdown(parts)


def selective_solve(
    game: Game,
    half_round_horizon: int,
    config: ExactSearchConfig | None = None,
    *,
    candidates: CandidateActions | None = None,
) -> SelectiveSearchResult:
    """Selective candidate-only minimax over exact-second matrix games."""
    config = config or ExactSearchConfig()
    terminal = terminal_value(game, perspective_name=config.perspective_name)
    if terminal is not None or half_round_horizon <= 0:
        breakdown = _terminal_breakdown(terminal)
        return SelectiveSearchResult(
            value_for_hal=breakdown.value,
            breakdown=breakdown,
            unresolved_probability=breakdown.unresolved_probability,
            half_round_horizon=half_round_horizon,
            drop_seconds=(),
            check_seconds=(),
            dropper_strategy=np.zeros(0),
            checker_strategy=np.zeros(0),
            payoff_for_hal=None,
            candidate_count=0,
        )

    if candidates is None:
        candidates = generate_candidates(game, config)
    drop_actions = candidates.drop_seconds
    check_actions = candidates.check_seconds
    d_index = {s: i for i, s in enumerate(drop_actions)}
    c_index = {s: i for i, s in enumerate(check_actions)}

    dropper, _checker = game.get_roles_for_half(game.current_half)
    hal_is_dropper = dropper.name.lower() == config.perspective_name.lower()

    hal_payoff = np.zeros((len(drop_actions), len(check_actions)), dtype=np.float64)
    breakdowns: dict[tuple[int, int], UtilityBreakdown] = {}

    for d in drop_actions:
        for c in check_actions:
            action = ExactJointAction(d, c)
            breakdown = _evaluate_joint_action_selective(game, action, half_round_horizon, config)
            i = d_index[d]
            j = c_index[c]
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
    return SelectiveSearchResult(
        value_for_hal=value,
        breakdown=breakdown,
        unresolved_probability=breakdown.unresolved_probability,
        half_round_horizon=half_round_horizon,
        drop_seconds=drop_actions,
        check_seconds=check_actions,
        dropper_strategy=dropper_strategy,
        checker_strategy=checker_strategy,
        payoff_for_hal=hal_payoff,
        candidate_count=len(drop_actions) * len(check_actions),
    )


def audit_against_full_width(
    game: Game,
    half_round_horizon: int,
    config: ExactSearchConfig | None = None,
) -> SelectiveAuditResult:
    """Run selective and full-width on the same state; report the value gap."""
    config = config or ExactSearchConfig()
    selective = selective_solve(game, half_round_horizon, config)
    full = solve_exact_finite_horizon(game, half_round_horizon, config)
    return SelectiveAuditResult(
        selective=selective,
        full_width_value=full.value_for_hal,
        value_gap=abs(selective.value_for_hal - full.value_for_hal),
        candidate_joint_count=selective.candidate_count,
        full_width_joint_count=len(full.drop_actions) * len(full.check_actions),
    )
