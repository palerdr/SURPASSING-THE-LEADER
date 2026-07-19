"""Finite-horizon exact dynamic programming for ToySTL."""

from __future__ import annotations

from dataclasses import dataclass
from typing import MutableMapping

import numpy as np

from toy.matrix import solve_matrix
from toy.rules import ToyRuleset
from toy.state import ToyBranch, ToyState


@dataclass(frozen=True, slots=True)
class ToyCellTransition:
    drop_action: int
    check_action: int
    branches: tuple[ToyBranch, ...]
    value_for_hal: float


@dataclass(frozen=True, slots=True)
class ToyExactResult:
    value_for_hal: float
    dropper_strategy: np.ndarray
    checker_strategy: np.ndarray
    drop_actions: tuple[int, ...]
    check_actions: tuple[int, ...]
    payoff_for_hal: np.ndarray
    saddle_gap: float
    truncated_probability: float
    hal_win_probability: float
    baku_win_probability: float
    transitions: tuple[ToyCellTransition, ...] = ()

    def __post_init__(self) -> None:
        for name in (
            "dropper_strategy",
            "checker_strategy",
            "payoff_for_hal",
        ):
            value = np.asarray(getattr(self, name), dtype=np.float64).copy()
            value.setflags(write=False)
            object.__setattr__(self, name, value)


def _terminal_breakdown(value: float) -> tuple[float, float, float, float]:
    if value > 0.0:
        return value, 1.0, 0.0, 0.0
    if value < 0.0:
        return value, 0.0, 1.0, 0.0
    return value, 0.0, 0.0, 0.0


def _child_breakdown(
    branch: ToyBranch,
    horizon: int,
    rules: ToyRuleset,
    cache: MutableMapping[tuple[object, ...], ToyExactResult],
    *,
    include_transitions: bool,
) -> tuple[float, float, float, float]:
    if branch.terminal_value is not None:
        return _terminal_breakdown(branch.terminal_value)
    assert branch.state is not None
    child = _solve_exact(
        branch.state,
        horizon,
        rules,
        cache,
        include_transitions=include_transitions,
    )
    return (
        child.value_for_hal,
        child.hal_win_probability,
        child.baku_win_probability,
        child.truncated_probability,
    )


def _state_key(
    state: ToyState,
    horizon: int,
    rules: ToyRuleset,
    *,
    include_transitions: bool,
) -> tuple[object, ...]:
    # Keep inspection-heavy and tablebase-compact results in separate caches.
    # This prevents an exhaustive build from retaining a transition object for
    # every recursive matrix cell while preserving full expansion for callers
    # that explicitly request it.
    return (
        rules.ruleset_id,
        rules.state_fields(state),
        int(horizon),
        bool(include_transitions),
    )


def _solve_exact(
    state: ToyState,
    horizon: int,
    rules: ToyRuleset,
    cache: MutableMapping[tuple[object, ...], ToyExactResult],
    *,
    include_transitions: bool,
) -> ToyExactResult:
    if horizon < 0:
        raise ValueError("horizon must be nonnegative")
    key = _state_key(
        state,
        horizon,
        rules,
        include_transitions=include_transitions,
    )
    cached = cache.get(key)
    if cached is not None:
        return cached

    if horizon == 0:
        result = ToyExactResult(
            value_for_hal=0.0,
            dropper_strategy=np.zeros(0, dtype=np.float64),
            checker_strategy=np.zeros(0, dtype=np.float64),
            drop_actions=(),
            check_actions=(),
            payoff_for_hal=np.zeros((0, 0), dtype=np.float64),
            saddle_gap=0.0,
            truncated_probability=1.0,
            hal_win_probability=0.0,
            baku_win_probability=0.0,
        )
        cache[key] = result
        return result

    drop_actions = tuple(rules.legal_drop_actions(state))
    check_actions = tuple(rules.legal_check_actions(state))
    payoff = np.zeros((len(drop_actions), len(check_actions)), dtype=np.float64)
    cell_hal_win = np.zeros_like(payoff)
    cell_baku_win = np.zeros_like(payoff)
    cell_truncated = np.zeros_like(payoff)
    transitions: list[ToyCellTransition] = []

    for d_index, drop in enumerate(drop_actions):
        for c_index, check in enumerate(check_actions):
            branches = tuple(rules.expand_joint_action(state, drop, check))
            probability_total = sum(branch.probability for branch in branches)
            if not np.isclose(probability_total, 1.0, atol=1e-12):
                raise ValueError(
                    f"chance probabilities do not sum to one for {drop},{check}: "
                    f"{probability_total}"
                )

            value = 0.0
            hal_win = 0.0
            baku_win = 0.0
            truncated = 0.0
            for branch in branches:
                child_value, child_hal, child_baku, child_truncated = _child_breakdown(
                    branch,
                    horizon - 1,
                    rules,
                    cache,
                    include_transitions=include_transitions,
                )
                value += branch.probability * child_value
                hal_win += branch.probability * child_hal
                baku_win += branch.probability * child_baku
                truncated += branch.probability * child_truncated
            payoff[d_index, c_index] = value
            cell_hal_win[d_index, c_index] = hal_win
            cell_baku_win[d_index, c_index] = baku_win
            cell_truncated[d_index, c_index] = truncated
            if include_transitions:
                transitions.append(
                    ToyCellTransition(
                        drop_action=drop,
                        check_action=check,
                        branches=branches,
                        value_for_hal=value,
                    )
                )

    equilibrium = solve_matrix(payoff, row_is_hal=state.hal_is_dropper)
    joint_policy = np.outer(equilibrium.row_strategy, equilibrium.column_strategy)
    result = ToyExactResult(
        value_for_hal=equilibrium.value_for_hal,
        dropper_strategy=equilibrium.row_strategy,
        checker_strategy=equilibrium.column_strategy,
        drop_actions=drop_actions,
        check_actions=check_actions,
        payoff_for_hal=payoff,
        saddle_gap=equilibrium.saddle_gap,
        truncated_probability=float(np.sum(joint_policy * cell_truncated)),
        hal_win_probability=float(np.sum(joint_policy * cell_hal_win)),
        baku_win_probability=float(np.sum(joint_policy * cell_baku_win)),
        transitions=tuple(transitions),
    )
    cache[key] = result
    return result


def solve_exact(
    state: ToyState,
    horizon: int,
    rules: ToyRuleset,
    *,
    cache: MutableMapping[tuple[object, ...], ToyExactResult] | None = None,
    include_transitions: bool = True,
) -> ToyExactResult:
    """Solve one ToySTL public state exactly for a finite horizon."""

    active_cache: MutableMapping[tuple[object, ...], ToyExactResult]
    active_cache = {} if cache is None else cache
    return _solve_exact(
        state,
        horizon,
        rules,
        active_cache,
        include_transitions=include_transitions,
    )


def solve_all_states(
    rules: ToyRuleset,
    *,
    max_horizon: int | None = None,
) -> tuple[tuple[ToyState, int, ToyExactResult], ...]:
    """Build the complete v0 state/horizon table in deterministic order."""

    if rules.ruleset_id != "bucket12_fixed50":
        raise ValueError("solve_all_states is exhaustive only for ToySTL-v0")
    horizon_limit = rules.max_half_rounds if max_horizon is None else max_horizon
    if not 0 <= horizon_limit <= rules.max_half_rounds:
        raise ValueError("max_horizon is outside the ruleset limit")
    cache: dict[tuple[object, ...], ToyExactResult] = {}
    rows: list[tuple[ToyState, int, ToyExactResult]] = []
    for horizon in range(horizon_limit + 1):
        for state in rules.enumerate_states():
                rows.append(
                    (
                        state,
                        horizon,
                        solve_exact(
                            state,
                            horizon,
                            rules,
                            cache=cache,
                            include_transitions=False,
                        ),
                    )
                )
    return tuple(rows)
