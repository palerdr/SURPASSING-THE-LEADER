"""Exhaustive exact solution of the finite acyclic abstract game graph."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import MutableMapping

import numpy as np

from abstract.matrix import solve_matrix
from abstract.rules import AbstractRuleset
from abstract.state import AbstractBranch, AbstractState


@dataclass(frozen=True, slots=True)
class AbstractCellTransition:
    drop_action: int
    check_action: int
    branches: tuple[AbstractBranch, ...]
    value_for_dropper: float


@dataclass(frozen=True, slots=True)
class AbstractExactResult:
    value_for_dropper: float
    dropper_strategy: np.ndarray
    checker_strategy: np.ndarray
    drop_actions: tuple[int, ...]
    check_actions: tuple[int, ...]
    payoff_for_dropper: np.ndarray
    saddle_gap: float
    dropper_win_probability: float
    checker_win_probability: float
    transitions: tuple[AbstractCellTransition, ...] = ()

    def __post_init__(self) -> None:
        for name in ("dropper_strategy", "checker_strategy", "payoff_for_dropper"):
            frozen = np.asarray(getattr(self, name), dtype=np.float64).copy()
            frozen.setflags(write=False)
            object.__setattr__(self, name, frozen)


def _terminal_breakdown(value: float) -> tuple[float, float, float]:
    if value > 0.0:
        return value, 1.0, 0.0
    if value < 0.0:
        return value, 0.0, 1.0
    return value, 0.0, 0.0


def _child_breakdown(
    branch: AbstractBranch,
    rules: AbstractRuleset,
    cache: MutableMapping[tuple[AbstractState, bool], AbstractExactResult],
    *,
    include_transitions: bool,
) -> tuple[float, float, float]:
    if branch.terminal_value is not None:
        return _terminal_breakdown(branch.terminal_value)
    assert branch.state is not None
    child = _solve_exact(branch.state, rules, cache, include_transitions=include_transitions)
    # Every live successor is expressed from the old Checker's now-Dropper
    # perspective, so swap players and negate the zero-sum value.
    return -child.value_for_dropper, child.checker_win_probability, child.dropper_win_probability


def _solve_exact(
    state: AbstractState,
    rules: AbstractRuleset,
    cache: MutableMapping[tuple[AbstractState, bool], AbstractExactResult],
    *,
    include_transitions: bool,
) -> AbstractExactResult:
    key = (state, include_transitions)
    cached = cache.get(key)
    if cached is not None:
        return cached

    drop_actions = rules.legal_drop_actions(state)
    check_actions = rules.legal_check_actions(state)
    payoff = np.zeros((len(drop_actions), len(check_actions)), dtype=np.float64)
    cell_dropper_win = np.zeros_like(payoff)
    cell_checker_win = np.zeros_like(payoff)
    transitions: list[AbstractCellTransition] = []

    for d_index, drop in enumerate(drop_actions):
        for c_index, check in enumerate(check_actions):
            branches = rules.expand_joint_action(state, drop, check)
            probability_total = sum(branch.probability for branch in branches)
            if not np.isclose(probability_total, 1.0, atol=1e-12):
                raise ValueError(f"chance probabilities do not sum to one for {drop},{check}: {probability_total}")

            value = 0.0
            dropper_win = 0.0
            checker_win = 0.0
            for branch in branches:
                if branch.state is not None and branch.state.potential <= state.potential:
                    raise RuntimeError("abstract successor does not increase the acyclic potential")
                child_value, child_dropper, child_checker = _child_breakdown(
                    branch,
                    rules,
                    cache,
                    include_transitions=include_transitions,
                )
                value += branch.probability * child_value
                dropper_win += branch.probability * child_dropper
                checker_win += branch.probability * child_checker
            payoff[d_index, c_index] = value
            cell_dropper_win[d_index, c_index] = dropper_win
            cell_checker_win[d_index, c_index] = checker_win
            if include_transitions:
                transitions.append(AbstractCellTransition(drop, check, branches, value))

    equilibrium = solve_matrix(payoff)
    joint_policy = np.outer(equilibrium.row_strategy, equilibrium.column_strategy)
    result = AbstractExactResult(
        value_for_dropper=equilibrium.value,
        dropper_strategy=equilibrium.row_strategy,
        checker_strategy=equilibrium.column_strategy,
        drop_actions=drop_actions,
        check_actions=check_actions,
        payoff_for_dropper=payoff,
        saddle_gap=equilibrium.saddle_gap,
        dropper_win_probability=float(np.sum(joint_policy * cell_dropper_win)),
        checker_win_probability=float(np.sum(joint_policy * cell_checker_win)),
        transitions=tuple(transitions),
    )
    cache[key] = result
    return result


def solve_exact(
    state: AbstractState,
    rules: AbstractRuleset,
    *,
    cache: MutableMapping[tuple[AbstractState, bool], AbstractExactResult] | None = None,
    include_transitions: bool = True,
) -> AbstractExactResult:
    """Solve a state to terminal outcomes without a depth cutoff."""

    active_cache: MutableMapping[tuple[AbstractState, bool], AbstractExactResult]
    active_cache = {} if cache is None else cache
    return _solve_exact(state, rules, active_cache, include_transitions=include_transitions)


def enumerate_reachable_states(
    rules: AbstractRuleset,
    *,
    root: AbstractState | None = None,
) -> tuple[AbstractState, ...]:
    """Enumerate the complete nonterminal closure reachable from ``root``."""

    root = rules.initial_state() if root is None else root
    seen = {root}
    pending = deque([root])
    while pending:
        state = pending.popleft()
        for drop in rules.legal_drop_actions(state):
            for check in rules.legal_check_actions(state):
                for branch in rules.expand_joint_action(state, drop, check):
                    if branch.state is None:
                        continue
                    if branch.state.potential <= state.potential:
                        raise RuntimeError("abstract successor does not increase the acyclic potential")
                    if branch.state not in seen:
                        seen.add(branch.state)
                        pending.append(branch.state)
    return tuple(sorted(seen, key=lambda state: rules.state_fields(state)))


def solve_all_reachable(
    rules: AbstractRuleset,
    *,
    root: AbstractState | None = None,
) -> tuple[tuple[AbstractState, AbstractExactResult], ...]:
    """Solve every state in the root's complete reachable closure exactly."""

    states = enumerate_reachable_states(rules, root=root)
    cache: dict[tuple[AbstractState, bool], AbstractExactResult] = {}
    return tuple(
        (state, solve_exact(state, rules, cache=cache, include_transitions=False))
        for state in states
    )
