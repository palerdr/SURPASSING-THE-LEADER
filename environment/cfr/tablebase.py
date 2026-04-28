"""Lazy registry of exact tablebase targets.

Each entry in ``REGISTRY`` is a factory that produces a fresh
``TacticalScenario``. Call ``solve_target(name)`` to materialise a scenario
and run ``solve_exact_finite_horizon`` against it. Scenarios with a non-None
``expected_value`` are pinned: the solver value is exact by construction
(forced terminal sequences) and any drift is a regression. Relational
scenarios (``expected_value`` None) ship in pairs; their tests assert
monotonicity, action dominance, or threshold inequalities rather than a
single fixed number.

This module is part of the rigorous CFR core: no shaping, no value-net
frontier, no curriculum or stage labels. Targets exist purely as supervised
labels for downstream MCCFR / MCTS / value-net training.
"""

from __future__ import annotations

import math
from collections.abc import Callable

from .exact_solver import ExactSolveResult, solve_exact_finite_horizon
from .tactical_scenarios import (
    TacticalScenario,
    cpr_degradation_fatigued_referee,
    cpr_degradation_fresh_referee,
    forced_baku_overflow_death,
    forced_hal_overflow_death,
    leap_second_check_61_probe,
    safe_budget_pressure_at_cylinder_240,
    safe_budget_pressure_at_cylinder_241,
)


ScenarioFactory = Callable[[], TacticalScenario]


REGISTRY: dict[str, ScenarioFactory] = {
    factory.__name__: factory
    for factory in (
        forced_baku_overflow_death,
        forced_hal_overflow_death,
        leap_second_check_61_probe,
        safe_budget_pressure_at_cylinder_241,
        safe_budget_pressure_at_cylinder_240,
        cpr_degradation_fresh_referee,
        cpr_degradation_fatigued_referee,
    )
}


def scenario_names() -> tuple[str, ...]:
    return tuple(REGISTRY.keys())


def get_scenario(name: str) -> TacticalScenario:
    if name not in REGISTRY:
        raise KeyError(f"unknown tablebase scenario: {name}")
    return REGISTRY[name]()


def materialize_all() -> tuple[TacticalScenario, ...]:
    return tuple(factory() for factory in REGISTRY.values())


def pinned_scenarios() -> tuple[TacticalScenario, ...]:
    return tuple(s for s in materialize_all() if s.expected_value is not None)


def scenarios_by_tag(tag: str) -> tuple[TacticalScenario, ...]:
    return tuple(s for s in materialize_all() if tag in s.tags)


def solve_target(name: str) -> ExactSolveResult:
    """Run the exact finite-horizon solver against a registered scenario."""
    scenario = get_scenario(name)
    return solve_exact_finite_horizon(
        scenario.game,
        scenario.half_round_horizon,
        scenario.config,
    )


def verify_pinned_value(name: str, *, abs_tol: float = 1e-9) -> ExactSolveResult:
    """Solve a pinned scenario and raise if the value drifted from its pin."""
    scenario = get_scenario(name)
    if scenario.expected_value is None:
        raise ValueError(f"scenario {name!r} has no pinned expected_value")
    result = solve_target(name)
    if not math.isclose(result.value_for_hal, scenario.expected_value, abs_tol=abs_tol):
        raise AssertionError(
            f"tablebase value drift for {name!r}: "
            f"got {result.value_for_hal!r}, expected {scenario.expected_value!r}"
        )
    return result
