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

from .exact import ExactSolveResult, solve_exact_finite_horizon
from .tactical_scenarios import (
    TacticalScenario,
    baku_dropper_leap_window_alignment,
    both_overflow_baku_dies_first,
    both_overflow_hal_dies_first,
    cpr_degradation_fatigued_referee,
    cpr_degradation_fresh_referee,
    death_trade_double_pressure,
    forced_baku_overflow_death,
    forced_baku_overflow_fatigued_referee,
    forced_baku_overflow_high_ttd,
    forced_baku_overflow_leap_window_late,
    forced_baku_overflow_leap_window_open,
    forced_baku_overflow_mid_clock,
    forced_baku_overflow_post_leap,
    forced_baku_overflow_pre_leap,
    forced_baku_overflow_with_baku_deaths,
    forced_baku_overflow_with_hal_deaths,
    forced_hal_overflow_death,
    forced_hal_overflow_fatigued_referee,
    forced_hal_overflow_leap_window_open,
    forced_hal_overflow_mid_clock,
    forced_hal_overflow_pre_leap,
    forced_hal_overflow_with_baku_deaths,
    forced_hal_overflow_with_hal_deaths,
    hal_dropper_leap_window_asymmetry,
    near_overflow_marginal_baku_294,
    role_alignment_active_lsr_runway,
    role_alignment_variation4_post_engineering,
    safe_budget_pressure_at_cylinder_240,
    safe_budget_pressure_at_cylinder_241,
)


ScenarioFactory = Callable[[], TacticalScenario]


REGISTRY: dict[str, ScenarioFactory] = {
    factory.__name__: factory
    for factory in (
        # Original pinned (Phase 8)
        forced_baku_overflow_death,
        forced_hal_overflow_death,
        # Original relational (Phase 8)
        safe_budget_pressure_at_cylinder_241,
        safe_budget_pressure_at_cylinder_240,
        cpr_degradation_fresh_referee,
        cpr_degradation_fatigued_referee,
        # Original holdout diagnostics (Phase 8)
        baku_dropper_leap_window_alignment,
        hal_dropper_leap_window_asymmetry,
        near_overflow_marginal_baku_294,
        death_trade_double_pressure,
        role_alignment_active_lsr_runway,
        role_alignment_variation4_post_engineering,
        # Phase F: pinned-tablebase expansion (17 new) ───────────────────
        # Clock variants
        forced_baku_overflow_mid_clock,
        forced_hal_overflow_mid_clock,
        forced_baku_overflow_pre_leap,
        forced_hal_overflow_pre_leap,
        # Leap-window variants
        forced_baku_overflow_leap_window_open,
        forced_hal_overflow_leap_window_open,
        forced_baku_overflow_leap_window_late,
        forced_baku_overflow_post_leap,
        # Fatigue / TTD pressure
        forced_baku_overflow_fatigued_referee,
        forced_hal_overflow_fatigued_referee,
        forced_baku_overflow_high_ttd,
        # Asymmetric death pins
        forced_baku_overflow_with_baku_deaths,
        forced_hal_overflow_with_hal_deaths,
        forced_baku_overflow_with_hal_deaths,
        forced_hal_overflow_with_baku_deaths,
        # Double-overflow
        both_overflow_baku_dies_first,
        both_overflow_hal_dies_first,
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


def pinned_scenarios(*, include_holdout: bool = False) -> tuple[TacticalScenario, ...]:
    return tuple(
        s
        for s in materialize_all()
        if s.expected_value is not None and (include_holdout or not s.holdout)
    )


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
