"""Deterministic P3 conformance reports for simultaneous-move MCTS.

The acceptance run treats each fixture as a one-half-round matrix game.  The
full exact solver supplies the oracle matrix and value, while MCTS is capped at
``max_depth=1`` so its root samples have the same terminal-only frontier.  A
candidate-mode search policy is lifted onto the oracle's literal-second action
sets before any best-response or policy metric is computed.

Policy total variation is deliberately narrow.  It is reported only when the
oracle matrix has a *strict unique pure saddle*: each role's action is the
unique best response to the other role's action by more than the configured
tolerance.  This is a sufficient certificate of a unique equilibrium.  The
report makes no uniqueness claim for mixed or degenerate equilibria.
"""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Protocol, Sequence

import numpy as np

from stl.engine.game import Game
from stl.solver.exact import ExactSearchConfig, solve_exact_finite_horizon
from stl.solver.search import (
    MCTSActionMode,
    MCTSConfig,
    TerminalOnlyEvaluator,
    mcts_search,
)


SCHEMA_VERSION = "stl.mcts_conformance.v1"
BOUNDED_SCHEMA_VERSION = "stl.mcts_bounded_conformance.v1"
UNIQUENESS_TEST = "strict_unique_pure_saddle_v1"
FROZEN_BUDGETS = (64, 256, 1024)
FROZEN_SEEDS = tuple(range(10))
FROZEN_HORIZON_ONE_SCENARIOS = (
    "forced_baku_overflow_death",
    "forced_hal_overflow_death",
    "safe_budget_pressure_at_cylinder_239",
    "safe_budget_pressure_at_cylinder_240",
)


class HorizonOneScenario(Protocol):
    """Structural type required by :func:`run_mcts_conformance`."""

    name: str
    game: Game
    config: ExactSearchConfig


@dataclass(frozen=True)
class MCTSConformanceRecord:
    scenario_name: str
    budget: int
    seed: int
    action_mode: str
    exact_dropper_actions: int
    exact_checker_actions: int
    search_dropper_actions: int
    search_checker_actions: int
    exact_value_for_hal: float
    search_value_for_hal: float
    absolute_value_error: float
    lifted_policy_value_for_hal: float
    full_width_saddle_gap: float
    dropper_exploitability: float
    checker_exploitability: float
    dropper_normalized_entropy: float
    checker_normalized_entropy: float
    root_visits: int
    root_unique_cells_visited: int
    unique_equilibrium_certified: bool
    uniqueness_test: str
    dropper_policy_tv: float | None
    checker_policy_tv: float | None


@dataclass(frozen=True)
class MCTSConformanceReport:
    schema_version: str
    scenario_names: tuple[str, ...]
    budgets: tuple[int, ...]
    seeds: tuple[int, ...]
    action_mode: str
    exploration_c: float
    records: tuple[MCTSConformanceRecord, ...]


@dataclass(frozen=True)
class MCTSConformanceGateThresholds:
    """Frozen P3 acceptance thresholds from ``stl/docs/REGEN2RL.md``."""

    evaluation_budget: int = 1024
    comparison_budget: int = 256
    median_absolute_value_error: float = 0.05
    p95_absolute_value_error: float = 0.10
    maximum_absolute_value_error: float = 0.05
    maximum_saddle_gap: float = 0.05
    maximum_root_value_std: float = 0.03
    maximum_fixture_median_worsening: float = 0.02


@dataclass(frozen=True)
class MCTSConformanceGateEvaluation:
    passed: bool
    evaluation_budget: int
    comparison_budget: int
    median_absolute_value_error: float | None
    p95_absolute_value_error: float | None
    maximum_absolute_value_error: float | None
    maximum_saddle_gap: float | None
    maximum_root_value_std: float | None
    maximum_fixture_median_worsening: float | None
    root_value_std_by_scenario: tuple[tuple[str, float], ...]
    fixture_median_worsening: tuple[tuple[str, float], ...]
    failures: tuple[str, ...]


def lift_policy_to_full_actions(
    search_actions: Sequence[int],
    search_policy: Sequence[float] | np.ndarray,
    full_actions: Sequence[int],
) -> np.ndarray:
    """Embed one legal search marginal in the full literal-second action set."""

    actions = tuple(int(action) for action in search_actions)
    full = tuple(int(action) for action in full_actions)
    policy = np.asarray(search_policy, dtype=np.float64).reshape(-1)
    if len(actions) != len(policy):
        raise ValueError("search action and policy lengths differ")
    if len(set(actions)) != len(actions) or len(set(full)) != len(full):
        raise ValueError("action sets must contain distinct seconds")
    if not np.all(np.isfinite(policy)) or np.any(policy < 0.0):
        raise ValueError("search policy must be finite and nonnegative")
    total = float(policy.sum())
    if total <= 1e-12:
        raise ValueError("search policy has zero mass")
    policy = policy / total

    full_index = {action: index for index, action in enumerate(full)}
    lifted = np.zeros(len(full), dtype=np.float64)
    for action, probability in zip(actions, policy):
        if action not in full_index:
            raise ValueError(
                f"search action {action} is absent from the full action set"
            )
        lifted[full_index[action]] = float(probability)
    if not np.isclose(float(lifted.sum()), 1.0, atol=1e-8):
        raise ValueError("lifted policy failed normalization")
    return lifted


def certify_strict_unique_pure_saddle(
    payoff_for_hal: np.ndarray,
    *,
    hal_is_dropper: bool,
    tolerance: float = 1e-9,
) -> tuple[int, int] | None:
    """Return the sole strict pure saddle, or ``None`` when not certified.

    Rows are dropper actions and columns are checker actions.  When Hal is the
    dropper, rows maximize and columns minimize Hal value; the orientation is
    reversed when Hal is the checker.  Strict mutual best responses imply that
    neither optimal marginal can put mass on another action, which certifies a
    unique pure equilibrium.  This test intentionally does not attempt to
    certify uniqueness of a mixed equilibrium.
    """

    matrix = np.asarray(payoff_for_hal, dtype=np.float64)
    if matrix.ndim != 2 or 0 in matrix.shape or not np.all(np.isfinite(matrix)):
        raise ValueError("payoff must be a finite, non-empty matrix")
    if not np.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError("tolerance must be finite and nonnegative")

    saddles: list[tuple[int, int]] = []
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            value = float(matrix[row, column])
            other_rows = np.delete(matrix[:, column], row)
            other_columns = np.delete(matrix[row, :], column)
            if hal_is_dropper:
                row_is_strict = (
                    not other_rows.size or value > float(other_rows.max()) + tolerance
                )
                column_is_strict = (
                    not other_columns.size
                    or value < float(other_columns.min()) - tolerance
                )
            else:
                row_is_strict = (
                    not other_rows.size or value < float(other_rows.min()) - tolerance
                )
                column_is_strict = (
                    not other_columns.size
                    or value > float(other_columns.max()) + tolerance
                )
            if row_is_strict and column_is_strict:
                saddles.append((row, column))
    return saddles[0] if len(saddles) == 1 else None


def unique_equilibrium_policy_tv(
    payoff_for_hal: np.ndarray,
    dropper_policy: np.ndarray,
    checker_policy: np.ndarray,
    *,
    hal_is_dropper: bool,
    tolerance: float = 1e-9,
) -> tuple[float, float] | None:
    """Return role-policy TV only under the strict uniqueness certificate."""

    certificate = certify_strict_unique_pure_saddle(
        payoff_for_hal,
        hal_is_dropper=hal_is_dropper,
        tolerance=tolerance,
    )
    if certificate is None:
        return None
    row, column = certificate
    exact_dropper = np.zeros(len(dropper_policy), dtype=np.float64)
    exact_checker = np.zeros(len(checker_policy), dtype=np.float64)
    exact_dropper[row] = 1.0
    exact_checker[column] = 1.0
    dropper_tv = 0.5 * float(np.abs(dropper_policy - exact_dropper).sum())
    checker_tv = 0.5 * float(np.abs(checker_policy - exact_checker).sum())
    return dropper_tv, checker_tv


def _normalized_entropy(policy: np.ndarray) -> float:
    positive = np.asarray(policy, dtype=np.float64)
    positive = positive[positive > 0.0]
    if len(policy) <= 1 or positive.size <= 1:
        return 0.0
    return float(-(positive * np.log(positive)).sum() / np.log(len(policy)))


def _full_width_diagnostics(
    payoff_for_hal: np.ndarray,
    dropper_policy: np.ndarray,
    checker_policy: np.ndarray,
    *,
    hal_is_dropper: bool,
) -> tuple[float, float, float, float]:
    matrix = np.asarray(payoff_for_hal, dtype=np.float64)
    expected = float(dropper_policy @ matrix @ checker_policy)
    dropper_values = matrix @ checker_policy
    checker_values = dropper_policy @ matrix
    if hal_is_dropper:
        dropper_gain = max(0.0, float(dropper_values.max()) - expected)
        checker_gain = max(0.0, expected - float(checker_values.min()))
    else:
        dropper_gain = max(0.0, expected - float(dropper_values.min()))
        checker_gain = max(0.0, float(checker_values.max()) - expected)
    return expected, dropper_gain, checker_gain, dropper_gain + checker_gain


def _one_record(
    scenario: HorizonOneScenario,
    *,
    budget: int,
    seed: int,
    action_mode: MCTSActionMode,
    exploration_c: float,
    uniqueness_tolerance: float,
    evaluator=None,
) -> MCTSConformanceRecord:
    exact_game = deepcopy(scenario.game)
    exact = solve_exact_finite_horizon(exact_game, 1, scenario.config)
    if exact.payoff_for_hal is None:
        raise ValueError(f"scenario {scenario.name!r} is terminal at the root")

    search_game = deepcopy(scenario.game)
    try:
        result = mcts_search(
            search_game,
            MCTSConfig(
                iterations=budget,
                exploration_c=exploration_c,
                action_mode=action_mode,
                max_depth=1,
            ),
            (
                TerminalOnlyEvaluator(
                    perspective_name=scenario.config.perspective_name
                )
                if evaluator is None
                else evaluator
            ),
            np.random.default_rng(seed),
            scenario.config,
        )
    except RuntimeError as exc:
        raise RuntimeError(
            "MCTS conformance failed at "
            f"scenario={scenario.name!r}, budget={budget}, seed={seed}: {exc}"
        ) from exc
    dropper_policy = lift_policy_to_full_actions(
        result.root_drop_seconds,
        result.improved_dropper_policy,
        exact.drop_actions,
    )
    checker_policy = lift_policy_to_full_actions(
        result.root_check_seconds,
        result.improved_checker_policy,
        exact.check_actions,
    )

    dropper, _checker = scenario.game.get_roles_for_half(scenario.game.current_half)
    hal_is_dropper = dropper.name.lower() == scenario.config.perspective_name.lower()
    expected, dropper_gain, checker_gain, saddle_gap = _full_width_diagnostics(
        exact.payoff_for_hal,
        dropper_policy,
        checker_policy,
        hal_is_dropper=hal_is_dropper,
    )
    policy_tv = unique_equilibrium_policy_tv(
        exact.payoff_for_hal,
        dropper_policy,
        checker_policy,
        hal_is_dropper=hal_is_dropper,
        tolerance=uniqueness_tolerance,
    )

    return MCTSConformanceRecord(
        scenario_name=scenario.name,
        budget=budget,
        seed=seed,
        action_mode=result.action_mode,
        exact_dropper_actions=len(exact.drop_actions),
        exact_checker_actions=len(exact.check_actions),
        search_dropper_actions=len(result.root_drop_seconds),
        search_checker_actions=len(result.root_check_seconds),
        exact_value_for_hal=float(exact.value_for_hal),
        search_value_for_hal=float(result.root_value_for_hal),
        absolute_value_error=abs(
            float(result.root_value_for_hal) - float(exact.value_for_hal)
        ),
        lifted_policy_value_for_hal=expected,
        full_width_saddle_gap=saddle_gap,
        dropper_exploitability=dropper_gain,
        checker_exploitability=checker_gain,
        dropper_normalized_entropy=_normalized_entropy(dropper_policy),
        checker_normalized_entropy=_normalized_entropy(checker_policy),
        root_visits=int(result.root_visits),
        root_unique_cells_visited=int(result.root_unique_cells_visited),
        unique_equilibrium_certified=policy_tv is not None,
        uniqueness_test=UNIQUENESS_TEST,
        dropper_policy_tv=None if policy_tv is None else policy_tv[0],
        checker_policy_tv=None if policy_tv is None else policy_tv[1],
    )


def run_mcts_conformance(
    scenarios: Sequence[HorizonOneScenario],
    *,
    budgets: Sequence[int] = FROZEN_BUDGETS,
    seeds: Sequence[int] = FROZEN_SEEDS,
    action_mode: MCTSActionMode = "candidate",
    exploration_c: float = 1.0,
    uniqueness_tolerance: float = 1e-9,
    evaluator_factory: Callable[[], object] | None = None,
) -> MCTSConformanceReport:
    """Run a deterministic horizon-one conformance matrix.

    The frozen defaults are the P3 acceptance budgets and seeds and may be
    expensive.  Unit tests must pass explicitly small sequences.
    """

    ordered_scenarios = tuple(sorted(scenarios, key=lambda scenario: scenario.name))
    ordered_budgets = tuple(sorted(int(budget) for budget in budgets))
    ordered_seeds = tuple(sorted(int(seed) for seed in seeds))
    scenario_names = tuple(scenario.name for scenario in ordered_scenarios)
    if not ordered_scenarios or not ordered_budgets or not ordered_seeds:
        raise ValueError("scenarios, budgets, and seeds must be non-empty")
    if len(set(scenario_names)) != len(scenario_names):
        raise ValueError("scenario names must be unique")
    if len(set(ordered_budgets)) != len(ordered_budgets) or any(
        budget <= 0 for budget in ordered_budgets
    ):
        raise ValueError("budgets must be distinct positive integers")
    if len(set(ordered_seeds)) != len(ordered_seeds) or any(
        seed < 0 for seed in ordered_seeds
    ):
        raise ValueError("seeds must be distinct nonnegative integers")

    records = tuple(
        _one_record(
            scenario,
            budget=budget,
            seed=seed,
            action_mode=action_mode,
            exploration_c=exploration_c,
            uniqueness_tolerance=uniqueness_tolerance,
            evaluator=(None if evaluator_factory is None else evaluator_factory()),
        )
        for scenario in ordered_scenarios
        for budget in ordered_budgets
        for seed in ordered_seeds
    )
    return MCTSConformanceReport(
        schema_version=SCHEMA_VERSION,
        scenario_names=scenario_names,
        budgets=ordered_budgets,
        seeds=ordered_seeds,
        action_mode=action_mode,
        exploration_c=float(exploration_c),
        records=records,
    )


def frozen_horizon_one_scenarios() -> tuple[HorizonOneScenario, ...]:
    """Materialize the named P3 acceptance fixtures without running search."""

    from stl.solver.tablebase import get_scenario

    return tuple(get_scenario(name) for name in FROZEN_HORIZON_ONE_SCENARIOS)


class BellmanLookupEvaluator:
    """Exact V3-root/V2-leaf evaluator backed by one Bellman bundle."""

    supports_horizon_context = True

    def __init__(self, bundle) -> None:
        from stl.learning.replay import exact_state_hash

        self._roots = {root.state_hash: root for root in bundle.roots}
        self._successors = {
            row.state_hash: row for row in bundle.successors
        }
        self._hash = exact_state_hash

    def __call__(self, game: Game, *, value_horizon: int | None = None):
        from stl.engine.actions import ACTION_SIZE
        from stl.solver.exact import exact_public_state, solve_minimax, terminal_value
        from stl.solver.search import uniform_policy_for_current_roles

        terminal = terminal_value(game)
        if terminal is not None:
            drop, check = uniform_policy_for_current_roles(game)
            return float(terminal), drop, check
        state_hash = self._hash(exact_public_state(game))
        if value_horizon == 2 and state_hash in self._successors:
            row = self._successors[state_hash]
            drop, check = uniform_policy_for_current_roles(game)
            return float(row.value_h2), drop, check
        if value_horizon == 2 and state_hash in self._roots:
            row = self._roots[state_hash]
            drop, check = uniform_policy_for_current_roles(game)
            return float(row.value_h2), drop, check
        if value_horizon == 3 and state_hash in self._roots:
            row = self._roots[state_hash]
            dropper, _checker = game.get_roles_for_half(game.current_half)
            hal_is_dropper = dropper.name.lower() == "hal"
            if hal_is_dropper:
                drop_strategy, _ = solve_minimax(row.q3_for_hal)
                check_strategy, _ = solve_minimax((-row.q3_for_hal).T)
            else:
                drop_strategy, _ = solve_minimax(-row.q3_for_hal)
                check_strategy, _ = solve_minimax(row.q3_for_hal.T)
            drop = np.zeros(ACTION_SIZE, dtype=np.float64)
            check = np.zeros(ACTION_SIZE, dtype=np.float64)
            drop[list(row.drop_actions)] = drop_strategy
            check[list(row.check_actions)] = check_strategy
            return float(row.value_h3), drop, check
        raise KeyError(
            f"Bellman lookup misses state={state_hash} horizon={value_horizon}"
        )


def run_bounded_mcts_conformance(
    bundle,
    *,
    root_hashes: Sequence[str] | None = None,
    budgets: Sequence[int] = FROZEN_BUDGETS,
    seeds: Sequence[int] = FROZEN_SEEDS,
    action_mode: MCTSActionMode = "candidate",
    exploration_c: float = 1.0,
    evaluator_factory: Callable[[], object] | None = None,
) -> MCTSConformanceReport:
    """Run depth-one MCTS against exact V3 roots with V2 leaf context."""

    from stl.learning.replay import reconstruct_game

    selected = set(root_hashes or (root.state_hash for root in bundle.roots))
    roots = sorted(
        [root for root in bundle.roots if root.state_hash in selected],
        key=lambda row: row.name,
    )
    if {root.state_hash for root in roots} != selected:
        raise ValueError("bounded conformance root selection misses sealed hashes")
    ordered_budgets = tuple(sorted(int(value) for value in budgets))
    ordered_seeds = tuple(sorted(int(value) for value in seeds))
    if not roots or not ordered_budgets or not ordered_seeds:
        raise ValueError("bounded scenarios, budgets, and seeds must be non-empty")
    records: list[MCTSConformanceRecord] = []
    for root in roots:
        for budget in ordered_budgets:
            for seed in ordered_seeds:
                game = reconstruct_game(root.state)
                evaluator = (
                    BellmanLookupEvaluator(bundle)
                    if evaluator_factory is None
                    else evaluator_factory()
                )
                result = mcts_search(
                    game,
                    MCTSConfig(
                        iterations=budget,
                        exploration_c=exploration_c,
                        action_mode=action_mode,
                        max_depth=1,
                        root_value_horizon=3,
                    ),
                    evaluator,
                    np.random.default_rng(seed),
                    ExactSearchConfig(),
                )
                dropper_policy = lift_policy_to_full_actions(
                    result.root_drop_seconds,
                    result.improved_dropper_policy,
                    root.drop_actions,
                )
                checker_policy = lift_policy_to_full_actions(
                    result.root_check_seconds,
                    result.improved_checker_policy,
                    root.check_actions,
                )
                dropper, _checker = game.get_roles_for_half(game.current_half)
                hal_is_dropper = dropper.name.lower() == "hal"
                expected, drop_gain, check_gain, gap = _full_width_diagnostics(
                    root.q3_for_hal,
                    dropper_policy,
                    checker_policy,
                    hal_is_dropper=hal_is_dropper,
                )
                policy_tv = unique_equilibrium_policy_tv(
                    root.q3_for_hal,
                    dropper_policy,
                    checker_policy,
                    hal_is_dropper=hal_is_dropper,
                )
                records.append(
                    MCTSConformanceRecord(
                        scenario_name=root.name,
                        budget=budget,
                        seed=seed,
                        action_mode=result.action_mode,
                        exact_dropper_actions=len(root.drop_actions),
                        exact_checker_actions=len(root.check_actions),
                        search_dropper_actions=len(result.root_drop_seconds),
                        search_checker_actions=len(result.root_check_seconds),
                        exact_value_for_hal=float(root.value_h3),
                        search_value_for_hal=float(result.root_value_for_hal),
                        absolute_value_error=abs(
                            float(result.root_value_for_hal) - float(root.value_h3)
                        ),
                        lifted_policy_value_for_hal=expected,
                        full_width_saddle_gap=gap,
                        dropper_exploitability=drop_gain,
                        checker_exploitability=check_gain,
                        dropper_normalized_entropy=_normalized_entropy(dropper_policy),
                        checker_normalized_entropy=_normalized_entropy(checker_policy),
                        root_visits=int(result.root_visits),
                        root_unique_cells_visited=int(result.root_unique_cells_visited),
                        unique_equilibrium_certified=policy_tv is not None,
                        uniqueness_test=UNIQUENESS_TEST,
                        dropper_policy_tv=None if policy_tv is None else policy_tv[0],
                        checker_policy_tv=None if policy_tv is None else policy_tv[1],
                    )
                )
    return MCTSConformanceReport(
        schema_version=BOUNDED_SCHEMA_VERSION,
        scenario_names=tuple(root.name for root in roots),
        budgets=ordered_budgets,
        seeds=ordered_seeds,
        action_mode=action_mode,
        exploration_c=float(exploration_c),
        records=tuple(records),
    )


def evaluate_conformance_gate(
    report: MCTSConformanceReport,
    thresholds: MCTSConformanceGateThresholds | None = None,
) -> MCTSConformanceGateEvaluation:
    """Evaluate the frozen P3 aggregate gates without rerunning MCTS.

    Budget-to-budget worsening is the increase in each fixture's median
    absolute value error over the common seed set.  Seed stability is the
    population standard deviation of root values at the evaluation budget.
    """

    limits = thresholds or MCTSConformanceGateThresholds()
    failures: list[str] = []
    evaluation_records = [
        record for record in report.records if record.budget == limits.evaluation_budget
    ]
    comparison_records = [
        record for record in report.records if record.budget == limits.comparison_budget
    ]

    expected_pairs = {
        (scenario, seed) for scenario in report.scenario_names for seed in report.seeds
    }
    evaluation_pairs = {
        (record.scenario_name, record.seed) for record in evaluation_records
    }
    comparison_pairs = {
        (record.scenario_name, record.seed) for record in comparison_records
    }
    if evaluation_pairs != expected_pairs:
        failures.append(f"evaluation budget {limits.evaluation_budget} is incomplete")
    if comparison_pairs != expected_pairs:
        failures.append(f"comparison budget {limits.comparison_budget} is incomplete")

    median_error: float | None = None
    p95_error: float | None = None
    maximum_error: float | None = None
    maximum_gap: float | None = None
    if evaluation_records:
        errors = np.asarray(
            [record.absolute_value_error for record in evaluation_records],
            dtype=np.float64,
        )
        gaps = np.asarray(
            [record.full_width_saddle_gap for record in evaluation_records],
            dtype=np.float64,
        )
        median_error = float(np.median(errors))
        p95_error = float(np.percentile(errors, 95))
        maximum_error = float(np.max(errors))
        maximum_gap = float(np.max(gaps))
        if median_error > limits.median_absolute_value_error:
            failures.append("median absolute value error exceeds threshold")
        if p95_error > limits.p95_absolute_value_error:
            failures.append("95th-percentile absolute value error exceeds threshold")
        if maximum_error > limits.maximum_absolute_value_error:
            failures.append("maximum absolute value error exceeds threshold")
        if maximum_gap > limits.maximum_saddle_gap:
            failures.append("full-width saddle gap exceeds threshold")

    std_rows: list[tuple[str, float]] = []
    worsening_rows: list[tuple[str, float]] = []
    for scenario in report.scenario_names:
        evaluation_for_scenario = [
            record for record in evaluation_records if record.scenario_name == scenario
        ]
        comparison_for_scenario = [
            record for record in comparison_records if record.scenario_name == scenario
        ]
        if evaluation_for_scenario:
            root_std = float(
                np.std(
                    [record.search_value_for_hal for record in evaluation_for_scenario]
                )
            )
            std_rows.append((scenario, root_std))
            if root_std > limits.maximum_root_value_std:
                failures.append(
                    f"root value standard deviation exceeds threshold: {scenario}"
                )
        if evaluation_for_scenario and comparison_for_scenario:
            final_median = float(
                np.median(
                    [record.absolute_value_error for record in evaluation_for_scenario]
                )
            )
            earlier_median = float(
                np.median(
                    [record.absolute_value_error for record in comparison_for_scenario]
                )
            )
            worsening = final_median - earlier_median
            worsening_rows.append((scenario, worsening))
            if worsening > limits.maximum_fixture_median_worsening:
                failures.append(f"fixture median value error worsened: {scenario}")

    maximum_std = max((value for _name, value in std_rows), default=None)
    maximum_worsening = max((value for _name, value in worsening_rows), default=None)
    return MCTSConformanceGateEvaluation(
        passed=not failures,
        evaluation_budget=limits.evaluation_budget,
        comparison_budget=limits.comparison_budget,
        median_absolute_value_error=median_error,
        p95_absolute_value_error=p95_error,
        maximum_absolute_value_error=maximum_error,
        maximum_saddle_gap=maximum_gap,
        maximum_root_value_std=maximum_std,
        maximum_fixture_median_worsening=maximum_worsening,
        root_value_std_by_scenario=tuple(std_rows),
        fixture_median_worsening=tuple(worsening_rows),
        failures=tuple(failures),
    )


def _report_payload(report: MCTSConformanceReport) -> dict:
    return asdict(report)


def _canonical_json(payload: dict) -> str:
    return json.dumps(
        payload,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )


def conformance_report_digest(report: MCTSConformanceReport) -> str:
    """Return SHA-256 over the timestamp-free canonical report payload."""

    return hashlib.sha256(
        _canonical_json(_report_payload(report)).encode("utf-8")
    ).hexdigest()


def write_conformance_report(
    report: MCTSConformanceReport,
    path: str | Path,
    *,
    gate: MCTSConformanceGateEvaluation | None = None,
) -> str:
    """Atomically write stable JSON and return its canonical report digest."""

    digest = conformance_report_digest(report)
    envelope = {
        "gate": None if gate is None else asdict(gate),
        "report": _report_payload(report),
        "report_sha256": digest,
    }
    rendered = (
        json.dumps(
            envelope,
            allow_nan=False,
            ensure_ascii=True,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    temporary.write_text(rendered, encoding="utf-8", newline="\n")
    temporary.replace(destination)
    return digest


__all__ = [
    "FROZEN_BUDGETS",
    "FROZEN_HORIZON_ONE_SCENARIOS",
    "FROZEN_SEEDS",
    "MCTSConformanceGateEvaluation",
    "MCTSConformanceGateThresholds",
    "MCTSConformanceRecord",
    "MCTSConformanceReport",
    "SCHEMA_VERSION",
    "UNIQUENESS_TEST",
    "certify_strict_unique_pure_saddle",
    "conformance_report_digest",
    "evaluate_conformance_gate",
    "frozen_horizon_one_scenarios",
    "lift_policy_to_full_actions",
    "run_mcts_conformance",
    "unique_equilibrium_policy_tv",
    "write_conformance_report",
]
