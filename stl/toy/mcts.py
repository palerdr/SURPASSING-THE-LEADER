"""Standalone simultaneous-move matrix-game MCTS for ToySTL."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Callable, Protocol

import numpy as np

from stl.toy.exact import solve_exact
from stl.toy.matrix import saddle_gap, solve_matrix
from stl.toy.rules import ToyRuleset
from stl.toy.state import ToyState


class ToyLeafEvaluator(Protocol):
    def __call__(
        self,
        state: ToyState,
        remaining_horizon: int,
        rules: ToyRuleset,
    ) -> tuple[float, np.ndarray, np.ndarray]: ...


def _uniform(length: int) -> np.ndarray:
    if length <= 0:
        return np.zeros(0, dtype=np.float64)
    return np.full(length, 1.0 / length, dtype=np.float64)


def _normalize(policy: np.ndarray, length: int, *, name: str) -> np.ndarray:
    values = np.asarray(policy, dtype=np.float64).reshape(-1)
    if values.shape != (length,):
        raise ValueError(f"{name} shape {values.shape} does not match {(length,)}")
    if not np.all(np.isfinite(values)) or np.any(values < 0.0):
        raise ValueError(f"{name} must be finite and nonnegative")
    total = float(values.sum())
    if total <= 1e-12:
        raise ValueError(f"{name} has no probability mass")
    return values / total


def uniform_evaluator(
    state: ToyState,
    remaining_horizon: int,
    rules: ToyRuleset,
) -> tuple[float, np.ndarray, np.ndarray]:
    del remaining_horizon
    return 0.0, _uniform(len(rules.legal_drop_actions(state))), _uniform(
        len(rules.legal_check_actions(state))
    )


def exact_evaluator(
    state: ToyState,
    remaining_horizon: int,
    rules: ToyRuleset,
) -> tuple[float, np.ndarray, np.ndarray]:
    result = solve_exact(state, remaining_horizon, rules, include_transitions=False)
    return _exact_result_as_full_policy(result, rules)


def _exact_result_as_full_policy(
    result: object,
    rules: ToyRuleset,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Adapt a compact exact result to the MCTS full-head evaluator contract."""
    # Keep the adapter duck-typed so cached and direct exact results share it
    # without importing tablebase-specific types into MCTS.
    drop = np.zeros(rules.action_size, dtype=np.float64)
    check = np.zeros(rules.action_size, dtype=np.float64)
    if result.drop_actions:  # type: ignore[attr-defined]
        drop[np.asarray(result.drop_actions) - 1] = result.dropper_strategy  # type: ignore[attr-defined]
        check[np.asarray(result.check_actions) - 1] = result.checker_strategy  # type: ignore[attr-defined]
    return float(result.value_for_hal), drop, check  # type: ignore[attr-defined]


def make_exact_evaluator(rules: ToyRuleset) -> ToyLeafEvaluator:
    """Return a memoized exact leaf evaluator for convergence audits."""

    cache: dict[tuple[object, ...], object] = {}

    def evaluate(
        state: ToyState,
        remaining_horizon: int,
        active_rules: ToyRuleset,
    ) -> tuple[float, np.ndarray, np.ndarray]:
        if active_rules.ruleset_id != rules.ruleset_id:
            raise ValueError("exact evaluator ruleset does not match search rules")
        key = (rules.ruleset_id, rules.state_fields(state), int(remaining_horizon))
        result = cache.get(key)
        if result is None:
            result = solve_exact(
                state,
                remaining_horizon,
                rules,
                cache=cache,  # type: ignore[arg-type]
                include_transitions=False,
            )
            cache[key] = result
        return _exact_result_as_full_policy(result, rules)

    return evaluate


def make_tablebase_evaluator(
    tablebase: dict[str, object],
    rules: ToyRuleset,
) -> ToyLeafEvaluator:
    """Adapt a validated tablebase artifact to the MCTS leaf contract."""

    arrays = tablebase["arrays"]
    if not isinstance(arrays, dict):
        raise TypeError("tablebase arrays must be a mapping")
    metadata = tablebase.get("metadata", {})
    if isinstance(metadata, dict) and metadata.get("ruleset_id") not in (None, rules.ruleset_id):
        raise ValueError("tablebase ruleset does not match active rules")
    states = np.asarray(arrays["states"])
    horizons = np.asarray(arrays["horizon"])
    values = np.asarray(arrays["value"], dtype=np.float64)
    drop_policies = np.asarray(arrays["drop_policy"], dtype=np.float64)
    check_policies = np.asarray(arrays["check_policy"], dtype=np.float64)
    index = {
        (tuple(int(value) for value in state), int(horizon)): row
        for row, (state, horizon) in enumerate(zip(states, horizons))
    }

    def evaluate(
        state: ToyState,
        remaining_horizon: int,
        active_rules: ToyRuleset,
    ) -> tuple[float, np.ndarray, np.ndarray]:
        if active_rules.ruleset_id != rules.ruleset_id:
            raise ValueError("tablebase evaluator ruleset does not match search rules")
        key = (rules.state_fields(state), int(remaining_horizon))
        try:
            row = index[key]
        except KeyError as exc:
            raise KeyError(f"state/horizon is absent from tablebase: {key}") from exc
        return float(values[row]), drop_policies[row].copy(), check_policies[row].copy()

    return evaluate


@dataclass(frozen=True, slots=True)
class ToyMCTSConfig:
    rules: ToyRuleset
    iterations: int = 256
    exploration_c: float = 1.0
    max_depth: int | None = None
    root_noise_epsilon: float = 0.0
    root_dirichlet_alpha_scale: float = 10.0
    full_width_bootstrap: bool = False
    bootstrap_prior_count: int = 32

    def __post_init__(self) -> None:
        if self.iterations <= 0:
            raise ValueError("iterations must be positive")
        if not np.isfinite(self.exploration_c) or self.exploration_c < 0.0:
            raise ValueError("exploration_c must be finite and nonnegative")
        if self.max_depth is not None and self.max_depth <= 0:
            raise ValueError("max_depth must be positive")
        if not 0.0 <= self.root_noise_epsilon <= 1.0:
            raise ValueError("root_noise_epsilon must be in [0, 1]")
        if self.root_dirichlet_alpha_scale <= 0.0:
            raise ValueError("root_dirichlet_alpha_scale must be positive")
        if self.bootstrap_prior_count <= 0:
            raise ValueError("bootstrap prior count must be positive")
        if self.full_width_bootstrap and self.rules.ruleset_id != "bucket12_fixed50":
            raise ValueError("full-width bootstrap is defined only for ToySTL-v0")


@dataclass
class _ToyNode:
    state: ToyState
    remaining_horizon: int
    drop_actions: tuple[int, ...]
    check_actions: tuple[int, ...]
    q_values: np.ndarray
    q_prior_counts: np.ndarray
    visits: np.ndarray
    node_visits: int
    drop_prior: np.ndarray
    check_prior: np.ndarray
    prior: np.ndarray
    expanded: bool
    children: dict[tuple[int, int, int], "_ToyNode"]


@dataclass(frozen=True, slots=True)
class ToyMCTSResult:
    improved_dropper_policy: np.ndarray
    improved_checker_policy: np.ndarray
    mean_q_dropper_policy: np.ndarray
    mean_q_checker_policy: np.ndarray
    root_value_for_hal: float
    mean_q_value_for_hal: float
    root_visits: int
    root_unique_cells_visited: int
    cells_used: int
    root_drop_actions: tuple[int, ...]
    root_check_actions: tuple[int, ...]

    def __post_init__(self) -> None:
        for name in (
            "improved_dropper_policy",
            "improved_checker_policy",
            "mean_q_dropper_policy",
            "mean_q_checker_policy",
        ):
            value = np.asarray(getattr(self, name), dtype=np.float64).copy()
            value.setflags(write=False)
            object.__setattr__(self, name, value)


def _state_key(state: ToyState, horizon: int, rules: ToyRuleset) -> tuple[object, ...]:
    return (rules.ruleset_id, rules.state_fields(state), horizon)


def _evaluate(
    evaluator: ToyLeafEvaluator | None,
    state: ToyState,
    horizon: int,
    rules: ToyRuleset,
    drop_actions: tuple[int, ...],
    check_actions: tuple[int, ...],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    if horizon <= 0:
        value = 0.0
        drop = _uniform(len(drop_actions))
        check = _uniform(len(check_actions))
    elif evaluator is None:
        value = 0.0
        drop = _uniform(len(drop_actions))
        check = _uniform(len(check_actions))
    else:
        value, drop_full, check_full = evaluator(state, horizon, rules)
        drop_full = np.asarray(drop_full, dtype=np.float64)
        check_full = np.asarray(check_full, dtype=np.float64)
        if drop_full.shape != (rules.action_size,) or check_full.shape != (rules.action_size,):
            raise ValueError("toy evaluator must return full action-size priors")
        drop = _normalize(drop_full[np.asarray(drop_actions) - 1], len(drop_actions), name="drop prior")
        check = _normalize(check_full[np.asarray(check_actions) - 1], len(check_actions), name="check prior")
    if not -1.0 <= float(value) <= 1.0:
        raise ValueError("toy evaluator value must be in [-1, 1]")
    return drop, check, np.outer(drop, check), float(value)


def _make_node(
    state: ToyState,
    remaining_horizon: int,
    rules: ToyRuleset,
    evaluator: ToyLeafEvaluator | None,
) -> _ToyNode:
    drop_actions = tuple(rules.legal_drop_actions(state))
    check_actions = tuple(rules.legal_check_actions(state))
    drop_prior, check_prior, prior, value = _evaluate(
        evaluator, state, remaining_horizon, rules, drop_actions, check_actions
    )
    return _ToyNode(
        state=state,
        remaining_horizon=remaining_horizon,
        drop_actions=drop_actions,
        check_actions=check_actions,
        q_values=np.full((len(drop_actions), len(check_actions)), value, dtype=np.float64),
        q_prior_counts=np.zeros((len(drop_actions), len(check_actions)), dtype=np.int64),
        visits=np.zeros((len(drop_actions), len(check_actions)), dtype=np.int64),
        node_visits=0,
        drop_prior=drop_prior,
        check_prior=check_prior,
        prior=prior,
        expanded=False,
        children={},
    )


def _full_width_bootstrap(
    node: _ToyNode,
    rules: ToyRuleset,
    evaluator: ToyLeafEvaluator | None,
    prior_count: int,
) -> None:
    """Seed a v0 h=1 root with exact chance expectations as virtual visits."""

    if node.remaining_horizon != 1:
        return
    for d_index, drop in enumerate(node.drop_actions):
        for c_index, check in enumerate(node.check_actions):
            branches = rules.expand_joint_action(node.state, drop, check)
            value = 0.0
            for branch in branches:
                if branch.terminal_value is not None:
                    child_value = float(branch.terminal_value)
                elif node.remaining_horizon - 1 <= 0 or evaluator is None:
                    child_value = 0.0
                else:
                    assert branch.state is not None
                    child_value, _drop, _check = evaluator(branch.state, 0, rules)
                value += branch.probability * child_value
            node.q_values[d_index, c_index] = value
            node.q_prior_counts[d_index, c_index] = prior_count


def _selection_strategies(node: _ToyNode, exploration_c: float) -> tuple[np.ndarray, np.ndarray]:
    bonus = exploration_c * node.prior * np.sqrt(node.node_visits) / (1.0 + node.visits)
    optimistic_for_hal = node.q_values + bonus
    optimistic_for_baku = -node.q_values + bonus
    if node.state.hal_is_dropper:
        drop = solve_matrix(optimistic_for_hal, row_is_hal=True).row_strategy
        check = solve_matrix(optimistic_for_baku.T, row_is_hal=True).row_strategy
    else:
        drop = solve_matrix(optimistic_for_baku, row_is_hal=True).row_strategy
        check = solve_matrix(optimistic_for_hal.T, row_is_hal=True).row_strategy
    return drop, check


def _mean_q_strategies(node: _ToyNode) -> tuple[np.ndarray, np.ndarray, float, float]:
    if node.state.hal_is_dropper:
        equilibrium = solve_matrix(node.q_values, row_is_hal=True)
        drop = equilibrium.row_strategy
        check = equilibrium.column_strategy
    else:
        equilibrium = solve_matrix(node.q_values, row_is_hal=False)
        drop = equilibrium.row_strategy
        check = equilibrium.column_strategy
    value, _row_gain, _column_gain, _gap = saddle_gap(
        node.q_values,
        drop,
        check,
        row_is_hal=node.state.hal_is_dropper,
    )
    return drop, check, value, equilibrium.saddle_gap


def _step_into_child(
    node: _ToyNode,
    rules: ToyRuleset,
    d_index: int,
    c_index: int,
    chance_rng: np.random.Generator,
    evaluator: ToyLeafEvaluator | None,
    transposition: dict[tuple[object, ...], _ToyNode],
) -> tuple[_ToyNode | None, float | None]:
    drop = node.drop_actions[d_index]
    check = node.check_actions[c_index]
    branches = rules.expand_joint_action(node.state, drop, check)
    probabilities = np.asarray([branch.probability for branch in branches], dtype=np.float64)
    branch_index = int(chance_rng.choice(len(branches), p=probabilities / probabilities.sum()))
    branch = branches[branch_index]
    if branch.terminal_value is not None:
        return None, float(branch.terminal_value)
    assert branch.state is not None
    child_horizon = max(0, node.remaining_horizon - 1)
    child_key = _state_key(branch.state, child_horizon, rules)
    child = node.children.get((d_index, c_index, branch_index))
    if child is None:
        child = transposition.get(child_key)
        if child is None:
            child = _make_node(branch.state, child_horizon, rules, evaluator)
            transposition[child_key] = child
        node.children[(d_index, c_index, branch_index)] = child
    return child, None


def _backup(path: list[tuple[_ToyNode, int, int]], value: float) -> None:
    for node, d_index, c_index in path:
        node.node_visits += 1
        node.visits[d_index, c_index] += 1
        count = node.q_prior_counts[d_index, c_index] + node.visits[d_index, c_index]
        node.q_values[d_index, c_index] += (
            value - node.q_values[d_index, c_index]
        ) / count


def mcts_search(
    state: ToyState,
    remaining_horizon: int,
    evaluator: ToyLeafEvaluator | None,
    config: ToyMCTSConfig,
    action_rng: np.random.Generator,
    chance_rng: np.random.Generator,
    root_noise_rng: np.random.Generator | None = None,
) -> ToyMCTSResult:
    """Run simultaneous matrix-game MCTS from one toy state."""

    if remaining_horizon <= 0:
        raise ValueError("MCTS requires a positive remaining horizon")
    rules = config.rules
    root = _make_node(state, remaining_horizon, rules, evaluator)
    if config.full_width_bootstrap:
        _full_width_bootstrap(
            root,
            rules,
            evaluator,
            config.bootstrap_prior_count,
        )
    # Root expansion happens before simulations; every budgeted simulation
    # therefore contributes one root visit and one matrix cell sample.
    root.expanded = True
    if config.root_noise_epsilon > 0.0:
        if root_noise_rng is None:
            raise ValueError("root_noise_rng is required when root noise is enabled")
        alpha = config.root_dirichlet_alpha_scale / len(root.drop_actions)
        drop_noise = root_noise_rng.dirichlet(np.full(len(root.drop_actions), alpha))
        check_noise = root_noise_rng.dirichlet(np.full(len(root.check_actions), alpha))
        epsilon = config.root_noise_epsilon
        root.drop_prior = (1.0 - epsilon) * root.drop_prior + epsilon * drop_noise
        root.check_prior = (1.0 - epsilon) * root.check_prior + epsilon * check_noise
        root.prior = np.outer(root.drop_prior, root.check_prior)

    transposition = {_state_key(state, remaining_horizon, rules): root}
    root_drop_sum = np.zeros(len(root.drop_actions), dtype=np.float64)
    root_check_sum = np.zeros(len(root.check_actions), dtype=np.float64)
    root_weight = 0.0

    for _iteration in range(config.iterations):
        node = root
        path: list[tuple[_ToyNode, int, int]] = []
        depth = 0
        while True:
            if node.remaining_horizon <= 0:
                leaf_value = 0.0
                break
            if not node.expanded:
                _drop, _check, _prior, leaf_value = _evaluate(
                    evaluator,
                    node.state,
                    node.remaining_horizon,
                    rules,
                    node.drop_actions,
                    node.check_actions,
                )
                node.expanded = True
                break
            if config.max_depth is not None and depth >= config.max_depth:
                _drop, _check, _prior, leaf_value = _evaluate(
                    evaluator,
                    node.state,
                    node.remaining_horizon,
                    rules,
                    node.drop_actions,
                    node.check_actions,
                )
                break

            if node is root:
                mean_drop, mean_check, _mean_value, _gap = _mean_q_strategies(node)
                # The improvement target is a linearly weighted running mean:
                # weights are 1, 2, ..., N rather than recursively doubling
                # the accumulated denominator.
                weight = float(_iteration + 1)
                root_drop_sum += weight * mean_drop
                root_check_sum += weight * mean_check
                root_weight += weight
                selection_drop, selection_check = _selection_strategies(node, config.exploration_c)
            else:
                selection_drop, selection_check = _selection_strategies(node, config.exploration_c)
            d_index = int(action_rng.choice(len(node.drop_actions), p=selection_drop))
            c_index = int(action_rng.choice(len(node.check_actions), p=selection_check))
            path.append((node, d_index, c_index))
            child, terminal_value = _step_into_child(
                node,
                rules,
                d_index,
                c_index,
                chance_rng,
                evaluator,
                transposition,
            )
            if terminal_value is not None:
                leaf_value = terminal_value
                break
            assert child is not None
            node = child
            depth += 1
        _backup(path, float(leaf_value))

    final_drop, final_check, final_value, _final_gap = _mean_q_strategies(root)
    if root_weight > 0.0:
        improved_drop = root_drop_sum / root_weight
        improved_check = root_check_sum / root_weight
    else:
        improved_drop = final_drop.copy()
        improved_check = final_check.copy()
    improved_drop = _normalize(improved_drop, len(root.drop_actions), name="improved drop policy")
    improved_check = _normalize(improved_check, len(root.check_actions), name="improved check policy")
    root_value, _row_gain, _column_gain, _saddle = saddle_gap(
        root.q_values,
        improved_drop,
        improved_check,
        row_is_hal=state.hal_is_dropper,
    )
    return ToyMCTSResult(
        improved_dropper_policy=improved_drop,
        improved_checker_policy=improved_check,
        mean_q_dropper_policy=final_drop,
        mean_q_checker_policy=final_check,
        root_value_for_hal=root_value,
        mean_q_value_for_hal=final_value,
        root_visits=root.node_visits,
        root_unique_cells_visited=int(np.count_nonzero(root.visits)),
        cells_used=int(root.visits.sum()),
        root_drop_actions=root.drop_actions,
        root_check_actions=root.check_actions,
    )


@dataclass(frozen=True, slots=True)
class ConformanceRecord:
    state_id: str
    horizon: int
    budget: int
    seed: int
    exact_value: float
    search_value: float
    value_error: float
    saddle_gap: float
    exact_saddle_gap: float
    drop_policy_tv: float
    check_policy_tv: float
    root_visits: int
    root_unique_cells_visited: int


def run_conformance(
    rules: ToyRuleset,
    states: tuple[ToyState, ...],
    *,
    horizons: tuple[int, ...] = (1, 4, 8),
    budgets: tuple[int, ...] = (64, 256, 1024),
    seeds: tuple[int, ...] = tuple(range(10)),
    evaluator: ToyLeafEvaluator | None = None,
) -> tuple[ConformanceRecord, ...]:
    """Run exact-leaf MCTS audits over a declared root pack."""

    records: list[ConformanceRecord] = []
    active_evaluator = evaluator or make_exact_evaluator(rules)
    for state_index, state in enumerate(states):
        for horizon in horizons:
            exact = solve_exact(state, horizon, rules, include_transitions=False)
            for budget in budgets:
                for seed in seeds:
                    config = ToyMCTSConfig(
                        rules=rules,
                        iterations=budget,
                        exploration_c=1.0,
                        max_depth=horizon,
                        full_width_bootstrap=(horizon == 1),
                    )
                    result = mcts_search(
                        state,
                        horizon,
                        active_evaluator,
                        config,
                        np.random.default_rng(seed),
                        np.random.default_rng(seed + 100_000),
                    )
                    _expected, _row_gain, _column_gain, gap = saddle_gap(
                        exact.payoff_for_hal,
                        result.improved_dropper_policy,
                        result.improved_checker_policy,
                        row_is_hal=state.hal_is_dropper,
                    )
                    records.append(
                        ConformanceRecord(
                            state_id=f"root-{state_index}",
                            horizon=horizon,
                            budget=budget,
                            seed=seed,
                            exact_value=exact.value_for_hal,
                            search_value=result.root_value_for_hal,
                            value_error=abs(result.root_value_for_hal - exact.value_for_hal),
                            saddle_gap=gap,
                            exact_saddle_gap=exact.saddle_gap,
                            drop_policy_tv=0.5
                            * float(
                                np.abs(
                                    result.improved_dropper_policy[
                                        np.asarray(exact.drop_actions) - 1
                                    ]
                                    - exact.dropper_strategy
                                ).sum()
                            ),
                            check_policy_tv=0.5
                            * float(
                                np.abs(
                                    result.improved_checker_policy[
                                        np.asarray(exact.check_actions) - 1
                                    ]
                                    - exact.checker_strategy
                                ).sum()
                            ),
                            root_visits=result.root_visits,
                            root_unique_cells_visited=result.root_unique_cells_visited,
                        )
                    )
    return tuple(records)


def conformance_summary(records: tuple[ConformanceRecord, ...]) -> dict:
    """Summarize the requested P3-style value and equilibrium gates."""

    if not records:
        return {
            "records": 0,
            "median_value_error": 0.0,
            "p95_value_error": 0.0,
            "max_saddle_gap": 0.0,
            "max_root_value_seed_std": 0.0,
            "budget_median_value_errors": {},
            "value_error_trend_nonworsening": True,
            "gates": {},
        }
    value_errors = np.asarray([record.value_error for record in records], dtype=np.float64)
    saddle_gaps = np.asarray([record.saddle_gap for record in records], dtype=np.float64)
    grouped: dict[tuple[str, int, int], list[float]] = {}
    for record in records:
        grouped.setdefault((record.state_id, record.horizon, record.budget), []).append(
            record.search_value
        )
    seed_stds = [float(np.std(values)) for values in grouped.values()]
    by_budget: dict[int, float] = {}
    for budget in sorted({record.budget for record in records}):
        by_budget[budget] = float(
            np.median([record.value_error for record in records if record.budget == budget])
        )
    budgets = sorted(by_budget)
    trend_nonworsening = all(
        by_budget[later] <= by_budget[earlier] + 0.01
        for earlier, later in zip(budgets, budgets[1:])
    )
    gates = {
        "median_value_error_le_0_05": float(np.median(value_errors)) <= 0.05,
        "p95_value_error_le_0_10": float(np.quantile(value_errors, 0.95)) <= 0.10,
        "max_saddle_gap_le_0_05": float(np.max(saddle_gaps)) <= 0.05,
        "max_root_value_seed_std_le_0_03": (max(seed_stds) if seed_stds else 0.0) <= 0.03,
        "value_error_not_systematically_worsening": trend_nonworsening,
    }
    return {
        "records": len(records),
        "median_value_error": float(np.median(value_errors)),
        "p95_value_error": float(np.quantile(value_errors, 0.95)),
        "max_saddle_gap": float(np.max(saddle_gaps)),
        "max_root_value_seed_std": max(seed_stds) if seed_stds else 0.0,
        "budget_median_value_errors": {str(key): value for key, value in by_budget.items()},
        "value_error_trend_nonworsening": trend_nonworsening,
        "gates": gates,
    }


def conformance_report(records: tuple[ConformanceRecord, ...]) -> dict:
    payload = {
        "schema_version": "toy.mcts_conformance.v2",
        "policy_tv_is_report_only": True,
        "records": [asdict(record) for record in records],
        "summary": conformance_summary(records),
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    payload["report_sha256"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return payload
