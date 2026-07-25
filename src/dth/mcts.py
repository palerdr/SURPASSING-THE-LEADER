from dth.solver import (
    CHECKER_ACTIONS,
    DROPPER_ACTIONS,
    Distribution,
    NTState,
    State,
    TState,
    reward,
    solve_matrix,
    solve,
    payoff,
    transition,
)
from dth.network import DTHNetworkConfig, DTHPolicyValueNet
from dataclasses import dataclass
from pathlib import Path
import json
import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from typing import Protocol

UNIFORM_POLICY = np.full(60, 1.0 / 60)

@dataclass(frozen=True)
class Evaluation:
    value: float
    drop_prior: np.ndarray
    check_prior: np.ndarray
    exact: bool = False

@dataclass(frozen=True)
class MCTSConfig:
    simulations: int = 4096
    confidence_constant: float = 1.5
    exploration_scale: float = 1.0
    prior_uniform_mix: float = 0.05
    policy_update_interval: int = 16
    root_warmup_cells: int = 0
    internal_warmup_cells: int = 0
    internal_warmup_horizons: frozenset[int] | None = None
    max_depth: int | None = None
    root_noise_epsilon: float = 0.0
    root_dirichlet_alpha_scale: float = 10.0

    def __post_init__(self) -> None:
        action_cells = len(DROPPER_ACTIONS) * len(CHECKER_ACTIONS)
        if self.simulations < 0:
            raise ValueError("simulations must be nonnegative")
        if not 0 <= self.root_warmup_cells <= action_cells:
            raise ValueError(f"root_warmup_cells must be in 0..{action_cells}")
        if not 0 <= self.internal_warmup_cells <= action_cells:
            raise ValueError(f"internal_warmup_cells must be in 0..{action_cells}")
        if self.internal_warmup_horizons is not None and any(
            horizon <= 0 for horizon in self.internal_warmup_horizons
        ):
            raise ValueError("internal_warmup_horizons must be positive")
        if self.policy_update_interval <= 0:
            raise ValueError("policy_update_interval must be positive")
        if self.max_depth is not None and self.max_depth <= 0:
            raise ValueError("max_depth must be positive")
        if not 0.0 <= self.root_noise_epsilon <= 1.0:
            raise ValueError("root_noise_epsilon must be in [0, 1]")
        if self.root_dirichlet_alpha_scale <= 0.0:
            raise ValueError("root_dirichlet_alpha_scale must be positive")

@dataclass
class Node:
    state: NTState
    remaining_horizon: int

    expanded: bool
    visits: int

    joint_visits: np.ndarray
    joint_value_sum: np.ndarray

    prior_value: float
    exact_value: float | None
    drop_prior: np.ndarray
    check_prior: np.ndarray

    mean_q_drop: np.ndarray
    mean_q_check: np.ndarray
    mean_q_value: float
    selection_drop: np.ndarray
    selection_check: np.ndarray

    weighted_drop_sum: np.ndarray
    weighted_check_sum: np.ndarray
    policy_weight_sum: float

def make_node(state: NTState, horizon: int) -> Node:
    """Creates a blank node with all the relevant statistics given state and horizon"""
    action_shape = (
        len(DROPPER_ACTIONS),
        len(CHECKER_ACTIONS),
    )

    return Node(
        state=state,
        remaining_horizon=horizon,
        expanded=False,
        visits=0,
        joint_visits=np.zeros(action_shape, dtype=np.uint32),
        joint_value_sum=np.zeros(action_shape, dtype=np.float64),
        prior_value=0.0,
        exact_value=None,
        drop_prior=UNIFORM_POLICY.copy(),
        check_prior=UNIFORM_POLICY.copy(),
        mean_q_drop=UNIFORM_POLICY.copy(),
        mean_q_check=UNIFORM_POLICY.copy(),
        mean_q_value=0.0,
        selection_drop=UNIFORM_POLICY.copy(),
        selection_check=UNIFORM_POLICY.copy(),
        weighted_drop_sum=np.zeros(len(DROPPER_ACTIONS)),
        weighted_check_sum=np.zeros(len(CHECKER_ACTIONS)),
        policy_weight_sum=0.0,
    )

@dataclass(frozen=True)
class MCTSResult:
    value: float
    drop_policy: np.ndarray
    check_policy: np.ndarray

    mean_q_drop_policy: np.ndarray
    mean_q_check_policy: np.ndarray

    joint_visits: np.ndarray
    unique_cells: int
    root_visits: int

    warmup_visits: int
    search_visits: int

class Evaluator(Protocol):
    def __call__(
            self,
            state: NTState,
            remaining_horizon: int,
    ) -> Evaluation:
        ...

class NetworkEvaluator:
    def __init__(
            self,
            checkpoint_path: str | Path,
            device: str | torch.device = "cpu",
            ) -> None:
        self.device = torch.device(device)
        # ``...`` is an Ellipsis value, not a placeholder accepted by torch.load.
        checkpoint = torch.load(
            checkpoint_path,
            map_location=self.device,
            weights_only=False,
        )
        config = DTHNetworkConfig(**checkpoint["model_config"])

        self.model = DTHPolicyValueNet(config)
        self.model.load_state_dict(checkpoint["state_dict"])
        self.model.to(self.device)
        self.model.eval()

    def __call__(self, state: NTState, horizon: int) -> Evaluation:
        # The network API expects a batch, so these have leading dimension one.
        states = torch.tensor([state], dtype=torch.float32, device=self.device)
        horizons = torch.tensor([horizon], dtype=torch.float32, device=self.device)

        with torch.inference_mode():
            features = self.model.encode(states, horizons)
            value, drop_logits, check_logits = self.model(features)

        return Evaluation(
            value=float(value[0].item()),
            drop_prior=torch.softmax(drop_logits[0], dim=-1).cpu().numpy(),
            check_prior=torch.softmax(check_logits[0], dim=-1).cpu().numpy(),
        )


@dataclass(frozen=True)
class ExactTargetStore:
    values: dict[tuple[NTState, int], float]

    @classmethod
    def load(cls, path: str | Path) -> "ExactTargetStore":
        with np.load(Path(path), allow_pickle=False) as artifact:
            states = artifact["states"]
            horizons = artifact["horizons"]
            target_values = artifact["values"]
            if not (
                len(states) == len(horizons) == len(target_values)
            ):
                raise ValueError("exact target arrays must have equal lengths")
            values: dict[tuple[NTState, int], float] = {}
            for state_row, horizon, value in zip(
                states, horizons, target_values, strict=True
            ):
                state = tuple(int(item) for item in state_row)
                key = (state, int(horizon))
                if key in values:
                    raise ValueError("exact targets contain duplicate rows")
                values[key] = float(value)
        return cls(values)


class AnchoredLeafEvaluator:
    """Use certified shallow values below one searched root, then fall back."""

    def __init__(
        self,
        fallback: Evaluator,
        targets: ExactTargetStore,
        root_state: NTState,
        root_horizon: int,
        anchor_horizons: frozenset[int] | None = None,
    ) -> None:
        self.fallback = fallback
        self.targets = targets
        self.root_key = (root_state, root_horizon)
        self.anchor_horizons = anchor_horizons

    def __call__(self, state: NTState, horizon: int) -> Evaluation:
        key = (state, horizon)
        horizon_enabled = (
            self.anchor_horizons is None or horizon in self.anchor_horizons
        )
        if key != self.root_key and horizon_enabled and key in self.targets.values:
            return Evaluation(
                value=self.targets.values[key],
                drop_prior=UNIFORM_POLICY.copy(),
                check_prior=UNIFORM_POLICY.copy(),
                exact=True,
            )
        return self.fallback(state, horizon)


def payoff_from_exact_targets(
    state: NTState,
    horizon: int,
    targets: ExactTargetStore,
) -> np.ndarray:
    """Reconstruct one exact matrix from certified child values."""

    matrix = np.empty((len(DROPPER_ACTIONS), len(CHECKER_ACTIONS)))
    for drop_index, drop in enumerate(DROPPER_ACTIONS):
        for check_index, check in enumerate(CHECKER_ACTIONS):
            action_total = 0.0
            for probability, child in transition(state, drop, check):
                if isinstance(child, TState):
                    branch_value = float(reward(child))
                elif horizon == 1:
                    branch_value = 0.0
                else:
                    key = (child, horizon - 1)
                    if key not in targets.values:
                        raise KeyError(key)
                    branch_value = -targets.values[key]
                action_total += probability * branch_value
            matrix[drop_index, check_index] = action_total
    return matrix


class ExactValueEvaluator:
    def __call__(self, state: NTState, horizon: int) -> Evaluation:
        solution = solve(state, horizon)

        return Evaluation(
            value=solution.value,
            drop_prior=UNIFORM_POLICY.copy(),
            check_prior=UNIFORM_POLICY.copy(),
            exact=True,
        )


def expand(
        node: Node,
        evaluator: Evaluator,
        config: MCTSConfig
        ) -> float:
    """Calls the selected evaluator network, stores the values and priors, and marks node as expanded"""
    if node.remaining_horizon == 0:
        node.prior_value = 0.0
        node.drop_prior = UNIFORM_POLICY.copy()
        node.check_prior = UNIFORM_POLICY.copy()
        node.expanded = True
        return 0.0

    evaluation = evaluator(node.state, node.remaining_horizon)
    node.prior_value = evaluation.value
    node.exact_value = evaluation.value if evaluation.exact else None

    mix = config.prior_uniform_mix

    node.drop_prior = (
        (1.0 - mix) * evaluation.drop_prior
        + mix * UNIFORM_POLICY
    )
    node.check_prior = (
        (1.0 - mix) * evaluation.check_prior
        + mix * UNIFORM_POLICY
    )
    node.drop_prior /= node.drop_prior.sum()
    node.check_prior /= node.check_prior.sum()
    node.expanded = True

    return evaluation.value

def mean_q_matrix(node: Node) -> np.ndarray:
    """Converts the visit/value sums for a node in to the approximate payoff matrix Q = W/N"""
    matrix = np.full(
        node.joint_visits.shape,
        node.prior_value,
        dtype=np.float64,
    )
    visited = node.joint_visits > 0
    matrix[visited] = (
        node.joint_value_sum[visited]
        / node.joint_visits[visited]
    )
    return matrix

def refresh_policies(node: Node, config: MCTSConfig) -> None:
    Q = mean_q_matrix(node)
    mean_value, mean_drop, mean_check = solve_matrix(Q)

    joint_prior = np.outer(node.drop_prior, node.check_prior)

    # \(U_{ij}=cP_{ij}\frac{\sqrt{N}}{1+N_{ij}}.\)
    bonus = (
        config.confidence_constant
        * joint_prior
        * np.sqrt(max(node.visits, 1))
        / (1 + node.joint_visits)
    )

    try:
        _, optimistic_drop, _ = solve_matrix(Q + bonus)
    except RuntimeError as error:
        if "LP saddle gap too large" not in str(error):
            raise
        optimistic_drop = mean_drop
    try:
        _, _, optimistic_check = solve_matrix(Q - bonus)
    except RuntimeError as error:
        if "LP saddle gap too large" not in str(error):
            raise
        optimistic_check = mean_check

    node.mean_q_drop = mean_drop
    node.mean_q_check = mean_check
    node.mean_q_value = float(mean_value)

    # np.max(policy) was one scalar and could not be passed to rng.choice.
    # Keep the complete policies and mix in a decaying amount of prior exploration.
    epsilon = min(
        1.0,
        config.exploration_scale / np.sqrt(node.visits + 1),
    )
    node.selection_drop = (
        (1.0 - epsilon) * optimistic_drop
        + epsilon * node.drop_prior
    )
    node.selection_check = (
        (1.0 - epsilon) * optimistic_check
        + epsilon * node.check_prior
    )
    node.selection_drop /= node.selection_drop.sum()
    node.selection_check /= node.selection_check.sum()

def sample_chance_branch(
        branches: Distribution,
        rng: np.random.Generator,
        ) -> State:
    probabilities = np.asarray([probability for probability, _ in branches])

    index = int(rng.choice(len(branches), p=probabilities))
    return branches[index][1]

def simulate(
        node: Node,
        table: dict[tuple[NTState, int], Node],
        evaluator: Evaluator,
        config: MCTSConfig,
        action_rng: np.random.Generator,
        chance_rng: np.random.Generator,
        forced_joint_action: tuple[int, int] | None = None,
        record_policy: bool = True,
        depth: int = 0,
) -> float:
    if node.remaining_horizon == 0:
        return 0.0
    if not node.expanded:
        value = expand(node, evaluator, config)
        if (
            depth > 0
            and config.internal_warmup_cells
            and (
                config.internal_warmup_horizons is None
                or node.remaining_horizon in config.internal_warmup_horizons
            )
            and (config.max_depth is None or depth < config.max_depth)
        ):
            warmup_node(
                node,
                table,
                evaluator,
                config,
                action_rng,
                chance_rng,
                cells=config.internal_warmup_cells,
                depth=depth,
            )
            value, _, _ = solve_matrix(mean_q_matrix(node))
        return value
    if node.exact_value is not None:
        return node.exact_value
    if config.max_depth is not None and depth >= config.max_depth:
        return node.prior_value

    if (
        forced_joint_action is None
        and node.visits % config.policy_update_interval == 0
    ):
        refresh_policies(node, config)

    drop_index = int(action_rng.choice(len(DROPPER_ACTIONS), p=node.selection_drop)) if forced_joint_action is None else forced_joint_action[0]
    check_index = int(action_rng.choice(len(CHECKER_ACTIONS), p=node.selection_check)) if forced_joint_action is None else forced_joint_action[1]

    drop = DROPPER_ACTIONS[drop_index]
    check = CHECKER_ACTIONS[check_index]

    outcome = 0.0
    for probability, child in transition(node.state, drop, check):
        if isinstance(child, TState):
            branch_value = float(reward(child))
        else:
            child_key = (child, node.remaining_horizon - 1)
            if child_key not in table:
                table[child_key] = make_node(*child_key)
            # A live transition swaps roles, so the child value changes sign once.
            branch_value = -simulate(
                table[child_key],
                table,
                evaluator,
                config,
                action_rng,
                chance_rng,
                depth=depth + 1,
            )
        outcome += probability * branch_value

    node.joint_visits[drop_index, check_index] += 1
    node.joint_value_sum[drop_index, check_index] += outcome
    node.visits += 1

    if record_policy:
        weight = float(node.visits)
        # Both sums must use the same weight; the Dropper weight was previously missing.
        node.weighted_check_sum += weight * node.mean_q_check
        node.weighted_drop_sum += weight * node.mean_q_drop
        node.policy_weight_sum += weight

    # Parents need this matrix node's minimax estimate, not one sampled
    # optimistic-policy trajectory through the node.
    return node.mean_q_value


def warmup_node(
    node: Node,
    table: dict[tuple[NTState, int], Node],
    evaluator: Evaluator,
    config: MCTSConfig,
    action_rng: np.random.Generator,
    chance_rng: np.random.Generator,
    *,
    cells: int,
    depth: int,
) -> int:
    cell_count = len(DROPPER_ACTIONS) * len(CHECKER_ACTIONS)
    visits = 0
    for flat_index in action_rng.permutation(cell_count)[:cells]:
        drop_index, check_index = divmod(int(flat_index), len(CHECKER_ACTIONS))
        simulate(
            node,
            table,
            evaluator,
            config,
            action_rng,
            chance_rng,
            forced_joint_action=(drop_index, check_index),
            record_policy=False,
            depth=depth,
        )
        visits += 1
    refresh_policies(node, config)
    return visits


def _result_from_root(
    root: Node,
    *,
    warmup_visits: int,
    search_visits: int,
) -> MCTSResult:
    """Snapshot one root without mutating its accumulated search statistics."""

    value, mean_q_drop, mean_q_check = solve_matrix(mean_q_matrix(root))
    if root.policy_weight_sum > 0.0:
        drop_policy = root.weighted_drop_sum / root.policy_weight_sum
        check_policy = root.weighted_check_sum / root.policy_weight_sum
        drop_policy /= drop_policy.sum()
        check_policy /= check_policy.sum()
    else:
        drop_policy = mean_q_drop.copy()
        check_policy = mean_q_check.copy()

    return MCTSResult(
        value=float(value),
        drop_policy=drop_policy.copy(),
        check_policy=check_policy.copy(),
        mean_q_drop_policy=mean_q_drop,
        mean_q_check_policy=mean_q_check,
        joint_visits=root.joint_visits.copy(),
        unique_cells=int(np.count_nonzero(root.joint_visits)),
        root_visits=root.visits,
        warmup_visits=warmup_visits,
        search_visits=search_visits,
    )


def mcts_search_ladder(
    state: NTState,
    remaining_horizon: int,
    evaluator: Evaluator,
    config: MCTSConfig,
    action_rng: np.random.Generator,
    chance_rng: np.random.Generator,
    root_noise_rng: np.random.Generator | None = None,
    *,
    budgets: list[int] | tuple[int, ...],
) -> dict[int, MCTSResult]:
    """Run one shared random prefix and snapshot requested simulation budgets."""

    ordered_budgets = sorted(set(int(budget) for budget in budgets))
    if not ordered_budgets or ordered_budgets[0] < 0:
        raise ValueError("budgets must contain nonnegative integers")

    root = make_node(state, remaining_horizon)
    table: dict[tuple[NTState, int], Node] = {
        (state, remaining_horizon): root
    }

    _ = expand(root, evaluator, config)
    if config.root_noise_epsilon > 0.0:
        if root_noise_rng is None:
            raise ValueError("root_noise_rng is required when root noise is enabled")
        alpha = config.root_dirichlet_alpha_scale / len(DROPPER_ACTIONS)
        drop_noise = root_noise_rng.dirichlet(
            np.full(len(DROPPER_ACTIONS), alpha)
        )
        check_noise = root_noise_rng.dirichlet(
            np.full(len(CHECKER_ACTIONS), alpha)
        )
        epsilon = config.root_noise_epsilon
        root.drop_prior = (
            (1.0 - epsilon) * root.drop_prior + epsilon * drop_noise
        )
        root.check_prior = (
            (1.0 - epsilon) * root.check_prior + epsilon * check_noise
        )
        root.drop_prior /= root.drop_prior.sum()
        root.check_prior /= root.check_prior.sum()
    warmup_visits = 0
    if config.root_warmup_cells:
        warmup_visits = warmup_node(
            root,
            table,
            evaluator,
            config,
            action_rng,
            chance_rng,
            cells=config.root_warmup_cells,
            depth=0,
        )

    results: dict[int, MCTSResult] = {}
    if 0 in ordered_budgets:
        results[0] = _result_from_root(
            root,
            warmup_visits=warmup_visits,
            search_visits=0,
        )

    requested = set(ordered_budgets)
    for completed in range(1, ordered_budgets[-1] + 1):
        simulate(
            root,
            table,
            evaluator,
            config,
            action_rng,
            chance_rng,
            )
        if completed in requested:
            results[completed] = _result_from_root(
                root,
                warmup_visits=warmup_visits,
                search_visits=completed,
            )

    return results


# public MCTS function
def mcts_search(
    state: NTState,
    remaining_horizon: int,
    evaluator: Evaluator,
    config: MCTSConfig,
    action_rng: np.random.Generator,
    chance_rng: np.random.Generator,
    root_noise_rng: np.random.Generator | None = None,
) -> MCTSResult:
    return mcts_search_ladder(
        state,
        remaining_horizon,
        evaluator,
        config,
        action_rng,
        chance_rng,
        root_noise_rng,
        budgets=[config.simulations],
    )[config.simulations]


# convergence audit
class ExactLeafEvaluator:
    """Uses exact child values without giving the root's exact value to MCTS."""
    def __init__(
        self,
        root_state: NTState,
        root_horizon: int,
        targets: ExactTargetStore | None = None,
    ) -> None:
        self.root_key = (root_state, root_horizon)
        self.targets = targets

    def __call__(self, state: NTState, horizon: int) -> Evaluation:
        is_root = (state, horizon) == self.root_key
        if is_root:
            value = 0.0
        elif self.targets is not None and (state, horizon) in self.targets.values:
            value = self.targets.values[(state, horizon)]
        else:
            value = solve(state, horizon).value

        return Evaluation(
            value=value,
            drop_prior=UNIFORM_POLICY.copy(),
            check_prior=UNIFORM_POLICY.copy(),
            exact=not is_root,
        )


def saddle_gap(
        matrix: np.ndarray,
        drop_policy: np.ndarray,
        check_policy: np.ndarray,
        ) -> float:
    """Exploitability of the two returned policies in an exact payoff matrix."""
    lower = np.min(matrix.T @ drop_policy)
    upper = np.max(matrix @ check_policy)
    return float(upper - lower)


def summarize_audit(records: list[dict]) -> dict:
    """Aggregate the per-root audit records by evaluator and budget."""
    summary = {}
    evaluators = sorted({record["evaluator"] for record in records})

    warmup_values = sorted({int(record["warmup_cells"]) for record in records})
    is_ladder = len(warmup_values) > 1
    for evaluator_name in evaluators:
        summary[evaluator_name] = {}
        budgets = sorted({
            record["budget"]
            for record in records
            if record["evaluator"] == evaluator_name
        })

        groups = (
            [(warmup, budget) for warmup in warmup_values for budget in budgets]
            if is_ladder
            else [(warmup_values[0], budget) for budget in budgets]
        )
        for warmup, budget in groups:
            group = [
                record for record in records
                if record["evaluator"] == evaluator_name
                and record["budget"] == budget
                and record["warmup_cells"] == warmup
            ]
            if not group:
                continue
            value_errors = np.asarray([
                record["value_error"] for record in group
            ])
            saddle_gaps = np.asarray([
                record["saddle_gap"] for record in group
            ])
            mean_q_saddle_gaps = np.asarray([
                record["mean_q_saddle_gap"] for record in group
            ])
            coverage_fractions = np.asarray([
                record["coverage_fraction"] for record in group
            ])

            values_by_root = {}
            for record in group:
                root_key = (
                    tuple(record["state"]),
                    record["horizon"],
                )
                values_by_root.setdefault(root_key, []).append(
                    record["mcts_value"]
                )
            seed_stds = [
                np.std(values)
                for values in values_by_root.values()
            ]

            destination = summary[evaluator_name]
            if is_ladder:
                destination = destination.setdefault(str(warmup), {})
            destination[str(budget)] = {
                "runs": len(group),
                "median_value_error": float(np.median(value_errors)),
                "p95_value_error": float(np.quantile(value_errors, 0.95)),
                "median_saddle_gap": float(np.median(saddle_gaps)),
                "p95_saddle_gap": float(np.quantile(saddle_gaps, 0.95)),
                "max_saddle_gap": float(np.max(saddle_gaps)),
                "median_mean_q_saddle_gap": float(
                    np.median(mean_q_saddle_gaps)
                ),
                "p95_mean_q_saddle_gap": float(
                    np.quantile(mean_q_saddle_gaps, 0.95)
                ),
                "max_mean_q_saddle_gap": float(np.max(mean_q_saddle_gaps)),
                "max_root_value_seed_std": float(np.max(seed_stds)),
                "mean_unique_cells": float(np.mean([
                    record["unique_cells"] for record in group
                ])),
                "mean_coverage_fraction": float(np.mean(coverage_fractions)),
            }

    return summary


def run_convergence_audit(config: dict) -> dict:
    """Run exact comparisons and write one deterministic JSON report."""
    checkpoint_path = Path(config["checkpoint"])
    evaluator_names = list(config["evaluators"])
    network_evaluator = (
        NetworkEvaluator(checkpoint_path, device=str(config.get("device", "cpu")))
        if {"network", "anchored_network"}.intersection(evaluator_names)
        else None
    )
    exact_targets = (
        ExactTargetStore.load(config["exact_targets"])
        if "anchored_network" in evaluator_names
        else None
    )
    anchor_horizons = (
        frozenset(int(value) for value in config["anchor_horizons"])
        if config.get("anchor_horizons") is not None
        else None
    )
    reference_targets = (
        ExactTargetStore.load(
            config.get("reference_targets", config.get("exact_targets"))
        )
        if config.get("reference_targets", config.get("exact_targets"))
        is not None
        else None
    )

    mcts_values = config["mcts"]
    budgets = [int(value) for value in config["budgets"]]
    seeds = [int(value) for value in config["seeds"]]
    warmup_values = [
        int(value) for value in config.get(
            "warmup_cells",
            [mcts_values.get("root_warmup_cells", 0)],
        )
    ]
    records = []
    references = []

    # Reuse exact Bellman results across all roots and runs in this process.
    solve.cache_clear()
    for root_index, root_config in enumerate(config["roots"]):
        state = tuple(int(value) for value in root_config["state"])
        horizon = int(root_config["horizon"])
        exact_value: float
        exact_drop_policy: np.ndarray | tuple[float, ...]
        exact_check_policy: np.ndarray | tuple[float, ...]
        if reference_targets is not None and (state, horizon) in reference_targets.values:
            try:
                exact_matrix = payoff_from_exact_targets(
                    state,
                    horizon,
                    reference_targets,
                )
            except KeyError:
                exact_solution = solve(state, horizon)
                exact_matrix = payoff(state, horizon)
                exact_value = exact_solution.value
                exact_drop_policy = exact_solution.drop_policy
                exact_check_policy = exact_solution.check_policy
            else:
                exact_value, exact_drop_policy, exact_check_policy = solve_matrix(
                    exact_matrix
                )
                if not np.isclose(
                    exact_value,
                    reference_targets.values[(state, horizon)],
                    atol=1e-6,
                ):
                    raise ValueError("reconstructed exact root value does not match target")
        else:
            exact_solution = solve(state, horizon)
            exact_matrix = payoff(state, horizon)
            exact_value = exact_solution.value
            exact_drop_policy = exact_solution.drop_policy
            exact_check_policy = exact_solution.check_policy

        references.append({
            "state": list(state),
            "horizon": horizon,
            "exact_value": exact_value,
            "exact_drop_policy": list(exact_drop_policy),
            "exact_check_policy": list(exact_check_policy),
        })

        for evaluator_index, evaluator_name in enumerate(evaluator_names):
            if evaluator_name == "exact_leaf":
                evaluator = ExactLeafEvaluator(
                    state,
                    horizon,
                    reference_targets,
                )
            elif evaluator_name == "network" and network_evaluator is not None:
                evaluator = network_evaluator
            elif (
                evaluator_name == "anchored_network"
                and network_evaluator is not None
                and exact_targets is not None
            ):
                evaluator = AnchoredLeafEvaluator(
                    network_evaluator,
                    exact_targets,
                    state,
                    horizon,
                    anchor_horizons,
                )
            else:
                raise ValueError(f"unknown evaluator {evaluator_name!r}")

            for warmup_cells in warmup_values:
                search_config = MCTSConfig(
                    simulations=max(budgets),
                    confidence_constant=float(
                        mcts_values["confidence_constant"]
                    ),
                    exploration_scale=float(mcts_values["exploration_scale"]),
                    prior_uniform_mix=float(mcts_values["prior_uniform_mix"]),
                    policy_update_interval=int(
                        mcts_values["policy_update_interval"]
                    ),
                    root_warmup_cells=warmup_cells,
                    internal_warmup_cells=int(
                        mcts_values.get("internal_warmup_cells", 0)
                    ),
                    internal_warmup_horizons=(
                        frozenset(
                            int(value)
                            for value in mcts_values[
                                "internal_warmup_horizons"
                            ]
                        )
                        if mcts_values.get("internal_warmup_horizons")
                        is not None
                        else None
                    ),
                    max_depth=(
                        None
                        if mcts_values.get("max_depth") is None
                        else int(mcts_values["max_depth"])
                    ),
                )

                for seed in seeds:
                    stream_seed = (
                        seed
                        + 10_000 * root_index
                        + 1_000_000 * evaluator_index
                    )
                    action_rng = np.random.default_rng(2 * stream_seed + 1)
                    chance_rng = np.random.default_rng(2 * stream_seed + 2)
                    ladder = mcts_search_ladder(
                        state,
                        horizon,
                        evaluator,
                        search_config,
                        action_rng,
                        chance_rng,
                        budgets=budgets,
                    )

                    for budget in budgets:
                        result = ladder[budget]
                        record = {
                            "state": list(state),
                            "horizon": horizon,
                            "evaluator": evaluator_name,
                            "budget": budget,
                            "warmup_cells": warmup_cells,
                            "seed": seed,
                            "exact_value": exact_value,
                            "mcts_value": result.value,
                            "value_error": abs(
                                result.value - exact_value
                            ),
                            "saddle_gap": saddle_gap(
                                exact_matrix,
                                result.drop_policy,
                                result.check_policy,
                            ),
                            "mean_q_saddle_gap": saddle_gap(
                                exact_matrix,
                                result.mean_q_drop_policy,
                                result.mean_q_check_policy,
                            ),
                            "root_visits": result.root_visits,
                            "warmup_visits": result.warmup_visits,
                            "search_visits": result.search_visits,
                            "unique_cells": result.unique_cells,
                            "coverage_fraction": result.unique_cells / (
                                len(DROPPER_ACTIONS) * len(CHECKER_ACTIONS)
                            ),
                            "drop_policy": result.drop_policy.tolist(),
                            "check_policy": result.check_policy.tolist(),
                            "mean_q_drop_policy": (
                                result.mean_q_drop_policy.tolist()
                            ),
                            "mean_q_check_policy": (
                                result.mean_q_check_policy.tolist()
                            ),
                        }
                        records.append(record)
                        print(
                            f"{evaluator_name} h={horizon} warmup={warmup_cells} "
                            f"budget={budget} "
                            f"seed={seed} gap={record['saddle_gap']:.4f} "
                            f"error={record['value_error']:.4f}",
                            flush=True,
                        )

    report = {
        "schema_version": "dth-mcts-convergence-v1",
        "checkpoint": str(checkpoint_path),
        "config": config,
        "references": references,
        "summary": summarize_audit(records),
        "records": records,
    }

    output_path = Path(config["output"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote MCTS audit to {output_path}", flush=True)
    return report


@hydra.main(version_base="1.3", config_path="config", config_name="mcts_audit")
def main(config: DictConfig) -> None:
    values = OmegaConf.to_container(config, resolve=True)
    if not isinstance(values, dict):
        raise TypeError("MCTS audit config must resolve to a mapping")
    run_convergence_audit(values)


if __name__ == "__main__":
    main()
