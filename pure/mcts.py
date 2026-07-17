from pure.solver import (
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
from pure.network import PureNetworkConfig, PurePolicyValueNet
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

@dataclass(frozen=True)
class MCTSConfig:
    simulations: int = 4096
    confidence_constant: float = 1.5
    exploration_scale: float = 1.0
    prior_uniform_mix: float = 0.05
    policy_update_interval: int = 16

@dataclass
class Node:
    state: NTState
    remaining_horizon: int

    expanded: bool
    visits: int

    joint_visits: np.ndarray
    joint_value_sum: np.ndarray

    prior_value: float
    drop_prior: np.ndarray
    check_prior: np.ndarray

    mean_q_drop: np.ndarray
    mean_q_check: np.ndarray
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
        drop_prior=UNIFORM_POLICY.copy(),
        check_prior=UNIFORM_POLICY.copy(),
        mean_q_drop=UNIFORM_POLICY.copy(),
        mean_q_check=UNIFORM_POLICY.copy(),
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

class Evaluator(Protocol):
    def __call__(
            self,
            state: NTState,
            remaining_horizon: int,
    ) -> Evaluation:
        ...

class NetworkEvaluator:
    def __init__(self, checkpoint_path: str | Path) -> None:
        # ``...`` is an Ellipsis value, not a placeholder accepted by torch.load.
        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=False,
        )
        config = PureNetworkConfig(**checkpoint["model_config"])

        self.model = PurePolicyValueNet(config)
        self.model.load_state_dict(checkpoint["state_dict"])
        self.model.eval()

    def __call__(self, state: NTState, horizon: int) -> Evaluation:
        # The network API expects a batch, so these have leading dimension one.
        states = torch.tensor([state], dtype=torch.float32)
        horizons = torch.tensor([horizon], dtype=torch.float32)

        with torch.inference_mode():
            features = self.model.encode(states, horizons)
            value, drop_logits, check_logits = self.model(features)

        return Evaluation(
            value=float(value[0].item()),
            drop_prior=torch.softmax(drop_logits[0], dim=-1).numpy(),
            check_prior=torch.softmax(check_logits[0], dim=-1).numpy(),
        )


class ExactValueEvaluator:
    def __call__(self, state: NTState, horizon: int) -> Evaluation:
        solution = solve(state, horizon)

        return Evaluation(
            value=solution.value,
            drop_prior=UNIFORM_POLICY.copy(),
            check_prior=UNIFORM_POLICY.copy(),
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
    _, mean_drop, mean_check = solve_matrix(Q)

    bonus = config.confidence_constant * np.sqrt(
        np.log(node.visits + 2) / (node.joint_visits + 1)
    )
    bonus = np.minimum(bonus, 1.0)

    _, optimistic_drop, _ = solve_matrix(np.clip(Q + bonus, -1.0, 1.0))
    _, _, optimistic_check = solve_matrix(np.clip(Q - bonus, -1.0, 1.0))

    node.mean_q_drop = mean_drop
    node.mean_q_check = mean_check

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
) -> float:
    if node.remaining_horizon == 0:
        return 0.0
    if not node.expanded:
        return expand(node, evaluator, config)

    if node.visits % config.policy_update_interval == 0:
        refresh_policies(node, config)

    drop_index = int(
        action_rng.choice(len(DROPPER_ACTIONS), p=node.selection_drop)
    )
    check_index = int(
        action_rng.choice(len(CHECKER_ACTIONS), p=node.selection_check)
    )

    drop = DROPPER_ACTIONS[drop_index]
    check = CHECKER_ACTIONS[check_index]

    child = sample_chance_branch(transition(node.state, drop, check), chance_rng)

    if isinstance(child, TState):
        outcome = float(reward(child))
    else:
        child_key = (child, node.remaining_horizon - 1)
        if child_key not in table:
            table[child_key] = make_node(*child_key)
        # A live transition swaps roles, so the child value changes sign once.
        outcome = -simulate(
            table[child_key],
            table,
            evaluator,
            config,
            action_rng,
            chance_rng,
        )

    node.joint_visits[drop_index, check_index] += 1
    node.joint_value_sum[drop_index, check_index] += outcome
    node.visits += 1

    weight = float(node.visits)
    # Both sums must use the same weight; the Dropper weight was previously missing.
    node.weighted_check_sum += weight * node.mean_q_check
    node.weighted_drop_sum += weight * node.mean_q_drop
    node.policy_weight_sum += weight

    return outcome


# public MCTS function
def mcts_search(
    state: NTState,
    remaining_horizon: int,
    evaluator: Evaluator,
    config: MCTSConfig,
    action_rng: np.random.Generator,
    chance_rng: np.random.Generator,
) -> MCTSResult:

    root = make_node(state, remaining_horizon)
    table: dict[tuple[NTState, int], Node] = {
        (state, remaining_horizon): root
    }

    _ = expand(root, evaluator, config)
    for _ in range(config.simulations):
        simulate(
            root,
            table,
            evaluator,
            config,
            action_rng,
            chance_rng,
            )

    # One final solve gives the final estimated matrix value and diagnostic policies.
    value, mean_q_drop, mean_q_check = solve_matrix(mean_q_matrix(root))

    # These are the policy-improvement targets accumulated throughout search.
    # If no simulations ran, use the final mean-Q policies as the sensible fallback.
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
        drop_policy=drop_policy,
        check_policy=check_policy,
        mean_q_drop_policy=mean_q_drop,
        mean_q_check_policy=mean_q_check,
        joint_visits=root.joint_visits.copy(),
        unique_cells=int(np.count_nonzero(root.joint_visits)),
        root_visits=root.visits,
    )


# convergence audit
class ExactLeafEvaluator:
    """Uses exact child values without giving the root's exact value to MCTS."""
    def __init__(self, root_state: NTState, root_horizon: int) -> None:
        self.root_key = (root_state, root_horizon)

    def __call__(self, state: NTState, horizon: int) -> Evaluation:
        if (state, horizon) == self.root_key:
            value = 0.0
        else:
            value = solve(state, horizon).value

        return Evaluation(
            value=value,
            drop_prior=UNIFORM_POLICY.copy(),
            check_prior=UNIFORM_POLICY.copy(),
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

    for evaluator_name in evaluators:
        summary[evaluator_name] = {}
        budgets = sorted({
            record["budget"]
            for record in records
            if record["evaluator"] == evaluator_name
        })

        for budget in budgets:
            group = [
                record for record in records
                if record["evaluator"] == evaluator_name
                and record["budget"] == budget
            ]
            value_errors = np.asarray([
                record["value_error"] for record in group
            ])
            saddle_gaps = np.asarray([
                record["saddle_gap"] for record in group
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

            summary[evaluator_name][str(budget)] = {
                "runs": len(group),
                "median_value_error": float(np.median(value_errors)),
                "p95_value_error": float(np.quantile(value_errors, 0.95)),
                "median_saddle_gap": float(np.median(saddle_gaps)),
                "p95_saddle_gap": float(np.quantile(saddle_gaps, 0.95)),
                "max_saddle_gap": float(np.max(saddle_gaps)),
                "max_root_value_seed_std": float(np.max(seed_stds)),
                "mean_unique_cells": float(np.mean([
                    record["unique_cells"] for record in group
                ])),
            }

    return summary


def run_convergence_audit(config: dict) -> dict:
    """Run exact comparisons and write one deterministic JSON report."""
    checkpoint_path = Path(config["checkpoint"])
    evaluator_names = list(config["evaluators"])
    network_evaluator = (
        NetworkEvaluator(checkpoint_path)
        if "network" in evaluator_names
        else None
    )

    mcts_values = config["mcts"]
    budgets = [int(value) for value in config["budgets"]]
    seeds = [int(value) for value in config["seeds"]]
    records = []
    references = []

    # Reuse exact Bellman results across all roots and runs in this process.
    solve.cache_clear()
    for root_index, root_config in enumerate(config["roots"]):
        state = tuple(int(value) for value in root_config["state"])
        horizon = int(root_config["horizon"])
        exact_solution = solve(state, horizon)
        exact_matrix = payoff(state, horizon)

        references.append({
            "state": list(state),
            "horizon": horizon,
            "exact_value": exact_solution.value,
            "exact_drop_policy": list(exact_solution.drop_policy),
            "exact_check_policy": list(exact_solution.check_policy),
        })

        for evaluator_index, evaluator_name in enumerate(evaluator_names):
            if evaluator_name == "exact_leaf":
                evaluator = ExactLeafEvaluator(state, horizon)
            elif evaluator_name == "network" and network_evaluator is not None:
                evaluator = network_evaluator
            else:
                raise ValueError(f"unknown evaluator {evaluator_name!r}")

            for budget in budgets:
                search_config = MCTSConfig(
                    simulations=budget,
                    confidence_constant=float(
                        mcts_values["confidence_constant"]
                    ),
                    exploration_scale=float(mcts_values["exploration_scale"]),
                    prior_uniform_mix=float(mcts_values["prior_uniform_mix"]),
                    policy_update_interval=int(
                        mcts_values["policy_update_interval"]
                    ),
                )

                for seed in seeds:
                    # Budgets intentionally do not affect these seeds.  Fresh
                    # searches with the same seed therefore share the same
                    # random prefix, making budget comparisons less noisy.
                    stream_seed = (
                        seed
                        + 10_000 * root_index
                        + 1_000_000 * evaluator_index
                    )
                    action_rng = np.random.default_rng(2 * stream_seed + 1)
                    chance_rng = np.random.default_rng(2 * stream_seed + 2)

                    result = mcts_search(
                        state,
                        horizon,
                        evaluator,
                        search_config,
                        action_rng,
                        chance_rng,
                    )

                    record = {
                        "state": list(state),
                        "horizon": horizon,
                        "evaluator": evaluator_name,
                        "budget": budget,
                        "seed": seed,
                        "exact_value": exact_solution.value,
                        "mcts_value": result.value,
                        "value_error": abs(
                            result.value - exact_solution.value
                        ),
                        "saddle_gap": saddle_gap(
                            exact_matrix,
                            result.drop_policy,
                            result.check_policy,
                        ),
                        "root_visits": result.root_visits,
                        "unique_cells": result.unique_cells,
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
                        f"{evaluator_name} h={horizon} budget={budget} "
                        f"seed={seed} gap={record['saddle_gap']:.4f} "
                        f"error={record['value_error']:.4f}",
                        flush=True,
                    )

    report = {
        "schema_version": "pure-mcts-convergence-v1",
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
