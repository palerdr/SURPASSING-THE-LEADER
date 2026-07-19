"""Hydra-configured supervised training on DTH exact targets."""

from __future__ import annotations

import hashlib
import json
import random
import shutil
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import hydra
import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf
from torch import Tensor
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset

from dth.generate_dataset import TARGET_SCHEMA, live_successors
from dth.network import FEATURE_SCHEMA, DTHNetworkConfig, DTHPolicyValueNet
from dth.self_play import validate_replay
from dth.mcts import ExactTargetStore, payoff_from_exact_targets
from dth.solver import (
    CHECKER_ACTIONS,
    DROPPER_ACTIONS,
    NTState,
    TState,
    payoff,
    reward,
    solve,
    solve_matrix,
    transition,
)


@dataclass(frozen=True)
class ExactTargets:
    states: np.ndarray
    horizons: np.ndarray
    values: np.ndarray
    drop_policies: np.ndarray
    check_policies: np.ndarray
    value_weights: np.ndarray
    dataset_version: str
    schema_version: str

    def __len__(self) -> int:
        return int(self.states.shape[0])


@dataclass(frozen=True)
class SelfPlayTargets:
    states: np.ndarray
    horizons: np.ndarray
    values: np.ndarray
    drop_policies: np.ndarray
    check_policies: np.ndarray
    value_weights: np.ndarray

    def __len__(self) -> int:
        return int(self.states.shape[0])


class TargetRows(Dataset[tuple[Tensor, ...]]):
    def __init__(
        self,
        targets: ExactTargets | SelfPlayTargets,
        indices: np.ndarray,
        *,
        horizon_scale: float,
    ) -> None:
        states = torch.from_numpy(targets.states[indices].astype(np.float32))
        horizons = torch.from_numpy(targets.horizons[indices].astype(np.float32))
        self.features = torch.cat(
            (
                states / 300.0,
                horizons.reshape(-1, 1) / float(horizon_scale),
            ),
            dim=1,
        )
        self.horizons = horizons.to(dtype=torch.int64)
        self.values = torch.from_numpy(targets.values[indices].astype(np.float32))
        self.drop_policies = torch.from_numpy(
            targets.drop_policies[indices].astype(np.float32)
        )
        self.check_policies = torch.from_numpy(
            targets.check_policies[indices].astype(np.float32)
        )
        self.value_weights = torch.from_numpy(
            targets.value_weights[indices].astype(np.float32)
        )

        counts = defaultdict(int)
        for horizon in self.horizons.tolist():
            counts[int(horizon)] += 1
        weights = np.asarray(
            [1.0 / counts[int(horizon)] for horizon in self.horizons.tolist()],
            dtype=np.float32,
        )
        weights /= weights.mean()
        self.weights = torch.from_numpy(weights)

    def __len__(self) -> int:
        return int(self.values.shape[0])

    def __getitem__(self, index: int) -> tuple[Tensor, ...]:
        return (
            self.features[index],
            self.values[index],
            self.drop_policies[index],
            self.check_policies[index],
            self.weights[index],
            self.value_weights[index],
            self.horizons[index],
        )


def _scalar(array: np.ndarray) -> str:
    return str(np.asarray(array).item())


def load_exact_targets(path: str | Path) -> ExactTargets:
    source = Path(path)
    with np.load(source, allow_pickle=False) as artifact:
        required = {
            "states",
            "horizons",
            "values",
            "drop_policies",
            "check_policies",
            "drop_actions",
            "check_actions",
            "schema_version",
        }
        missing = required.difference(artifact.files)
        if missing:
            raise ValueError(f"target artifact is missing arrays: {sorted(missing)}")

        states = artifact["states"].astype(np.int16, copy=True)
        horizons = artifact["horizons"].astype(np.uint8, copy=True)
        values = artifact["values"].astype(np.float32, copy=True)
        drop_policies = artifact["drop_policies"].astype(np.float32, copy=True)
        check_policies = artifact["check_policies"].astype(np.float32, copy=True)
        drop_actions = artifact["drop_actions"].astype(np.int16, copy=False)
        check_actions = artifact["check_actions"].astype(np.int16, copy=False)
        schema_version = _scalar(artifact["schema_version"])
        dataset_version = (
            _scalar(artifact["dataset_version"])
            if "dataset_version" in artifact.files
            else source.stem
        )

    rows = len(states)
    if states.shape != (rows, 4):
        raise ValueError(f"states must have shape (N, 4), got {states.shape}")
    if horizons.shape != (rows,) or values.shape != (rows,):
        raise ValueError("horizons and values must each have shape (N,)")
    if drop_policies.shape != (rows, len(DROPPER_ACTIONS)):
        raise ValueError(f"invalid drop policy shape {drop_policies.shape}")
    if check_policies.shape != (rows, len(CHECKER_ACTIONS)):
        raise ValueError(f"invalid check policy shape {check_policies.shape}")
    if not np.array_equal(drop_actions, np.asarray(DROPPER_ACTIONS)):
        raise ValueError("drop action schema does not match the DTH solver")
    if not np.array_equal(check_actions, np.asarray(CHECKER_ACTIONS)):
        raise ValueError("check action schema does not match the DTH solver")
    if schema_version != TARGET_SCHEMA:
        raise ValueError(
            f"target schema {schema_version!r} does not match {TARGET_SCHEMA!r}"
        )
    for name, policies in (
        ("drop", drop_policies),
        ("check", check_policies),
    ):
        if not np.isfinite(policies).all() or (policies < -1e-7).any():
            raise ValueError(f"{name} policies contain invalid probabilities")
        if not np.allclose(policies.sum(axis=1), 1.0, atol=1e-5):
            raise ValueError(f"{name} policies are not normalized")

    return ExactTargets(
        states=states,
        horizons=horizons,
        values=values,
        drop_policies=drop_policies,
        check_policies=check_policies,
        value_weights=np.ones(rows, dtype=np.float32),
        dataset_version=dataset_version,
        schema_version=schema_version,
    )


def load_self_play_targets(
    path: str | Path,
    *,
    max_saddle_gap: float,
    min_coverage_fraction: float,
    exact_targets: str | Path | None = None,
) -> tuple[SelfPlayTargets, dict[str, Any]]:
    """Load audited approximate targets without relabeling them as exact."""

    source = Path(path)
    with np.load(source, allow_pickle=False) as artifact:
        arrays = {name: artifact[name].copy() for name in artifact.files}
    validation = validate_replay(arrays)
    required = {
        "states",
        "horizon",
        "outcome",
        "truncated",
        "drop_policy",
        "check_policy",
        "unique_cells",
    }
    missing = required.difference(arrays)
    if missing:
        raise ValueError(f"self-play artifact is missing arrays: {sorted(missing)}")
    if not 0.0 <= min_coverage_fraction <= 1.0:
        raise ValueError("minimum coverage fraction must be in [0, 1]")
    if max_saddle_gap < 0.0:
        raise ValueError("maximum saddle gap must be nonnegative")

    matrix_cache: dict[tuple[NTState, int], np.ndarray] = {}
    target_store = (
        ExactTargetStore.load(exact_targets)
        if exact_targets is not None
        else None
    )
    gaps = np.empty(len(arrays["states"]), dtype=np.float64)
    for index, (raw_state, raw_horizon) in enumerate(
        zip(arrays["states"], arrays["horizon"])
    ):
        state = tuple(int(value) for value in raw_state)
        horizon = int(raw_horizon)
        key = (state, horizon)
        if key not in matrix_cache:
            if target_store is not None and key in target_store.values:
                try:
                    matrix_cache[key] = payoff_from_exact_targets(
                        state,
                        horizon,
                        target_store,
                    )
                except KeyError:
                    matrix_cache[key] = payoff(state, horizon)
            else:
                matrix_cache[key] = payoff(state, horizon)
        matrix = matrix_cache[key]
        drop_policy = arrays["drop_policy"][index]
        check_policy = arrays["check_policy"][index]
        lower = float(np.min(matrix.T @ drop_policy))
        upper = float(np.max(matrix @ check_policy))
        gaps[index] = max(0.0, upper - lower)

    action_cells = len(DROPPER_ACTIONS) * len(CHECKER_ACTIONS)
    coverage = arrays["unique_cells"].astype(np.float64) / action_cells
    accepted = (gaps <= max_saddle_gap) & (
        coverage >= min_coverage_fraction
    )
    if not np.any(accepted):
        raise ValueError("no self-play rows pass the configured audit gate")

    accepted_truncated = arrays["truncated"][accepted]
    targets = SelfPlayTargets(
        states=arrays["states"][accepted].astype(np.int16, copy=True),
        horizons=arrays["horizon"][accepted].astype(np.uint8, copy=True),
        values=arrays["outcome"][accepted].astype(np.float32, copy=True),
        drop_policies=arrays["drop_policy"][accepted].astype(np.float32, copy=True),
        check_policies=arrays["check_policy"][accepted].astype(np.float32, copy=True),
        value_weights=(~accepted_truncated).astype(np.float32, copy=True),
    )
    accepted_values = arrays["outcome"][accepted]
    summary = {
        "source": str(source),
        "source_type": "mcts_self_play",
        "rows": int(len(arrays["states"])),
        "accepted_rows": len(targets),
        "rejected_rows": int((~accepted).sum()),
        "terminal_rows": int((~accepted_truncated).sum()),
        "truncated_zero_rows": int(accepted_truncated.sum()),
        "effective_value_rows": int((~accepted_truncated).sum()),
        "masked_value_rows": int(accepted_truncated.sum()),
        "negative_outcomes": int((accepted_values < 0.0).sum()),
        "zero_outcomes": int((accepted_values == 0.0).sum()),
        "positive_outcomes": int((accepted_values > 0.0).sum()),
        "max_saddle_gap": max_saddle_gap,
        "min_coverage_fraction": min_coverage_fraction,
        "accepted_max_saddle_gap": float(gaps[accepted].max()),
        "validation": validation,
    }
    return targets, summary


def grouped_state_split(
    states: np.ndarray,
    *,
    validation_fraction: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Split physical states as indivisible groups to prevent horizon leakage."""

    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must lie strictly between zero and one")
    groups: dict[tuple[int, ...], list[int]] = defaultdict(list)
    for index, state in enumerate(states):
        groups[tuple(int(value) for value in state)].append(index)
    keys = sorted(groups)
    if len(keys) < 2:
        raise ValueError("at least two distinct physical states are required")

    rng = np.random.default_rng(seed)
    rng.shuffle(keys)
    validation_groups = max(
        1,
        min(len(keys) - 1, round(len(keys) * validation_fraction)),
    )
    validation_keys = set(keys[:validation_groups])
    train = [index for key in keys if key not in validation_keys for index in groups[key]]
    validation = [index for key in keys if key in validation_keys for index in groups[key]]
    return np.asarray(sorted(train), dtype=np.int64), np.asarray(
        sorted(validation), dtype=np.int64
    )


def exact_frontier_indices(
    targets: ExactTargets,
    roots: Iterable[dict[str, object]],
) -> np.ndarray:
    """Select exact child rows needed by configured MCTS leaf frontiers."""

    lookup = {
        (int(horizon), tuple(int(value) for value in state)): index
        for index, (state, horizon) in enumerate(
            zip(targets.states, targets.horizons)
        )
    }
    selected: set[int] = set()
    for root in roots:
        state = tuple(int(value) for value in root["state"])
        horizon = int(root["horizon"])
        if horizon <= 1:
            raise ValueError("exact frontier roots must have horizon at least two")
        for child in live_successors(state):
            key = (horizon - 1, child)
            if key not in lookup:
                raise ValueError(f"exact dataset is missing frontier row {key!r}")
            selected.add(lookup[key])
    if not selected:
        raise ValueError("exact frontier selection is empty")
    return np.asarray(sorted(selected), dtype=np.int64)


def exact_frontier_replay_indices(
    targets: ExactTargets,
    config: dict[str, Any],
) -> tuple[np.ndarray, dict[str, Any]]:
    """Build independently repeated exact frontier groups for one replay pass."""

    configured_groups = config.get("groups")
    groups = list(configured_groups) if configured_groups is not None else [config]
    if not groups:
        raise ValueError("exact frontier replay requires at least one group")

    repeated: list[np.ndarray] = []
    selected: list[np.ndarray] = []
    summaries: list[dict[str, Any]] = []
    for group_index, group in enumerate(groups):
        indices = exact_frontier_indices(targets, group["roots"])
        repeats = int(group["repeats"])
        if repeats <= 0:
            raise ValueError("exact frontier repeats must be positive")
        repeated_indices = np.tile(indices, repeats)
        selected.append(indices)
        repeated.append(repeated_indices)
        summaries.append(
            {
                "name": str(group.get("name", f"group_{group_index}")),
                "rows": len(indices),
                "repeats": repeats,
                "training_rows": len(repeated_indices),
                "roots": list(group["roots"]),
            }
        )

    selected_rows = np.unique(np.concatenate(selected))
    training_indices = np.concatenate(repeated)
    return training_indices, {
        "source_type": "exact_frontier_replay",
        "rows": len(selected_rows),
        "training_rows": len(training_indices),
        "groups": summaries,
    }


def soft_cross_entropy(logits: Tensor, targets: Tensor) -> Tensor:
    return -(targets * F.log_softmax(logits, dim=-1)).sum(dim=-1)


def _weighted_mean(values: Tensor, weights: Tensor) -> Tensor:
    weight_total = weights.sum()
    if float(weight_total.detach().item()) == 0.0:
        return values.sum() * 0.0
    return (values * weights).sum() / weight_total


def _batch_loss(
    model: DTHPolicyValueNet,
    batch: tuple[Tensor, ...],
    *,
    policy_weight: float,
    device: torch.device,
) -> Tensor:
    features, values, drop_targets, check_targets, weights, value_weights, _ = (
        tensor.to(device) for tensor in batch
    )
    predicted_values, drop_logits, check_logits = model(features)
    value_mse = _weighted_mean(
        (predicted_values - values) ** 2,
        weights * value_weights,
    )
    drop_ce = _weighted_mean(soft_cross_entropy(drop_logits, drop_targets), weights)
    check_ce = _weighted_mean(soft_cross_entropy(check_logits, check_targets), weights)
    return value_mse + float(policy_weight) * 0.5 * (drop_ce + check_ce)


@torch.no_grad()
def evaluate(
    model: DTHPolicyValueNet,
    loader: DataLoader[tuple[Tensor, ...]],
    *,
    policy_weight: float,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    numerators = defaultdict(float)
    policy_weight_total = 0.0
    value_weight_total = 0.0
    for batch in loader:
        features, values, drop_targets, check_targets, weights, value_weights, _ = (
            tensor.to(device) for tensor in batch
        )
        predicted_values, drop_logits, check_logits = model(features)
        value_loss_weights = weights * value_weights
        numerators["value_mse"] += float(
            (((predicted_values - values) ** 2) * value_loss_weights).sum().item()
        )
        numerators["drop_ce"] += float(
            (soft_cross_entropy(drop_logits, drop_targets) * weights).sum().item()
        )
        numerators["check_ce"] += float(
            (soft_cross_entropy(check_logits, check_targets) * weights).sum().item()
        )
        policy_weight_total += float(weights.sum().item())
        value_weight_total += float(value_loss_weights.sum().item())
    result = {
        "value_mse": (
            numerators["value_mse"] / value_weight_total
            if value_weight_total > 0.0
            else 0.0
        ),
        "drop_ce": numerators["drop_ce"] / policy_weight_total,
        "check_ce": numerators["check_ce"] / policy_weight_total,
        "effective_value_weight": value_weight_total,
        "effective_policy_weight": policy_weight_total,
    }
    result["total_loss"] = result["value_mse"] + float(policy_weight) * 0.5 * (
        result["drop_ce"] + result["check_ce"]
    )
    return result


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_safe(value: Any) -> Any:
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


@torch.no_grad()
def audit_saddle_gaps(
    model: DTHPolicyValueNet,
    targets: ExactTargets,
    indices: Iterable[int],
    *,
    max_rows_per_horizon: int,
    device: torch.device,
) -> dict[str, Any]:
    """Measure exploitability of predicted marginals in exact payoff matrices."""

    if max_rows_per_horizon <= 0:
        return {"rows": 0, "overall": {}, "by_horizon": {}}
    selected: list[int] = []
    by_horizon: dict[int, list[int]] = defaultdict(list)
    for index in indices:
        by_horizon[int(targets.horizons[index])].append(int(index))
    for horizon in sorted(by_horizon):
        selected.extend(by_horizon[horizon][:max_rows_per_horizon])

    model.eval()
    solve.cache_clear()
    records: list[tuple[int, float, float]] = []
    for index in selected:
        state = tuple(int(value) for value in targets.states[index])
        horizon = int(targets.horizons[index])
        states = torch.tensor([state], dtype=torch.float32, device=device)
        horizons = torch.tensor([horizon], dtype=torch.float32, device=device)
        features = model.encode(states, horizons)
        predicted_value, drop_logits, check_logits = model(features)
        drop_policy = torch.softmax(drop_logits, dim=-1)[0].cpu().numpy()
        check_policy = torch.softmax(check_logits, dim=-1)[0].cpu().numpy()
        matrix = payoff(state, horizon)
        lower = float(np.min(matrix.T @ drop_policy))
        upper = float(np.max(matrix @ check_policy))
        gap = max(0.0, upper - lower)
        value_error = abs(float(predicted_value.item()) - float(targets.values[index]))
        records.append((horizon, gap, value_error))

    def summarize(rows: list[tuple[int, float, float]]) -> dict[str, float | int]:
        gaps = np.asarray([row[1] for row in rows], dtype=np.float64)
        errors = np.asarray([row[2] for row in rows], dtype=np.float64)
        return {
            "count": len(rows),
            "median_saddle_gap": float(np.median(gaps)),
            "p95_saddle_gap": float(np.quantile(gaps, 0.95)),
            "max_saddle_gap": float(np.max(gaps)),
            "mean_value_error": float(np.mean(errors)),
        }

    return {
        "rows": len(records),
        "overall": summarize(records) if records else {},
        "by_horizon": {
            str(horizon): summarize([row for row in records if row[0] == horizon])
            for horizon in sorted({row[0] for row in records})
        },
    }


@dataclass(frozen=True)
class DecisionRoot:
    state: NTState
    horizon: int
    exact_value: float
    exact_matrix: np.ndarray
    value_preservation_weight: float = 0.0


@dataclass(frozen=True)
class DecisionLossRoot:
    """Device-local tensors for differentiable root/frontier supervision."""

    state: NTState
    horizon: int
    root_features: Tensor
    child_features: Tensor
    exact_matrix: Tensor
    exact_value: Tensor
    exact_drop_policy: Tensor
    exact_check_policy: Tensor
    value_preservation_weight: float
    matrix_base: Tensor
    child_coefficients: Tensor


def build_decision_roots(
    targets: ExactTargetStore,
    roots: Iterable[dict[str, object]],
) -> tuple[DecisionRoot, ...]:
    """Precompute certified matrices for decision-aware selection roots."""

    prepared: list[DecisionRoot] = []
    for root in roots:
        state = tuple(int(value) for value in root["state"])
        horizon = int(root["horizon"])
        value_preservation_weight = float(
            root.get("value_preservation_weight", 0.0)
        )
        if value_preservation_weight < 0.0:
            raise ValueError("root value-preservation weight must be nonnegative")
        key = (state, horizon)
        if key not in targets.values:
            raise ValueError(f"decision root is missing from exact targets: {key!r}")
        try:
            matrix = payoff_from_exact_targets(state, horizon, targets)
        except KeyError as error:
            raise ValueError(
                f"decision root frontier is missing exact target {error.args[0]!r}"
            ) from error
        prepared.append(
            DecisionRoot(
                state=state,
                horizon=horizon,
                exact_value=targets.values[key],
                exact_matrix=matrix,
                value_preservation_weight=value_preservation_weight,
            )
        )
    if not prepared:
        raise ValueError("decision selection requires at least one root")
    return tuple(prepared)


def prepare_decision_loss_roots(
    model: DTHPolicyValueNet,
    roots: Iterable[DecisionRoot],
    *,
    device: torch.device,
) -> tuple[DecisionLossRoot, ...]:
    """Precompute the linear child-value map for each exact root matrix."""

    prepared: list[DecisionLossRoot] = []
    action_shape = (len(DROPPER_ACTIONS), len(CHECKER_ACTIONS))
    for root in roots:
        _, exact_drop_policy, exact_check_policy = solve_matrix(root.exact_matrix)
        children = sorted(live_successors(root.state)) if root.horizon > 1 else []
        child_lookup = {child: index for index, child in enumerate(children)}
        matrix_base = np.zeros(action_shape, dtype=np.float32)
        child_coefficients = np.zeros(
            (len(DROPPER_ACTIONS) * len(CHECKER_ACTIONS), len(children)),
            dtype=np.float32,
        )
        for drop_index, drop in enumerate(DROPPER_ACTIONS):
            for check_index, check in enumerate(CHECKER_ACTIONS):
                flat_index = drop_index * len(CHECKER_ACTIONS) + check_index
                for probability, child in transition(root.state, drop, check):
                    if isinstance(child, TState):
                        matrix_base[drop_index, check_index] += (
                            float(probability) * float(reward(child))
                        )
                    elif root.horizon > 1:
                        child_coefficients[flat_index, child_lookup[child]] -= float(
                            probability
                        )

        root_states = torch.tensor([root.state], dtype=torch.float32, device=device)
        root_horizons = torch.tensor(
            [root.horizon], dtype=torch.float32, device=device
        )
        if children:
            child_states = torch.tensor(children, dtype=torch.float32, device=device)
            child_horizons = torch.full(
                (len(children),),
                float(root.horizon - 1),
                dtype=torch.float32,
                device=device,
            )
            child_features = model.encode(child_states, child_horizons)
        else:
            child_features = torch.empty(
                (0, len(FEATURE_SCHEMA)), dtype=torch.float32, device=device
            )
        prepared.append(
            DecisionLossRoot(
                state=root.state,
                horizon=root.horizon,
                root_features=model.encode(root_states, root_horizons),
                child_features=child_features,
                exact_matrix=torch.as_tensor(
                    root.exact_matrix, dtype=torch.float32, device=device
                ),
                exact_value=torch.tensor(
                    root.exact_value, dtype=torch.float32, device=device
                ),
                exact_drop_policy=torch.as_tensor(
                    exact_drop_policy, dtype=torch.float32, device=device
                ),
                exact_check_policy=torch.as_tensor(
                    exact_check_policy, dtype=torch.float32, device=device
                ),
                value_preservation_weight=root.value_preservation_weight,
                matrix_base=torch.as_tensor(
                    matrix_base.reshape(-1), dtype=torch.float32, device=device
                ),
                child_coefficients=torch.as_tensor(
                    child_coefficients, dtype=torch.float32, device=device
                ),
            )
        )
    if not prepared:
        raise ValueError("decision loss requires at least one root")
    return tuple(prepared)


def decision_training_objective(
    model: DTHPolicyValueNet,
    roots: Iterable[DecisionLossRoot],
    *,
    saddle_gap_weight: float,
    matrix_weight: float,
    matrix_top_k: int,
) -> tuple[Tensor, dict[str, Tensor]]:
    """Minimize the worst root's exploitability, matrix, and value error."""

    if saddle_gap_weight < 0.0 or matrix_weight < 0.0:
        raise ValueError("decision loss weights must be nonnegative")
    if matrix_top_k <= 0:
        raise ValueError("matrix top-k must be positive")

    prepared_roots = tuple(roots)
    if not prepared_roots:
        raise ValueError("decision loss requires at least one root")
    if (
        saddle_gap_weight == 0.0
        and matrix_weight == 0.0
        and not any(root.value_preservation_weight > 0.0 for root in prepared_roots)
    ):
        raise ValueError("at least one decision loss weight must be positive")
    feature_batches = [root.root_features for root in prepared_roots]
    feature_batches.extend(
        root.child_features
        for root in prepared_roots
        if root.child_features.shape[0] > 0
    )
    predicted_values, all_drop_logits, all_check_logits = model(
        torch.cat(feature_batches, dim=0)
    )

    gaps: list[Tensor] = []
    top_k_errors: list[Tensor] = []
    max_errors: list[Tensor] = []
    value_errors: list[Tensor] = []
    root_losses: list[Tensor] = []
    child_offset = len(prepared_roots)
    for root_index, root in enumerate(prepared_roots):
        drop_policy = torch.softmax(all_drop_logits[root_index], dim=-1)
        check_policy = torch.softmax(all_check_logits[root_index], dim=-1)
        lower = torch.min(root.exact_matrix.T @ drop_policy)
        upper = torch.max(root.exact_matrix @ check_policy)
        gaps.append(torch.clamp(upper - lower, min=0.0))

        if root.child_features.shape[0] > 0:
            child_count = int(root.child_features.shape[0])
            child_values = predicted_values[
                child_offset : child_offset + child_count
            ]
            child_offset += child_count
            approximate_flat = (
                root.matrix_base + root.child_coefficients @ child_values
            )
        else:
            approximate_flat = root.matrix_base
        approximate_matrix = approximate_flat.reshape(root.exact_matrix.shape)
        cell_errors = torch.abs(approximate_matrix - root.exact_matrix).reshape(-1)
        top_k = min(matrix_top_k, int(cell_errors.numel()))
        root_top_k_error = torch.topk(cell_errors, top_k).values.mean()
        root_max_error = torch.max(cell_errors)
        top_k_errors.append(root_top_k_error)
        max_errors.append(root_max_error)

        exact_drop_lower = torch.min(
            approximate_matrix.T @ root.exact_drop_policy
        )
        exact_check_upper = torch.max(
            approximate_matrix @ root.exact_check_policy
        )
        root_value_error = torch.maximum(
            torch.abs(exact_drop_lower - root.exact_value),
            torch.abs(exact_check_upper - root.exact_value),
        )
        value_errors.append(root_value_error)
        root_losses.append(
            float(saddle_gap_weight) * gaps[-1]
            + float(matrix_weight) * root_top_k_error
            + float(root.value_preservation_weight) * root_value_error
        )

    mean_gap = torch.stack(gaps).mean()
    mean_top_k = torch.stack(top_k_errors).mean()
    mean_max = torch.stack(max_errors).mean()
    mean_value_error = torch.stack(value_errors).mean()
    total = torch.stack(root_losses).max()
    return total, {
        "total_loss": total,
        "worst_root_loss": total,
        "worst_saddle_gap": torch.stack(gaps).max(),
        "worst_matrix_top_k_error": torch.stack(top_k_errors).max(),
        "worst_matrix_max_error": torch.stack(max_errors).max(),
        "worst_root_value_error": torch.stack(value_errors).max(),
        "mean_saddle_gap": mean_gap,
        "mean_matrix_top_k_error": mean_top_k,
        "mean_matrix_max_error": mean_max,
        "mean_root_value_error": mean_value_error,
    }


@torch.no_grad()
def evaluate_decision_training_objective(
    model: DTHPolicyValueNet,
    roots: Iterable[DecisionLossRoot],
    *,
    saddle_gap_weight: float,
    matrix_weight: float,
    matrix_top_k: int,
) -> dict[str, float]:
    model.eval()
    _, metrics = decision_training_objective(
        model,
        roots,
        saddle_gap_weight=saddle_gap_weight,
        matrix_weight=matrix_weight,
        matrix_top_k=matrix_top_k,
    )
    return {name: float(value.item()) for name, value in metrics.items()}


@torch.no_grad()
def approximate_payoff_from_network(
    model: DTHPolicyValueNet,
    state: NTState,
    horizon: int,
    *,
    device: torch.device,
) -> np.ndarray:
    """Build a root matrix from the network's live-child value estimates."""

    if horizon <= 0:
        raise ValueError("decision roots must have positive horizon")
    children = sorted(live_successors(state)) if horizon > 1 else []
    child_values: dict[NTState, float] = {}
    if children:
        states = torch.tensor(children, dtype=torch.float32, device=device)
        horizons = torch.full(
            (len(children),),
            float(horizon - 1),
            dtype=torch.float32,
            device=device,
        )
        predicted, _, _ = model(model.encode(states, horizons))
        child_values = {
            child: float(value)
            for child, value in zip(children, predicted.cpu().tolist(), strict=True)
        }

    matrix = np.empty((len(DROPPER_ACTIONS), len(CHECKER_ACTIONS)))
    for drop_index, drop in enumerate(DROPPER_ACTIONS):
        for check_index, check in enumerate(CHECKER_ACTIONS):
            total = 0.0
            for probability, child in transition(state, drop, check):
                if isinstance(child, TState):
                    branch_value = float(reward(child))
                elif horizon == 1:
                    branch_value = 0.0
                else:
                    branch_value = -child_values[child]
                total += probability * branch_value
            matrix[drop_index, check_index] = total
    return matrix


@torch.no_grad()
def evaluate_decision_roots(
    model: DTHPolicyValueNet,
    roots: Iterable[DecisionRoot],
    *,
    device: torch.device,
) -> dict[str, Any]:
    """Evaluate approximate root decisions against certified root matrices."""

    model.eval()
    records: list[dict[str, Any]] = []
    for root in roots:
        approximate_matrix = approximate_payoff_from_network(
            model,
            root.state,
            root.horizon,
            device=device,
        )
        approximate_value, drop_policy, check_policy = solve_matrix(
            approximate_matrix
        )
        lower = float(np.min(root.exact_matrix.T @ drop_policy))
        upper = float(np.max(root.exact_matrix @ check_policy))
        records.append(
            {
                "state": list(root.state),
                "horizon": root.horizon,
                "exact_value": root.exact_value,
                "approximate_value": float(approximate_value),
                "value_error": abs(float(approximate_value) - root.exact_value),
                "saddle_gap": max(0.0, upper - lower),
            }
        )
    return {
        "roots": records,
        "max_saddle_gap": max(record["saddle_gap"] for record in records),
        "max_value_error": max(record["value_error"] for record in records),
    }


def decision_guard_passes(
    candidate: dict[str, Any],
    baseline: dict[str, Any],
    *,
    tolerance: float,
) -> bool:
    return (
        candidate["max_saddle_gap"]
        <= baseline["max_saddle_gap"] + tolerance
        and candidate["max_value_error"]
        <= baseline["max_value_error"] + tolerance
    )


def decision_checkpoint_better(
    candidate: dict[str, Any],
    incumbent: dict[str, Any],
    baseline: dict[str, Any],
    *,
    candidate_validation_loss: float,
    incumbent_validation_loss: float,
    minimum_gap_improvement: float,
    tie_tolerance: float,
) -> bool:
    """Gap-first ranking with an epoch-zero minimum-improvement floor."""

    if (
        candidate["max_saddle_gap"]
        > baseline["max_saddle_gap"] - minimum_gap_improvement
    ):
        return False
    candidate_gap = float(candidate["max_saddle_gap"])
    incumbent_gap = float(incumbent["max_saddle_gap"])
    if candidate_gap < incumbent_gap - tie_tolerance:
        return True
    if abs(candidate_gap - incumbent_gap) > tie_tolerance:
        return False
    candidate_error = float(candidate["max_value_error"])
    incumbent_error = float(incumbent["max_value_error"])
    if candidate_error < incumbent_error - tie_tolerance:
        return True
    if abs(candidate_error - incumbent_error) > tie_tolerance:
        return False
    return candidate_validation_loss < incumbent_validation_loss - tie_tolerance


def train_exact(config: dict[str, Any]) -> dict[str, Any]:
    seed = int(config["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)

    dataset_path = Path(config["dataset"])
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    targets = load_exact_targets(dataset_path)

    model_values = config["model"]
    model_config = DTHNetworkConfig(
        hidden_width=int(model_values["hidden_width"]),
        hidden_layers=int(model_values["hidden_layers"]),
        action_count=len(DROPPER_ACTIONS),
        horizon_scale=float(model_values["horizon_scale"]),
    )
    model = DTHPolicyValueNet(model_config)
    device = torch.device(str(config["device"]))
    model.to(device)
    initial_checkpoint = config.get("initial_checkpoint")
    if initial_checkpoint:
        initial = torch.load(
            Path(initial_checkpoint),
            map_location=device,
            weights_only=False,
        )
        if dict(initial["model_config"]) != model_config.to_dict():
            raise ValueError("initial checkpoint model configuration does not match")
        model.load_state_dict(initial["state_dict"])

    training = config["training"]
    train_indices, validation_indices = grouped_state_split(
        targets.states,
        validation_fraction=float(training["validation_fraction"]),
        seed=seed,
    )
    train_rows = TargetRows(
        targets,
        train_indices,
        horizon_scale=model_config.horizon_scale,
    )
    validation_rows = TargetRows(
        targets,
        validation_indices,
        horizon_scale=model_config.horizon_scale,
    )
    batch_size = int(training["batch_size"])
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        train_rows,
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
        num_workers=0,
    )
    validation_loader = DataLoader(
        validation_rows,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )
    self_play_loader = None
    self_play_summary = None
    self_play_values = config.get("self_play")
    if self_play_values:
        self_play_targets, self_play_summary = load_self_play_targets(
            self_play_values["dataset"],
            max_saddle_gap=float(self_play_values["max_saddle_gap"]),
            min_coverage_fraction=float(
                self_play_values["min_coverage_fraction"]
            ),
            exact_targets=self_play_values.get("exact_targets"),
        )
        repeats = int(self_play_values["repeats"])
        if repeats <= 0:
            raise ValueError("self-play repeats must be positive")
        self_play_indices = np.tile(np.arange(len(self_play_targets)), repeats)
        self_play_rows = TargetRows(
            self_play_targets,
            self_play_indices,
            horizon_scale=model_config.horizon_scale,
        )
        self_play_loader = DataLoader(
            self_play_rows,
            batch_size=batch_size,
            shuffle=True,
            generator=torch.Generator().manual_seed(seed + 1),
            num_workers=0,
        )
        self_play_summary["repeats"] = repeats
        self_play_summary["training_rows"] = len(self_play_indices)
    exact_frontier_loader = None
    exact_frontier_summary = None
    exact_frontier_values = config.get("exact_frontiers")
    if exact_frontier_values:
        repeated_frontier_indices, exact_frontier_summary = exact_frontier_replay_indices(
            targets,
            exact_frontier_values,
        )
        exact_frontier_rows = TargetRows(
            targets,
            repeated_frontier_indices,
            horizon_scale=model_config.horizon_scale,
        )
        exact_frontier_loader = DataLoader(
            exact_frontier_rows,
            batch_size=batch_size,
            shuffle=True,
            generator=torch.Generator().manual_seed(seed + 2),
            num_workers=0,
        )
    optimizer = Adam(
        model.parameters(),
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    policy_weight = float(training["policy_weight"])
    selection_metric = str(training.get("selection_metric", "validation"))
    if selection_metric not in {
        "validation",
        "exact_frontier",
        "self_play",
        "all_sources",
        "decision",
    }:
        raise ValueError(
            "training.selection_metric must be validation, exact_frontier, "
            "self_play, all_sources, or decision"
        )
    if selection_metric == "exact_frontier" and exact_frontier_loader is None:
        raise ValueError(
            "exact_frontier selection requires configured exact_frontiers"
        )
    if selection_metric == "self_play" and self_play_loader is None:
        raise ValueError("self_play selection requires configured self_play")
    decision_values = config.get("decision_selection")
    if selection_metric == "decision" and not decision_values:
        raise ValueError("decision selection requires decision_selection config")
    decision_roots = None
    decision_guard_roots = None
    decision_store = None
    if decision_values:
        decision_store = ExactTargetStore.load(
            decision_values.get("exact_targets", dataset_path)
        )
        decision_roots = build_decision_roots(
            decision_store,
            decision_values["roots"],
        )
        decision_guard_roots = build_decision_roots(
            decision_store,
            decision_values["guard_roots"],
        )
    decision_loss_values = config.get("decision_loss")
    decision_loss_roots = None
    decision_loss_repeats = 0
    if decision_loss_values:
        decision_loss_store = (
            decision_store
            if decision_store is not None
            and decision_loss_values.get("exact_targets", dataset_path)
            == decision_values.get("exact_targets", dataset_path)
            else ExactTargetStore.load(
                decision_loss_values.get("exact_targets", dataset_path)
            )
        )
        configured_loss_roots = build_decision_roots(
            decision_loss_store,
            decision_loss_values["roots"],
        )
        decision_loss_roots = prepare_decision_loss_roots(
            model,
            configured_loss_roots,
            device=device,
        )
        decision_loss_repeats = int(decision_loss_values["repeats"])
        if decision_loss_repeats <= 0:
            raise ValueError("decision loss repeats must be positive")
        decision_training_objective(
            model,
            decision_loss_roots,
            saddle_gap_weight=float(decision_loss_values["saddle_gap_weight"]),
            matrix_weight=float(decision_loss_values["matrix_weight"]),
            matrix_top_k=int(decision_loss_values["matrix_top_k"]),
        )
    best_loss = float("inf")
    best_epoch = 0
    best_decision_metrics = None
    best_guard_metrics = None
    best_validation_loss = float("inf")
    stale_epochs = 0
    history: list[dict[str, Any]] = []
    checkpoint_path = output_dir / "best.pt"

    def save_checkpoint(
        *,
        epoch: int,
        validation_metrics: dict[str, float],
        exact_frontier_metrics: dict[str, float] | None,
        self_play_metrics: dict[str, float] | None,
        selection_loss: float,
        decision_metrics: dict[str, Any] | None,
        guard_metrics: dict[str, Any] | None,
        guard_passed: bool | None,
        equilibrium_loss_metrics: dict[str, float] | None,
        destination: Path = checkpoint_path,
        diagnostic_only: bool = False,
    ) -> None:
        torch.save(
            {
                "state_dict": model.state_dict(),
                "model_config": model_config.to_dict(),
                "feature_schema": FEATURE_SCHEMA,
                "dataset_version": targets.dataset_version,
                "dataset_sha256": _sha256(dataset_path),
                "target_schema": targets.schema_version,
                "drop_actions": DROPPER_ACTIONS,
                "check_actions": CHECKER_ACTIONS,
                "seed": seed,
                "device": str(device),
                "epoch": epoch,
                "validation_metrics": validation_metrics,
                "exact_frontier_metrics": exact_frontier_metrics,
                "self_play_metrics": self_play_metrics,
                "selection_metric": selection_metric,
                "selection_loss": selection_loss,
                "decision_metrics": decision_metrics,
                "decision_guard_metrics": guard_metrics,
                "decision_guard_passed": guard_passed,
                "equilibrium_loss_metrics": equilibrium_loss_metrics,
                "decision_loss": (
                    dict(decision_loss_values) if decision_loss_values else None
                ),
                "self_play": self_play_summary,
                "exact_frontiers": exact_frontier_summary,
                "initial_checkpoint": initial_checkpoint,
                "diagnostic_only": diagnostic_only,
            },
            destination,
        )

    baseline_decision_metrics = None
    baseline_guard_metrics = None
    baseline_checkpoint = (
        Path(decision_values["baseline_checkpoint"])
        if decision_values and decision_values.get("baseline_checkpoint")
        else None
    )
    diagnostic_checkpoint_path = output_dir / "best_h5_attempt.pt"
    best_attempt_epoch = 0
    best_attempt_metrics = None
    best_attempt_guard_metrics = None
    best_attempt_guard_passed = True
    best_attempt_gap = float("inf")
    if selection_metric == "decision":
        baseline_model = model
        if baseline_checkpoint is not None:
            baseline_payload = torch.load(
                baseline_checkpoint,
                map_location=device,
                weights_only=False,
            )
            baseline_model_config = DTHNetworkConfig(
                **dict(baseline_payload["model_config"])
            )
            if (
                baseline_model_config.action_count != model_config.action_count
                or baseline_model_config.horizon_scale
                != model_config.horizon_scale
            ):
                raise ValueError(
                    "decision baseline must use the candidate action and feature schema"
                )
            baseline_model = DTHPolicyValueNet(baseline_model_config)
            baseline_model.load_state_dict(baseline_payload["state_dict"])
            baseline_model.to(device)
        baseline_validation = evaluate(
            baseline_model,
            validation_loader,
            policy_weight=policy_weight,
            device=device,
        )
        baseline_frontier = (
            evaluate(
                baseline_model,
                exact_frontier_loader,
                policy_weight=policy_weight,
                device=device,
            )
            if exact_frontier_loader is not None
            else None
        )
        baseline_self_play = (
            evaluate(
                baseline_model,
                self_play_loader,
                policy_weight=policy_weight,
                device=device,
            )
            if self_play_loader is not None
            else None
        )
        baseline_decision_metrics = evaluate_decision_roots(
            baseline_model,
            decision_roots,
            device=device,
        )
        baseline_guard_metrics = evaluate_decision_roots(
            baseline_model,
            decision_guard_roots,
            device=device,
        )
        baseline_equilibrium_loss = (
            evaluate_decision_training_objective(
                baseline_model,
                decision_loss_roots,
                saddle_gap_weight=float(
                    decision_loss_values["saddle_gap_weight"]
                ),
                matrix_weight=float(decision_loss_values["matrix_weight"]),
                matrix_top_k=int(decision_loss_values["matrix_top_k"]),
            )
            if decision_loss_roots is not None
            else None
        )
        best_decision_metrics = baseline_decision_metrics
        best_guard_metrics = baseline_guard_metrics
        best_validation_loss = baseline_validation["total_loss"]
        best_loss = baseline_decision_metrics["max_saddle_gap"]
        best_attempt_metrics = baseline_decision_metrics
        best_attempt_guard_metrics = baseline_guard_metrics
        best_attempt_gap = float(baseline_decision_metrics["max_saddle_gap"])
        history.append(
            {
                "epoch": 0,
                "train": None,
                "validation": baseline_validation,
                "exact_frontier": baseline_frontier,
                "self_play": baseline_self_play,
                "decision": baseline_decision_metrics,
                "decision_guard": baseline_guard_metrics,
                "decision_guard_passed": True,
                "equilibrium_loss": baseline_equilibrium_loss,
                "selection_loss": best_loss,
            }
        )
        if baseline_checkpoint is not None:
            shutil.copyfile(baseline_checkpoint, checkpoint_path)
        else:
            save_checkpoint(
                epoch=0,
                validation_metrics=baseline_validation,
                exact_frontier_metrics=baseline_frontier,
                self_play_metrics=baseline_self_play,
                selection_loss=best_loss,
                decision_metrics=baseline_decision_metrics,
                guard_metrics=baseline_guard_metrics,
                guard_passed=True,
                equilibrium_loss_metrics=baseline_equilibrium_loss,
            )

    epochs = int(training["epochs"])
    patience = int(training["patience"])
    log_every = int(training["log_every"])
    for epoch in range(1, epochs + 1):
        model.train()
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            loss = _batch_loss(
                model,
                batch,
                policy_weight=policy_weight,
                device=device,
            )
            loss.backward()
            optimizer.step()
        if self_play_loader is not None:
            for batch in self_play_loader:
                optimizer.zero_grad(set_to_none=True)
                loss = _batch_loss(
                    model,
                    batch,
                    policy_weight=policy_weight,
                    device=device,
                )
                loss.backward()
                optimizer.step()
        if exact_frontier_loader is not None:
            for batch in exact_frontier_loader:
                optimizer.zero_grad(set_to_none=True)
                loss = _batch_loss(
                    model,
                    batch,
                    policy_weight=policy_weight,
                    device=device,
                )
                loss.backward()
                optimizer.step()
        if decision_loss_roots is not None:
            for _ in range(decision_loss_repeats):
                optimizer.zero_grad(set_to_none=True)
                loss, _ = decision_training_objective(
                    model,
                    decision_loss_roots,
                    saddle_gap_weight=float(
                        decision_loss_values["saddle_gap_weight"]
                    ),
                    matrix_weight=float(decision_loss_values["matrix_weight"]),
                    matrix_top_k=int(decision_loss_values["matrix_top_k"]),
                )
                loss.backward()
                optimizer.step()

        train_metrics = evaluate(
            model,
            train_loader,
            policy_weight=policy_weight,
            device=device,
        )
        validation_metrics = evaluate(
            model,
            validation_loader,
            policy_weight=policy_weight,
            device=device,
        )
        exact_frontier_metrics = (
            evaluate(
                model,
                exact_frontier_loader,
                policy_weight=policy_weight,
                device=device,
            )
            if exact_frontier_loader is not None
            else None
        )
        self_play_metrics = (
            evaluate(
                model,
                self_play_loader,
                policy_weight=policy_weight,
                device=device,
            )
            if self_play_loader is not None
            else None
        )
        decision_metrics = (
            evaluate_decision_roots(model, decision_roots, device=device)
            if decision_roots is not None
            else None
        )
        guard_metrics = (
            evaluate_decision_roots(model, decision_guard_roots, device=device)
            if decision_guard_roots is not None
            else None
        )
        guard_passed = (
            decision_guard_passes(
                guard_metrics,
                baseline_guard_metrics,
                tolerance=float(decision_values["guard_tolerance"]),
            )
            if guard_metrics is not None
            else None
        )
        equilibrium_loss_metrics = (
            evaluate_decision_training_objective(
                model,
                decision_loss_roots,
                saddle_gap_weight=float(
                    decision_loss_values["saddle_gap_weight"]
                ),
                matrix_weight=float(decision_loss_values["matrix_weight"]),
                matrix_top_k=int(decision_loss_values["matrix_top_k"]),
            )
            if decision_loss_roots is not None
            else None
        )
        if selection_metric == "decision":
            selection_loss = decision_metrics["max_saddle_gap"]
        elif selection_metric == "exact_frontier":
            selection_loss = exact_frontier_metrics["total_loss"]
        elif selection_metric == "self_play":
            selection_loss = self_play_metrics["total_loss"]
        elif selection_metric == "all_sources":
            source_losses = [validation_metrics["total_loss"]]
            if exact_frontier_metrics is not None:
                source_losses.append(exact_frontier_metrics["total_loss"])
            if self_play_metrics is not None:
                source_losses.append(self_play_metrics["total_loss"])
            selection_loss = float(np.mean(source_losses))
        else:
            selection_loss = validation_metrics["total_loss"]
        history.append(
            {
                "epoch": epoch,
                "train": train_metrics,
                "validation": validation_metrics,
                "exact_frontier": exact_frontier_metrics,
                "self_play": self_play_metrics,
                "decision": decision_metrics,
                "decision_guard": guard_metrics,
                "decision_guard_passed": guard_passed,
                "equilibrium_loss": equilibrium_loss_metrics,
                "selection_loss": selection_loss,
            }
        )

        if (
            selection_metric == "decision"
            and decision_metrics["max_saddle_gap"]
            < best_attempt_gap - float(decision_values["tie_tolerance"])
        ):
            best_attempt_epoch = epoch
            best_attempt_metrics = decision_metrics
            best_attempt_guard_metrics = guard_metrics
            best_attempt_guard_passed = bool(guard_passed)
            best_attempt_gap = float(decision_metrics["max_saddle_gap"])
            save_checkpoint(
                epoch=epoch,
                validation_metrics=validation_metrics,
                exact_frontier_metrics=exact_frontier_metrics,
                self_play_metrics=self_play_metrics,
                selection_loss=selection_loss,
                decision_metrics=decision_metrics,
                guard_metrics=guard_metrics,
                guard_passed=guard_passed,
                equilibrium_loss_metrics=equilibrium_loss_metrics,
                destination=diagnostic_checkpoint_path,
                diagnostic_only=True,
            )

        if selection_metric == "decision":
            improved = bool(guard_passed) and decision_checkpoint_better(
                decision_metrics,
                best_decision_metrics,
                baseline_decision_metrics,
                candidate_validation_loss=validation_metrics["total_loss"],
                incumbent_validation_loss=best_validation_loss,
                minimum_gap_improvement=float(
                    decision_values["minimum_gap_improvement"]
                ),
                tie_tolerance=float(decision_values["tie_tolerance"]),
            )
        else:
            improved = selection_loss < best_loss - float(training["minimum_delta"])
        if improved:
            best_loss = selection_loss
            best_epoch = epoch
            best_decision_metrics = decision_metrics
            best_guard_metrics = guard_metrics
            best_validation_loss = validation_metrics["total_loss"]
            stale_epochs = 0
            save_checkpoint(
                epoch=epoch,
                validation_metrics=validation_metrics,
                exact_frontier_metrics=exact_frontier_metrics,
                self_play_metrics=self_play_metrics,
                selection_loss=selection_loss,
                decision_metrics=decision_metrics,
                guard_metrics=guard_metrics,
                guard_passed=guard_passed,
                equilibrium_loss_metrics=equilibrium_loss_metrics,
            )
        else:
            stale_epochs += 1

        if epoch == 1 or epoch % log_every == 0:
            print(
                f"epoch={epoch} train={train_metrics['total_loss']:.6f} "
                f"validation={validation_metrics['total_loss']:.6f} "
                f"selection={selection_loss:.6f} "
                f"value_mse={validation_metrics['value_mse']:.6f}",
                flush=True,
            )
        if patience > 0 and stale_epochs >= patience:
            print(f"Early stopping at epoch {epoch}", flush=True)
            break

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    selected_model_config = DTHNetworkConfig(**dict(checkpoint["model_config"]))
    selected_model = DTHPolicyValueNet(selected_model_config)
    selected_model.load_state_dict(checkpoint["state_dict"])
    selected_model.to(device)
    final_train = evaluate(
        selected_model,
        train_loader,
        policy_weight=policy_weight,
        device=device,
    )
    final_validation = evaluate(
        selected_model,
        validation_loader,
        policy_weight=policy_weight,
        device=device,
    )
    final_equilibrium_loss = (
        evaluate_decision_training_objective(
            selected_model,
            decision_loss_roots,
            saddle_gap_weight=float(decision_loss_values["saddle_gap_weight"]),
            matrix_weight=float(decision_loss_values["matrix_weight"]),
            matrix_top_k=int(decision_loss_values["matrix_top_k"]),
        )
        if decision_loss_roots is not None
        else None
    )
    audit = audit_saddle_gaps(
        selected_model,
        targets,
        validation_indices,
        max_rows_per_horizon=int(config["audit"]["max_rows_per_horizon"]),
        device=device,
    )

    report = {
        "dataset": str(dataset_path),
        "dataset_sha256": _sha256(dataset_path),
        "dataset_version": targets.dataset_version,
        "target_schema": targets.schema_version,
        "rows": len(targets),
        "train_rows": len(train_indices),
        "validation_rows": len(validation_indices),
        "seed": seed,
        "device": str(device),
        "model": model_config.to_dict(),
        "selected_model": selected_model_config.to_dict(),
        "training": dict(training),
        "selection_metric": selection_metric,
        "best_epoch": best_epoch,
        "decision_selection": (
            {
                "config": dict(decision_values),
                "baseline": baseline_decision_metrics,
                "baseline_guard": baseline_guard_metrics,
                "baseline_checkpoint": (
                    str(baseline_checkpoint)
                    if baseline_checkpoint is not None
                    else None
                ),
                "selected": best_decision_metrics,
                "selected_guard": best_guard_metrics,
                "selected_guard_passed": True,
                "improved_from_epoch_zero": best_epoch > 0,
                "best_attempt": {
                    "epoch": best_attempt_epoch,
                    "decision": best_attempt_metrics,
                    "guard": best_attempt_guard_metrics,
                    "guard_passed": best_attempt_guard_passed,
                    "diagnostic_checkpoint": (
                        str(diagnostic_checkpoint_path)
                        if best_attempt_epoch > 0
                        else None
                    ),
                    "diagnostic_checkpoint_sha256": (
                        _sha256(diagnostic_checkpoint_path)
                        if best_attempt_epoch > 0
                        else None
                    ),
                },
            }
            if decision_values
            else None
        ),
        "train_metrics": final_train,
        "validation_metrics": final_validation,
        "exact_matrix_audit": audit,
        "self_play": self_play_summary,
        "exact_frontiers": exact_frontier_summary,
        "decision_loss": (
            {
                "config": dict(decision_loss_values),
                "repeats": decision_loss_repeats,
                "selected_metrics": final_equilibrium_loss,
            }
            if decision_loss_values
            else None
        ),
        "initial_checkpoint": initial_checkpoint,
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": _sha256(checkpoint_path),
        "history": history,
    }
    report_path = output_dir / "training_report.json"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True, default=_json_safe) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote checkpoint to {checkpoint_path}", flush=True)
    print(f"Wrote report to {report_path}", flush=True)
    return report


@hydra.main(version_base="1.3", config_path="config", config_name="train")
def main(config: DictConfig) -> None:
    values = OmegaConf.to_container(config, resolve=True)
    if not isinstance(values, dict):
        raise TypeError("training config must resolve to a mapping")
    train_exact(values)


if __name__ == "__main__":
    main()
