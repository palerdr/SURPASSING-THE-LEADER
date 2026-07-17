"""Hydra-configured supervised training on pure exact targets."""

from __future__ import annotations

import hashlib
import json
import random
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

from pure.generate_dataset import TARGET_SCHEMA
from pure.network import FEATURE_SCHEMA, PureNetworkConfig, PurePolicyValueNet
from pure.solver import CHECKER_ACTIONS, DROPPER_ACTIONS, payoff, solve


@dataclass(frozen=True)
class ExactTargets:
    states: np.ndarray
    horizons: np.ndarray
    values: np.ndarray
    drop_policies: np.ndarray
    check_policies: np.ndarray
    dataset_version: str
    schema_version: str

    def __len__(self) -> int:
        return int(self.states.shape[0])


class TargetRows(Dataset[tuple[Tensor, ...]]):
    def __init__(
        self,
        targets: ExactTargets,
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
        raise ValueError("drop action schema does not match the pure solver")
    if not np.array_equal(check_actions, np.asarray(CHECKER_ACTIONS)):
        raise ValueError("check action schema does not match the pure solver")
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
        dataset_version=dataset_version,
        schema_version=schema_version,
    )


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


def soft_cross_entropy(logits: Tensor, targets: Tensor) -> Tensor:
    return -(targets * F.log_softmax(logits, dim=-1)).sum(dim=-1)


def _weighted_mean(values: Tensor, weights: Tensor) -> Tensor:
    return (values * weights).sum() / weights.sum()


def _batch_loss(
    model: PurePolicyValueNet,
    batch: tuple[Tensor, ...],
    *,
    policy_weight: float,
    device: torch.device,
) -> Tensor:
    features, values, drop_targets, check_targets, weights, _ = (
        tensor.to(device) for tensor in batch
    )
    predicted_values, drop_logits, check_logits = model(features)
    value_mse = _weighted_mean((predicted_values - values) ** 2, weights)
    drop_ce = _weighted_mean(soft_cross_entropy(drop_logits, drop_targets), weights)
    check_ce = _weighted_mean(soft_cross_entropy(check_logits, check_targets), weights)
    return value_mse + float(policy_weight) * 0.5 * (drop_ce + check_ce)


@torch.no_grad()
def evaluate(
    model: PurePolicyValueNet,
    loader: DataLoader[tuple[Tensor, ...]],
    *,
    policy_weight: float,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    numerators = defaultdict(float)
    weight_total = 0.0
    for batch in loader:
        features, values, drop_targets, check_targets, weights, _ = (
            tensor.to(device) for tensor in batch
        )
        predicted_values, drop_logits, check_logits = model(features)
        metrics = {
            "value_mse": (predicted_values - values) ** 2,
            "drop_ce": soft_cross_entropy(drop_logits, drop_targets),
            "check_ce": soft_cross_entropy(check_logits, check_targets),
        }
        for name, rows in metrics.items():
            numerators[name] += float((rows * weights).sum().item())
        weight_total += float(weights.sum().item())
    result = {name: value / weight_total for name, value in numerators.items()}
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
    model: PurePolicyValueNet,
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
    model_config = PureNetworkConfig(
        hidden_width=int(model_values["hidden_width"]),
        hidden_layers=int(model_values["hidden_layers"]),
        action_count=len(DROPPER_ACTIONS),
        horizon_scale=float(model_values["horizon_scale"]),
    )
    model = PurePolicyValueNet(model_config)
    device = torch.device(str(config["device"]))
    model.to(device)

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
    optimizer = Adam(
        model.parameters(),
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    policy_weight = float(training["policy_weight"])
    best_loss = float("inf")
    best_epoch = 0
    stale_epochs = 0
    history: list[dict[str, Any]] = []
    checkpoint_path = output_dir / "best.pt"

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
        history.append(
            {"epoch": epoch, "train": train_metrics, "validation": validation_metrics}
        )

        validation_loss = validation_metrics["total_loss"]
        if validation_loss < best_loss - float(training["minimum_delta"]):
            best_loss = validation_loss
            best_epoch = epoch
            stale_epochs = 0
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
                    "epoch": epoch,
                    "validation_metrics": validation_metrics,
                },
                checkpoint_path,
            )
        else:
            stale_epochs += 1

        if epoch == 1 or epoch % log_every == 0:
            print(
                f"epoch={epoch} train={train_metrics['total_loss']:.6f} "
                f"validation={validation_loss:.6f} "
                f"value_mse={validation_metrics['value_mse']:.6f}",
                flush=True,
            )
        if patience > 0 and stale_epochs >= patience:
            print(f"Early stopping at epoch {epoch}", flush=True)
            break

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["state_dict"])
    final_train = evaluate(
        model,
        train_loader,
        policy_weight=policy_weight,
        device=device,
    )
    final_validation = evaluate(
        model,
        validation_loader,
        policy_weight=policy_weight,
        device=device,
    )
    audit = audit_saddle_gaps(
        model,
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
        "model": model_config.to_dict(),
        "training": dict(training),
        "best_epoch": best_epoch,
        "train_metrics": final_train,
        "validation_metrics": final_validation,
        "exact_matrix_audit": audit,
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
