"""Small deterministic supervised trainer for exact ToySTL targets."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch
from torch import nn

from toy.artifacts import digest_files, sha256_file
from toy.network import ToyModelConfig, ToyPolicyValueNet, feature_dim
from toy.rules import ToyRuleset
from toy.state import ToyState
from toy.targets import load_exact_targets


@dataclass(frozen=True, slots=True)
class ToyTrainConfig:
    epochs: int = 100
    batch_size: int = 256
    learning_rate: float = 1e-3
    policy_loss_weight: float = 1.0
    hidden_dim: int = 64
    val_fraction: float = 0.2
    seed: int = 4
    device: str = "cpu"


@dataclass(slots=True)
class ToyTrainResult:
    checkpoint_path: Path
    checkpoint_manifest_path: Path
    best_epoch: int
    best_validation_loss: float
    history: list[dict] = field(default_factory=list)


def _seed_all(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def _source_digest() -> str:
    root = Path(__file__).resolve().parent
    digest = hashlib.sha256()
    for name in ("state.py", "rules.py", "exact.py", "matrix.py", "network.py", "train.py"):
        digest.update((root / name).read_bytes())
    return digest.hexdigest()


def _source_config_digest(config: ToyTrainConfig) -> str:
    root = Path(__file__).resolve().parent
    return digest_files(
        [root / name for name in ("state.py", "rules.py", "matrix.py", "network.py", "targets.py", "train.py")],
        config=asdict(config),
    )


def _split_grouped(
    physical_ids: np.ndarray,
    val_fraction: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    if not 0.0 < val_fraction < 1.0:
        raise ValueError("val_fraction must be between zero and one")
    groups = np.unique(physical_ids)
    rng = np.random.default_rng(seed)
    rng.shuffle(groups)
    val_count = max(1, int(round(len(groups) * val_fraction)))
    val_groups = set(groups[:val_count].tolist())
    val_mask = np.asarray([value in val_groups for value in physical_ids], dtype=bool)
    train_idx = np.flatnonzero(~val_mask)
    val_idx = np.flatnonzero(val_mask)
    if len(train_idx) == 0 or len(val_idx) == 0:
        raise ValueError("grouped split produced an empty partition")
    return train_idx, val_idx


def _soft_policy_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    return -(targets * torch.log_softmax(logits, dim=-1)).sum(dim=-1).mean()


def _evaluate(
    model: ToyPolicyValueNet,
    features: torch.Tensor,
    values: torch.Tensor,
    drop_targets: torch.Tensor,
    check_targets: torch.Tensor,
    policy_weight: float,
) -> dict[str, float]:
    with torch.no_grad():
        predicted_value, drop_logits, check_logits = model(features)
        value_mse = torch.mean((predicted_value.squeeze(-1) - values) ** 2)
        drop_loss = _soft_policy_loss(drop_logits, drop_targets)
        check_loss = _soft_policy_loss(check_logits, check_targets)
        policy_loss = 0.5 * (drop_loss + check_loss)
        total = value_mse + policy_weight * policy_loss
    return {
        "value_mse": float(value_mse.item()),
        "policy_loss": float(policy_loss.item()),
        "total_loss": float(total.item()),
    }


def train_exact_targets(
    target_npz: str | Path,
    target_manifest: str | Path,
    rules: ToyRuleset,
    output_dir: str | Path,
    *,
    config: ToyTrainConfig | None = None,
) -> ToyTrainResult:
    config = config or ToyTrainConfig()
    if config.epochs <= 0 or config.batch_size <= 0:
        raise ValueError("epochs and batch_size must be positive")
    if config.learning_rate <= 0.0 or config.hidden_dim <= 0:
        raise ValueError("learning rate and hidden width must be positive")
    _seed_all(config.seed)
    arrays, target_metadata = load_exact_targets(target_npz, target_manifest)
    states = np.asarray(arrays["states"], dtype=np.float32)
    horizons = np.asarray(arrays["horizon"], dtype=np.int64)
    values = np.asarray(arrays["value"], dtype=np.float32)
    drop_targets = np.asarray(arrays["drop_policy"], dtype=np.float32)
    check_targets = np.asarray(arrays["check_policy"], dtype=np.float32)
    physical_ids = np.asarray(arrays["physical_state_id"])
    if states.ndim != 2 or states.shape[1] != 3:
        raise ValueError("v0 targets must store three physical state fields")
    if drop_targets.shape != check_targets.shape or drop_targets.shape[1] != rules.action_size:
        raise ValueError("target policy shape does not match the active ruleset")
    if len(states) == 0 or len(states) != len(horizons) or len(states) != len(values):
        raise ValueError("target arrays must have a common nonzero row count")
    if not np.all(np.isfinite(values)) or np.any(np.abs(values) > 1.0 + 1e-6):
        raise ValueError("target values must lie in [-1, 1]")
    for name, policy in (("drop", drop_targets), ("check", check_targets)):
        if not np.all(np.isfinite(policy)) or np.any(policy < -1e-7):
            raise ValueError(f"{name} targets must be finite and nonnegative")
        if not np.allclose(policy.sum(axis=1), 1.0, atol=1e-5):
            raise ValueError(f"{name} targets must be normalized")

    features = np.stack(
        [
            rules.encode_state(
                ToyState(*[int(value) for value in row]),
                int(horizon),
            )
            for row, horizon in zip(states, horizons)
        ],
        axis=0,
    ).astype(np.float32)
    train_idx, val_idx = _split_grouped(physical_ids, config.val_fraction, config.seed)

    model_config = ToyModelConfig(
        feature_dim=feature_dim(rules),
        action_size=rules.action_size,
        hidden_dim=config.hidden_dim,
    )
    model = ToyPolicyValueNet(**model_config.to_dict()).to(config.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    feature_tensor = torch.as_tensor(features, dtype=torch.float32, device=config.device)
    value_tensor = torch.as_tensor(values, dtype=torch.float32, device=config.device)
    drop_tensor = torch.as_tensor(drop_targets, dtype=torch.float32, device=config.device)
    check_tensor = torch.as_tensor(check_targets, dtype=torch.float32, device=config.device)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "best.pt"
    checkpoint_manifest_path = output_dir / "best.json"
    best_loss = float("inf")
    best_epoch = -1
    history: list[dict] = []
    source_config_digest = _source_config_digest(config)

    rng = np.random.default_rng(config.seed)
    for epoch in range(config.epochs):
        model.train()
        order = train_idx.copy()
        rng.shuffle(order)
        for start in range(0, len(order), config.batch_size):
            batch = order[start : start + config.batch_size]
            optimizer.zero_grad(set_to_none=True)
            predicted, drop_logits, check_logits = model(feature_tensor[batch])
            value_loss = torch.mean((predicted.squeeze(-1) - value_tensor[batch]) ** 2)
            policy_loss = 0.5 * (
                _soft_policy_loss(drop_logits, drop_tensor[batch])
                + _soft_policy_loss(check_logits, check_tensor[batch])
            )
            loss = value_loss + config.policy_loss_weight * policy_loss
            loss.backward()
            optimizer.step()

        model.eval()
        train_metrics = _evaluate(
            model,
            feature_tensor[train_idx],
            value_tensor[train_idx],
            drop_tensor[train_idx],
            check_tensor[train_idx],
            config.policy_loss_weight,
        )
        val_metrics = _evaluate(
            model,
            feature_tensor[val_idx],
            value_tensor[val_idx],
            drop_tensor[val_idx],
            check_tensor[val_idx],
            config.policy_loss_weight,
        )
        record = {
            "epoch": epoch,
            **{f"train_{key}": value for key, value in train_metrics.items()},
            **{f"val_{key}": value for key, value in val_metrics.items()},
        }
        history.append(record)
        if val_metrics["total_loss"] < best_loss:
            best_loss = val_metrics["total_loss"]
            best_epoch = epoch
            payload = {
                "state_dict": model.state_dict(),
                "model_config": model_config.to_dict(),
                "ruleset_id": rules.ruleset_id,
                "feature_schema": list(rules.feature_names),
                "action_schema": {
                    "action_size": rules.action_size,
                    "actions": list(rules.action_values),
                },
                "target_manifest_digest": sha256_file(target_manifest),
                "target_npz_digest": sha256_file(target_npz),
                "target_metadata": target_metadata.get("metadata", {}),
                "train_config": asdict(config),
                "best_epoch": best_epoch,
                "best_validation_loss": best_loss,
                "source_tree_digest": _source_digest(),
                "source_config_digest": source_config_digest,
            }
            torch.save(payload, checkpoint_path)

    if best_epoch < 0:
        raise RuntimeError("training produced no checkpoint")
    best_record = history[best_epoch]
    legal_mask = np.zeros(rules.action_size, dtype=bool)
    legal_mask[np.asarray(rules.action_values, dtype=np.int64) - 1] = True
    illegal_target_mass = max(
        float(np.max(drop_targets[:, ~legal_mask].sum(axis=1))) if np.any(~legal_mask) else 0.0,
        float(np.max(check_targets[:, ~legal_mask].sum(axis=1))) if np.any(~legal_mask) else 0.0,
    )
    manifest = {
        "schema_version": "toy.checkpoint.v2",
        "ruleset_id": rules.ruleset_id,
        "feature_schema": list(rules.feature_names),
        "action_schema": {
            "action_size": rules.action_size,
            "actions": list(rules.action_values),
        },
        "model_config": model_config.to_dict(),
        "train_config": asdict(config),
        "target_manifest_sha256": sha256_file(target_manifest),
        "target_npz_sha256": sha256_file(target_npz),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "best_epoch": best_epoch,
        "best_validation_loss": best_loss,
        "source_tree_digest": _source_digest(),
        "source_config_digest": source_config_digest,
        "training_gates": {
            "value_loss_decreased": history[-1]["train_value_mse"] < history[0]["train_value_mse"],
            "policy_loss_decreased": history[-1]["train_policy_loss"] < history[0]["train_policy_loss"],
            "heldout_value_mse": best_record["val_value_mse"],
            "heldout_value_mse_le_0_02": best_record["val_value_mse"] <= 0.02,
            "max_illegal_target_mass": illegal_target_mass,
            "no_illegal_target_mass": illegal_target_mass <= 1e-7,
        },
        "history": history,
    }
    checkpoint_manifest_path.write_text(
        json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    return ToyTrainResult(
        checkpoint_path=checkpoint_path,
        checkpoint_manifest_path=checkpoint_manifest_path,
        best_epoch=best_epoch,
        best_validation_loss=best_loss,
        history=history,
    )


def load_toy_checkpoint(
    checkpoint_path: str | Path,
    rules: ToyRuleset,
    *,
    device: str = "cpu",
) -> ToyPolicyValueNet:
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if payload.get("ruleset_id") != rules.ruleset_id:
        raise ValueError("checkpoint ruleset does not match active rules")
    if payload.get("feature_schema") != list(rules.feature_names):
        raise ValueError("checkpoint feature schema does not match active rules")
    action_schema = payload.get("action_schema", {})
    if action_schema.get("action_size") != rules.action_size:
        raise ValueError("checkpoint action schema does not match active rules")
    model_config = payload["model_config"]
    model = ToyPolicyValueNet(**model_config)
    model.load_state_dict(payload["state_dict"])
    model.to(device)
    model.eval()
    return model
