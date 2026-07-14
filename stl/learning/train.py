"""Supervised policy/value training for the exact-to-self-play bridge.

The default path consumes a validated ``TrainingRecordV3`` shard, performs a
grouped train/validation split, optimizes Hal-perspective value MSE plus masked
dropper/checker cross entropy, and selects the best epoch by grouped validation
MSE. Manifestless corpora require an explicit legacy opt-in.

``train()`` writes ``best.pt`` and ``last.pt`` as strict
``stl.checkpoint.v2`` bundles containing model, optimizer, scheduler, schemas,
provenance, resolved config, training history, and RNG state. ``log.json``
contains the per-epoch metrics, and ``TrainResult`` exposes the summary in
memory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from stl.learning.contracts import canonical_config_json, config_digest
from stl.learning.model import (
    FEATURE_DIM,
    FEATURE_SCHEMA_VERSION,
    HIDDEN_DIM,
    ValueNet,
    value_output,
)
from stl.engine.actions import ACTION_SIZE


CHECKPOINT_FORMAT_VERSION = "stl.checkpoint.v2"
ACTION_SCHEMA_VERSION = "stl.literal-seconds.v1"


class CheckpointFormatError(ValueError):
    """Raised when checkpoint semantics do not match the active model."""


@dataclass(frozen=True)
class TrainConfig:
    """Hyperparameters for the supervised training loop."""

    epochs: int = 100
    batch_size: int = 64
    learning_rate: float = 1.0e-3
    weight_decay: float = 0.0  # Adam L2 regularization; 0.0 = original behavior.
    policy_loss_weight: float = 1.0
    val_fraction: float = 0.1
    early_stopping_patience: int = 25
    seed: int = 0
    device: str = "cpu"
    # Per-class value-loss weight overrides. Sources not listed default to 1.0.
    # Default upweights extreme-valued anchors (terminal/tablebase) so the net
    # actually fits them despite their small share of the corpus. Without
    # this, the calibration-gate tablebase MSE threshold is unreachable
    # whenever tablebase is <5% of training records.
    source_weights: tuple[tuple[str, float], ...] = (
        ("terminal", 20.0),
        ("tablebase", 20.0),
    )
    # Per-class policy-loss weight overrides. Sources not listed default to 1.0.
    # This is intentionally separate from value loss: a tablebase- or MCTS-
    # produced source can be useful as a policy prior without deserving the
    # same weight as a value target on the held-out ruler.
    policy_source_weights: tuple[tuple[str, float], ...] = ()
    # Net architecture knobs (Phase I). hidden_dim=64 is the original 13.7K-
    # param trunk; hidden_dim=128 gives 35.5K params (2.6× capacity) for
    # fitting more diverse tablebase pins. hidden_dim=192 (65K) exceeds the
    # 50K guard; raise the guard only with explicit justification.
    hidden_dim: int = 64
    # Optional warm start for fine-tuning from an accepted checkpoint. When
    # omitted, training starts from a seeded random initialization as before.
    init_checkpoint: str | None = None
    # Which module subset may update during training. Use "value_head" for
    # conservative calibration tweaks that should leave policy priors and the
    # shared feature trunk intact.
    trainable_parts: str = "all"
    # Optional trust-region reference. When set with value_distill_weight > 0,
    # training penalizes value-output drift from this checkpoint on each batch.
    reference_checkpoint: str | None = None
    value_distill_weight: float = 0.0
    # Optional integrity assertion for warm starts.  When set, the digest of
    # ``init_checkpoint`` must match before any optimizer work begins.
    required_parent_digest: str | None = None
    # Manifestless ValueTarget NPZ files are legacy.  New training requires a
    # validated TrainingRecordV3 shard unless a caller opts into migration.
    allow_legacy_targets: bool = False


@dataclass
class TrainResult:
    """Outcome of a training run."""

    best_val_mse: float
    best_epoch: int
    best_per_source_mse: dict[str, float]
    final_train_mse: float
    final_val_mse: float
    train_history: list[dict] = field(default_factory=list)
    checkpoint_path: str = ""


def _seed_all(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def _split_indices(X: np.ndarray, val_fraction: float, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Deterministic, group-aware train/val split under a seeded rng.

    Rows are grouped by unique state identity (``features.tobytes()``) and
    whole groups are assigned to train or val. Callers (e.g.
    ``scripts/run_gen_iteration.py``) replicate tablebase/interior anchor
    records 30-660x before training; a raw row permutation scattered those
    identical replicas across both splits, so the validation MSE (and hence
    best-epoch selection) was leakage-biased. Group assignment guarantees no
    state appears on both sides. Same seed -> same split (dict insertion
    order is row order, so the grouping is deterministic for a given X).
    """
    groups: dict[bytes, list[int]] = {}
    for i in range(len(X)):
        groups.setdefault(X[i].tobytes(), []).append(i)
    keys = list(groups.keys())
    order = rng.permutation(len(keys))
    n_val_groups = max(1, int(round(len(keys) * val_fraction)))
    val_idx: list[int] = []
    train_idx: list[int] = []
    for rank, key_pos in enumerate(order):
        bucket = val_idx if rank < n_val_groups else train_idx
        bucket.extend(groups[keys[key_pos]])
    return np.array(train_idx, dtype=np.int64), np.array(val_idx, dtype=np.int64)


def _per_source_mse(
    preds: np.ndarray,
    labels: np.ndarray,
    sources: np.ndarray,
) -> dict[str, float]:
    """Mean squared error keyed by ``ValueTarget.source`` tag."""
    out: dict[str, float] = {}
    squared_error = (preds - labels) ** 2
    for source in np.unique(sources):
        mask = sources == source
        out[str(source)] = float(squared_error[mask].mean())
    return out


def _evaluate(
    model: nn.Module,
    X: torch.Tensor,
    y: torch.Tensor,
    sources: np.ndarray,
) -> tuple[float, dict[str, float]]:
    """Run the model on (X, y) and return (overall MSE, per-source MSE)."""
    model.eval()
    with torch.no_grad():
        preds = value_output(model(X)).squeeze(-1).cpu().numpy()
    labels = y.cpu().numpy()
    overall = float(((preds - labels) ** 2).mean())
    per_source = _per_source_mse(preds, labels, sources)
    return overall, per_source


def _load_targets_npz(
    path: str | Path,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    list | None,
]:
    """Load value and policy targets from a saved .npz file."""
    from stl.learning.replay import load_replay_shard, manifest_path_for

    if manifest_path_for(path).exists():
        records = load_replay_shard(path, for_training=True)
        n = len(records)
        X = (
            np.stack([record.features for record in records]).astype(np.float32)
            if records
            else np.zeros((0, FEATURE_DIM), dtype=np.float32)
        )
        y = np.asarray([record.value for record in records], dtype=np.float32)
        sources = np.asarray([record.source for record in records], dtype=np.str_)
        dropper_dists = (
            np.stack([record.dropper_dist for record in records]).astype(np.float32)
            if records
            else np.zeros((0, ACTION_SIZE), dtype=np.float32)
        )
        checker_dists = (
            np.stack([record.checker_dist for record in records]).astype(np.float32)
            if records
            else np.zeros((0, ACTION_SIZE), dtype=np.float32)
        )
        dropper_masks = (
            np.stack([record.dropper_legal_mask for record in records]).astype(np.float32)
            if records
            else np.zeros((0, ACTION_SIZE), dtype=np.float32)
        )
        checker_masks = (
            np.stack([record.checker_legal_mask for record in records]).astype(np.float32)
            if records
            else np.zeros((0, ACTION_SIZE), dtype=np.float32)
        )
        return (
            X,
            y,
            sources,
            dropper_dists,
            checker_dists,
            dropper_masks,
            checker_masks,
            records,
        )

    data = np.load(path, allow_pickle=False)
    X = data["X"].astype(np.float32)
    y = data["y"].astype(np.float32)
    sources = np.array(data["sources"]).astype(str)
    n = len(X)
    dropper_dists = data["dropper_dists"].astype(np.float32) if "dropper_dists" in data else np.zeros((n, ACTION_SIZE), dtype=np.float32)
    checker_dists = data["checker_dists"].astype(np.float32) if "checker_dists" in data else np.zeros((n, ACTION_SIZE), dtype=np.float32)
    dropper_masks = data["dropper_legal_masks"].astype(np.float32) if "dropper_legal_masks" in data else (dropper_dists > 0.0).astype(np.float32)
    checker_masks = data["checker_legal_masks"].astype(np.float32) if "checker_legal_masks" in data else (checker_dists > 0.0).astype(np.float32)
    if X.shape[1] != FEATURE_DIM:
        raise ValueError(
            f"Expected FEATURE_DIM={FEATURE_DIM} columns in X, got {X.shape[1]}"
        )
    for name, arr in (
        ("dropper_dists", dropper_dists),
        ("checker_dists", checker_dists),
        ("dropper_legal_masks", dropper_masks),
        ("checker_legal_masks", checker_masks),
    ):
        if arr.shape != (n, ACTION_SIZE):
            raise ValueError(f"Expected {name} shape {(n, ACTION_SIZE)}, got {arr.shape}")
    return X, y, sources, dropper_dists, checker_dists, dropper_masks, checker_masks, None


def _configure_trainable_parts(model: ValueNet, mode: str) -> None:
    valid = {"all", "value_head", "policy_head", "heads", "trunk_value"}
    if mode not in valid:
        raise ValueError(f"trainable_parts must be one of {sorted(valid)}, got {mode!r}")
    for parameter in model.parameters():
        parameter.requires_grad = False
    if mode == "all":
        for parameter in model.parameters():
            parameter.requires_grad = True
    elif mode == "value_head":
        for parameter in model.value_head.parameters():
            parameter.requires_grad = True
    elif mode == "policy_head":
        for parameter in model.policy_head.parameters():
            parameter.requires_grad = True
    elif mode == "heads":
        for parameter in model.value_head.parameters():
            parameter.requires_grad = True
        for parameter in model.policy_head.parameters():
            parameter.requires_grad = True
    elif mode == "trunk_value":
        for parameter in model.trunk.parameters():
            parameter.requires_grad = True
        for parameter in model.value_head.parameters():
            parameter.requires_grad = True


def _masked_policy_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    legal_mask: torch.Tensor,
    sample_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    active = target.sum(dim=1) > 1e-8
    if not bool(active.any()):
        return logits.sum() * 0.0

    logits_active = logits[active]
    target_active = target[active]
    mask_active = legal_mask[active] > 0.5
    target_active = target_active / target_active.sum(dim=1, keepdim=True).clamp_min(1e-8)
    masked_logits = logits_active.masked_fill(~mask_active, -1.0e9)
    log_probs = torch.log_softmax(masked_logits, dim=1)
    per_sample = -(target_active * log_probs).sum(dim=1)
    if sample_weight is None:
        return per_sample.mean()
    weight_active = sample_weight[active]
    return (per_sample * weight_active).sum() / weight_active.sum().clamp(min=1e-8)


def train(
    targets_npz_path: str | Path,
    output_dir: str | Path,
    config: TrainConfig | None = None,
) -> TrainResult:
    """Train ValueNet against labeled targets and write checkpoints.

    Args:
        targets_npz_path: Path to ``.npz`` produced by
            ``training.value_targets.save_targets``.
        output_dir: Directory for checkpoints and the training log.
        config: Hyperparameters; defaults to ``TrainConfig()``.

    Returns:
        ``TrainResult`` with best/final MSE values and the path of the
        best-epoch checkpoint.
    """
    config = config or TrainConfig()
    _seed_all(config.seed)
    rng = np.random.default_rng(config.seed)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    (
        X_np,
        y_np,
        sources_np,
        dropper_dists_np,
        checker_dists_np,
        dropper_masks_np,
        checker_masks_np,
        replay_records,
    ) = _load_targets_npz(targets_npz_path)
    if replay_records is None:
        if not config.allow_legacy_targets:
            raise ValueError(
                "manifestless target corpus rejected; set allow_legacy_targets=True "
                "only for an explicit V1 migration"
            )
        train_idx, val_idx = _split_indices(X_np, config.val_fraction, rng)
    else:
        from stl.learning.replay import grouped_split_indices

        train_idx, val_idx = grouped_split_indices(
            replay_records,
            validation_fraction=config.val_fraction,
            seed=config.seed,
        )

    device = torch.device(config.device)
    X_train = torch.from_numpy(X_np[train_idx]).to(device)
    y_train = torch.from_numpy(y_np[train_idx]).to(device)
    dropper_train = torch.from_numpy(dropper_dists_np[train_idx]).to(device)
    checker_train = torch.from_numpy(checker_dists_np[train_idx]).to(device)
    dropper_mask_train = torch.from_numpy(dropper_masks_np[train_idx]).to(device)
    checker_mask_train = torch.from_numpy(checker_masks_np[train_idx]).to(device)
    X_val = torch.from_numpy(X_np[val_idx]).to(device)
    y_val = torch.from_numpy(y_np[val_idx]).to(device)
    val_sources = sources_np[val_idx]

    weight_lookup = dict(config.source_weights)
    policy_weight_lookup = dict(config.policy_source_weights)
    train_sources = sources_np[train_idx]
    train_weights_np = np.array(
        [weight_lookup.get(str(s), 1.0) for s in train_sources],
        dtype=np.float32,
    )
    train_policy_weights_np = np.array(
        [policy_weight_lookup.get(str(s), 1.0) for s in train_sources],
        dtype=np.float32,
    )
    weights_train = torch.from_numpy(train_weights_np).to(device)
    policy_weights_train = torch.from_numpy(train_policy_weights_np).to(device)

    train_ds = TensorDataset(
        X_train,
        y_train,
        dropper_train,
        checker_train,
        dropper_mask_train,
        checker_mask_train,
        weights_train,
        policy_weights_train,
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=config.batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(config.seed),
    )

    parent_digest = None
    if config.init_checkpoint is not None:
        parent_digest = checkpoint_digest(config.init_checkpoint)
        if (
            config.required_parent_digest is not None
            and parent_digest != config.required_parent_digest
        ):
            raise CheckpointFormatError(
                "init_checkpoint digest does not match required_parent_digest"
            )
        model = load_checkpoint(config.init_checkpoint, device=config.device).to(device)
        loaded_hidden = int(model.trunk[0].out_features)
        if loaded_hidden != config.hidden_dim:
            raise ValueError(
                f"init_checkpoint hidden_dim={loaded_hidden} does not match "
                f"TrainConfig.hidden_dim={config.hidden_dim}"
            )
    else:
        model = ValueNet(input_dim=FEATURE_DIM, hidden_dim=config.hidden_dim).to(device)
    _configure_trainable_parts(model, config.trainable_parts)
    trainable_parameters = [p for p in model.parameters() if p.requires_grad]
    if not trainable_parameters:
        raise ValueError(f"trainable_parts={config.trainable_parts!r} left no trainable parameters")
    reference_model = None
    if config.reference_checkpoint is not None and config.value_distill_weight > 0.0:
        reference_model = load_checkpoint(config.reference_checkpoint, device=config.device).to(device)
        reference_model.eval()
    optimizer = torch.optim.Adam(
        trainable_parameters, lr=config.learning_rate, weight_decay=config.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, config.epochs)
    )
    loss_fn = nn.MSELoss(reduction="none")

    best_val = math.inf
    best_epoch = 0
    best_per_source: dict[str, float] = {}
    best_path = output_dir / "best.pt"
    last_path = output_dir / "last.pt"
    log_path = output_dir / "log.json"
    history: list[dict] = []
    epochs_since_improvement = 0

    final_train_mse = 0.0
    val_mse = math.inf
    for epoch in range(config.epochs):
        model.train()
        total_loss = 0.0
        n_seen = 0
        total_policy = 0.0
        total_distill = 0.0
        for (
            x_batch,
            y_batch,
            dropper_batch,
            checker_batch,
            dropper_mask_batch,
            checker_mask_batch,
            weight_batch,
            policy_weight_batch,
        ) in train_loader:
            optimizer.zero_grad()
            preds, dropper_logits, checker_logits = model(x_batch)
            per_sample_se = loss_fn(preds.squeeze(-1), y_batch)
            value_loss = (per_sample_se * weight_batch).sum() / weight_batch.sum().clamp(min=1e-8)
            if reference_model is not None:
                with torch.no_grad():
                    ref_preds = value_output(reference_model(x_batch)).squeeze(-1)
                value_distill_loss = loss_fn(preds.squeeze(-1), ref_preds).mean()
            else:
                value_distill_loss = preds.sum() * 0.0
            dropper_loss = _masked_policy_loss(
                dropper_logits,
                dropper_batch,
                dropper_mask_batch,
                policy_weight_batch,
            )
            checker_loss = _masked_policy_loss(
                checker_logits,
                checker_batch,
                checker_mask_batch,
                policy_weight_batch,
            )
            policy_loss = dropper_loss + checker_loss
            loss = (
                value_loss
                + config.policy_loss_weight * policy_loss
                + config.value_distill_weight * value_distill_loss
            )
            loss.backward()
            optimizer.step()
            batch_size = x_batch.shape[0]
            total_loss += float(value_loss.item()) * batch_size
            total_policy += float(policy_loss.item()) * batch_size
            total_distill += float(value_distill_loss.item()) * batch_size
            n_seen += batch_size
        scheduler.step()

        train_mse = total_loss / max(1, n_seen)
        train_policy_nll = total_policy / max(1, n_seen)
        train_value_distill_mse = total_distill / max(1, n_seen)
        val_mse, per_source = _evaluate(model, X_val, y_val, val_sources)
        history.append(
            {
                "epoch": epoch,
                "train_mse": train_mse,
                "train_policy_nll": train_policy_nll,
                "train_value_distill_mse": train_value_distill_mse,
                "val_mse": val_mse,
                "per_source_val_mse": per_source,
                "lr": optimizer.param_groups[0]["lr"],
            }
        )
        final_train_mse = train_mse

        if val_mse < best_val:
            best_val = val_mse
            best_epoch = epoch
            best_per_source = per_source
            save_checkpoint_bundle(
                best_path,
                model,
                optimizer=optimizer,
                scheduler=scheduler,
                train_config=config,
                parent_digest=parent_digest,
                corpus_digests=(checkpoint_digest(targets_npz_path),),
                history=history,
                epoch=epoch,
            )
            epochs_since_improvement = 0
        else:
            epochs_since_improvement += 1
            if epochs_since_improvement >= config.early_stopping_patience:
                break

    save_checkpoint_bundle(
        last_path,
        model,
        optimizer=optimizer,
        scheduler=scheduler,
        train_config=config,
        parent_digest=parent_digest,
        corpus_digests=(checkpoint_digest(targets_npz_path),),
        history=history,
        epoch=len(history) - 1,
    )
    with open(log_path, "w") as f:
        json.dump(
            {
                "config": asdict(config),
                "history": history,
                "best_val_mse": best_val,
                "best_epoch": best_epoch,
                "best_per_source_mse": best_per_source,
            },
            f,
            indent=2,
        )

    return TrainResult(
        best_val_mse=best_val,
        best_epoch=best_epoch,
        best_per_source_mse=best_per_source,
        final_train_mse=final_train_mse,
        final_val_mse=val_mse,
        train_history=history,
        checkpoint_path=str(best_path),
    )


def checkpoint_digest(path: str | Path) -> str:
    """SHA-256 digest for a checkpoint, corpus, or other immutable artifact."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def _atomic_torch_save(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(fd)
    temporary = Path(temporary_name)
    try:
        torch.save(payload, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def save_checkpoint_bundle(
    path: str | Path,
    model: ValueNet,
    *,
    optimizer=None,
    scheduler=None,
    train_config: TrainConfig | None = None,
    parent_digest: str | None = None,
    corpus_digests: tuple[str, ...] = (),
    history: list[dict] | None = None,
    epoch: int | None = None,
) -> None:
    """Atomically write a self-describing V2 checkpoint bundle."""

    hidden_dim = int(model.trunk[0].out_features)
    resolved_config = asdict(train_config) if train_config is not None else {}
    payload = {
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "model_schema": {
            "feature_schema": FEATURE_SCHEMA_VERSION,
            "feature_dim": FEATURE_DIM,
            "action_schema": ACTION_SCHEMA_VERSION,
            "action_size": ACTION_SIZE,
            "hidden_dim": hidden_dim,
        },
        "provenance": {
            "parent_digest": parent_digest,
            "corpus_digests": list(corpus_digests),
            "resolved_config": resolved_config,
            "resolved_config_json": canonical_config_json(resolved_config),
            "resolved_config_digest": config_digest(resolved_config),
            "git_commit": _git_commit(),
            "seed": train_config.seed if train_config is not None else None,
        },
        "training": {
            "epoch": epoch,
            "history": list(history or ()),
            "numpy_rng_state": np.random.get_state(),
            "torch_rng_state": torch.get_rng_state(),
        },
    }
    _atomic_torch_save(payload, Path(path))


def load_checkpoint_bundle(
    checkpoint_path: str | Path,
    *,
    device: str = "cpu",
    expected_parent_digest: str | None = None,
) -> dict:
    """Load and validate checkpoint metadata before constructing a model."""

    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if not isinstance(payload, dict) or payload.get("format_version") != CHECKPOINT_FORMAT_VERSION:
        raise CheckpointFormatError(
            "bare or legacy checkpoint rejected; use load_legacy_checkpoint() "
            "only for an explicit migration"
        )
    required = {"model_state_dict", "model_schema", "provenance", "training"}
    missing = required - set(payload)
    if missing:
        raise CheckpointFormatError(f"checkpoint bundle missing fields: {sorted(missing)}")

    schema = payload["model_schema"]
    expected_schema = {
        "feature_schema": FEATURE_SCHEMA_VERSION,
        "feature_dim": FEATURE_DIM,
        "action_schema": ACTION_SCHEMA_VERSION,
        "action_size": ACTION_SIZE,
    }
    for key, expected in expected_schema.items():
        if schema.get(key) != expected:
            raise CheckpointFormatError(
                f"checkpoint {key}={schema.get(key)!r}, expected {expected!r}"
            )
    parent_digest = payload["provenance"].get("parent_digest")
    if expected_parent_digest is not None and parent_digest != expected_parent_digest:
        raise CheckpointFormatError("checkpoint parent digest mismatch")

    state = payload["model_state_dict"]
    if not isinstance(state, dict) or "trunk.0.weight" not in state:
        raise CheckpointFormatError("checkpoint has no modern trunk.0.weight")
    hidden_dim = int(schema.get("hidden_dim", 0))
    if hidden_dim <= 0:
        raise CheckpointFormatError("checkpoint hidden_dim must be positive")
    first_weight = state["trunk.0.weight"]
    policy_weight = state.get("policy_head.weight")
    if tuple(first_weight.shape) != (hidden_dim, FEATURE_DIM):
        raise CheckpointFormatError("checkpoint input layer does not match feature schema")
    if policy_weight is None or tuple(policy_weight.shape) != (2 * ACTION_SIZE, hidden_dim):
        raise CheckpointFormatError("checkpoint policy head does not match action schema")
    return payload


def load_checkpoint(checkpoint_path: str | Path, device: str = "cpu") -> ValueNet:
    """Restore a strictly validated V2 ValueNet checkpoint bundle."""

    payload = load_checkpoint_bundle(checkpoint_path, device=device)
    hidden_dim = int(payload["model_schema"]["hidden_dim"])
    model = ValueNet(input_dim=FEATURE_DIM, hidden_dim=hidden_dim)
    try:
        model.load_state_dict(payload["model_state_dict"], strict=True)
    except RuntimeError as exc:
        raise CheckpointFormatError(f"checkpoint state_dict is incompatible: {exc}") from exc
    model.to(device)
    model.eval()
    return model


def load_legacy_checkpoint(checkpoint_path: str | Path, device: str = "cpu") -> ValueNet:
    """Explicit V1/bare-state adapter for inspection and one-time migration."""

    state = torch.load(checkpoint_path, map_location=device, weights_only=True)
    if not isinstance(state, dict) or "trunk.0.weight" not in state:
        raise CheckpointFormatError("unrecognized legacy checkpoint")
    input_dim = int(state["trunk.0.weight"].shape[1])
    if input_dim != FEATURE_DIM:
        raise CheckpointFormatError(
            f"legacy checkpoint input_dim={input_dim}; active V2 requires {FEATURE_DIM}"
        )
    hidden_dim = int(state["trunk.0.weight"].shape[0])
    model = ValueNet(input_dim=FEATURE_DIM, hidden_dim=hidden_dim)
    model.load_state_dict(state, strict=True)
    model.to(device)
    model.eval()
    return model


def make_predict_fn(model: ValueNet, device: str = "cpu"):
    """Return a callable ``(game) -> (value, dropper_dist, checker_dist)``.

    Suitable for ``ValueNetEvaluator(model_fn=predict_fn)`` and for
    ``generate_mcts_bootstrap_targets(predict_fn=...)``.
    """
    from stl.learning.model import extract_features

    torch_device = torch.device(device)
    model = model.to(torch_device)
    model.eval()

    def predict(game) -> float:
        features = extract_features(game)
        x = torch.from_numpy(features).unsqueeze(0).to(torch_device)
        with torch.no_grad():
            value, dropper_logits, checker_logits = model(x)
            y = value.squeeze().cpu().item()

        from stl.solver.search import normalize_policy_vector
        from stl.solver.exact import ExactSearchConfig, legal_seconds_for_current_role

        def masked_softmax(logits, seconds: tuple[int, ...]) -> np.ndarray:
            dist = np.zeros(ACTION_SIZE, dtype=np.float64)
            if not seconds:
                return dist
            raw = logits.squeeze(0).detach().cpu().numpy().astype(np.float64)
            legal = np.array([raw[second] for second in seconds], dtype=np.float64)
            legal -= float(legal.max())
            probs = np.exp(legal)
            probs /= probs.sum()
            for second, probability in zip(seconds, probs):
                dist[second] = float(probability)
            return normalize_policy_vector(dist)

        cfg = ExactSearchConfig()
        if game.game_over:
            dropper_dist = np.zeros(ACTION_SIZE, dtype=np.float64)
            checker_dist = np.zeros(ACTION_SIZE, dtype=np.float64)
        else:
            dropper, checker = game.get_roles_for_half(game.current_half)
            drop_seconds = tuple(legal_seconds_for_current_role(game, dropper.name, "dropper", cfg))
            check_seconds = tuple(legal_seconds_for_current_role(game, checker.name, "checker", cfg))
            dropper_dist = masked_softmax(dropper_logits, drop_seconds)
            checker_dist = masked_softmax(checker_logits, check_seconds)
        return float(y), dropper_dist, checker_dist

    return predict


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser used directly and by the Hydra dispatcher."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--targets", required=True, help="Path to .npz from value_targets")
    parser.add_argument("--out", required=True, help="Output directory for checkpoints + log")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--allow-legacy-targets",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    return parser


def main() -> int:
    """Train from command-line arguments and return a process exit code."""

    args = build_parser().parse_args()

    cfg = TrainConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        seed=args.seed,
        device=args.device,
        allow_legacy_targets=args.allow_legacy_targets,
    )
    result = train(args.targets, args.out, cfg)
    print(
        f"Best val MSE {result.best_val_mse:.5f} at epoch {result.best_epoch}; "
        f"checkpoint={result.checkpoint_path}"
    )
    if result.best_per_source_mse:
        print("Per-source val MSE:")
        for source, mse in sorted(result.best_per_source_mse.items()):
            print(f"  {source}: {mse:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
