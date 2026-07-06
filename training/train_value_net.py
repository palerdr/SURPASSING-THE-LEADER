"""Phase 4: supervised training pipeline for the value net.

Trains ``hal.value_net.ValueNet`` against labeled targets produced by
``training.value_targets``. MSE regression with a cosine learning-rate
schedule, a held-out validation split, per-source MSE callbacks, and
model checkpointing on best held-out MSE. No reward shaping, no
imitation; targets are exact equilibrium values or MCTS-bootstrap
values.

Usage (script):

    python training/train_value_net.py --targets gen0.npz --out checkpoints/

API (importable):

    from training.train_value_net import train, TrainConfig
    result = train("gen0.npz", "checkpoints/", TrainConfig(epochs=50))

Outputs:
    - ``<out>/best.pt`` — model state_dict snapshot at best val epoch.
    - ``<out>/last.pt`` — model state_dict snapshot at final epoch.
    - ``<out>/log.json`` — per-epoch metrics including train/val MSE
      and per-source val MSE.
    - ``TrainResult`` carrying the same metrics in memory.
"""

from __future__ import annotations

import json
import math
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from hal.value_net import FEATURE_DIM, HIDDEN_DIM, ValueNet, value_output


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
    # Optional trust-region reference. When set with a positive distillation
    # weight, training penalizes value and/or policy drift from this checkpoint
    # on each batch.
    reference_checkpoint: str | None = None
    value_distill_weight: float = 0.0
    policy_distill_weight: float = 0.0
    # Select which validation metric determines best.pt. Default preserves the
    # calibration-first contract; policy_nll is for policy-head-only repair
    # runs where value MSE is intentionally unchanged.
    selection_metric: str = "value_mse"
    # Sparse trace-repair rows are often the point of a focused fine-tune. Move
    # whole feature groups containing these sources into train after the
    # group-aware split so one-off guards cannot land only in validation.
    force_train_sources: tuple[str, ...] = ()


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
    best_selection_score: float = math.inf


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


def _force_sources_into_train(
    *,
    X: np.ndarray,
    sources: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    force_sources: tuple[str, ...],
) -> tuple[np.ndarray, np.ndarray]:
    """Move validation feature groups containing selected sources to train."""
    if not force_sources or val_idx.size == 0:
        return train_idx, val_idx

    force_set = {str(source) for source in force_sources}
    forced_keys = {
        X[int(idx)].tobytes()
        for idx in val_idx
        if str(sources[int(idx)]) in force_set
    }
    if not forced_keys:
        return train_idx, val_idx

    move_mask = np.array(
        [X[int(idx)].tobytes() in forced_keys for idx in val_idx],
        dtype=bool,
    )
    moved = val_idx[move_mask]
    kept_val = val_idx[~move_mask]
    if moved.size == 0:
        return train_idx, val_idx
    return np.concatenate([train_idx, moved]).astype(np.int64), kept_val.astype(np.int64)


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
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load value and policy targets from a saved .npz file."""
    data = np.load(path, allow_pickle=True)
    X = data["X"].astype(np.float32)
    y = data["y"].astype(np.float32)
    sources = np.array(data["sources"]).astype(str)
    n = len(X)
    dropper_dists = data["dropper_dists"].astype(np.float32) if "dropper_dists" in data else np.zeros((n, 61), dtype=np.float32)
    checker_dists = data["checker_dists"].astype(np.float32) if "checker_dists" in data else np.zeros((n, 61), dtype=np.float32)
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
        if arr.shape != (n, 61):
            raise ValueError(f"Expected {name} shape {(n, 61)}, got {arr.shape}")
    return X, y, sources, dropper_dists, checker_dists, dropper_masks, checker_masks


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


def _masked_policy_distillation_loss(
    logits: torch.Tensor,
    reference_logits: torch.Tensor,
    legal_mask: torch.Tensor,
) -> torch.Tensor:
    active = legal_mask.sum(dim=1) > 0.5
    if not bool(active.any()):
        return logits.sum() * 0.0

    mask_active = legal_mask[active] > 0.5
    student_logits = logits[active].masked_fill(~mask_active, -1.0e9)
    teacher_logits = reference_logits[active].masked_fill(~mask_active, -1.0e9)
    with torch.no_grad():
        teacher_probs = torch.softmax(teacher_logits, dim=1)
    student_log_probs = torch.log_softmax(student_logits, dim=1)
    return -(teacher_probs * student_log_probs).sum(dim=1).mean()


def _evaluate_policy_nll(
    model: nn.Module,
    X: torch.Tensor,
    dropper_target: torch.Tensor,
    checker_target: torch.Tensor,
    dropper_mask: torch.Tensor,
    checker_mask: torch.Tensor,
    policy_weights: torch.Tensor,
) -> float:
    """Validation NLL for both policy heads under the same masking contract."""
    model.eval()
    with torch.no_grad():
        _value, dropper_logits, checker_logits = model(X)
        dropper_loss = _masked_policy_loss(
            dropper_logits,
            dropper_target,
            dropper_mask,
            policy_weights,
        )
        checker_loss = _masked_policy_loss(
            checker_logits,
            checker_target,
            checker_mask,
            policy_weights,
        )
    return float((dropper_loss + checker_loss).item())


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
    if config.selection_metric not in {"value_mse", "policy_nll", "value_plus_policy"}:
        raise ValueError(
            "selection_metric must be one of "
            "'value_mse', 'policy_nll', or 'value_plus_policy'"
        )
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
    ) = _load_targets_npz(targets_npz_path)
    train_idx, val_idx = _split_indices(X_np, config.val_fraction, rng)
    train_idx, val_idx = _force_sources_into_train(
        X=X_np,
        sources=sources_np,
        train_idx=train_idx,
        val_idx=val_idx,
        force_sources=config.force_train_sources,
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
    dropper_val = torch.from_numpy(dropper_dists_np[val_idx]).to(device)
    checker_val = torch.from_numpy(checker_dists_np[val_idx]).to(device)
    dropper_mask_val = torch.from_numpy(dropper_masks_np[val_idx]).to(device)
    checker_mask_val = torch.from_numpy(checker_masks_np[val_idx]).to(device)
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
    val_policy_weights_np = np.array(
        [policy_weight_lookup.get(str(s), 1.0) for s in val_sources],
        dtype=np.float32,
    )
    weights_train = torch.from_numpy(train_weights_np).to(device)
    policy_weights_train = torch.from_numpy(train_policy_weights_np).to(device)
    policy_weights_val = torch.from_numpy(val_policy_weights_np).to(device)

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

    if config.init_checkpoint is not None:
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
    if config.reference_checkpoint is not None and (
        config.value_distill_weight > 0.0 or config.policy_distill_weight > 0.0
    ):
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
    best_selection_score = math.inf
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
        total_value_distill = 0.0
        total_policy_distill = 0.0
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
                    ref_value, ref_dropper_logits, ref_checker_logits = reference_model(x_batch)
                    ref_preds = ref_value.squeeze(-1)
                value_distill_loss = loss_fn(preds.squeeze(-1), ref_preds).mean()
                policy_distill_loss = _masked_policy_distillation_loss(
                    dropper_logits,
                    ref_dropper_logits,
                    dropper_mask_batch,
                ) + _masked_policy_distillation_loss(
                    checker_logits,
                    ref_checker_logits,
                    checker_mask_batch,
                )
            else:
                value_distill_loss = preds.sum() * 0.0
                policy_distill_loss = preds.sum() * 0.0
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
                + config.policy_distill_weight * policy_distill_loss
            )
            loss.backward()
            optimizer.step()
            batch_size = x_batch.shape[0]
            total_loss += float(value_loss.item()) * batch_size
            total_policy += float(policy_loss.item()) * batch_size
            total_value_distill += float(value_distill_loss.item()) * batch_size
            total_policy_distill += float(policy_distill_loss.item()) * batch_size
            n_seen += batch_size
        scheduler.step()

        train_mse = total_loss / max(1, n_seen)
        train_policy_nll = total_policy / max(1, n_seen)
        train_value_distill_mse = total_value_distill / max(1, n_seen)
        train_policy_distill_nll = total_policy_distill / max(1, n_seen)
        val_mse, per_source = _evaluate(model, X_val, y_val, val_sources)
        val_policy_nll = _evaluate_policy_nll(
            model,
            X_val,
            dropper_val,
            checker_val,
            dropper_mask_val,
            checker_mask_val,
            policy_weights_val,
        )
        if config.selection_metric == "policy_nll":
            selection_score = val_policy_nll
        elif config.selection_metric == "value_plus_policy":
            selection_score = val_mse + config.policy_loss_weight * val_policy_nll
        else:
            selection_score = val_mse
        history.append(
            {
                "epoch": epoch,
                "train_mse": train_mse,
                "train_policy_nll": train_policy_nll,
                "train_value_distill_mse": train_value_distill_mse,
                "train_policy_distill_nll": train_policy_distill_nll,
                "val_mse": val_mse,
                "val_policy_nll": val_policy_nll,
                "selection_score": selection_score,
                "per_source_val_mse": per_source,
                "lr": optimizer.param_groups[0]["lr"],
            }
        )
        final_train_mse = train_mse

        if selection_score < best_selection_score:
            best_val = val_mse
            best_selection_score = selection_score
            best_epoch = epoch
            best_per_source = per_source
            torch.save(model.state_dict(), best_path)
            epochs_since_improvement = 0
        else:
            epochs_since_improvement += 1
            if epochs_since_improvement >= config.early_stopping_patience:
                break

    torch.save(model.state_dict(), last_path)
    with open(log_path, "w") as f:
        json.dump(
            {
                "config": asdict(config),
                "history": history,
                "best_val_mse": best_val,
                "best_selection_score": best_selection_score,
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
        best_selection_score=best_selection_score,
    )


def load_checkpoint(checkpoint_path: str | Path, device: str = "cpu") -> ValueNet:
    """Restore a ValueNet from a saved state_dict.

    Infers ``hidden_dim`` from the checkpoint's ``trunk.0.weight`` shape so
    Phase I (hidden=128) and pre-Phase-I (hidden=64) checkpoints both load
    transparently. Falls back to legacy 'layers.*' key migration for very
    early checkpoints written before the trunk/heads split.
    """
    state = torch.load(checkpoint_path, map_location=device)
    # Infer hidden_dim from the first linear layer's output dimension.
    inferred_hidden_dim = HIDDEN_DIM
    for key in ("trunk.0.weight", "layers.0.weight"):
        if key in state:
            inferred_hidden_dim = int(state[key].shape[0])
            break
    model = ValueNet(input_dim=FEATURE_DIM, hidden_dim=inferred_hidden_dim)
    try:
        model.load_state_dict(state)
    except RuntimeError:
        migrated = {}
        for key, value in state.items():
            if key.startswith("layers.0."):
                migrated[key.replace("layers.0.", "trunk.0.")] = value
            elif key.startswith("layers.2."):
                migrated[key.replace("layers.2.", "trunk.2.")] = value
            elif key.startswith("layers.4."):
                migrated[key.replace("layers.4.", "value_head.0.")] = value
        model.load_state_dict(migrated, strict=False)
    model.to(device)
    model.eval()
    return model


def make_predict_fn(model: ValueNet, device: str = "cpu"):
    """Return a callable ``(game) -> (value, dropper_dist, checker_dist)``.

    Suitable for ``ValueNetEvaluator(model_fn=predict_fn)`` and for
    ``generate_mcts_bootstrap_targets(predict_fn=...)``.
    """
    from hal.value_net import extract_features

    torch_device = torch.device(device)
    model = model.to(torch_device)
    model.eval()

    def predict(game) -> float:
        features = extract_features(game)
        x = torch.from_numpy(features).unsqueeze(0).to(torch_device)
        with torch.no_grad():
            value, dropper_logits, checker_logits = model(x)
            y = value.squeeze().cpu().item()

        from environment.cfr.evaluator import normalize_policy_vector
        from environment.cfr.exact import ExactSearchConfig, legal_seconds_for_current_role

        def masked_softmax(logits, seconds: tuple[int, ...]) -> np.ndarray:
            dist = np.zeros(61, dtype=np.float64)
            if not seconds:
                return dist
            raw = logits.squeeze(0).detach().cpu().numpy().astype(np.float64)
            legal = np.array([raw[second - 1] for second in seconds], dtype=np.float64)
            legal -= float(legal.max())
            probs = np.exp(legal)
            probs /= probs.sum()
            for second, probability in zip(seconds, probs):
                dist[second - 1] = float(probability)
            return normalize_policy_vector(dist)

        cfg = ExactSearchConfig()
        if game.game_over:
            dropper_dist = np.zeros(61, dtype=np.float64)
            checker_dist = np.zeros(61, dtype=np.float64)
        else:
            dropper, checker = game.get_roles_for_half(game.current_half)
            drop_seconds = tuple(legal_seconds_for_current_role(game, dropper.name, "dropper", cfg))
            check_seconds = tuple(legal_seconds_for_current_role(game, checker.name, "checker", cfg))
            dropper_dist = masked_softmax(dropper_logits, drop_seconds)
            checker_dist = masked_softmax(checker_logits, check_seconds)
        return float(y), dropper_dist, checker_dist

    return predict


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--targets", required=True, help="Path to .npz from value_targets")
    parser.add_argument("--out", required=True, help="Output directory for checkpoints + log")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--selection-metric",
        choices=("value_mse", "policy_nll", "value_plus_policy"),
        default="value_mse",
    )
    parser.add_argument(
        "--force-train-source",
        action="append",
        default=None,
        metavar="SOURCE",
        help="Move validation feature groups containing this source into training.",
    )
    args = parser.parse_args()

    cfg = TrainConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        seed=args.seed,
        device=args.device,
        selection_metric=args.selection_metric,
        force_train_sources=tuple(args.force_train_source or ()),
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
