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
from dataclasses import asdict, dataclass, field
from pathlib import Path

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
    # Net architecture knobs (Phase I). hidden_dim=64 is the original 13.7K-
    # param trunk; hidden_dim=128 gives 35.5K params (2.6× capacity) for
    # fitting more diverse tablebase pins. hidden_dim=192 (65K) exceeds the
    # 50K guard; raise the guard only with explicit justification.
    hidden_dim: int = 64


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


def _split_indices(n: int, val_fraction: float, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Deterministic train/val split via permutation under a seeded rng."""
    indices = rng.permutation(n)
    n_val = max(1, int(round(n * val_fraction)))
    val_idx = indices[:n_val]
    train_idx = indices[n_val:]
    return train_idx, val_idx


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


def _masked_policy_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    legal_mask: torch.Tensor,
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
    return -(target_active * log_probs).sum(dim=1).mean()


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
    ) = _load_targets_npz(targets_npz_path)
    train_idx, val_idx = _split_indices(len(X_np), config.val_fraction, rng)

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
    train_sources = sources_np[train_idx]
    train_weights_np = np.array(
        [weight_lookup.get(str(s), 1.0) for s in train_sources],
        dtype=np.float32,
    )
    weights_train = torch.from_numpy(train_weights_np).to(device)

    train_ds = TensorDataset(
        X_train,
        y_train,
        dropper_train,
        checker_train,
        dropper_mask_train,
        checker_mask_train,
        weights_train,
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=config.batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(config.seed),
    )

    model = ValueNet(input_dim=FEATURE_DIM, hidden_dim=config.hidden_dim).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
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
        for (
            x_batch,
            y_batch,
            dropper_batch,
            checker_batch,
            dropper_mask_batch,
            checker_mask_batch,
            weight_batch,
        ) in train_loader:
            optimizer.zero_grad()
            preds, dropper_logits, checker_logits = model(x_batch)
            per_sample_se = loss_fn(preds.squeeze(-1), y_batch)
            value_loss = (per_sample_se * weight_batch).sum() / weight_batch.sum().clamp(min=1e-8)
            dropper_loss = _masked_policy_loss(dropper_logits, dropper_batch, dropper_mask_batch)
            checker_loss = _masked_policy_loss(checker_logits, checker_batch, checker_mask_batch)
            policy_loss = dropper_loss + checker_loss
            loss = value_loss + config.policy_loss_weight * policy_loss
            loss.backward()
            optimizer.step()
            batch_size = x_batch.shape[0]
            total_loss += float(value_loss.item()) * batch_size
            total_policy += float(policy_loss.item()) * batch_size
            n_seen += batch_size
        scheduler.step()

        train_mse = total_loss / max(1, n_seen)
        train_policy_nll = total_policy / max(1, n_seen)
        val_mse, per_source = _evaluate(model, X_val, y_val, val_sources)
        history.append(
            {
                "epoch": epoch,
                "train_mse": train_mse,
                "train_policy_nll": train_policy_nll,
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
    args = parser.parse_args()

    cfg = TrainConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        seed=args.seed,
        device=args.device,
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
