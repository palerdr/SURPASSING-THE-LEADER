"""Supervised policy/value training for the exact-to-self-play bridge.

The default path consumes a validated ``TrainingRecordV3`` shard, performs a
grouped train/validation split, optimizes Hal-perspective value MSE plus masked
dropper/checker cross entropy, and selects the best epoch by grouped validation
MSE. Manifestless corpora require an explicit legacy opt-in.

``train()`` writes ``best.pt`` and ``last.pt`` as strict
``stl.checkpoint.v3`` bundles containing model, optimizer, scheduler, schemas,
provenance, resolved config, training history, and RNG state. ``log.json``
contains the per-epoch metrics, and ``TrainResult`` exposes the summary in
memory.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
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
from torch.utils.data import DataLoader, Sampler, TensorDataset

from stl.learning.contracts import canonical_config_json, config_digest
from stl.learning.model import (
    FEATURE_DIM,
    FEATURE_SCHEMA_VERSION,
    HORIZON_DIM,
    HORIZON_SCHEMA_VERSION,
    MODEL_INPUT_DIM,
    SUPPORTED_HORIZONS,
    ValueNet,
    value_output,
)
from stl.engine.actions import ACTION_SIZE


CHECKPOINT_FORMAT_VERSION = "stl.checkpoint.v3"
HORIZON_BLIND_CHECKPOINT_FORMAT_VERSION = "stl.checkpoint.v2"
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
    # Direct matrix-game objective. Cross-entropy imitates one equilibrium
    # returned by the LP; this term instead penalizes the actual best-response
    # gap of the network's two role policies against the certified payoff
    # matrix. The optional tail term targets the maximum gap in each batch.
    policy_saddle_gap_weight: float = 0.0
    policy_saddle_gap_tail_weight: float = 0.0
    checkpoint_selection_metric: str = "value_mse"
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
    # Optional exact per-batch source composition. When non-empty, every
    # optimizer batch contains exactly these counts (sampling a small source
    # cyclically as needed). This prevents a large exact source from drowning
    # out sparse terminal or tablebase anchors.
    source_batch_counts: tuple[tuple[str, int], ...] = ()
    # Net architecture knobs (Phase I). hidden_dim=64 is the original 13.7K-
    # param trunk; hidden_dim=128 gives 35.5K params (2.6× capacity) for
    # fitting more diverse tablebase pins. hidden_dim=192 (65K) exceeds the
    # 50K guard; raise the guard only with explicit justification.
    hidden_dim: int = 64
    # Optional warm start for fine-tuning from an accepted checkpoint. When
    # omitted, training starts from a seeded random initialization as before.
    init_checkpoint: str | None = None
    # Explicit one-time importer for a horizon-blind V2 bundle. It cannot be
    # combined with init_checkpoint and never makes the V2 bundle strict-load
    # as a horizon-aware model.
    horizon_blind_init_checkpoint: str | None = None
    # Which module subset may update during training. Use "value_head" for
    # conservative calibration tweaks that should leave policy priors and the
    # shared feature trunk intact.
    trainable_parts: str = "all"
    # Optional trust-region reference. When set with value_distill_weight > 0,
    # training penalizes value-output drift from this checkpoint on each batch.
    reference_checkpoint: str | None = None
    value_distill_weight: float = 0.0
    # When non-empty, trust-region loss applies only to these sources. This
    # prevents a known-bad parent prediction from fighting a tablebase repair.
    value_distill_sources: tuple[str, ...] = ()
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
    best_val_saddle_gap: float | None = None


def _seed_all(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def _split_indices(
    X: np.ndarray, val_fraction: float, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
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
    horizons: torch.Tensor,
    y: torch.Tensor,
    sources: np.ndarray,
) -> tuple[float, dict[str, float]]:
    """Run the model on (X, y) and return (overall MSE, per-source MSE)."""
    model.eval()
    with torch.no_grad():
        preds = value_output(model(X, horizons)).squeeze(-1).cpu().numpy()
    labels = y.cpu().numpy()
    overall = float(((preds - labels) ** 2).mean())
    per_source = _per_source_mse(preds, labels, sources)
    return overall, per_source


def _per_source_policy_ce(
    model: nn.Module,
    X: torch.Tensor,
    horizons: torch.Tensor,
    dropper_targets: torch.Tensor,
    checker_targets: torch.Tensor,
    dropper_masks: torch.Tensor,
    checker_masks: torch.Tensor,
    sources: np.ndarray,
) -> dict[str, dict[str, float | None]]:
    """Report unweighted role-policy cross entropy for every source."""

    model.eval()
    with torch.no_grad():
        _value, dropper_logits, checker_logits = model(X, horizons)

    def role_loss(
        logits: torch.Tensor,
        targets: torch.Tensor,
        masks: torch.Tensor,
        selected: np.ndarray,
    ) -> float | None:
        source_indices = torch.from_numpy(np.flatnonzero(selected)).to(X.device)
        source_targets = targets[source_indices]
        active = source_targets.sum(dim=1) > 1e-8
        if not bool(active.any()):
            return None
        active_indices = source_indices[active]
        target = targets[active_indices]
        target = target / target.sum(dim=1, keepdim=True).clamp_min(1e-8)
        masked_logits = logits[active_indices].masked_fill(
            ~(masks[active_indices] > 0.5), -1.0e9
        )
        return float(
            (-(target * torch.log_softmax(masked_logits, dim=1)).sum(dim=1)).mean()
        )

    return {
        str(source): {
            "dropper_ce": role_loss(
                dropper_logits,
                dropper_targets,
                dropper_masks,
                sources == source,
            ),
            "checker_ce": role_loss(
                checker_logits,
                checker_targets,
                checker_masks,
                sources == source,
            ),
        }
        for source in np.unique(sources)
    }


def _load_targets_npz(
    path: str | Path,
    *,
    for_training: bool = True,
    expected_role=None,
) -> tuple[
    np.ndarray,
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
        records = load_replay_shard(
            path,
            for_training=for_training,
            expected_role=expected_role,
        )
        n = len(records)
        X = (
            np.stack([record.features for record in records]).astype(np.float32)
            if records
            else np.zeros((0, FEATURE_DIM), dtype=np.float32)
        )
        y = np.asarray([record.value for record in records], dtype=np.float32)
        sources = np.asarray([record.source for record in records], dtype=np.str_)
        horizons = np.asarray(
            [record.value_horizon_half_rounds for record in records],
            dtype=np.int64,
        )
        unsupported = sorted(
            set(int(value) for value in horizons) - set(SUPPORTED_HORIZONS)
        )
        if unsupported:
            raise ValueError(
                f"replay shard contains unsupported horizons {unsupported}; "
                f"supported={SUPPORTED_HORIZONS}"
            )
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
            np.stack([record.dropper_legal_mask for record in records]).astype(
                np.float32
            )
            if records
            else np.zeros((0, ACTION_SIZE), dtype=np.float32)
        )
        checker_masks = (
            np.stack([record.checker_legal_mask for record in records]).astype(
                np.float32
            )
            if records
            else np.zeros((0, ACTION_SIZE), dtype=np.float32)
        )
        return (
            X,
            y,
            sources,
            horizons,
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
    horizons = (
        data["horizons"].astype(np.int64)
        if "horizons" in data
        else np.zeros(len(X), dtype=np.int64)
    )
    n = len(X)
    dropper_dists = (
        data["dropper_dists"].astype(np.float32)
        if "dropper_dists" in data
        else np.zeros((n, ACTION_SIZE), dtype=np.float32)
    )
    checker_dists = (
        data["checker_dists"].astype(np.float32)
        if "checker_dists" in data
        else np.zeros((n, ACTION_SIZE), dtype=np.float32)
    )
    dropper_masks = (
        data["dropper_legal_masks"].astype(np.float32)
        if "dropper_legal_masks" in data
        else (dropper_dists > 0.0).astype(np.float32)
    )
    checker_masks = (
        data["checker_legal_masks"].astype(np.float32)
        if "checker_legal_masks" in data
        else (checker_dists > 0.0).astype(np.float32)
    )
    data.close()
    if X.shape[1] != FEATURE_DIM:
        raise ValueError(
            f"Expected FEATURE_DIM={FEATURE_DIM} columns in X, got {X.shape[1]}"
        )
    unsupported = sorted(set(int(value) for value in horizons) - set(SUPPORTED_HORIZONS))
    if unsupported:
        raise ValueError(
            f"target corpus contains unsupported horizons {unsupported}; "
            f"supported={SUPPORTED_HORIZONS}"
        )
    for name, arr in (
        ("dropper_dists", dropper_dists),
        ("checker_dists", checker_dists),
        ("dropper_legal_masks", dropper_masks),
        ("checker_legal_masks", checker_masks),
    ):
        if arr.shape != (n, ACTION_SIZE):
            raise ValueError(
                f"Expected {name} shape {(n, ACTION_SIZE)}, got {arr.shape}"
            )
    return (
        X,
        y,
        sources,
        horizons,
        dropper_dists,
        checker_dists,
        dropper_masks,
        checker_masks,
        None,
    )


class _SourceBalancedBatchSampler(Sampler[list[int]]):
    """Deterministic batches with an exact, declared source composition."""

    def __init__(
        self,
        sources: np.ndarray,
        source_counts: tuple[tuple[str, int], ...],
        *,
        seed: int,
    ) -> None:
        counts = dict(source_counts)
        if len(counts) != len(source_counts):
            raise ValueError("source_batch_counts contains duplicate source names")
        if not counts or any(int(count) <= 0 for count in counts.values()):
            raise ValueError("source_batch_counts must contain positive counts")
        observed = {str(source) for source in np.unique(sources)}
        declared = set(counts)
        if observed != declared:
            raise ValueError(
                "source_batch_counts must name every training source exactly; "
                f"observed={sorted(observed)}, declared={sorted(declared)}"
            )
        self._indices = {
            source: np.flatnonzero(sources == source).astype(np.int64)
            for source in sorted(declared)
        }
        self._counts = tuple(
            (source, int(counts[source])) for source in sorted(declared)
        )
        self._batch_size = sum(count for _, count in self._counts)
        self._batch_count = max(1, math.ceil(len(sources) / self._batch_size))
        self._seed = int(seed)
        self._epoch = 0

    def __len__(self) -> int:
        return self._batch_count

    def __iter__(self):
        rng = np.random.default_rng(self._seed + self._epoch)
        self._epoch += 1
        orders = {
            source: rng.permutation(indices)
            for source, indices in self._indices.items()
        }
        offsets = {source: 0 for source in self._indices}

        def draw(source: str, count: int) -> list[int]:
            selected: list[int] = []
            while len(selected) < count:
                order = orders[source]
                offset = offsets[source]
                if offset == len(order):
                    order = rng.permutation(self._indices[source])
                    orders[source] = order
                    offset = 0
                take = min(count - len(selected), len(order) - offset)
                selected.extend(int(value) for value in order[offset : offset + take])
                offsets[source] = offset + take
            return selected

        for _ in range(self._batch_count):
            batch: list[int] = []
            for source, count in self._counts:
                batch.extend(draw(source, count))
            rng.shuffle(batch)
            yield batch


def _configure_trainable_parts(model: ValueNet, mode: str) -> None:
    valid = {"all", "value_head", "policy_head", "heads", "trunk_value"}
    if mode not in valid:
        raise ValueError(
            f"trainable_parts must be one of {sorted(valid)}, got {mode!r}"
        )
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
    target_active = target_active / target_active.sum(dim=1, keepdim=True).clamp_min(
        1e-8
    )
    masked_logits = logits_active.masked_fill(~mask_active, -1.0e9)
    log_probs = torch.log_softmax(masked_logits, dim=1)
    per_sample = -(target_active * log_probs).sum(dim=1)
    if sample_weight is None:
        return per_sample.mean()
    weight_active = sample_weight[active]
    return (per_sample * weight_active).sum() / weight_active.sum().clamp(min=1e-8)


@dataclass(frozen=True)
class _CertificatePolicyData:
    """Full-action payoff matrices aligned one-to-one with replay rows."""

    payoffs_for_hal: np.ndarray
    active: np.ndarray
    hal_is_dropper: np.ndarray


def _slice_certificate_policy_data(
    data: _CertificatePolicyData, indices: np.ndarray
) -> _CertificatePolicyData:
    return _CertificatePolicyData(
        payoffs_for_hal=data.payoffs_for_hal[indices],
        active=data.active[indices],
        hal_is_dropper=data.hal_is_dropper[indices],
    )


def _masked_policy_probabilities(
    logits: torch.Tensor, legal_mask: torch.Tensor
) -> torch.Tensor:
    mask = legal_mask > 0.5
    if not bool(mask.any(dim=1).all()):
        raise ValueError("every policy row must contain at least one legal action")
    return torch.softmax(logits.masked_fill(~mask, -1.0e9), dim=1)


def _certificate_saddle_gaps(
    dropper_logits: torch.Tensor,
    checker_logits: torch.Tensor,
    dropper_legal_mask: torch.Tensor,
    checker_legal_mask: torch.Tensor,
    payoffs_for_hal: torch.Tensor,
    certificate_active: torch.Tensor,
    hal_is_dropper: torch.Tensor,
) -> torch.Tensor:
    """Return the exact one-step best-response gap for every active row.

    For Hal-as-dropper matrices, rows maximize and columns minimize, giving

        max_a (A y)_a - min_b (x^T A)_b.

    When Baku drops, rows minimize and columns maximize, so the signs reverse.
    This is the same gap used by ``diagnose_exact_strategy`` and the immutable
    development/ruler gate. Inactive rows return zero and do not contribute to
    the certificate-backed loss.
    """

    if payoffs_for_hal.ndim != 3 or payoffs_for_hal.shape[1:] != (
        ACTION_SIZE,
        ACTION_SIZE,
    ):
        raise ValueError("certificate payoff tensors must be [batch, action, action]")
    batch = dropper_logits.shape[0]
    if payoffs_for_hal.shape[0] != batch:
        raise ValueError("certificate payoff batch does not match policy logits")

    active = certificate_active.bool()
    if not bool(active.any()):
        return dropper_logits.new_zeros(batch)
    drop_mask = dropper_legal_mask[active] > 0.5
    check_mask = checker_legal_mask[active] > 0.5
    dropper = _masked_policy_probabilities(dropper_logits[active], drop_mask)
    checker = _masked_policy_probabilities(checker_logits[active], check_mask)
    active_payoffs = payoffs_for_hal[active]
    drop_action_values = torch.bmm(
        active_payoffs, checker.unsqueeze(2)
    ).squeeze(2)
    check_action_values = torch.bmm(
        dropper.unsqueeze(1), active_payoffs
    ).squeeze(1)

    positive_inf = torch.tensor(
        torch.inf, dtype=payoffs_for_hal.dtype, device=payoffs_for_hal.device
    )
    negative_inf = -positive_inf
    best_drop_max = drop_action_values.masked_fill(~drop_mask, negative_inf).max(1).values
    best_drop_min = drop_action_values.masked_fill(~drop_mask, positive_inf).min(1).values
    best_check_max = check_action_values.masked_fill(~check_mask, negative_inf).max(1).values
    best_check_min = check_action_values.masked_fill(~check_mask, positive_inf).min(1).values

    gap_when_hal_drops = best_drop_max - best_check_min
    gap_when_baku_drops = best_check_max - best_drop_min
    active_gaps = torch.where(
        hal_is_dropper[active].bool(), gap_when_hal_drops, gap_when_baku_drops
    ).clamp_min(0.0)
    gaps = dropper_logits.new_zeros(batch)
    active_indices = torch.nonzero(active, as_tuple=False).flatten()
    return gaps.index_copy(0, active_indices, active_gaps)


def _load_certificate_policy_data(
    records: Sequence[object],
    certificate_path: str | Path,
    *,
    expected_role,
) -> _CertificatePolicyData:
    """Load and integrity-bind exact matrices to their replay rows."""

    from stl.learning.certificates import certificates_by_state, load_certificate_shard
    from stl.learning.replay import exact_state_hash, reconstruct_game

    certificates = certificates_by_state(
        load_certificate_shard(certificate_path, expected_role=expected_role)
    )
    active_hashes = {
        exact_state_hash(record.exact_state)
        for record in records
        if float(record.dropper_dist.sum(dtype=np.float64)) > 0.0
        or float(record.checker_dist.sum(dtype=np.float64)) > 0.0
    }
    if set(certificates) != active_hashes:
        raise ValueError(
            "policy certificates must exactly cover active replay policy rows"
        )

    count = len(records)
    payoffs = np.zeros((count, ACTION_SIZE, ACTION_SIZE), dtype=np.float32)
    active = np.zeros(count, dtype=np.bool_)
    hal_is_dropper = np.zeros(count, dtype=np.bool_)
    for index, record in enumerate(records):
        state_hash = exact_state_hash(record.exact_state)
        certificate = certificates.get(state_hash)
        if certificate is None:
            continue
        if certificate.search_config_digest != record.search_config_digest:
            raise ValueError("policy certificate search config digest mismatch")
        if certificate.horizon != record.value_horizon_half_rounds:
            raise ValueError("policy certificate horizon mismatch")

        drop_actions = tuple(
            int(action) for action in np.flatnonzero(record.dropper_legal_mask)
        )
        check_actions = tuple(
            int(action) for action in np.flatnonzero(record.checker_legal_mask)
        )
        if certificate.drop_actions != drop_actions:
            raise ValueError("policy certificate drop actions do not match replay legality")
        if certificate.check_actions != check_actions:
            raise ValueError("policy certificate check actions do not match replay legality")

        certified_dropper = np.zeros(ACTION_SIZE, dtype=np.float64)
        certified_checker = np.zeros(ACTION_SIZE, dtype=np.float64)
        certified_dropper[list(certificate.drop_actions)] = certificate.dropper_strategy
        certified_checker[list(certificate.check_actions)] = certificate.checker_strategy
        if not np.allclose(
            certified_dropper, record.dropper_dist, rtol=0.0, atol=1e-6
        ):
            raise ValueError("policy certificate dropper strategy mismatch")
        if not np.allclose(
            certified_checker, record.checker_dist, rtol=0.0, atol=1e-6
        ):
            raise ValueError("policy certificate checker strategy mismatch")

        payoffs[index][np.ix_(certificate.drop_actions, certificate.check_actions)] = (
            certificate.payoff_for_hal.astype(np.float32, copy=False)
        )
        game = reconstruct_game(record.exact_state)
        dropper, _checker = game.get_roles_for_half(game.current_half)
        hal_is_dropper[index] = dropper.name.lower() == "hal"
        active[index] = True
    return _CertificatePolicyData(
        payoffs_for_hal=payoffs,
        active=active,
        hal_is_dropper=hal_is_dropper,
    )


def _evaluate_certificate_saddle_gap(
    model: ValueNet,
    features: torch.Tensor,
    horizons: torch.Tensor,
    dropper_masks: torch.Tensor,
    checker_masks: torch.Tensor,
    certificate_data: _CertificatePolicyData,
    *,
    device: torch.device,
) -> tuple[float, float]:
    active = torch.from_numpy(certificate_data.active).to(device)
    if not bool(active.any()):
        return 0.0, 0.0
    payoffs = torch.from_numpy(certificate_data.payoffs_for_hal).to(device)
    orientations = torch.from_numpy(certificate_data.hal_is_dropper).to(device)
    model.eval()
    with torch.no_grad():
        _values, dropper_logits, checker_logits = model(features, horizons)
        gaps = _certificate_saddle_gaps(
            dropper_logits,
            checker_logits,
            dropper_masks,
            checker_masks,
            payoffs,
            active,
            orientations,
        )[active]
    return float(gaps.mean().item()), float(gaps.max().item())


def train(
    targets_npz_path: str | Path,
    output_dir: str | Path,
    config: TrainConfig | None = None,
    *,
    validation_npz_path: str | Path | None = None,
    policy_certificate_path: str | Path | None = None,
    validation_policy_certificate_path: str | Path | None = None,
    training_taxonomy_path: str | Path | None = None,
) -> TrainResult:
    """Train ValueNet against labeled targets and write checkpoints.

    Args:
        targets_npz_path: Path to ``.npz`` produced by
            ``training.value_targets.save_targets``.
        output_dir: Directory for checkpoints and the training log.
        config: Hyperparameters; defaults to ``TrainConfig()``.
        validation_npz_path: Optional immutable development shard used in full
            for checkpoint selection. It must have role ``development`` and is
            never passed to the optimizer. When omitted, the legacy grouped
            split within the replay shard is retained.
        policy_certificate_path: Exact payoff matrices for active training
            policy rows. Required when direct saddle-gap loss is enabled.
        validation_policy_certificate_path: Development payoff matrices used
            only to measure/select checkpoints; never passed to the optimizer.

    Returns:
        ``TrainResult`` with best/final MSE values and the path of the
        best-epoch checkpoint.
    """
    config = config or TrainConfig()
    if (
        config.init_checkpoint is not None
        and config.horizon_blind_init_checkpoint is not None
    ):
        raise ValueError(
            "init_checkpoint and horizon_blind_init_checkpoint are mutually exclusive"
        )
    if config.policy_saddle_gap_weight < 0.0:
        raise ValueError("policy_saddle_gap_weight must be nonnegative")
    if config.policy_saddle_gap_tail_weight < 0.0:
        raise ValueError("policy_saddle_gap_tail_weight must be nonnegative")
    if config.checkpoint_selection_metric not in {
        "value_mse",
        "saddle_gap_then_value",
    }:
        raise ValueError(
            "checkpoint_selection_metric must be 'value_mse' or "
            "'saddle_gap_then_value'"
        )
    if config.policy_saddle_gap_weight > 0.0 and policy_certificate_path is None:
        raise ValueError(
            "direct saddle-gap training requires policy_certificate_path"
        )
    if (
        config.checkpoint_selection_metric == "saddle_gap_then_value"
        and validation_policy_certificate_path is None
    ):
        raise ValueError(
            "saddle-gap checkpoint selection requires development certificates"
        )
    _seed_all(config.seed)
    rng = np.random.default_rng(config.seed)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    (
        X_np,
        y_np,
        sources_np,
        horizons_np,
        dropper_dists_np,
        checker_dists_np,
        dropper_masks_np,
        checker_masks_np,
        replay_records,
    ) = _load_targets_npz(targets_npz_path, for_training=True)
    if validation_npz_path is not None:
        from stl.learning.replay import ShardRole, exact_state_hash

        (
            X_val_np,
            y_val_np,
            val_sources,
            val_horizons_np,
            val_dropper_np,
            val_checker_np,
            val_dropper_masks_np,
            val_checker_masks_np,
            validation_records,
        ) = _load_targets_npz(
            validation_npz_path,
            for_training=False,
            expected_role=ShardRole.DEVELOPMENT,
        )
        if replay_records is None or validation_records is None:
            raise ValueError(
                "external development validation requires V3 manifested shards"
            )
        train_state_hashes = {
            exact_state_hash(record.exact_state) for record in replay_records
        }
        validation_state_hashes = {
            exact_state_hash(record.exact_state) for record in validation_records
        }
        if train_state_hashes & validation_state_hashes:
            raise ValueError("training and development exact states overlap")
        train_episodes = {record.episode_id for record in replay_records}
        validation_episodes = {record.episode_id for record in validation_records}
        if train_episodes & validation_episodes:
            raise ValueError("training and development episodes overlap")
        train_features = {record.features.tobytes() for record in replay_records}
        validation_features = {
            record.features.tobytes() for record in validation_records
        }
        if train_features & validation_features:
            raise ValueError("training and development feature vectors overlap")
        train_idx = np.arange(len(X_np), dtype=np.int64)
        val_idx = None
    elif replay_records is None:
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
    horizons_train = torch.from_numpy(horizons_np[train_idx]).to(device)
    y_train = torch.from_numpy(y_np[train_idx]).to(device)
    dropper_train = torch.from_numpy(dropper_dists_np[train_idx]).to(device)
    checker_train = torch.from_numpy(checker_dists_np[train_idx]).to(device)
    dropper_mask_train = torch.from_numpy(dropper_masks_np[train_idx]).to(device)
    checker_mask_train = torch.from_numpy(checker_masks_np[train_idx]).to(device)
    if val_idx is None:
        X_val = torch.from_numpy(X_val_np).to(device)
        horizons_val = torch.from_numpy(val_horizons_np).to(device)
        y_val = torch.from_numpy(y_val_np).to(device)
    else:
        X_val = torch.from_numpy(X_np[val_idx]).to(device)
        horizons_val = torch.from_numpy(horizons_np[val_idx]).to(device)
        y_val = torch.from_numpy(y_np[val_idx]).to(device)
        val_sources = sources_np[val_idx]
        val_dropper_np = dropper_dists_np[val_idx]
        val_checker_np = checker_dists_np[val_idx]
        val_dropper_masks_np = dropper_masks_np[val_idx]
        val_checker_masks_np = checker_masks_np[val_idx]
    dropper_val = torch.from_numpy(val_dropper_np).to(device)
    checker_val = torch.from_numpy(val_checker_np).to(device)
    dropper_mask_val = torch.from_numpy(val_dropper_masks_np).to(device)
    checker_mask_val = torch.from_numpy(val_checker_masks_np).to(device)

    train_certificate_data = None
    validation_certificate_data = None
    if policy_certificate_path is not None:
        if replay_records is None:
            raise ValueError("policy certificates require a manifested V3 replay shard")
        from stl.learning.replay import ShardRole

        all_training_certificates = _load_certificate_policy_data(
            replay_records,
            policy_certificate_path,
            expected_role=ShardRole.REPLAY,
        )
        train_certificate_data = _slice_certificate_policy_data(
            all_training_certificates, train_idx
        )
        if val_idx is not None:
            validation_certificate_data = _slice_certificate_policy_data(
                all_training_certificates, val_idx
            )
    if validation_policy_certificate_path is not None:
        if validation_npz_path is None or validation_records is None:
            raise ValueError(
                "development certificates require a dedicated development shard"
            )
        from stl.learning.replay import ShardRole

        validation_certificate_data = _load_certificate_policy_data(
            validation_records,
            validation_policy_certificate_path,
            expected_role=ShardRole.DEVELOPMENT,
        )
    if (
        config.checkpoint_selection_metric == "saddle_gap_then_value"
        and validation_certificate_data is None
    ):
        raise ValueError("no certificate data is available for checkpoint selection")

    weight_lookup = dict(config.source_weights)
    policy_weight_lookup = dict(config.policy_source_weights)
    train_sources = sources_np[train_idx]
    sampler_sources = train_sources.copy()
    if training_taxonomy_path is not None:
        if replay_records is None:
            raise ValueError("training taxonomy requires a manifested replay shard")
        from stl.learning.holdout import load_taxonomy
        from stl.learning.replay import exact_state_hash

        taxonomy = load_taxonomy(training_taxonomy_path, replay_path=targets_npz_path)
        indexed_records = [replay_records[int(index)] for index in train_idx]
        sampler_sources = np.asarray(train_sources, dtype=object)
        for index, record in enumerate(indexed_records):
            entry = taxonomy.get(exact_state_hash(record.exact_state))
            if entry is None:
                continue
            family = str(entry.get("family", ""))
            if family == "boundary_double_overflow":
                sampler_sources[index] = "tablebase_double_overflow"
            elif record.source == "tablebase":
                sampler_sources[index] = "tablebase_single_overflow"
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
    distill_sources = set(config.value_distill_sources)
    train_distill_weights_np = np.asarray(
        [
            1.0
            if not distill_sources or str(source) in distill_sources
            else 0.0
            for source in train_sources
        ],
        dtype=np.float32,
    )
    distill_weights_train = torch.from_numpy(train_distill_weights_np).to(device)

    training_tensors: list[torch.Tensor] = [
        X_train,
        horizons_train,
        y_train,
        dropper_train,
        checker_train,
        dropper_mask_train,
        checker_mask_train,
        weights_train,
        policy_weights_train,
        distill_weights_train,
    ]
    if train_certificate_data is not None:
        training_tensors.extend(
            (
                torch.from_numpy(train_certificate_data.payoffs_for_hal).to(device),
                torch.from_numpy(train_certificate_data.active).to(device),
                torch.from_numpy(train_certificate_data.hal_is_dropper).to(device),
            )
        )
    train_ds = TensorDataset(*training_tensors)
    if config.source_batch_counts:
        declared_batch_size = sum(count for _, count in config.source_batch_counts)
        if declared_batch_size != config.batch_size:
            raise ValueError(
                "source_batch_counts sum must equal batch_size: "
                f"{declared_batch_size} != {config.batch_size}"
            )
        train_loader = DataLoader(
            train_ds,
            batch_sampler=_SourceBalancedBatchSampler(
                sampler_sources,
                config.source_batch_counts,
                seed=config.seed,
            ),
        )
    else:
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
    elif config.horizon_blind_init_checkpoint is not None:
        model = initialize_from_horizon_blind_checkpoint(
            config.horizon_blind_init_checkpoint,
            device=config.device,
        ).to(device)
        parent_digest = checkpoint_digest(config.horizon_blind_init_checkpoint)
        loaded_hidden = int(model.trunk[0].out_features)
        if loaded_hidden != config.hidden_dim:
            raise ValueError(
                f"horizon-blind init hidden_dim={loaded_hidden} does not match "
                f"TrainConfig.hidden_dim={config.hidden_dim}"
            )
    else:
        model = ValueNet(input_dim=FEATURE_DIM, hidden_dim=config.hidden_dim).to(device)
    _configure_trainable_parts(model, config.trainable_parts)
    trainable_parameters = [p for p in model.parameters() if p.requires_grad]
    if not trainable_parameters:
        raise ValueError(
            f"trainable_parts={config.trainable_parts!r} left no trainable parameters"
        )
    reference_model = None
    if config.reference_checkpoint is not None and config.value_distill_weight > 0.0:
        reference_model = load_checkpoint(
            config.reference_checkpoint, device=config.device
        ).to(device)
        reference_model.eval()
    optimizer = torch.optim.Adam(
        trainable_parameters, lr=config.learning_rate, weight_decay=config.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, config.epochs)
    )
    loss_fn = nn.MSELoss(reduction="none")

    best_val = math.inf
    best_val_saddle_gap: float | None = None
    best_selection_key: tuple[float, ...] | None = None
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
        for batch in train_loader:
            (
                x_batch,
                horizon_batch,
                y_batch,
                dropper_batch,
                checker_batch,
                dropper_mask_batch,
                checker_mask_batch,
                weight_batch,
                policy_weight_batch,
                distill_weight_batch,
            ) = batch[:10]
            optimizer.zero_grad()
            preds, dropper_logits, checker_logits = model(x_batch, horizon_batch)
            per_sample_se = loss_fn(preds.squeeze(-1), y_batch)
            value_loss = (
                per_sample_se * weight_batch
            ).sum() / weight_batch.sum().clamp(min=1e-8)
            if reference_model is not None:
                with torch.no_grad():
                    ref_preds = value_output(
                        reference_model(x_batch, horizon_batch)
                    ).squeeze(-1)
                per_sample_distill = loss_fn(preds.squeeze(-1), ref_preds)
                value_distill_loss = (
                    per_sample_distill * distill_weight_batch
                ).sum() / distill_weight_batch.sum().clamp(min=1e-8)
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
            saddle_gap_loss = preds.sum() * 0.0
            if len(batch) == 13:
                payoff_batch, certificate_active_batch, orientation_batch = batch[10:]
                gaps = _certificate_saddle_gaps(
                    dropper_logits,
                    checker_logits,
                    dropper_mask_batch,
                    checker_mask_batch,
                    payoff_batch,
                    certificate_active_batch,
                    orientation_batch,
                )
                active = certificate_active_batch.bool()
                if bool(active.any()):
                    active_weights = policy_weight_batch[active]
                    gap_mean = (
                        gaps[active] * active_weights
                    ).sum() / active_weights.sum().clamp(min=1e-8)
                    gap_tail = gaps[active].max()
                    saddle_gap_loss = (
                        gap_mean
                        + config.policy_saddle_gap_tail_weight * gap_tail
                    )
            loss = (
                value_loss
                + config.policy_loss_weight * policy_loss
                + config.policy_saddle_gap_weight * saddle_gap_loss
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
        val_mse, per_source = _evaluate(
            model, X_val, horizons_val, y_val, val_sources
        )
        if train_certificate_data is None:
            train_gap_mean = None
            train_gap_max = None
        else:
            train_gap_mean, train_gap_max = _evaluate_certificate_saddle_gap(
                model,
                X_train,
                horizons_train,
                dropper_mask_train,
                checker_mask_train,
                train_certificate_data,
                device=device,
            )
        if validation_certificate_data is None:
            val_gap_mean = None
            val_gap_max = None
        else:
            val_gap_mean, val_gap_max = _evaluate_certificate_saddle_gap(
                model,
                X_val,
                horizons_val,
                dropper_mask_val,
                checker_mask_val,
                validation_certificate_data,
                device=device,
            )
        _train_overall_unweighted, per_source_train_mse = _evaluate(
            model, X_train, horizons_train, y_train, train_sources
        )
        per_source_train_policy = _per_source_policy_ce(
            model,
            X_train,
            horizons_train,
            dropper_train,
            checker_train,
            dropper_mask_train,
            checker_mask_train,
            train_sources,
        )
        per_source_val_policy = _per_source_policy_ce(
            model,
            X_val,
            horizons_val,
            dropper_val,
            checker_val,
            dropper_mask_val,
            checker_mask_val,
            val_sources,
        )
        history.append(
            {
                "epoch": epoch,
                "train_mse": train_mse,
                "train_policy_nll": train_policy_nll,
                "train_saddle_gap_mean": train_gap_mean,
                "train_saddle_gap_max": train_gap_max,
                "train_value_distill_mse": train_value_distill_mse,
                "val_mse": val_mse,
                "val_saddle_gap_mean": val_gap_mean,
                "val_saddle_gap_max": val_gap_max,
                "per_source_val_mse": per_source,
                "per_source_train_mse": per_source_train_mse,
                "per_source_train_policy_ce": per_source_train_policy,
                "per_source_val_policy_ce": per_source_val_policy,
                "lr": optimizer.param_groups[0]["lr"],
            }
        )
        final_train_mse = train_mse

        selection_key = (
            (val_mse,)
            if config.checkpoint_selection_metric == "value_mse"
            else (float(val_gap_max), val_mse)
        )
        if best_selection_key is None or selection_key < best_selection_key:
            best_selection_key = selection_key
            best_val = val_mse
            best_val_saddle_gap = val_gap_max
            best_epoch = epoch
            best_per_source = per_source
            save_checkpoint_bundle(
                best_path,
                model,
                optimizer=optimizer,
                scheduler=scheduler,
                train_config=config,
                parent_digest=parent_digest,
                corpus_digests=tuple(
                    checkpoint_digest(path)
                    for path in (
                        targets_npz_path,
                        validation_npz_path,
                        policy_certificate_path,
                        validation_policy_certificate_path,
                        training_taxonomy_path,
                    )
                    if path is not None
                ),
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
        corpus_digests=tuple(
            checkpoint_digest(path)
            for path in (
                targets_npz_path,
                validation_npz_path,
                policy_certificate_path,
                validation_policy_certificate_path,
                training_taxonomy_path,
            )
            if path is not None
        ),
        history=history,
        epoch=len(history) - 1,
    )
    with open(log_path, "w") as f:
        json.dump(
            {
                "config": asdict(config),
                "history": history,
                "best_val_mse": best_val,
                "best_val_saddle_gap": best_val_saddle_gap,
                "checkpoint_selection_metric": config.checkpoint_selection_metric,
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
        best_val_saddle_gap=best_val_saddle_gap,
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
    """Atomically write a self-describing horizon-aware V3 bundle."""

    hidden_dim = int(model.trunk[0].out_features)
    resolved_config = asdict(train_config) if train_config is not None else {}
    payload = {
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict()
        if optimizer is not None
        else None,
        "scheduler_state_dict": scheduler.state_dict()
        if scheduler is not None
        else None,
        "model_schema": {
            "feature_schema": FEATURE_SCHEMA_VERSION,
            "feature_dim": FEATURE_DIM,
            "horizon_schema": HORIZON_SCHEMA_VERSION,
            "supported_horizons": list(SUPPORTED_HORIZONS),
            "horizon_dim": HORIZON_DIM,
            "model_input_dim": MODEL_INPUT_DIM,
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
    if (
        not isinstance(payload, dict)
        or payload.get("format_version") != CHECKPOINT_FORMAT_VERSION
    ):
        raise CheckpointFormatError(
            "bare or legacy checkpoint rejected; use load_legacy_checkpoint() "
            "only for an explicit migration"
        )
    required = {"model_state_dict", "model_schema", "provenance", "training"}
    missing = required - set(payload)
    if missing:
        raise CheckpointFormatError(
            f"checkpoint bundle missing fields: {sorted(missing)}"
        )

    schema = payload["model_schema"]
    expected_schema = {
        "feature_schema": FEATURE_SCHEMA_VERSION,
        "feature_dim": FEATURE_DIM,
        "horizon_schema": HORIZON_SCHEMA_VERSION,
        "supported_horizons": list(SUPPORTED_HORIZONS),
        "horizon_dim": HORIZON_DIM,
        "model_input_dim": MODEL_INPUT_DIM,
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
    if tuple(first_weight.shape) != (hidden_dim, MODEL_INPUT_DIM):
        raise CheckpointFormatError(
            "checkpoint input layer does not match feature schema"
        )
    if policy_weight is None or tuple(policy_weight.shape) != (
        2 * ACTION_SIZE,
        hidden_dim,
    ):
        raise CheckpointFormatError(
            "checkpoint policy head does not match action schema"
        )
    return payload


def load_checkpoint(checkpoint_path: str | Path, device: str = "cpu") -> ValueNet:
    """Restore a strictly validated horizon-aware ValueNet bundle."""

    payload = load_checkpoint_bundle(checkpoint_path, device=device)
    hidden_dim = int(payload["model_schema"]["hidden_dim"])
    model = ValueNet(input_dim=FEATURE_DIM, hidden_dim=hidden_dim)
    try:
        model.load_state_dict(payload["model_state_dict"], strict=True)
    except RuntimeError as exc:
        raise CheckpointFormatError(
            f"checkpoint state_dict is incompatible: {exc}"
        ) from exc
    model.to(device)
    model.eval()
    return model


def load_legacy_checkpoint(
    checkpoint_path: str | Path, device: str = "cpu"
) -> ValueNet:
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
    raise CheckpointFormatError(
        "horizon-blind state dicts cannot be used for inference; use "
        "initialize_from_horizon_blind_checkpoint() for explicit warm start"
    )


def initialize_from_horizon_blind_checkpoint(
    checkpoint_path: str | Path,
    device: str = "cpu",
) -> ValueNet:
    """Import V2 bundle weights without granting them V3 inference status."""

    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if not isinstance(payload, dict):
        raise CheckpointFormatError("unrecognized horizon-blind checkpoint")
    if payload.get("format_version") == HORIZON_BLIND_CHECKPOINT_FORMAT_VERSION:
        state = payload.get("model_state_dict")
    elif "trunk.0.weight" in payload:
        state = payload
    else:
        raise CheckpointFormatError("unrecognized horizon-blind checkpoint")
    if not isinstance(state, dict) or "trunk.0.weight" not in state:
        raise CheckpointFormatError("horizon-blind checkpoint has no trunk")
    old_weight = state["trunk.0.weight"]
    if tuple(old_weight.shape)[1:] != (FEATURE_DIM,):
        raise CheckpointFormatError(
            "horizon-blind checkpoint does not use the active physical features"
        )
    hidden_dim = int(old_weight.shape[0])
    model = ValueNet(hidden_dim=hidden_dim)
    imported = model.state_dict()
    for key, tensor in state.items():
        if key == "trunk.0.weight":
            imported[key].zero_()
            imported[key][:, :FEATURE_DIM].copy_(tensor)
        elif key in imported and tuple(imported[key].shape) == tuple(tensor.shape):
            imported[key].copy_(tensor)
        else:
            raise CheckpointFormatError(
                f"horizon-blind parameter {key!r} is incompatible"
            )
    model.load_state_dict(imported, strict=True)
    model.to(device)
    model.eval()
    return model


def make_predict_fn(model: ValueNet, device: str = "cpu"):
    """Return ``(game, horizon=...) -> (value, role policies)``.

    Suitable for ``ValueNetEvaluator(model_fn=predict_fn)`` and for
    ``generate_mcts_bootstrap_targets(predict_fn=...)``.
    """
    from stl.learning.model import extract_features

    torch_device = torch.device(device)
    model = model.to(torch_device)
    model.eval()

    def predict(game, *, horizon: int) -> float:
        features = extract_features(game)
        x = torch.from_numpy(features).unsqueeze(0).to(torch_device)
        with torch.no_grad():
            value, dropper_logits, checker_logits = model(x, horizon)
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
            drop_seconds = tuple(
                legal_seconds_for_current_role(game, dropper.name, "dropper", cfg)
            )
            check_seconds = tuple(
                legal_seconds_for_current_role(game, checker.name, "checker", cfg)
            )
            dropper_dist = masked_softmax(dropper_logits, drop_seconds)
            checker_dist = masked_softmax(checker_logits, check_seconds)
        return float(y), dropper_dist, checker_dist

    return predict


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser used directly and by the Hydra dispatcher."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--targets", required=True, help="Path to .npz from value_targets"
    )
    parser.add_argument(
        "--validation-targets",
        default=None,
        help="Dedicated V3 development shard used only for checkpoint selection.",
    )
    parser.add_argument(
        "--policy-certificates",
        default=None,
        help="Exact training payoff matrices used by direct saddle-gap loss.",
    )
    parser.add_argument(
        "--validation-policy-certificates",
        default=None,
        help="Exact development matrices used only for checkpoint selection.",
    )
    parser.add_argument("--training-taxonomy", default=None)
    parser.add_argument(
        "--out", required=True, help="Output directory for checkpoints + log"
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--policy-loss-weight", type=float, default=1.0)
    parser.add_argument("--policy-saddle-gap-weight", type=float, default=0.0)
    parser.add_argument("--policy-saddle-gap-tail-weight", type=float, default=0.0)
    parser.add_argument(
        "--checkpoint-selection-metric",
        choices=("value_mse", "saddle_gap_then_value"),
        default="value_mse",
    )
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--init-checkpoint", default=None)
    parser.add_argument("--horizon-blind-init-checkpoint", default=None)
    parser.add_argument("--reference-checkpoint", default=None)
    parser.add_argument("--value-distill-weight", type=float, default=0.0)
    parser.add_argument(
        "--value-distill-source",
        action="append",
        default=None,
    )
    parser.add_argument(
        "--trainable-parts",
        choices=("all", "value_head", "policy_head", "heads", "trunk_value"),
        default="all",
    )
    parser.add_argument("--required-parent-digest", default=None)
    parser.add_argument("--early-stopping-patience", type=int, default=25)
    parser.add_argument(
        "--source-weight",
        action="append",
        default=None,
        metavar="SOURCE:WEIGHT",
    )
    parser.add_argument(
        "--policy-source-weight",
        action="append",
        default=None,
        metavar="SOURCE:WEIGHT",
    )
    parser.add_argument(
        "--source-batch-count",
        action="append",
        default=None,
        metavar="SOURCE:COUNT",
    )
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

    def parse_pairs(items, *, value_type):
        if items is None:
            return None
        parsed = []
        for item in items:
            try:
                source, raw_value = item.rsplit(":", 1)
                value = value_type(raw_value)
            except (ValueError, TypeError) as exc:
                raise ValueError(f"invalid source setting {item!r}") from exc
            if not source:
                raise ValueError("source setting requires a non-empty source name")
            parsed.append((source, value))
        return tuple(parsed)

    source_weights = parse_pairs(args.source_weight, value_type=float)
    policy_source_weights = parse_pairs(args.policy_source_weight, value_type=float)
    source_batch_counts = parse_pairs(args.source_batch_count, value_type=int)
    defaults = TrainConfig()
    cfg = TrainConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        policy_loss_weight=args.policy_loss_weight,
        policy_saddle_gap_weight=args.policy_saddle_gap_weight,
        policy_saddle_gap_tail_weight=args.policy_saddle_gap_tail_weight,
        checkpoint_selection_metric=args.checkpoint_selection_metric,
        seed=args.seed,
        device=args.device,
        hidden_dim=args.hidden_dim,
        init_checkpoint=args.init_checkpoint,
        horizon_blind_init_checkpoint=args.horizon_blind_init_checkpoint,
        trainable_parts=args.trainable_parts,
        reference_checkpoint=args.reference_checkpoint,
        value_distill_weight=args.value_distill_weight,
        value_distill_sources=tuple(args.value_distill_source or ()),
        required_parent_digest=args.required_parent_digest,
        early_stopping_patience=args.early_stopping_patience,
        source_weights=(
            defaults.source_weights if source_weights is None else source_weights
        ),
        policy_source_weights=(
            defaults.policy_source_weights
            if policy_source_weights is None
            else policy_source_weights
        ),
        source_batch_counts=(
            defaults.source_batch_counts
            if source_batch_counts is None
            else source_batch_counts
        ),
        allow_legacy_targets=args.allow_legacy_targets,
    )
    optional_paths = {
        "validation_npz_path": args.validation_targets,
        "policy_certificate_path": args.policy_certificates,
        "validation_policy_certificate_path": args.validation_policy_certificates,
        "training_taxonomy_path": args.training_taxonomy,
    }
    supplied_paths = {
        name: value for name, value in optional_paths.items() if value is not None
    }
    result = train(args.targets, args.out, cfg, **supplied_paths)
    print(
        f"Best val MSE {result.best_val_mse:.5f} at epoch {result.best_epoch}; "
        f"checkpoint={result.checkpoint_path}"
    )
    if result.best_per_source_mse:
        print("Per-source val MSE:")
        for source, mse in sorted(result.best_per_source_mse.items()):
            print(f"  {source}: {mse:.5f}")
    best_gap = getattr(result, "best_val_saddle_gap", None)
    if best_gap is not None:
        print(f"Best development saddle gap: {best_gap:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
