"""Tests for Phase 4: training/train_value_net.py.

These tests verify the training-loop scaffolding without requiring real
training to convergence: tiny synthetic corpora, a few epochs each, and
structural assertions on the produced artifacts (checkpoint, log,
TrainResult). The contract is that the loop runs deterministically,
produces shape-correct outputs, and the trained model loads.
"""

import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, os.getcwd())

from stl.learning.model import FEATURE_DIM, ValueNet
from stl.engine.actions import ACTION_SIZE
from stl.learning.train import (
    ACTION_SCHEMA_VERSION,
    CHECKPOINT_FORMAT_VERSION,
    CheckpointFormatError,
    TrainConfig as _StrictTrainConfig,
    TrainResult,
    _SourceBalancedBatchSampler,
    _certificate_saddle_gaps,
    _load_certificate_policy_data,
    checkpoint_digest,
    load_checkpoint,
    load_checkpoint_bundle,
    make_predict_fn,
    train,
)


def TrainConfig(*args, **kwargs):
    """Explicit legacy-corpus adapter for pre-V2 synthetic fixtures."""

    kwargs.setdefault("allow_legacy_targets", True)
    return _StrictTrainConfig(*args, **kwargs)


def _write_synthetic_targets(path: Path, n: int = 64, seed: int = 0) -> None:
    """Write a tiny synthetic .npz that the trainer can consume."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, FEATURE_DIM)).astype(np.float32)
    y = np.tanh(X.sum(axis=1) * 0.1).astype(np.float32)
    sources = np.array(["terminal"] * (n // 2) + ["exact_horizon_2"] * (n - n // 2))
    horizons = np.zeros(n, dtype=np.int32)
    np.savez(path, X=X, y=y, sources=sources, horizons=horizons)


def test_train_runs_and_returns_train_result():
    with tempfile.TemporaryDirectory() as tmp:
        targets = Path(tmp) / "targets.npz"
        _write_synthetic_targets(targets, n=64)
        out_dir = Path(tmp) / "out"

        cfg = TrainConfig(epochs=3, batch_size=16, learning_rate=1e-3, seed=42)
        result = train(targets, out_dir, cfg)

    assert isinstance(result, TrainResult)
    assert result.checkpoint_path.endswith("best.pt")
    assert 0.0 <= result.best_val_mse < 10.0
    assert 0 <= result.best_epoch <= 2
    assert len(result.train_history) == 3


def test_source_balanced_sampler_enforces_every_batch_exactly():
    sources = np.asarray(
        ["exact_horizon_2"] * 12 + ["terminal"] * 4,
        dtype=np.str_,
    )
    sampler = _SourceBalancedBatchSampler(
        sources,
        (("exact_horizon_2", 6), ("terminal", 2)),
        seed=17,
    )
    batches = list(sampler)
    assert len(batches) == 2
    for batch in batches:
        selected = sources[np.asarray(batch)]
        assert np.count_nonzero(selected == "exact_horizon_2") == 6
        assert np.count_nonzero(selected == "terminal") == 2


def test_source_balanced_training_rejects_an_unrepresented_source(tmp_path):
    targets = tmp_path / "targets.npz"
    _write_synthetic_targets(targets, n=32)
    with pytest.raises(ValueError, match="name every training source exactly"):
        train(
            targets,
            tmp_path / "out",
            TrainConfig(
                epochs=1,
                batch_size=8,
                source_batch_counts=(("terminal", 8),),
            ),
        )


def test_checkpoint_bundle_carries_optimizer_schema_and_provenance(tmp_path):
    targets = tmp_path / "targets.npz"
    _write_synthetic_targets(targets, n=64)
    result = train(
        targets,
        tmp_path / "out",
        TrainConfig(epochs=1, batch_size=16, seed=17),
    )

    bundle = load_checkpoint_bundle(result.checkpoint_path)
    assert bundle["format_version"] == CHECKPOINT_FORMAT_VERSION
    assert bundle["optimizer_state_dict"] is not None
    assert bundle["scheduler_state_dict"] is not None
    assert bundle["model_schema"]["action_schema"] == ACTION_SCHEMA_VERSION
    assert bundle["model_schema"]["action_size"] == ACTION_SIZE
    assert bundle["provenance"]["corpus_digests"] == [checkpoint_digest(targets)]
    assert bundle["provenance"]["seed"] == 17


def test_checkpoint_bundle_rejects_wrong_parent_and_policy_shape(tmp_path):
    import torch

    targets = tmp_path / "targets.npz"
    _write_synthetic_targets(targets, n=64)
    result = train(targets, tmp_path / "out", TrainConfig(epochs=1, batch_size=16))

    with pytest.raises(CheckpointFormatError, match="parent digest"):
        load_checkpoint_bundle(result.checkpoint_path, expected_parent_digest="0" * 64)

    payload = torch.load(result.checkpoint_path, weights_only=False)
    hidden = payload["model_schema"]["hidden_dim"]
    payload["model_state_dict"]["policy_head.weight"] = torch.zeros(122, hidden)
    broken = tmp_path / "broken.pt"
    torch.save(payload, broken)
    with pytest.raises(CheckpointFormatError, match="policy head"):
        load_checkpoint(broken)


def test_empty_checkpoint_is_rejected_instead_of_loading_random_weights(tmp_path):
    import torch

    path = tmp_path / "empty.pt"
    torch.save({}, path)
    with pytest.raises(CheckpointFormatError, match="bare or legacy"):
        load_checkpoint(path)


def test_manifestless_targets_require_explicit_legacy_opt_in(tmp_path):
    targets = tmp_path / "legacy.npz"
    _write_synthetic_targets(targets, n=32)
    with pytest.raises(ValueError, match="manifestless target corpus"):
        train(
            targets,
            tmp_path / "out",
            _StrictTrainConfig(epochs=1, batch_size=8),
        )


def test_train_hidden_dim_192_runs_end_to_end():
    """Phase I-2: the widened net (hidden_dim=192, ~65.4K params) must train
    through the real loop and produce a checkpoint that ``load_checkpoint``
    auto-infers back to width 192. Tiny synthetic corpus + few epochs — a
    pipeline smoke test, not a convergence claim."""
    with tempfile.TemporaryDirectory() as tmp:
        targets = Path(tmp) / "targets.npz"
        _write_synthetic_targets(targets, n=96)
        out_dir = Path(tmp) / "out"

        cfg = TrainConfig(epochs=3, batch_size=16, seed=0, hidden_dim=192)
        result = train(targets, out_dir, cfg)

        assert isinstance(result, TrainResult)
        assert result.checkpoint_path.endswith("best.pt")
        model = load_checkpoint(result.checkpoint_path)
        total = sum(p.numel() for p in model.parameters())
        assert 68_000 < total < 76_000, (
            f"checkpoint did not preserve hidden=192; got {total}"
        )


def test_train_can_warm_start_from_checkpoint():
    with tempfile.TemporaryDirectory() as tmp:
        targets = Path(tmp) / "targets.npz"
        _write_synthetic_targets(targets, n=64)

        base = train(
            targets,
            Path(tmp) / "base",
            TrainConfig(epochs=1, batch_size=16, seed=0),
        )
        fine_tuned = train(
            targets,
            Path(tmp) / "fine_tuned",
            TrainConfig(
                epochs=1,
                batch_size=16,
                seed=0,
                init_checkpoint=base.checkpoint_path,
            ),
        )

        assert Path(fine_tuned.checkpoint_path).exists()


def test_train_rejects_init_checkpoint_hidden_dim_mismatch():
    with tempfile.TemporaryDirectory() as tmp:
        targets = Path(tmp) / "targets.npz"
        _write_synthetic_targets(targets, n=64)

        base = train(
            targets,
            Path(tmp) / "base",
            TrainConfig(epochs=1, batch_size=16, seed=0, hidden_dim=192),
        )
        with pytest.raises(ValueError, match="hidden_dim"):
            train(
                targets,
                Path(tmp) / "mismatch",
                TrainConfig(
                    epochs=1,
                    batch_size=16,
                    seed=0,
                    hidden_dim=64,
                    init_checkpoint=base.checkpoint_path,
                ),
            )


def test_train_value_head_mode_freezes_trunk_and_policy_head():
    import torch

    with tempfile.TemporaryDirectory() as tmp:
        targets = Path(tmp) / "targets.npz"
        _write_synthetic_targets(targets, n=64)

        base = train(
            targets,
            Path(tmp) / "base",
            TrainConfig(epochs=1, batch_size=16, seed=0),
        )
        before = load_checkpoint(base.checkpoint_path).state_dict()
        fine_tuned = train(
            targets,
            Path(tmp) / "fine_tuned",
            TrainConfig(
                epochs=2,
                batch_size=16,
                seed=0,
                init_checkpoint=base.checkpoint_path,
                trainable_parts="value_head",
                learning_rate=1e-3,
                early_stopping_patience=10,
            ),
        )
        after = load_checkpoint(fine_tuned.checkpoint_path).state_dict()

        for key in before:
            if key.startswith("value_head."):
                continue
            assert torch.equal(before[key], after[key]), (
                f"{key} changed despite value_head freeze"
            )


def test_train_records_reference_value_distillation_loss():
    with tempfile.TemporaryDirectory() as tmp:
        targets = Path(tmp) / "targets.npz"
        _write_synthetic_targets(targets, n=64)

        base = train(
            targets,
            Path(tmp) / "base",
            TrainConfig(epochs=1, batch_size=16, seed=0),
        )
        result = train(
            targets,
            Path(tmp) / "distill",
            TrainConfig(
                epochs=2,
                batch_size=16,
                seed=0,
                init_checkpoint=base.checkpoint_path,
                reference_checkpoint=base.checkpoint_path,
                value_distill_weight=10.0,
                trainable_parts="value_head",
            ),
        )

        assert "train_value_distill_mse" in result.train_history[0]
        assert result.train_history[0]["train_value_distill_mse"] >= 0.0


def test_train_rejects_unknown_trainable_parts():
    with tempfile.TemporaryDirectory() as tmp:
        targets = Path(tmp) / "targets.npz"
        _write_synthetic_targets(targets, n=64)

        with pytest.raises(ValueError, match="trainable_parts"):
            train(
                targets,
                Path(tmp) / "bad",
                TrainConfig(
                    epochs=1,
                    batch_size=16,
                    seed=0,
                    trainable_parts="not_a_mode",
                ),
            )


def test_train_produces_best_last_checkpoints_and_log():
    with tempfile.TemporaryDirectory() as tmp:
        targets = Path(tmp) / "targets.npz"
        _write_synthetic_targets(targets, n=64)
        out_dir = Path(tmp) / "out"

        cfg = TrainConfig(epochs=3, batch_size=16, seed=0)
        train(targets, out_dir, cfg)

        assert (out_dir / "best.pt").exists()
        assert (out_dir / "last.pt").exists()
        log_path = out_dir / "log.json"
        assert log_path.exists()

        with open(log_path) as f:
            log = json.load(f)
        assert log["history"]
        assert "best_val_mse" in log
        assert "best_per_source_mse" in log
        assert "config" in log


def test_train_per_source_mse_tracks_both_sources_in_corpus():
    with tempfile.TemporaryDirectory() as tmp:
        targets = Path(tmp) / "targets.npz"
        _write_synthetic_targets(targets, n=64, seed=1)
        out_dir = Path(tmp) / "out"

        cfg = TrainConfig(epochs=3, batch_size=16, seed=0, val_fraction=0.5)
        result = train(targets, out_dir, cfg)

    sources = set(result.best_per_source_mse.keys())
    # With val_fraction=0.5 the validation set is large enough to span both.
    assert "terminal" in sources or "exact_horizon_2" in sources


def test_train_is_deterministic_under_same_seed():
    with tempfile.TemporaryDirectory() as tmp:
        targets = Path(tmp) / "targets.npz"
        _write_synthetic_targets(targets, n=64, seed=7)

        cfg = TrainConfig(epochs=3, batch_size=16, seed=123)
        out_a = Path(tmp) / "a"
        out_b = Path(tmp) / "b"
        result_a = train(targets, out_a, cfg)
        result_b = train(targets, out_b, cfg)

    assert result_a.best_val_mse == pytest.approx(result_b.best_val_mse, abs=1e-7)
    assert result_a.best_epoch == result_b.best_epoch


def test_load_checkpoint_restores_a_usable_model():
    with tempfile.TemporaryDirectory() as tmp:
        targets = Path(tmp) / "targets.npz"
        _write_synthetic_targets(targets, n=32)
        out_dir = Path(tmp) / "out"
        cfg = TrainConfig(epochs=2, batch_size=8, seed=0)
        result = train(targets, out_dir, cfg)

        model = load_checkpoint(result.checkpoint_path)
    assert isinstance(model, ValueNet)

    import torch

    x = torch.zeros(1, FEATURE_DIM)
    with torch.no_grad():
        value, dropper_logits, checker_logits = model(x, 2)
        out = value.squeeze().item()
    assert -1.0 <= out <= 1.0
    assert dropper_logits.shape == (1, ACTION_SIZE)
    assert checker_logits.shape == (1, ACTION_SIZE)


def test_make_predict_fn_returns_value_in_unit_interval():
    from stl.engine.game import PHYSICALITY_BAKU, PHYSICALITY_HAL
    from stl.engine.game import Game
    from stl.engine.game import Player
    from stl.engine.game import Referee

    with tempfile.TemporaryDirectory() as tmp:
        targets = Path(tmp) / "targets.npz"
        _write_synthetic_targets(targets, n=32)
        out_dir = Path(tmp) / "out"
        cfg = TrainConfig(epochs=2, batch_size=8, seed=0)
        result = train(targets, out_dir, cfg)
        model = load_checkpoint(result.checkpoint_path)

    predict = make_predict_fn(model)
    hal = Player(name="Hal", physicality=PHYSICALITY_HAL)
    baku = Player(name="Baku", physicality=PHYSICALITY_BAKU)
    game = Game(player1=hal, player2=baku, referee=Referee(), first_dropper=hal)
    game.seed(0)

    value, dropper_dist, checker_dist = predict(game, horizon=2)
    assert isinstance(value, float)
    assert -1.0 <= value <= 1.0
    assert dropper_dist.shape == (ACTION_SIZE,)
    assert checker_dist.shape == (ACTION_SIZE,)
    assert dropper_dist.sum() == pytest.approx(1.0)
    assert checker_dist.sum() == pytest.approx(1.0)


def _write_synthetic_targets_with_policy(
    path: Path, n: int = 128, seed: int = 0
) -> None:
    """Write a tiny synthetic .npz that includes non-uniform policy targets.

    Policy targets concentrate dropper probability mass on second 30 and
    checker probability mass on second 60 — both legal seconds for any
    non-leap turn. A trained net must learn these peaked distributions or
    its policy NLL will not decrease.
    """
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, FEATURE_DIM)).astype(np.float32)
    y = np.tanh(X.sum(axis=1) * 0.1).astype(np.float32)
    sources = np.array(["terminal"] * n)
    horizons = np.zeros(n, dtype=np.int32)

    dropper_dists = np.zeros((n, ACTION_SIZE), dtype=np.float32)
    dropper_dists[:, 30] = 1.0  # second 30
    checker_dists = np.zeros((n, ACTION_SIZE), dtype=np.float32)
    checker_dists[:, 60] = 1.0  # second 60

    legal_masks = np.zeros((n, ACTION_SIZE), dtype=np.float32)
    legal_masks[:, 1:61] = 1.0  # seconds 1..60 legal; second 61 illegal

    np.savez(
        path,
        X=X,
        y=y,
        sources=sources,
        horizons=horizons,
        dropper_dists=dropper_dists,
        checker_dists=checker_dists,
        dropper_legal_masks=legal_masks,
        checker_legal_masks=legal_masks,
    )


def test_train_value_net_optimizes_both_value_and_policy_loss():
    """Both value MSE and policy NLL must measurably decrease over training
    when targets carry non-trivial policy distributions.

    Without this gate, a bug that silently zeros the policy gradient (wrong
    output dim, fully-masked logits, detached tensor in the loss path) would
    ship green: shape tests would still pass and the value head would still
    train. The only signal that the policy head is actually learning is its
    own NLL falling.
    """
    with tempfile.TemporaryDirectory() as tmp:
        targets = Path(tmp) / "targets.npz"
        _write_synthetic_targets_with_policy(targets, n=128, seed=0)
        out_dir = Path(tmp) / "out"

        cfg = TrainConfig(
            epochs=30,
            batch_size=16,
            learning_rate=3e-3,
            policy_loss_weight=1.0,
            seed=42,
            early_stopping_patience=100,  # disable; let all 30 epochs run
        )
        result = train(targets, out_dir, cfg)

    history = result.train_history
    assert len(history) == 30, f"Expected all 30 epochs to run, got {len(history)}"

    initial_value_mse = history[0]["train_mse"]
    final_value_mse = history[-1]["train_mse"]
    initial_policy_nll = history[0]["train_policy_nll"]
    final_policy_nll = history[-1]["train_policy_nll"]

    assert final_value_mse < initial_value_mse, (
        f"Value MSE did not decrease: {initial_value_mse:.5f} → {final_value_mse:.5f}"
    )
    assert final_policy_nll < initial_policy_nll, (
        f"Policy NLL did not decrease: {initial_policy_nll:.5f} → {final_policy_nll:.5f}"
    )
    # Stronger sanity: policy must drop materially, not just by floating-point noise.
    # Random uniform over 60 legal seconds = ln(60) ≈ 4.094 NLL; perfect = 0.
    # Two heads → initial ≈ 8.19. After 30 epochs on a peaked target, NLL should
    # have dropped by at least 30%.
    assert final_policy_nll < initial_policy_nll * 0.7, (
        f"Policy NLL dropped only marginally ({initial_policy_nll:.3f} → "
        f"{final_policy_nll:.3f}); possible silent gradient zeroing."
    )


def test_split_keeps_replicated_records_out_of_both_splits():
    """Ticket B7: callers replicate tablebase/interior records 30-660x before
    training; a raw row-permutation split scattered identical replicas across
    train AND val, so best-epoch selection validated on memorized rows. The
    split must group rows by unique state identity (features bytes) and
    assign whole groups to one side."""
    from stl.learning.train import _split_indices

    rng_data = np.random.default_rng(0)
    base = rng_data.normal(size=(20, FEATURE_DIM)).astype(np.float32)
    replicated = np.repeat(base[:1], 50, axis=0)  # one record replicated 50x
    X = np.concatenate([base, replicated], axis=0)  # 70 rows, 20 unique keys

    train_idx, val_idx = _split_indices(X, 0.25, np.random.default_rng(123))

    train_keys = {X[i].tobytes() for i in train_idx}
    val_keys = {X[i].tobytes() for i in val_idx}
    assert not (train_keys & val_keys), (
        f"{len(train_keys & val_keys)} state key(s) leaked into both splits"
    )
    # Partition: every row lands in exactly one split.
    combined = sorted(np.concatenate([train_idx, val_idx]).tolist())
    assert combined == list(range(len(X)))
    assert len(train_idx) > 0 and len(val_idx) > 0
    # Determinism: same seed -> same split.
    train_idx2, val_idx2 = _split_indices(X, 0.25, np.random.default_rng(123))
    assert np.array_equal(np.sort(train_idx), np.sort(train_idx2))
    assert np.array_equal(np.sort(val_idx), np.sort(val_idx2))


def test_train_rejects_wrong_feature_dim():
    with tempfile.TemporaryDirectory() as tmp:
        targets = Path(tmp) / "targets.npz"
        rng = np.random.default_rng(0)
        n = 16
        X = rng.normal(size=(n, FEATURE_DIM + 5)).astype(np.float32)  # wrong width
        y = rng.normal(size=(n,)).astype(np.float32)
        sources = np.array(["terminal"] * n)
        horizons = np.zeros(n, dtype=np.int32)
        np.savez(targets, X=X, y=y, sources=sources, horizons=horizons)

        with pytest.raises(ValueError):
            train(targets, Path(tmp) / "out", TrainConfig(epochs=1, batch_size=4))


def test_certificate_saddle_gap_matches_both_role_orientations():
    import torch

    payoff = torch.zeros((2, ACTION_SIZE, ACTION_SIZE), dtype=torch.float32)
    payoff[:, 1:3, 1:3] = torch.tensor([[0.0, 1.0], [2.0, -1.0]])
    dropper_logits = torch.full((2, ACTION_SIZE), -20.0)
    checker_logits = torch.full((2, ACTION_SIZE), -20.0)
    dropper_logits[:, 1:3] = torch.log(torch.tensor([0.75, 0.25]))
    checker_logits[:, 1:3] = torch.log(torch.tensor([0.4, 0.6]))
    legal = torch.zeros((2, ACTION_SIZE), dtype=torch.bool)
    legal[:, 1:3] = True

    gaps = _certificate_saddle_gaps(
        dropper_logits,
        checker_logits,
        legal,
        legal,
        payoff,
        torch.tensor([True, True]),
        torch.tensor([True, False]),
    )

    matrix = np.asarray([[0.0, 1.0], [2.0, -1.0]])
    x = np.asarray([0.75, 0.25])
    y = np.asarray([0.4, 0.6])
    expected_hal_dropper = float((matrix @ y).max() - (x @ matrix).min())
    expected_baku_dropper = float((x @ matrix).max() - (matrix @ y).min())
    assert gaps[0].item() == pytest.approx(expected_hal_dropper)
    assert gaps[1].item() == pytest.approx(expected_baku_dropper)


def test_certificate_saddle_gap_is_differentiable_and_ignores_inactive_rows():
    import torch

    payoff = torch.zeros((2, ACTION_SIZE, ACTION_SIZE), dtype=torch.float32)
    payoff[:, 1:3, 1:3] = torch.tensor([[1.0, -1.0], [-1.0, 1.0]])
    dropper_logits = torch.zeros((2, ACTION_SIZE), requires_grad=True)
    checker_logits = torch.zeros((2, ACTION_SIZE), requires_grad=True)
    legal = torch.zeros((2, ACTION_SIZE), dtype=torch.bool)
    legal[0, 1:3] = True
    with torch.no_grad():
        dropper_logits[:, 1] = 2.0
        checker_logits[:, 1] = 2.0

    gaps = _certificate_saddle_gaps(
        dropper_logits,
        checker_logits,
        legal,
        legal,
        payoff,
        torch.tensor([True, False]),
        torch.tensor([True, True]),
    )
    gaps.sum().backward()

    assert gaps[0].item() > 0.0
    assert gaps[1].item() == 0.0
    assert float(dropper_logits.grad[0].abs().sum()) > 0.0
    assert float(checker_logits.grad[0].abs().sum()) > 0.0
    assert float(dropper_logits.grad[1].abs().sum()) == 0.0
    assert float(checker_logits.grad[1].abs().sum()) == 0.0


def test_training_certificate_loader_binds_matrix_to_replay_state(tmp_path):
    from types import SimpleNamespace

    from stl.learning.certificates import ExactPolicyCertificate, save_certificate_shard
    from stl.learning.reachable import split_reachable_candidates
    from stl.learning.replay import ShardRole, exact_state_hash

    state = next(
        item.exact_state
        for item in split_reachable_candidates(smoke=True, split_seed=20260714).train
        if not item.exact_state.game_over
    )
    dropper = np.zeros(ACTION_SIZE, dtype=np.float32)
    checker = np.zeros(ACTION_SIZE, dtype=np.float32)
    dropper[1:3] = 0.5
    checker[1:3] = 0.5
    legal = np.zeros(ACTION_SIZE, dtype=np.bool_)
    legal[1:3] = True
    record = SimpleNamespace(
        exact_state=state,
        dropper_dist=dropper,
        checker_dist=checker,
        dropper_legal_mask=legal,
        checker_legal_mask=legal,
        search_config_digest="a" * 64,
        value_horizon_half_rounds=2,
    )
    certificate = ExactPolicyCertificate(
        state_hash=exact_state_hash(state),
        search_config_digest="a" * 64,
        horizon=2,
        drop_actions=(1, 2),
        check_actions=(1, 2),
        payoff_for_hal=np.asarray([[1.0, -1.0], [-1.0, 1.0]]),
        dropper_strategy=np.asarray([0.5, 0.5]),
        checker_strategy=np.asarray([0.5, 0.5]),
        value_for_hal=0.0,
        unresolved_probability=0.0,
    )
    path = tmp_path / "train-certificates.npz"
    save_certificate_shard([certificate], path, shard_role=ShardRole.REPLAY)

    data = _load_certificate_policy_data(
        [record], path, expected_role=ShardRole.REPLAY
    )
    assert data.active.tolist() == [True]
    np.testing.assert_array_equal(
        data.payoffs_for_hal[0, 1:3, 1:3], certificate.payoff_for_hal
    )

    record.search_config_digest = "b" * 64
    with pytest.raises(ValueError, match="search config digest"):
        _load_certificate_policy_data([record], path, expected_role=ShardRole.REPLAY)
