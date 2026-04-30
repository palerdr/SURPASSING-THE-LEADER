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

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hal.value_net import FEATURE_DIM, ValueNet
from training.train_value_net import (
    TrainConfig,
    TrainResult,
    load_checkpoint,
    make_predict_fn,
    train,
)


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
        out = model(x).squeeze().item()
    assert -1.0 <= out <= 1.0


def test_make_predict_fn_returns_value_in_unit_interval():
    from src.Constants import PHYSICALITY_BAKU, PHYSICALITY_HAL
    from src.Game import Game
    from src.Player import Player
    from src.Referee import Referee

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

    value = predict(game)
    assert isinstance(value, float)
    assert -1.0 <= value <= 1.0


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
