"""Tests for Phase 3: training/bootstrap_loop.py and the MCTS bootstrap target generator.

The full AlphaZero bootstrap loop runs MCTS at thousands of states for
thousands of iterations each — way too slow for unit tests. These tests
verify the *structural* contract:

  - generate_mcts_bootstrap_targets emits well-formed ValueTargets at
    tiny iteration counts and tiny grids.
  - Sources route correctly: terminal positions are tagged "terminal",
    pinned tablebase states are tagged "tablebase", everything else is
    tagged "mcts_bootstrap".
  - bootstrap_one_generation chains target-gen + train into a single
    pass.
  - calibration_check loads a saved checkpoint and produces a report.
  - Determinism under seeded RNG.

We use a stub predict_fn (returns 0.0) to avoid PyTorch overhead during
target generation.
"""

import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hal.value_net import FEATURE_DIM
from training.bootstrap_loop import (
    BootstrapConfig,
    bootstrap_one_generation,
    calibration_check,
)
from training.train_value_net import TrainConfig, train
from training.value_targets import (
    SOURCE_MCTS_BOOTSTRAP,
    SOURCE_TABLEBASE,
    SOURCE_TERMINAL,
    ValueTarget,
    generate_mcts_bootstrap_targets,
)


# ── generate_mcts_bootstrap_targets ─────────────────────────────────────


def _stub_predict_fn(_game) -> float:
    """A constant predictor; cheap, deterministic, doesn't need PyTorch."""
    return 0.0


def _tiny_grids() -> dict:
    """Minimal grid: forces one tablebase hit and one MCTS-bootstrap call."""
    return dict(
        baku_cylinder_grid=(0.0, 299.0),
        hal_cylinder_grid=(0.0,),
        clock_grid=(720.0,),
        half_grid=(1,),
        deaths_grid=(0,),
        cpr_grid=(0,),
    )


def test_bootstrap_targets_emit_correct_sources_for_tiny_grid():
    targets = generate_mcts_bootstrap_targets(
        _stub_predict_fn,
        iterations_per_state=20,
        seed=0,
        **_tiny_grids(),
    )
    sources = {t.source for t in targets}
    assert SOURCE_TABLEBASE in sources       # cyl=299 -> forced_baku_overflow_death
    assert SOURCE_MCTS_BOOTSTRAP in sources  # cyl=0   -> non-terminal, non-tablebase


def test_bootstrap_targets_have_correct_feature_shape_and_dtype():
    targets = generate_mcts_bootstrap_targets(
        _stub_predict_fn,
        iterations_per_state=20,
        seed=0,
        **_tiny_grids(),
    )
    assert len(targets) >= 2
    for t in targets:
        assert isinstance(t, ValueTarget)
        assert t.features.shape == (FEATURE_DIM,)
        assert t.features.dtype == np.float32
        assert -1.0 <= t.value <= 1.0


def test_bootstrap_targets_horizon_field_records_iteration_budget():
    iterations = 30
    targets = generate_mcts_bootstrap_targets(
        _stub_predict_fn,
        iterations_per_state=iterations,
        seed=0,
        **_tiny_grids(),
    )
    for t in targets:
        if t.source == SOURCE_MCTS_BOOTSTRAP:
            assert t.horizon == iterations
        else:
            # terminal / tablebase short-circuits keep horizon=0.
            assert t.horizon == 0


def test_bootstrap_targets_are_deterministic_under_same_seed():
    a = generate_mcts_bootstrap_targets(
        _stub_predict_fn, iterations_per_state=20, seed=42, **_tiny_grids()
    )
    b = generate_mcts_bootstrap_targets(
        _stub_predict_fn, iterations_per_state=20, seed=42, **_tiny_grids()
    )
    assert len(a) == len(b)
    for ta, tb in zip(a, b):
        assert ta.source == tb.source
        assert ta.value == pytest.approx(tb.value)


def test_bootstrap_terminal_source_pinned_to_known_value():
    """A game-over terminal returns value=1.0 (Hal wins) directly without MCTS."""
    grids = dict(
        baku_cylinder_grid=(299.0,),  # tablebase entry +1.0
        hal_cylinder_grid=(0.0,),
        clock_grid=(720.0,),
        half_grid=(1,),
        deaths_grid=(0,),
        cpr_grid=(0,),
    )
    targets = generate_mcts_bootstrap_targets(
        _stub_predict_fn, iterations_per_state=20, seed=0, **grids
    )
    tablebase_targets = [t for t in targets if t.source == SOURCE_TABLEBASE]
    assert len(tablebase_targets) == 1
    assert tablebase_targets[0].value == pytest.approx(1.0)


# ── bootstrap_one_generation ─────────────────────────────────────────────


def _write_initial_checkpoint(out_dir: Path) -> str:
    """Train a tiny model on synthetic data so we have a gen-0 checkpoint."""
    rng = np.random.default_rng(0)
    n = 32
    X = rng.normal(size=(n, FEATURE_DIM)).astype(np.float32)
    y = np.tanh(X.sum(axis=1) * 0.1).astype(np.float32)
    sources = np.array(["terminal"] * n)
    horizons = np.zeros(n, dtype=np.int32)
    targets_path = out_dir / "gen0_targets.npz"
    np.savez(targets_path, X=X, y=y, sources=sources, horizons=horizons)

    gen0_dir = out_dir / "gen0"
    cfg = TrainConfig(epochs=2, batch_size=8, seed=0)
    result = train(targets_path, gen0_dir, cfg)
    return result.checkpoint_path


def test_bootstrap_one_generation_chains_target_gen_and_training():
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        gen0_ckpt = _write_initial_checkpoint(tmp_path)
        gen1_targets_path = tmp_path / "gen1_targets.npz"
        gen1_dir = tmp_path / "gen1"

        cfg = BootstrapConfig(
            iterations_per_state=20,
            seed=0,
            train_config=TrainConfig(epochs=2, batch_size=4, seed=0),
        )
        result = bootstrap_one_generation(
            gen_in_checkpoint=gen0_ckpt,
            gen_out_targets=gen1_targets_path,
            gen_out_dir=gen1_dir,
            config=cfg,
            grids=_tiny_grids(),
        )

        assert result.targets_count >= 2
        assert result.bootstrap_count >= 1
        assert result.targets_path.endswith("gen1_targets.npz")
        assert Path(result.train_result.checkpoint_path).exists()


# ── calibration_check ────────────────────────────────────────────────────


def test_calibration_check_loads_checkpoint_and_returns_report():
    from training.value_targets import SOURCE_EXACT_HORIZON_2

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        ckpt = _write_initial_checkpoint(tmp_path)

        held_out = [
            ValueTarget(
                features=np.zeros(FEATURE_DIM, dtype=np.float32),
                value=0.0,
                source=SOURCE_TERMINAL,
                horizon=0,
            ),
            ValueTarget(
                features=np.ones(FEATURE_DIM, dtype=np.float32),
                value=0.5,
                source=SOURCE_EXACT_HORIZON_2,
                horizon=2,
            ),
        ]
        report = calibration_check(checkpoint_path=ckpt, held_out_targets=held_out)

    assert report.n_targets == 2
    assert SOURCE_TERMINAL in report.mse_per_source
    assert SOURCE_EXACT_HORIZON_2 in report.mse_per_source
    assert 0.0 <= report.brier_score <= 1.0
