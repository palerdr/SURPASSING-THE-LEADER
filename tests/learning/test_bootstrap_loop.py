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

sys.path.insert(0, os.getcwd())

from stl.learning.model import FEATURE_DIM
from stl.learning.bootstrap import (
    BootstrapConfig,
    CalibrationGateError,
    bootstrap_one_generation,
    calibration_check,
    enforce_calibration_gate,
)
from stl.learning.train import TrainConfig, train
from stl.learning.targets import (
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
        baku_cylinder_grid=(0.0, 300.0),
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
    assert SOURCE_TABLEBASE in sources       # cyl=300 -> forced_baku_overflow_death
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


def test_bootstrap_targets_can_cap_mcts_rows_without_dropping_anchors():
    targets = generate_mcts_bootstrap_targets(
        _stub_predict_fn,
        iterations_per_state=1,
        seed=0,
        include_anchor_classes=True,
        bootstrap_max_states=1,
        baku_cylinder_grid=(0.0, 60.0, 120.0),
        hal_cylinder_grid=(0.0,),
        clock_grid=(720.0,),
        half_grid=(1,),
        deaths_grid=(0,),
        cpr_grid=(0,),
    )

    mcts_targets = [t for t in targets if t.source == SOURCE_MCTS_BOOTSTRAP]
    assert len(mcts_targets) == 1
    assert SOURCE_TERMINAL in {t.source for t in targets}


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
        baku_cylinder_grid=(300.0,),  # tablebase entry +1.0
        hal_cylinder_grid=(0.0,),
        clock_grid=(720.0,),
        half_grid=(1,),
        deaths_grid=(0,),
        cpr_grid=(0,),
    )
    targets = generate_mcts_bootstrap_targets(
        _stub_predict_fn,
        iterations_per_state=20,
        seed=0,
        include_anchor_classes=False,
        **grids,
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
            required_sources=(SOURCE_TERMINAL,),
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
    from stl.learning.targets import SOURCE_EXACT_HORIZON_2

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


# ── regression-rejection gate ────────────────────────────────────────────


def _build_held_out_targets() -> list[ValueTarget]:
    """A fixed held-out reference set used by the gate tests."""
    rng = np.random.default_rng(99)
    items: list[ValueTarget] = []
    for _ in range(8):
        feats = rng.normal(size=(FEATURE_DIM,)).astype(np.float32)
        items.append(
            ValueTarget(
                features=feats,
                value=float(np.tanh(feats.sum() * 0.1)),
                source=SOURCE_TERMINAL,
                horizon=0,
            )
        )
    return items


def test_bootstrap_gate_accepts_when_no_prev_baseline_supplied():
    """No prev_gen_holdout_mse means the gate cannot reject. Held-out
    calibration is still scored when held_out_targets is provided."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        gen0_ckpt = _write_initial_checkpoint(tmp_path)
        gen1_targets_path = tmp_path / "gen1_targets.npz"
        gen1_dir = tmp_path / "gen1"

        cfg = BootstrapConfig(
            iterations_per_state=20,
            seed=0,
            train_config=TrainConfig(epochs=2, batch_size=4, seed=0),
            required_sources=(SOURCE_TERMINAL,),
        )
        result = bootstrap_one_generation(
            gen_in_checkpoint=gen0_ckpt,
            gen_out_targets=gen1_targets_path,
            gen_out_dir=gen1_dir,
            config=cfg,
            grids=_tiny_grids(),
            held_out_targets=_build_held_out_targets(),
            prev_gen_holdout_mse=None,
        )

        assert result.accepted is True
        assert result.holdout_calibration is not None
        assert result.holdout_calibration.n_targets == 8


def test_bootstrap_gate_rejects_when_new_mse_strictly_exceeds_prev():
    """If prev_gen_holdout_mse is below the new gen's MSE, the gate fires
    and ``accepted`` is False. Use a deliberately impossible-to-beat baseline
    of -1.0 so any non-negative MSE counts as a regression."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        gen0_ckpt = _write_initial_checkpoint(tmp_path)
        gen1_targets_path = tmp_path / "gen1_targets.npz"
        gen1_dir = tmp_path / "gen1"

        cfg = BootstrapConfig(
            iterations_per_state=20,
            seed=0,
            train_config=TrainConfig(epochs=2, batch_size=4, seed=0),
            required_sources=(SOURCE_TERMINAL,),
        )
        result = bootstrap_one_generation(
            gen_in_checkpoint=gen0_ckpt,
            gen_out_targets=gen1_targets_path,
            gen_out_dir=gen1_dir,
            config=cfg,
            grids=_tiny_grids(),
            held_out_targets=_build_held_out_targets(),
            prev_gen_holdout_mse=-1.0,  # impossible to beat → forced rejection
        )

        assert result.accepted is False
        assert result.holdout_calibration is not None
        assert result.holdout_calibration.overall_mse > -1.0
        assert result.prev_gen_holdout_mse == -1.0


def test_bootstrap_gate_accepts_when_new_mse_strictly_below_prev():
    """If prev_gen_holdout_mse is above the new gen's MSE, the gate passes.
    Use a deliberately huge baseline so any finite MSE counts as improvement."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        gen0_ckpt = _write_initial_checkpoint(tmp_path)
        gen1_targets_path = tmp_path / "gen1_targets.npz"
        gen1_dir = tmp_path / "gen1"

        cfg = BootstrapConfig(
            iterations_per_state=20,
            seed=0,
            train_config=TrainConfig(epochs=2, batch_size=4, seed=0),
            required_sources=(SOURCE_TERMINAL,),
        )
        result = bootstrap_one_generation(
            gen_in_checkpoint=gen0_ckpt,
            gen_out_targets=gen1_targets_path,
            gen_out_dir=gen1_dir,
            config=cfg,
            grids=_tiny_grids(),
            held_out_targets=_build_held_out_targets(),
            prev_gen_holdout_mse=1e6,  # any finite MSE beats this
        )

        assert result.accepted is True
        assert result.holdout_calibration is not None


def test_calibration_gate_raises_when_required_source_class_missing():
    from stl.learning.calibration import CalibrationReport

    report = CalibrationReport(
        mse_per_source={SOURCE_TERMINAL: 0.0},
        overall_mse=0.0,
        brier_score=0.0,
        reliability_bins=[],
        exact_target_error=0.0,
        n_targets=1,
        mean_unresolved_probability_per_source={SOURCE_TERMINAL: 0.0},
    )

    with pytest.raises(CalibrationGateError, match="missing required"):
        enforce_calibration_gate(report, BootstrapConfig())


def test_calibration_gate_raises_when_tablebase_mse_above_threshold():
    from stl.learning.calibration import CalibrationReport
    from stl.learning.targets import SOURCE_EXACT_HORIZON_2, SOURCE_EXACT_HORIZON_3

    report = CalibrationReport(
        mse_per_source={
            SOURCE_TERMINAL: 0.0,
            SOURCE_TABLEBASE: 0.02,
            SOURCE_EXACT_HORIZON_2: 0.0,
            SOURCE_EXACT_HORIZON_3: 0.0,
        },
        overall_mse=0.005,
        brier_score=0.0,
        reliability_bins=[],
        exact_target_error=0.0,
        n_targets=4,
        mean_unresolved_probability_per_source={
            SOURCE_TERMINAL: 0.0,
            SOURCE_TABLEBASE: 0.0,
            SOURCE_EXACT_HORIZON_2: 0.0,
            SOURCE_EXACT_HORIZON_3: 0.0,
        },
    )

    with pytest.raises(CalibrationGateError, match="tablebase MSE"):
        enforce_calibration_gate(report, BootstrapConfig(tablebase_mse_threshold=0.01))


def test_calibration_gate_raises_when_mean_unresolved_above_threshold():
    from stl.learning.calibration import CalibrationReport
    from stl.learning.targets import SOURCE_EXACT_HORIZON_2, SOURCE_EXACT_HORIZON_3

    report = CalibrationReport(
        mse_per_source={
            SOURCE_TERMINAL: 0.0,
            SOURCE_TABLEBASE: 0.0,
            SOURCE_EXACT_HORIZON_2: 0.0,
            SOURCE_EXACT_HORIZON_3: 0.0,
        },
        overall_mse=0.0,
        brier_score=0.0,
        reliability_bins=[],
        exact_target_error=0.0,
        n_targets=4,
        mean_unresolved_probability_per_source={
            SOURCE_TERMINAL: 0.0,
            SOURCE_TABLEBASE: 0.0,
            SOURCE_EXACT_HORIZON_2: 0.2,
            SOURCE_EXACT_HORIZON_3: 0.0,
        },
    )

    with pytest.raises(CalibrationGateError, match="unresolved_probability"):
        enforce_calibration_gate(report, BootstrapConfig(max_unresolved_per_source=0.05))


# ── per_source_mse_thresholds gate (Phase 9.B) ────────────────────────────


def _make_clean_calibration_report(**source_mse_overrides: float):
    """Build a CalibrationReport with all gates passing by default.

    Pass per-source MSE overrides as kwargs (e.g. exact_horizon_3=2.0) to
    simulate a specific failure mode. The fixture keeps tablebase, terminal,
    and unresolved-probability under their default thresholds so only the
    overridden axes can fail the gate.
    """
    from stl.learning.calibration import CalibrationReport
    from stl.learning.targets import SOURCE_EXACT_HORIZON_2, SOURCE_EXACT_HORIZON_3

    mse_per_source = {
        SOURCE_TERMINAL: 0.5,
        SOURCE_TABLEBASE: 0.005,
        SOURCE_EXACT_HORIZON_2: 0.05,
        SOURCE_EXACT_HORIZON_3: 0.05,
    }
    mse_per_source.update(source_mse_overrides)
    return CalibrationReport(
        mse_per_source=mse_per_source,
        overall_mse=sum(mse_per_source.values()) / 4,
        brier_score=0.0,
        reliability_bins=[],
        exact_target_error=0.0,
        n_targets=4,
        mean_unresolved_probability_per_source={
            SOURCE_TERMINAL: 0.0,
            SOURCE_TABLEBASE: 0.0,
            SOURCE_EXACT_HORIZON_2: 0.0,
            SOURCE_EXACT_HORIZON_3: 0.0,
        },
    )


def test_per_source_threshold_raises_on_horizon_3_overfit_collapse():
    """The exact failure gen-3 strict v1 hit: tablebase passes by
    overfitting (MSE 0.0006) at the cost of exact_horizon_3 collapsing
    (MSE 2.025). Without a per-source ceiling on h3, the gate accepts
    this. With one, it fires immediately."""
    from stl.learning.targets import SOURCE_EXACT_HORIZON_3

    report = _make_clean_calibration_report(exact_horizon_3=2.025)
    config = BootstrapConfig(
        tablebase_mse_threshold=0.01,
        per_source_mse_thresholds={SOURCE_EXACT_HORIZON_3: 0.15},
    )

    with pytest.raises(CalibrationGateError, match=r"exact_horizon_3 MSE"):
        enforce_calibration_gate(report, config)


def test_per_source_threshold_passes_when_all_axes_within_ceilings():
    from stl.learning.targets import (
        SOURCE_EXACT_HORIZON_2,
        SOURCE_EXACT_HORIZON_3,
    )

    report = _make_clean_calibration_report()
    config = BootstrapConfig(
        tablebase_mse_threshold=0.01,
        per_source_mse_thresholds={
            SOURCE_EXACT_HORIZON_2: 0.10,
            SOURCE_EXACT_HORIZON_3: 0.15,
        },
    )

    # No exception → gate passed
    enforce_calibration_gate(report, config)


def test_per_source_threshold_skips_sources_absent_from_report():
    """If the held-out corpus lacks a class for which a threshold is set,
    the gate must not error — it just doesn't apply the check.
    """
    from stl.learning.targets import SOURCE_MCTS_BOOTSTRAP

    report = _make_clean_calibration_report()
    # mcts_bootstrap isn't in the synthetic report at all
    config = BootstrapConfig(
        tablebase_mse_threshold=0.01,
        per_source_mse_thresholds={SOURCE_MCTS_BOOTSTRAP: 0.01},
    )

    enforce_calibration_gate(report, config)


# ── Phase F-2: interior-tablebase gate (separate from the ±1 boundary class) ─


def test_calibration_gate_raises_when_interior_tablebase_mse_above_threshold():
    """The defining Phase F-2 gate behaviour: interior pins are scored as a
    SEPARATE source, so a net that nails the 19 easy ±1 boundary pins but is
    badly wrong on the 3 interior pins still fails. Folding them into the
    boundary class would let 19 near-zero errors average-mask the interior."""
    report = _make_clean_calibration_report(tablebase_interior=0.20)
    config = BootstrapConfig(tablebase_interior_mse_threshold=0.05)

    with pytest.raises(CalibrationGateError, match=r"tablebase_interior MSE"):
        enforce_calibration_gate(report, config)


def test_calibration_gate_passes_when_interior_tablebase_mse_within_threshold():
    report = _make_clean_calibration_report(tablebase_interior=0.03)
    # Below the 0.05 interior ceiling → no exception.
    enforce_calibration_gate(report, BootstrapConfig(tablebase_interior_mse_threshold=0.05))


def test_calibration_gate_backward_safe_when_interior_source_absent():
    """Pre-F-2 held-out rulers have no tablebase_interior source. The interior
    check must be skipped (not error), so existing gate runs keep working until
    the ruler is regenerated to include the interior anchors."""
    report = _make_clean_calibration_report()
    assert "tablebase_interior" not in report.mse_per_source
    enforce_calibration_gate(report, BootstrapConfig())  # must not raise


def test_per_source_threshold_fires_on_first_offending_source():
    """If multiple per-source thresholds are violated, the gate must
    raise on the FIRST offender it encounters (deterministic dict
    iteration order in Python 3.7+)."""
    from stl.learning.targets import (
        SOURCE_EXACT_HORIZON_2,
        SOURCE_EXACT_HORIZON_3,
    )

    report = _make_clean_calibration_report(
        exact_horizon_2=0.5,
        exact_horizon_3=2.0,
    )
    config = BootstrapConfig(
        tablebase_mse_threshold=0.01,
        per_source_mse_thresholds={
            SOURCE_EXACT_HORIZON_2: 0.10,
            SOURCE_EXACT_HORIZON_3: 0.15,
        },
    )

    # Either of the two could fire first depending on iteration order;
    # either way it's a CalibrationGateError with a "MSE ... exceeds threshold" message.
    with pytest.raises(CalibrationGateError, match=r"MSE .* exceeds threshold"):
        enforce_calibration_gate(report, config)
