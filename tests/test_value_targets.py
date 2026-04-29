"""Tests for Phase 5 Step 1: value-net training-target generation."""

import os
import sys
import tempfile

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from environment.cfr.tactical_scenarios import forced_baku_overflow_death
from hal.value_net import FEATURE_DIM
from training.value_targets import (
    ValueTarget,
    _build_game,
    _build_pinned_table,
    generate_targets,
    label_state,
    load_targets,
    save_targets,
    source_breakdown,
)


def _tiny_grids() -> dict:
    """Minimal grid for fast tests: 2 states."""
    return dict(
        baku_cylinder_grid=(0.0, 299.0),
        hal_cylinder_grid=(0.0,),
        clock_grid=(720.0,),
        half_grid=(1,),
        deaths_grid=(0,),
        cpr_grid=(0,),
    )


def test_pinned_table_contains_two_overflow_scenarios():
    table = _build_pinned_table()
    assert len(table) == 2
    assert 1.0 in table.values()
    assert -1.0 in table.values()


def test_label_state_terminal_returns_terminal_source():
    game = _build_game()
    game.game_over = True
    game.winner = game.player1
    game.loser = game.player2

    value, source = label_state(game, horizon=1)
    assert value == 1.0
    assert source == "terminal"


def test_label_state_tablebase_match_returns_tablebase_source():
    scenario = forced_baku_overflow_death()
    value, source = label_state(scenario.game, horizon=1)
    assert value == 1.0
    assert source == "tablebase"


def test_label_state_arbitrary_position_uses_exact_solver():
    # A random non-terminal, non-pinned state. Should fall through to
    # solve_exact_finite_horizon and report source="exact".
    game = _build_game(baku_cylinder=120.0)
    value, source = label_state(game, horizon=1)
    assert source == "exact"
    assert -1.0 <= value <= 1.0


def test_generate_targets_tiny_grid_yields_expected_count():
    targets = generate_targets(**_tiny_grids(), horizon=1)
    assert len(targets) == 2  # 2 baku cylinders × 1 × 1 × 1 × 1 × 1


def test_generate_targets_features_have_correct_shape_and_dtype():
    targets = generate_targets(**_tiny_grids(), horizon=1)
    for t in targets:
        assert isinstance(t, ValueTarget)
        assert t.features.shape == (FEATURE_DIM,)
        assert t.features.dtype == np.float32


def test_generate_targets_values_in_unit_interval():
    targets = generate_targets(**_tiny_grids(), horizon=1)
    for t in targets:
        assert -1.0 <= t.value <= 1.0


def test_generate_targets_horizon_field_recorded():
    targets = generate_targets(**_tiny_grids(), horizon=1)
    for t in targets:
        assert t.horizon == 1


def test_generate_targets_picks_up_tablebase_source_on_match():
    # The cyl=299 state in the tiny grid matches forced_baku_overflow_death.
    targets = generate_targets(**_tiny_grids(), horizon=1)
    sources = {t.source for t in targets}
    assert "tablebase" in sources


def test_save_load_targets_round_trip():
    targets = generate_targets(**_tiny_grids(), horizon=1)
    with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f:
        path = f.name
    try:
        save_targets(targets, path)
        X, y = load_targets(path)
        assert X.shape == (len(targets), FEATURE_DIM)
        assert y.shape == (len(targets),)
        np.testing.assert_array_equal(X, np.stack([t.features for t in targets]))
        np.testing.assert_array_equal(y, np.array([t.value for t in targets], dtype=np.float32))
    finally:
        os.unlink(path)


def test_source_breakdown_counts_match_target_total():
    targets = generate_targets(**_tiny_grids(), horizon=1)
    breakdown = source_breakdown(targets)
    assert sum(breakdown.values()) == len(targets)


def test_generate_targets_horizon_param_threads_through():
    # Use a tablebase-pinned state so the horizon parameter doesn't trigger
    # a slow exact solve — it short-circuits via the pinned table. The point
    # of this test is just that horizon flows through to the ValueTarget.
    tiny = dict(_tiny_grids())
    tiny["baku_cylinder_grid"] = (299.0,)  # forced_baku_overflow_death
    targets = generate_targets(**tiny, horizon=2)
    assert len(targets) == 1
    assert targets[0].horizon == 2
    assert targets[0].source == "tablebase"
    assert targets[0].value == 1.0
