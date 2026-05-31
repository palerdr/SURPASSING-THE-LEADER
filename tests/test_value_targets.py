"""Tests for Phase 2: multi-horizon exact value-target generation.

Phase 2 dropped horizon=1 LP labels entirely. The label hierarchy is now:
    1. terminal      -> source="terminal"        horizon=0
    2. tablebase     -> source="tablebase"       horizon=0
    3. LSR-significant (rounds_until_leap_window <= 2 OR
       current_checker_fail_would_activate_lsr) -> source="exact_horizon_3"  horizon=3
    4. LSR-pressure (is_active_lsr OR any cylinder >= 240)
                                                  -> source="exact_horizon_2"  horizon=2
    5. otherwise -> excluded (label_state returns None,
       generate_targets skips).

These tests verify the routing decisions at a unit level, plus the
structural invariants of the corpus output (shape, dtype, save/load
round-trip).
"""

import os
import sys
import tempfile

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from environment.cfr.tactical_scenarios import forced_baku_overflow_death
from hal.value_net import FEATURE_DIM
from training.value_targets import (
    SOURCE_EXACT_HORIZON_2,
    SOURCE_EXACT_HORIZON_3,
    SOURCE_TABLEBASE,
    SOURCE_TERMINAL,
    VALID_SOURCES,
    LabelResult,
    ValueTarget,
    _build_game,
    _build_pinned_table,
    _is_lsr_pressure,
    _is_lsr_significant,
    generate_targets,
    label_state,
    load_targets,
    load_targets_as_records,
    load_rejected_pool,
    save_targets,
    source_breakdown,
)
import training.value_targets as vt


# ── State factories used by the tests ─────────────────────────────────────


def _terminal_game():
    """A game that has already ended with Hal as winner."""
    game = _build_game()
    game.game_over = True
    game.winner = game.player1
    game.loser = game.player2
    return game


def _tablebase_game():
    """A position that matches the forced_baku_overflow_death tablebase entry."""
    return _build_game(baku_cylinder=299.0)


def _lsr_significant_non_tablebase_game():
    """clock=720 half=2, Baku drops, Hal checks at cyl=299, cprs=1 (off-pin).

    - rounds_until_leap_window=11 (so the near-leap gate does NOT fire here),
      but current_checker_fail_would_activate_lsr=True at half=2 with these
      timings, which is enough to flag this state as LSR-significant.
    - hal_cylinder=299 forces every cell of the half-2 matrix to terminate,
      so horizon=3 collapses to a one-step LP that resolves in <100ms.
    - referee_cprs=1 differs from the pinned forced_hal_overflow_death entry
      (which has cprs=0), so the tablebase short-circuit does NOT fire.
    """
    return _build_game(
        clock=720.0,
        current_half=2,
        baku_cylinder=0.0,
        hal_cylinder=299.0,
        referee_cprs=1,
    )


def _lsr_pressure_non_significant_game():
    """clock=2000 half=1 baku=299 cprs=1: active LSR (var=2), not near leap.

    - is_active_lsr=True (clock=2000 -> minute 33 -> var 33%4+1=2).
    - rounds_until_leap_window=6, cflsr=False, so NOT LSR-significant.
    - baku_cylinder=299 forces every checker cell to terminate, so horizon=2
      collapses to a one-step LP that resolves in <100ms.
    """
    return _build_game(
        clock=2000.0,
        current_half=1,
        baku_cylinder=299.0,
        referee_cprs=1,
    )


def _excluded_game():
    """Default game (clock=720 half=1 cyls=0): no LSR signal in horizon."""
    return _build_game()


# ── Pinned-table loading ──────────────────────────────────────────────────


def test_pinned_table_contains_overflow_scenarios():
    """Phase F expanded the pinned table to 19 forced-terminal ±1 entries;
    Phase F-2 then added interior-valued pins (|value| < 1).

    The table must contain BOTH boundary signs (±1, the forced overflows) AND at
    least one genuinely interior value — the all-±1 invariant the old assertion
    encoded is exactly what F-2 set out to break.
    """
    table = _build_pinned_table()
    values = set(table.values())
    assert len(table) >= 22, f"expected >= 22 pinned entries after Phase F-2, got {len(table)}"
    # Boundary pins: both signs present.
    assert 1.0 in values
    assert -1.0 in values
    # Phase F-2: at least one genuinely interior pinned value (no longer all ±1).
    interior = [v for v in values if abs(v) < 1.0 - 1e-9]
    assert interior, "no interior-valued pins in the table; Phase F-2 regressed"


# ── Gate predicates (sanity) ──────────────────────────────────────────────


def test_default_state_is_not_lsr_significant_or_pressure():
    game = _excluded_game()
    assert _is_lsr_significant(game) is False
    assert _is_lsr_pressure(game) is False


def test_lsr_significant_fixture_actually_is_significant():
    game = _lsr_significant_non_tablebase_game()
    assert _is_lsr_significant(game) is True


def test_lsr_pressure_fixture_is_pressure_but_not_significant():
    game = _lsr_pressure_non_significant_game()
    assert _is_lsr_significant(game) is False
    assert _is_lsr_pressure(game) is True


# ── label_state routing ───────────────────────────────────────────────────


def test_label_state_terminal_returns_terminal_source():
    value, source, horizon = label_state(_terminal_game())
    assert value == 1.0
    assert source == SOURCE_TERMINAL
    assert horizon == 0


def test_label_state_tablebase_match_returns_tablebase_source():
    scenario = forced_baku_overflow_death()
    result = label_state(scenario.game)
    assert result is not None
    value, source, horizon = result
    assert value == 1.0
    assert source == SOURCE_TABLEBASE
    assert horizon == 0


def test_label_state_lsr_significant_uses_exact_horizon_3():
    result = label_state(_lsr_significant_non_tablebase_game())
    assert result is not None
    value, source, horizon = result
    assert source == SOURCE_EXACT_HORIZON_3
    assert horizon == 3
    assert -1.0 <= value <= 1.0


def test_label_state_lsr_pressure_uses_exact_horizon_2():
    result = label_state(_lsr_pressure_non_significant_game())
    assert result is not None
    value, source, horizon = result
    assert source == SOURCE_EXACT_HORIZON_2
    assert horizon == 2
    assert -1.0 <= value <= 1.0


def test_label_state_non_lsr_returns_none():
    """Non-terminal, non-tablebase, no LSR signal -> excluded."""
    assert label_state(_excluded_game()) is None


def test_label_state_never_emits_horizon_1_source():
    """Horizon=1 LP labels are dropped from the corpus entirely."""
    fixtures = [
        _terminal_game(),
        _tablebase_game(),
        _lsr_significant_non_tablebase_game(),
        _lsr_pressure_non_significant_game(),
    ]
    for game in fixtures:
        result = label_state(game)
        assert result is not None
        _, source, _ = result
        assert source != "exact_horizon_1"
        assert source != "exact"
        assert source in VALID_SOURCES


# ── generate_targets corpus ───────────────────────────────────────────────


def _tiny_grids() -> dict:
    """Minimal grid: one cyl=0 (excluded) + one cyl=299 (tablebase)."""
    return dict(
        baku_cylinder_grid=(0.0, 299.0),
        hal_cylinder_grid=(0.0,),
        clock_grid=(720.0,),
        half_grid=(1,),
        deaths_grid=(0,),
        cpr_grid=(0,),
    )


def _h2_only_grid() -> dict:
    """Single-state grid that emits exactly one horizon=2 label.

    clock=2000 half=1 baku=299 cprs=1: active LSR, not near leap, and
    forced-terminal so horizon=2 resolves quickly.
    """
    return dict(
        baku_cylinder_grid=(299.0,),
        hal_cylinder_grid=(0.0,),
        clock_grid=(2000.0,),
        half_grid=(1,),
        deaths_grid=(0,),
        cpr_grid=(1,),
    )


def _h3_only_grid() -> dict:
    """Single-state grid that emits exactly one horizon=3 label.

    clock=720 half=2 hal=299 cprs=1: cflsr=True (LSR-significant),
    forced-terminal so horizon=3 resolves quickly.
    """
    return dict(
        baku_cylinder_grid=(0.0,),
        hal_cylinder_grid=(299.0,),
        clock_grid=(720.0,),
        half_grid=(2,),
        deaths_grid=(0,),
        cpr_grid=(1,),
    )


def test_generate_targets_skips_excluded_states():
    """Default state alone yields zero targets; cyl=299 alone yields one."""
    targets_excluded_only = generate_targets(
        baku_cylinder_grid=(0.0,),
        hal_cylinder_grid=(0.0,),
        clock_grid=(720.0,),
        half_grid=(1,),
        deaths_grid=(0,),
        cpr_grid=(0,),
    )
    assert targets_excluded_only == []

    targets_tablebase = generate_targets(**_tiny_grids())
    # 2 candidate states, 1 excluded, 1 tablebase emit
    assert len(targets_tablebase) == 1
    assert targets_tablebase[0].source == SOURCE_TABLEBASE


def test_generate_targets_features_have_correct_shape_and_dtype():
    targets = generate_targets(**_tiny_grids())
    assert targets, "tiny grid should still produce at least one target"
    for t in targets:
        assert isinstance(t, ValueTarget)
        assert t.features.shape == (FEATURE_DIM,)
        assert t.features.dtype == np.float32


def test_generate_targets_values_in_unit_interval():
    targets = generate_targets(**_tiny_grids())
    for t in targets:
        assert -1.0 <= t.value <= 1.0


def test_generate_targets_sources_are_all_valid():
    """Every emitted target uses one of the four whitelisted sources."""
    targets = generate_targets(**_tiny_grids())
    for t in targets:
        assert t.source in VALID_SOURCES


def test_generate_targets_no_horizon_1_emitted():
    """Phase 2 gate: horizon=1 LP labels are gone."""
    targets = generate_targets(**_tiny_grids())
    for t in targets:
        assert t.horizon != 1
        assert t.source != "exact_horizon_1"
        assert t.source != "exact"


def test_generate_targets_picks_up_tablebase_source_on_match():
    targets = generate_targets(**_tiny_grids())
    sources = {t.source for t in targets}
    assert SOURCE_TABLEBASE in sources


def test_generate_targets_emits_exact_horizon_2_label():
    targets = generate_targets(**_h2_only_grid())
    assert len(targets) == 1
    target = targets[0]
    assert target.source == SOURCE_EXACT_HORIZON_2
    assert target.horizon == 2


def test_value_target_records_unresolved_probability_for_horizon_2(monkeypatch):
    class FakeResult:
        value_for_hal = 0.25
        unresolved_probability = 0.25
        drop_actions = (1, 60)
        check_actions = (1, 60)
        dropper_strategy = np.array([0.75, 0.25])
        checker_strategy = np.array([0.4, 0.6])

    monkeypatch.setattr(vt, "solve_exact_finite_horizon", lambda *args, **kwargs: FakeResult())

    result = label_state(_lsr_pressure_non_significant_game())
    assert result is not None
    assert result.source == SOURCE_EXACT_HORIZON_2
    assert result.unresolved_probability == pytest.approx(0.25)
    assert result.dropper_dist[0] == pytest.approx(0.75)
    assert result.checker_dist[59] == pytest.approx(0.6)


def test_high_unresolved_label_excluded_from_corpus(monkeypatch, tmp_path):
    class FakeResult:
        value_for_hal = 0.0
        unresolved_probability = 0.75
        drop_actions = (1,)
        check_actions = (1,)
        dropper_strategy = np.array([1.0])
        checker_strategy = np.array([1.0])

    monkeypatch.setattr(vt, "solve_exact_finite_horizon", lambda *args, **kwargs: FakeResult())
    rejected_path = tmp_path / "targets.rejected.npz"

    targets = generate_targets(
        **_h2_only_grid(),
        rejected_pool_path=rejected_path,
    )

    assert targets == []
    rejected = load_rejected_pool(rejected_path)
    assert len(rejected) == 1
    assert "game_clock" in rejected[0]


def test_generate_targets_explicit_death_axes_include_asymmetric_states(monkeypatch):
    zero = np.zeros(61, dtype=np.float32)

    def fake_label_state(*args, **kwargs):
        return LabelResult(
            value=0.0,
            source=SOURCE_TERMINAL,
            horizon=0,
            unresolved_probability=0.0,
            dropper_dist=zero,
            checker_dist=zero,
            dropper_legal_mask=zero,
            checker_legal_mask=zero,
        )

    monkeypatch.setattr(vt, "label_state", fake_label_state)
    targets = generate_targets(
        baku_cylinder_grid=(0.0,),
        hal_cylinder_grid=(0.0,),
        clock_grid=(720.0,),
        half_grid=(1,),
        baku_deaths_grid=(0, 1),
        hal_deaths_grid=(0, 1),
        cpr_grid=(0,),
    )

    assert len(targets) == 4
    death_feature_pairs = {(float(t.features[5]), float(t.features[4])) for t in targets}
    assert death_feature_pairs == {
        (0.0, 0.0),
        (0.25, 0.0),
        (0.0, 0.25),
        (0.25, 0.25),
    }


def test_generate_targets_legacy_deaths_grid_remains_symmetric(monkeypatch):
    zero = np.zeros(61, dtype=np.float32)

    def fake_label_state(*args, **kwargs):
        return LabelResult(
            value=0.0,
            source=SOURCE_TERMINAL,
            horizon=0,
            unresolved_probability=0.0,
            dropper_dist=zero,
            checker_dist=zero,
            dropper_legal_mask=zero,
            checker_legal_mask=zero,
        )

    monkeypatch.setattr(vt, "label_state", fake_label_state)
    targets = generate_targets(
        baku_cylinder_grid=(0.0,),
        hal_cylinder_grid=(0.0,),
        clock_grid=(720.0,),
        half_grid=(1,),
        deaths_grid=(0, 1),
        cpr_grid=(0,),
    )

    assert len(targets) == 2
    death_feature_pairs = {(float(t.features[5]), float(t.features[4])) for t in targets}
    assert death_feature_pairs == {(0.0, 0.0), (0.25, 0.25)}


def test_generate_targets_rejects_mixed_legacy_and_explicit_death_axes():
    with pytest.raises(ValueError):
        generate_targets(
            baku_cylinder_grid=(0.0,),
            hal_cylinder_grid=(0.0,),
            clock_grid=(720.0,),
            half_grid=(1,),
            deaths_grid=(0,),
            baku_deaths_grid=(0,),
            cpr_grid=(0,),
        )


def test_generate_targets_emits_exact_horizon_3_label():
    targets = generate_targets(**_h3_only_grid())
    assert len(targets) == 1
    target = targets[0]
    assert target.source == SOURCE_EXACT_HORIZON_3
    assert target.horizon == 3


def test_save_load_targets_round_trip():
    targets = generate_targets(**_tiny_grids())
    assert targets
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


def test_load_targets_as_records_round_trip_preserves_all_fields():
    """load_targets_as_records must reconstruct ValueTarget records bit-identical
    to the saved set, including policy distributions, legal masks, and
    unresolved_probability. This is what bootstrap_loop's --held-out-targets CLI
    consumes; any silent dimension mismatch would make calibration gates
    score against zeroed policy targets."""
    targets = generate_targets(**_tiny_grids())
    assert targets
    with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f:
        path = f.name
    try:
        save_targets(targets, path)
        loaded = load_targets_as_records(path)
        assert len(loaded) == len(targets)
        for orig, copy in zip(targets, loaded):
            assert orig.source == copy.source
            assert orig.horizon == copy.horizon
            assert orig.value == pytest.approx(copy.value)
            assert orig.unresolved_probability == pytest.approx(copy.unresolved_probability)
            np.testing.assert_array_equal(orig.features, copy.features)
            np.testing.assert_array_equal(orig.dropper_dist, copy.dropper_dist)
            np.testing.assert_array_equal(orig.checker_dist, copy.checker_dist)
            np.testing.assert_array_equal(orig.dropper_legal_mask, copy.dropper_legal_mask)
            np.testing.assert_array_equal(orig.checker_legal_mask, copy.checker_legal_mask)
    finally:
        os.unlink(path)


def test_source_breakdown_counts_match_target_total():
    targets = generate_targets(**_tiny_grids())
    breakdown = source_breakdown(targets)
    assert sum(breakdown.values()) == len(targets)


def test_source_breakdown_keys_are_valid_sources():
    targets = generate_targets(**_tiny_grids())
    breakdown = source_breakdown(targets)
    for source in breakdown.keys():
        assert source in VALID_SOURCES


# ── Phase F-2: held-out-only interior-pin source split ─────────────────────


def test_holdout_tablebase_targets_split_interior_into_separate_source():
    """On the held-out path, interior-valued pins are labeled
    SOURCE_TABLEBASE_INTERIOR so the calibration gate scores them as a distinct
    class; the ±1 boundary pins stay SOURCE_TABLEBASE."""
    targets = vt._generate_tablebase_targets(None, split_interior=True)
    by_source: dict[str, list] = {}
    for t in targets:
        by_source.setdefault(t.source, []).append(t)

    assert vt.SOURCE_TABLEBASE_INTERIOR in by_source, "interior pins were not split out"
    interior = by_source[vt.SOURCE_TABLEBASE_INTERIOR]
    assert len(interior) == 3
    assert all(abs(t.value) < 1.0 for t in interior), "an interior pin sits on the ±1 boundary"
    # Boundary pins remain and are all exactly ±1.
    boundary = by_source[SOURCE_TABLEBASE]
    assert boundary
    assert all(abs(t.value) == pytest.approx(1.0) for t in boundary)


def test_training_tablebase_targets_keep_interior_as_tablebase():
    """The split is held-out-ONLY. Without ``split_interior`` the interior pins
    stay SOURCE_TABLEBASE, so the Phase-G training corpus shape is unchanged."""
    targets = vt._generate_tablebase_targets(None)
    assert all(t.source != vt.SOURCE_TABLEBASE_INTERIOR for t in targets)
    assert any(t.source == SOURCE_TABLEBASE for t in targets)


def test_generate_targets_parallel_matches_serial():
    """workers=2 must produce bit-identical targets to workers=1 on the same grid.

    Pool.map preserves input ordering, _label_one_state is deterministic given
    its inputs, and the worker globals are set identically by ``_worker_init``.
    Any divergence here points at a stateful bug — non-determinism in
    ``solve_exact_finite_horizon``, a non-pickleable default, or a thread-unsafe
    cache somewhere in the call stack.
    """
    grids = dict(
        baku_cylinder_grid=(0.0, 240.0, 290.0, 299.0),
        hal_cylinder_grid=(0.0,),
        clock_grid=(720.0,),
        half_grid=(1,),
        deaths_grid=(0,),
        cpr_grid=(0,),
    )
    serial = generate_targets(workers=1, **grids)
    parallel = generate_targets(workers=2, **grids)

    assert len(serial) == len(parallel), (
        f"corpus length differs: serial={len(serial)} parallel={len(parallel)}"
    )
    for s, p in zip(serial, parallel):
        assert s.source == p.source
        assert s.horizon == p.horizon
        assert s.value == pytest.approx(p.value)
        assert s.unresolved_probability == pytest.approx(p.unresolved_probability)
        np.testing.assert_array_equal(s.features, p.features)
        np.testing.assert_array_equal(s.dropper_dist, p.dropper_dist)
        np.testing.assert_array_equal(s.checker_dist, p.checker_dist)
        np.testing.assert_array_equal(s.dropper_legal_mask, p.dropper_legal_mask)
        np.testing.assert_array_equal(s.checker_legal_mask, p.checker_legal_mask)
