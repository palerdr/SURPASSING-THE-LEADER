"""Epoch-sweep solver (plan Phase 3, ticket 15 machinery).

The decisive test: with survival forced to zero, an epoch IS the Tier-0
limit game, and the sweep must reproduce the pilot's table exactly. The
high-cylinder region (both >= 240) has p = 0 naturally, so REAL epochs
must also match the limit game there regardless of ttd/cprs.
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training.tablebase import EpochSpec, solve_epoch, survival_table
from training.tablebase.epoch_sweep import CYL, bracket_survive_value

LIMIT_GAME = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "checkpoints",
    "tablebase_pilot",
    "limit_game_v.npy",
)


def region(arr: np.ndarray, lo: int) -> np.ndarray:
    return arr[:, lo:, lo:]


def test_forced_fatal_epoch_equals_limit_game_region():
    """All-deaths-fatal degenerate case == Tier-0 limit game, exactly."""
    if not os.path.exists(LIMIT_GAME):
        pytest.skip("pilot limit-game artifact absent")
    limit = np.load(LIMIT_GAME)

    zeros = np.zeros(CYL)
    spec = EpochSpec(ttd_hal=0.0, ttd_baku=0.0, cprs=0)
    V_lo, V_hi = solve_epoch(
        spec,
        bracket_survive_value(),
        survival_overrides=(zeros, zeros),
        min_cyl=280,
    )

    np.testing.assert_array_equal(region(V_lo, 280), region(limit, 280))
    np.testing.assert_array_equal(region(V_hi, 280), region(limit, 280))


def test_real_epoch_matches_limit_game_where_fails_are_fatal():
    """At both-cylinders >= 280 the fail duration caps at 300 => p = 0 from
    the engine referee itself, so a REAL epoch equals the limit game there
    even with wide brackets at survived-death exits."""
    if not os.path.exists(LIMIT_GAME):
        pytest.skip("pilot limit-game artifact absent")
    limit = np.load(LIMIT_GAME)

    spec = EpochSpec(ttd_hal=120.0, ttd_baku=60.0, cprs=3)
    V_lo, V_hi = solve_epoch(spec, bracket_survive_value(), min_cyl=280)

    np.testing.assert_array_equal(region(V_lo, 280), region(limit, 280))
    np.testing.assert_array_equal(region(V_hi, 280), region(limit, 280))


def test_intervals_ordered_and_bounded():
    spec = EpochSpec(ttd_hal=60.0, ttd_baku=0.0, cprs=1)
    V_lo, V_hi = solve_epoch(spec, bracket_survive_value(), min_cyl=230)

    lo = region(V_lo, 230)
    hi = region(V_hi, 230)
    assert np.all(lo <= hi + 1e-12)
    assert np.all(lo >= -1.0 - 1e-12)
    assert np.all(hi <= 1.0 + 1e-12)
    # The p=0 band (cylinders >= 240, here region indices >= 240-230=10 on
    # both axes... use 20 for margin) must be exact (width 0); the
    # survivable-fail band (230-239) must carry real interval width.
    band_exact = hi[:, 20:, 20:] - lo[:, 20:, 20:]
    assert np.nanmax(band_exact) == 0.0
    band_survivable = hi[:, :10, :10] - lo[:, :10, :10]
    # Fail-survival probability at cyl 230 is ~7%, so the [-1,1] frontier
    # bracket shows up as a small-but-real width (~0.04 at the corner).
    assert np.nanmax(band_survivable) > 0.01


def test_survival_table_is_engine_derived_and_monotone():
    table = survival_table(name="Hal", ttd=120.0, cprs=2)
    assert table.shape == (CYL,)
    assert np.all(np.diff(table) <= 1e-12)  # longer death, lower survival
    assert table[240] == 0.0  # duration cap => exact zero
    assert table[0] > 0.5


def test_deterministic():
    spec = EpochSpec(ttd_hal=90.0, ttd_baku=0.0, cprs=1)
    a_lo, a_hi = solve_epoch(spec, bracket_survive_value(), min_cyl=290)
    b_lo, b_hi = solve_epoch(spec, bracket_survive_value(), min_cyl=290)
    np.testing.assert_array_equal(a_lo[:, 290:, 290:], b_lo[:, 290:, 290:])
    np.testing.assert_array_equal(a_hi[:, 290:, 290:], b_hi[:, 290:, 290:])
