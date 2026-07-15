"""Epoch-sweep solver (plan Phase 3, ticket 15 machinery).

The historical Tier-0 pilot artifact was generated under the old minimum-ST
rule and is intentionally not a migration target. These tests assert corrected
post-reset invariants directly: ST=0 creates a local role-swap cycle, fatal
fail bands have exact intervals, and survivable fails preserve brackets.
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.getcwd())

from stl.solver.tablebase import EpochSpec, solve_epoch, survival_table
from stl.solver.tablebase import CYL, bracket_survive_value


def region(arr: np.ndarray, lo: int) -> np.ndarray:
    return arr[:, lo:, lo:]


def test_forced_fatal_epoch_solves_diagonal_zero_cycle():
    """All-deaths-fatal cap state has only terminal off-diagonal outcomes."""
    zeros = np.zeros(CYL)
    spec = EpochSpec(ttd_hal=0.0, ttd_baku=0.0, cprs=0)
    V_lo, V_hi = solve_epoch(
        spec,
        bracket_survive_value(),
        survival_overrides=(zeros, zeros),
        min_cyl=298,
    )

    expected = 59.0 / 61.0
    np.testing.assert_allclose(region(V_lo, 298), region(V_hi, 298), atol=1e-10)
    assert V_lo[0, 299, 299] == pytest.approx(expected, abs=1e-9)
    assert V_lo[1, 299, 299] == pytest.approx(-expected, abs=1e-9)


def test_real_epoch_high_cap_band_is_exact_when_fails_are_fatal():
    """At both-cylinders >= 280 the fail duration caps at 300 => p = 0 from
    the engine referee itself, so a REAL epoch is exact there even with wide
    brackets at survived-death exits."""
    spec = EpochSpec(ttd_hal=120.0, ttd_baku=60.0, cprs=3)
    V_lo, V_hi = solve_epoch(spec, bracket_survive_value(), min_cyl=298)

    np.testing.assert_allclose(region(V_lo, 298), region(V_hi, 298), atol=1e-10)
    assert V_lo[0, 299, 299] == pytest.approx(59.0 / 61.0, abs=1e-9)
    assert V_lo[1, 299, 299] == pytest.approx(-59.0 / 61.0, abs=1e-9)


def test_intervals_ordered_and_bounded():
    p_hal = np.zeros(CYL)
    p_baku = np.zeros(CYL)
    p_hal[298] = 0.25
    p_baku[298] = 0.25
    spec = EpochSpec(ttd_hal=60.0, ttd_baku=0.0, cprs=1)
    V_lo, V_hi = solve_epoch(
        spec,
        bracket_survive_value(),
        survival_overrides=(p_hal, p_baku),
        min_cyl=298,
    )

    lo = region(V_lo, 298)
    hi = region(V_hi, 298)
    assert np.all(lo <= hi + 1e-12)
    assert np.all(lo >= -1.0 - 1e-12)
    assert np.all(hi <= 1.0 + 1e-12)
    assert np.nanmax(hi[:, 1:, 1:] - lo[:, 1:, 1:]) == 0.0
    assert np.nanmax(hi - lo) > 0.01


def test_survival_table_is_engine_derived_and_monotone():
    table = survival_table(name="Hal", ttd=120.0, cprs=2)
    assert table.shape == (CYL,)
    assert np.all(np.diff(table) <= 1e-12)  # longer death, lower survival
    assert table[240] == 0.0  # duration cap => exact zero
    assert table[0] > 0.5


def test_deterministic():
    spec = EpochSpec(ttd_hal=90.0, ttd_baku=0.0, cprs=1)
    a_lo, a_hi = solve_epoch(spec, bracket_survive_value(), min_cyl=298)
    b_lo, b_hi = solve_epoch(spec, bracket_survive_value(), min_cyl=298)
    np.testing.assert_array_equal(a_lo[:, 298:, 298:], b_lo[:, 298:, 298:])
    np.testing.assert_array_equal(a_hi[:, 298:, 298:], b_hi[:, 298:, 298:])
