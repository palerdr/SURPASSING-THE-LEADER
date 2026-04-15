"""Tests for full round CFR (two chained half-rounds)."""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from environment.cfr.full_round import resolve_half_round, simulate_round
from src.Constants import CYLINDER_MAX, FAILED_CHECK_PENALTY


# ── resolve_half_round ───────────────────────────────────────────────────


class TestResolveHalfRound:
    def test_successful_check_min_st(self):
        """drop=30, check=30 → success, ST=1 (Yakou minimum)."""
        r = resolve_half_round(0.0, drop_time=30, check_time=30)
        assert r["success"] is True
        assert r["st"] == 1
        assert r["new_cylinder"] == 1.0
        assert r["injection"] is False

    def test_successful_check_with_st(self):
        """drop=10, check=45 → ST=35."""
        r = resolve_half_round(0.0, drop_time=10, check_time=45)
        assert r["success"] is True
        assert r["st"] == 35
        assert r["new_cylinder"] == 35.0
        assert r["injection"] is False

    def test_successful_check_accumulates_on_cylinder(self):
        """cyl=100, drop=10, check=45 → ST=35, new_cyl=135."""
        r = resolve_half_round(100.0, drop_time=10, check_time=45)
        assert r["new_cylinder"] == 135.0
        assert r["injection"] is False

    def test_cylinder_overflow_triggers_injection(self):
        """cyl=280, drop=1, check=30 → ST=29, cyl+ST=309 >= 300."""
        r = resolve_half_round(280.0, drop_time=1, check_time=30)
        assert r["success"] is True
        assert r["injection"] is True
        assert r["injection_amount"] == CYLINDER_MAX

    def test_failed_check(self):
        """drop=30, check=20 → failed."""
        r = resolve_half_round(90.0, drop_time=30, check_time=20)
        assert r["success"] is False
        assert r["injection"] is True
        assert r["injection_amount"] == 90.0 + FAILED_CHECK_PENALTY

    def test_failed_check_zero_cylinder(self):
        """cyl=0, drop=60, check=1 → injection = 0 + 60 = 60."""
        r = resolve_half_round(0.0, drop_time=60, check_time=1)
        assert r["success"] is False
        assert r["injection_amount"] == FAILED_CHECK_PENALTY

    def test_check_at_drop_is_success(self):
        """Boundary: check_time == drop_time → success with ST=1."""
        r = resolve_half_round(0.0, drop_time=1, check_time=1)
        assert r["success"] is True
        assert r["st"] == 1

    def test_check_one_before_drop_is_failure(self):
        """Boundary: check_time == drop_time - 1 → failure."""
        r = resolve_half_round(0.0, drop_time=2, check_time=1)
        assert r["success"] is False


# ── simulate_round ───────────────────────────────────────────────────────


class TestSimulateRound:
    def test_returns_all_keys(self):
        result = simulate_round(0.0, 0.0, 0, 0, round_num=0, iterations=500)
        assert "h1_dropper_strat" in result
        assert "h1_checker_strat" in result
        assert "h1_game_value" in result
        assert "h2_dropper_strat" in result
        assert "h2_checker_strat" in result
        assert "h2_game_value" in result

    def test_strategies_are_valid_distributions(self):
        result = simulate_round(0.0, 0.0, 0, 0, round_num=0, iterations=500)
        for key in ["h1_dropper_strat", "h1_checker_strat",
                     "h2_dropper_strat", "h2_checker_strat"]:
            strat = result[key]
            assert strat.shape == (60,)
            np.testing.assert_almost_equal(strat.sum(), 1.0)
            assert np.all(strat >= 0)

    def test_half1_and_half2_are_independent(self):
        """With different cylinders, the two halves should produce different strategies."""
        result = simulate_round(0.0, 200.0, 0, 0, round_num=0, iterations=1000)
        # P2 checks in half 1 (cyl=200), P1 checks in half 2 (cyl=0)
        # These should be different since the checker's cylinder differs
        assert not np.allclose(result["h1_checker_strat"], result["h2_checker_strat"])

    def test_symmetric_cylinders_produce_similar_strategies(self):
        """Same cylinder for both → halves should have similar strategies."""
        result = simulate_round(100.0, 100.0, 0, 0, round_num=0, iterations=2000)
        np.testing.assert_allclose(
            result["h1_dropper_strat"], result["h2_dropper_strat"], atol=0.05)
        np.testing.assert_allclose(
            result["h1_checker_strat"], result["h2_checker_strat"], atol=0.05)

    def test_game_values_are_negative(self):
        """Checker always pays — both game values should be <= 0."""
        result = simulate_round(0.0, 0.0, 0, 0, round_num=0, iterations=1000)
        assert result["h1_game_value"] <= 0
        assert result["h2_game_value"] <= 0

    def test_high_cylinder_worse_game_value(self):
        """Higher checker cylinder → worse game value for that half's checker."""
        result = simulate_round(0.0, 200.0, 0, 0, round_num=0, iterations=2000)
        # Half 1: P2 checks with cyl=200 (bad). Half 2: P1 checks with cyl=0 (better).
        assert result["h1_game_value"] < result["h2_game_value"]
