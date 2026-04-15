"""Tests for single half-round CFR solver."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from environment.cfr.half_round import regret_match, compute_payoff_matrix, solve_half_round
from src.Constants import FAILED_CHECK_PENALTY, CYLINDER_MAX


# ── regret_match ─────────────────────────────────────────────────────────


class TestRegretMatch:
    def test_positive_regrets_normalize(self):
        strat = regret_match(np.array([5.0, -2.0, 3.0]))
        assert strat.shape == (3,)
        np.testing.assert_almost_equal(strat.sum(), 1.0)
        np.testing.assert_almost_equal(strat, [5/8, 0, 3/8])

    def test_all_negative_returns_uniform(self):
        strat = regret_match(np.array([-1.0, -5.0, -3.0]))
        np.testing.assert_almost_equal(strat, [1/3, 1/3, 1/3])

    def test_all_zero_returns_uniform(self):
        strat = regret_match(np.zeros(60))
        np.testing.assert_almost_equal(strat, np.ones(60) / 60)

    def test_single_positive(self):
        strat = regret_match(np.array([0.0, 0.0, 7.0, 0.0]))
        np.testing.assert_almost_equal(strat, [0, 0, 1, 0])

    def test_output_sums_to_one(self):
        rng = np.random.RandomState(42)
        for _ in range(20):
            regrets = rng.randn(60)
            strat = regret_match(regrets)
            np.testing.assert_almost_equal(strat.sum(), 1.0)
            assert np.all(strat >= 0)


# ── compute_payoff_matrix ────────────────────────────────────────────────


class TestPayoffMatrix:
    def test_shape(self):
        m = compute_payoff_matrix(0.0, turn_duration=60)
        assert m.shape == (60, 60)

    def test_shape_leap(self):
        m = compute_payoff_matrix(0.0, turn_duration=61)
        assert m.shape == (61, 61)

    def test_same_second_min_st(self):
        """drop=30, check=30 → successful check, ST=1 (Yakou minimum)."""
        m = compute_payoff_matrix(0.0)
        assert m[29][29] == -1.0  # 0-indexed: second 30

    def test_successful_check_st(self):
        """drop=10, check=45 → ST=35, payoff=-35."""
        m = compute_payoff_matrix(0.0)
        assert m[9][44] == -35.0

    def test_failed_check_penalty(self):
        """drop=30, check=20 → failed, payoff=-(cyl + 60)."""
        m = compute_payoff_matrix(90.0)
        assert m[29][19] == -(90.0 + FAILED_CHECK_PENALTY)

    def test_cylinder_overflow(self):
        """cyl=280, drop=1, check=30 → ST=29, cyl+ST=309 >= 300 → payoff=-300."""
        m = compute_payoff_matrix(280.0)
        assert m[0][29] == -CYLINDER_MAX

    def test_zero_cylinder_failed_check(self):
        """cyl=0, failed check → payoff = -(0 + 60) = -60."""
        m = compute_payoff_matrix(0.0)
        assert m[59][0] == -60.0  # drop=60, check=1 → failed

    def test_diagonal_is_min_st(self):
        """When drop_time == check_time, ST=1 (Yakou minimum), payoff=-1."""
        m = compute_payoff_matrix(0.0)
        for i in range(60):
            assert m[i][i] == -1.0

    def test_upper_triangle_is_successful(self):
        """check > drop → successful check, payoffs are -ST (<=0)."""
        m = compute_payoff_matrix(0.0)
        for d in range(60):
            for c in range(d + 1, 60):
                assert m[d][c] <= 0

    def test_lower_triangle_is_failed(self):
        """check < drop → failed check, all have same payoff for given cylinder."""
        m = compute_payoff_matrix(50.0)
        expected = -(50.0 + FAILED_CHECK_PENALTY)
        for d in range(60):
            for c in range(d):
                assert m[d][c] == expected

    def test_zero_sum(self):
        """Dropper payoff = -Checker payoff. Matrix should have no positive entries
        (checker always pays something or breaks even)."""
        m = compute_payoff_matrix(0.0)
        assert np.all(m <= 0)


# ── solve_half_round ─────────────────────────────────────────────────────


class TestSolveHalfRound:
    def test_returns_valid_distributions(self):
        d_strat, c_strat, gv = solve_half_round(0.0, iterations=1000)
        assert d_strat.shape == (60,)
        assert c_strat.shape == (60,)
        np.testing.assert_almost_equal(d_strat.sum(), 1.0)
        np.testing.assert_almost_equal(c_strat.sum(), 1.0)
        assert np.all(d_strat >= 0)
        assert np.all(c_strat >= 0)

    def test_game_value_is_negative(self):
        """Checker always pays something — game value should be <= 0."""
        _, _, gv = solve_half_round(0.0, iterations=5000)
        assert gv <= 0

    def test_higher_cylinder_worse_for_checker(self):
        """More cylinder → worse failed-check penalty → lower game value for checker."""
        _, _, gv_low = solve_half_round(0.0, iterations=5000)
        _, _, gv_high = solve_half_round(200.0, iterations=5000)
        assert gv_high < gv_low

    def test_checker_avoids_very_late_checks(self):
        """At equilibrium, checking at second 60 accumulates max ST if dropper
        drops early. The checker strategy should not put all weight on 60."""
        _, c_strat, _ = solve_half_round(0.0, iterations=5000)
        assert c_strat[59] < 0.5  # not all-in on second 60

    def test_dropper_spreads_probability(self):
        """Dropper should mix — putting all weight on one second is exploitable."""
        d_strat, _, _ = solve_half_round(0.0, iterations=5000)
        assert d_strat.max() < 0.5  # no single action dominates

    def test_leap_turn_61_actions(self):
        """During leap second, both players have 61 actions."""
        d_strat, c_strat, _ = solve_half_round(0.0, turn_duration=61, iterations=1000)
        assert d_strat.shape == (61,)
        assert c_strat.shape == (61,)
        np.testing.assert_almost_equal(d_strat.sum(), 1.0)

    def test_nash_condition_dropper_cannot_improve(self):
        """At Nash, no single Dropper action should beat the equilibrium EV."""
        d_strat, c_strat, gv = solve_half_round(0.0, iterations=10_000)
        payoff = compute_payoff_matrix(0.0)
        dropper_action_values = -payoff @ c_strat
        dropper_best = dropper_action_values.max()
        dropper_ev = d_strat @ dropper_action_values
        # Best single action shouldn't be much better than the mixed EV
        # (allow small epsilon for convergence)
        assert dropper_best - dropper_ev < 0.5

    def test_nash_condition_checker_cannot_improve(self):
        """At Nash, no single Checker action should beat the equilibrium EV."""
        d_strat, c_strat, gv = solve_half_round(0.0, iterations=10_000)
        payoff = compute_payoff_matrix(0.0)
        checker_action_values = payoff.T @ d_strat
        checker_best = checker_action_values.max()
        checker_ev = c_strat @ checker_action_values
        assert checker_best - checker_ev < 0.5

    def test_deterministic_with_fixed_seed(self):
        """Same inputs → same outputs (no hidden randomness)."""
        r1 = solve_half_round(100.0, iterations=2000)
        r2 = solve_half_round(100.0, iterations=2000)
        np.testing.assert_array_equal(r1[0], r2[0])
        np.testing.assert_array_equal(r1[1], r2[1])
        assert r1[2] == r2[2]
