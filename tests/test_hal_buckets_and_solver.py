"""Tests for tickets 1.1 (buckets) and 1.2 (LP solver)."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest
from random import Random

from src.Constants import CYLINDER_MAX, FAILED_CHECK_PENALTY
from hal.types import Bucket, BeliefState
from hal.buckets import (
    STANDARD_BUCKETS, LEAP_BUCKET, bucket_pair_payoff, resolve_bucket,
)
from hal.solver import solve_minimax, best_response


# ── Ticket 1.1: Bucket definitions ──

class TestBucketDefinitions:
    def test_standard_buckets_cover_1_to_60(self):
        covered = set()
        for b in STANDARD_BUCKETS:
            for s in range(b.lo, b.hi + 1):
                covered.add(s)
        assert covered == set(range(1, 61))

    def test_no_bucket_overlap(self):
        all_seconds = []
        for b in STANDARD_BUCKETS:
            for s in range(b.lo, b.hi + 1):
                all_seconds.append(s)
        assert len(all_seconds) == len(set(all_seconds))

    def test_leap_bucket_is_61(self):
        assert LEAP_BUCKET.lo == 61
        assert LEAP_BUCKET.hi == 61

    def test_seven_standard_buckets(self):
        assert len(STANDARD_BUCKETS) == 7


class TestResolve:
    def test_single_second_bucket(self):
        rng = Random(0)
        b = Bucket(1, 1, "instant")
        assert resolve_bucket(b, rng) == 1

    def test_resolve_within_range(self):
        rng = Random(42)
        b = Bucket(10, 20, "test")
        for _ in range(100):
            s = resolve_bucket(b, rng)
            assert 10 <= s <= 20

    def test_resolve_uniform_coverage(self):
        rng = Random(123)
        b = Bucket(1, 5, "small")
        seen = set()
        for _ in range(500):
            seen.add(resolve_bucket(b, rng))
        assert seen == {1, 2, 3, 4, 5}


# ── Ticket 1.1: Bucket pair payoffs ──

class TestBucketPairPayoff:
    def test_instant_vs_safe_no_cylinder(self):
        d = Bucket(1, 1, "instant")
        c = Bucket(59, 60, "safe")
        pay = bucket_pair_payoff(d, c, checker_cylinder=0.0)
        # check >= drop always (59>=1, 60>=1), ST = 58 or 59
        # avg = -(58 + 59) / 2 = -58.5
        assert pay == pytest.approx(-58.5)

    def test_safe_vs_instant_no_cylinder(self):
        d = Bucket(59, 60, "safe")
        c = Bucket(1, 1, "instant")
        pay = bucket_pair_payoff(d, c, checker_cylinder=0.0)
        # check < drop always (1 < 59, 1 < 60) → failed check
        # injection = min(0 + 60, 300) = 60
        # avg = -60
        assert pay == pytest.approx(-60.0)

    def test_same_bucket_no_cylinder(self):
        b = Bucket(30, 30, "single")
        pay = bucket_pair_payoff(b, b, checker_cylinder=0.0)
        # check == drop → ST = max(1, 0) = 1
        assert pay == pytest.approx(-1.0)

    def test_overflow_triggers(self):
        d = Bucket(1, 1, "instant")
        c = Bucket(59, 60, "safe")
        pay = bucket_pair_payoff(d, c, checker_cylinder=250.0)
        # ST = 58 or 59, both cause 250+58=308 ≥ 300 and 250+59=309 ≥ 300
        assert pay == pytest.approx(-CYLINDER_MAX)

    def test_payoff_matches_full_matrix(self):
        d = Bucket(2, 10, "early")
        c = Bucket(41, 52, "mid_late")
        cyl = 50.0
        bucket_pay = bucket_pair_payoff(d, c, cyl)

        total = 0.0
        count = 0
        for drop in range(2, 11):
            for check in range(41, 53):
                if check >= drop:
                    st = max(1, check - drop)
                    if cyl + st >= CYLINDER_MAX:
                        total += -CYLINDER_MAX
                    else:
                        total += -st
                else:
                    total += -min(cyl + FAILED_CHECK_PENALTY, CYLINDER_MAX)
                count += 1
        assert bucket_pay == pytest.approx(total / count)


# ── Ticket 1.2: LP Nash solver ──

class TestSolveMinimax:
    def test_2x2_pure_dominant(self):
        # Row player has a dominant strategy: row 0 is always better
        A = np.array([[3.0, 1.0],
                      [0.0, 0.0]])
        strat, val = solve_minimax(A)
        assert strat[0] == pytest.approx(1.0, abs=1e-6)
        assert val == pytest.approx(1.0, abs=1e-6)

    def test_2x2_matching_pennies(self):
        # Classic matching pennies: Nash is (0.5, 0.5), value = 0
        A = np.array([[ 1.0, -1.0],
                      [-1.0,  1.0]])
        strat, val = solve_minimax(A)
        assert strat[0] == pytest.approx(0.5, abs=1e-4)
        assert strat[1] == pytest.approx(0.5, abs=1e-4)
        assert val == pytest.approx(0.0, abs=1e-4)

    def test_2x2_rock_paper_scissors_subgame(self):
        # Asymmetric 2x2 where row 0 beats col 0 but loses to col 1
        A = np.array([[ 2.0, -1.0],
                      [-1.0,  1.0]])
        strat, val = solve_minimax(A)
        assert sum(strat) == pytest.approx(1.0)
        assert all(s >= -1e-9 for s in strat)
        # Value should be between -1 and 2
        assert -1.0 <= val <= 2.0

    def test_3x3_rps(self):
        # Full rock-paper-scissors: Nash is (1/3, 1/3, 1/3), value = 0
        A = np.array([[ 0.0, -1.0,  1.0],
                      [ 1.0,  0.0, -1.0],
                      [-1.0,  1.0,  0.0]])
        strat, val = solve_minimax(A)
        for s in strat:
            assert s == pytest.approx(1/3, abs=1e-4)
        assert val == pytest.approx(0.0, abs=1e-4)

    def test_strategy_is_probability_distribution(self):
        rng = np.random.RandomState(42)
        A = rng.randn(7, 7)
        strat, _ = solve_minimax(A)
        assert sum(strat) == pytest.approx(1.0, abs=1e-6)
        assert all(s >= -1e-9 for s in strat)

    def test_nonsquare_matrix(self):
        # 8 hal buckets vs 7 baku buckets (leap turn)
        rng = np.random.RandomState(99)
        A = rng.randn(8, 7)
        strat, val = solve_minimax(A)
        assert len(strat) == 8
        assert sum(strat) == pytest.approx(1.0, abs=1e-6)

    def test_value_bounded_by_payoff(self):
        A = np.array([[-5.0, -10.0],
                      [-3.0, -20.0]])
        _, val = solve_minimax(A)
        assert val >= A.min()
        assert val <= A.max()


class TestBestResponse:
    def test_pure_exploitation(self):
        # Baku always plays bucket 0 → Hal should pick the row with best A[i][0]
        A = np.array([[1.0, -5.0],
                      [3.0, -5.0]])
        belief = BeliefState(
            exploitation_mode=True,
            baku_predicted_bucket_probs=(1.0, 0.0),
        )
        strat, val = best_response(belief, A)
        assert strat[1] == pytest.approx(1.0)  # row 1 has payoff 3 vs col 0
        assert val == pytest.approx(3.0)

    def test_mixed_baku(self):
        A = np.array([[4.0, 0.0],
                      [0.0, 4.0]])
        belief = BeliefState(
            exploitation_mode=True,
            baku_predicted_bucket_probs=(0.8, 0.2),
        )
        strat, val = best_response(belief, A)
        # EV row 0 = 4*0.8 + 0*0.2 = 3.2
        # EV row 1 = 0*0.8 + 4*0.2 = 0.8
        assert strat[0] == pytest.approx(1.0)
        assert val == pytest.approx(3.2)

    def test_returns_valid_strategy(self):
        rng = np.random.RandomState(7)
        A = rng.randn(7, 7)
        probs = tuple(np.ones(7) / 7)
        belief = BeliefState(
            exploitation_mode=True,
            baku_predicted_bucket_probs=probs,
        )
        strat, _ = best_response(belief, A)
        assert sum(strat) == pytest.approx(1.0)
        assert sum(s == 1.0 for s in strat) == 1  # pure strategy
