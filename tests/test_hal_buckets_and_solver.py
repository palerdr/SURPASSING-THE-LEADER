"""Tests for tickets 1.1 (buckets) and 1.2 (LP solver)."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest
from random import Random

from src.Constants import CYLINDER_MAX, FAILED_CHECK_PENALTY
from hal.state import Bucket, BeliefState
from hal.action_model import (
    STANDARD_BUCKETS, LEAP_BUCKET, bucket_pair_payoff, resolve_bucket, get_legal_buckets,
)
from environment.cfr.exact import solve_minimax
from hal.action_model import best_response
from environment.legal_actions import legal_max_second, can_use_leap_second


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
        A = np.array([[1.0, -5.0],
                      [3.0, -5.0]])
        belief = BeliefState(
            check_exploit=True,
            baku_check_probs=(1.0, 0.0),
        )
        strat, val = best_response(belief, A, hal_is_dropper=True)
        assert strat[1] == pytest.approx(1.0)
        assert val == pytest.approx(3.0)

    def test_mixed_baku(self):
        A = np.array([[4.0, 0.0],
                      [0.0, 4.0]])
        belief = BeliefState(
            drop_exploit=True,
            baku_drop_probs=(0.8, 0.2),
        )
        strat, val = best_response(belief, A, hal_is_dropper=False)
        # EV row 0 = 4*0.8 + 0*0.2 = 3.2
        # EV row 1 = 0*0.8 + 4*0.2 = 0.8
        assert strat[0] == pytest.approx(1.0)
        assert val == pytest.approx(3.2)

    def test_returns_valid_strategy(self):
        rng = np.random.RandomState(7)
        A = rng.randn(7, 7)
        probs = tuple(np.ones(7) / 7)
        belief = BeliefState(
            check_exploit=True,
            baku_check_probs=probs,
        )
        strat, _ = best_response(belief, A, hal_is_dropper=True)
        assert sum(strat) == pytest.approx(1.0)
        assert sum(s == 1.0 for s in strat) == 1


class TestLegalActionAsymmetry:
    def test_hal_dropper_max_is_60_on_normal_turn(self):
        assert legal_max_second("hal", "dropper", 60) == 60

    def test_hal_dropper_max_is_60_on_leap_turn(self):
        assert legal_max_second("hal", "dropper", 61, hal_leap_deduced=True) == 60

    def test_hal_checker_max_is_60_when_unaware(self):
        assert legal_max_second("hal", "checker", 61, hal_leap_deduced=False) == 60

    def test_hal_checker_max_is_61_when_deduced(self):
        assert legal_max_second("hal", "checker", 61, hal_leap_deduced=True) == 61

    def test_hal_checker_max_is_60_when_amnesia(self):
        assert legal_max_second(
            "hal", "checker", 61, hal_leap_deduced=True, hal_memory_impaired=True,
        ) == 60

    def test_baku_dropper_max_is_61_on_leap_turn(self):
        assert legal_max_second("baku", "dropper", 61) == 61

    def test_baku_checker_max_is_60_even_when_aware(self):
        assert legal_max_second("baku", "checker", 61, hal_leap_deduced=True) == 60

    def test_can_use_leap_second_hal_dropper_never(self):
        assert not can_use_leap_second("hal", "dropper", hal_leap_deduced=True)

    def test_can_use_leap_second_hal_checker_requires_deduced(self):
        assert can_use_leap_second("hal", "checker", hal_leap_deduced=True)
        assert not can_use_leap_second("hal", "checker", hal_leap_deduced=False)
        assert not can_use_leap_second(
            "hal", "checker", hal_leap_deduced=True, hal_memory_impaired=True,
        )

    def test_can_use_leap_second_baku_dropper_always(self):
        assert can_use_leap_second("baku", "dropper")

    def test_can_use_leap_second_baku_checker_never(self):
        assert not can_use_leap_second("baku", "checker", hal_leap_deduced=True)


class TestLegalBucketSelection:
    def test_hal_dropper_buckets_omit_leap_on_leap_turn(self):
        buckets = get_legal_buckets("hal", "dropper", 61, hal_leap_deduced=True)
        assert LEAP_BUCKET not in buckets
        assert buckets == STANDARD_BUCKETS

    def test_hal_checker_buckets_include_leap_when_deduced(self):
        buckets = get_legal_buckets("hal", "checker", 61, hal_leap_deduced=True)
        assert LEAP_BUCKET in buckets

    def test_hal_checker_buckets_omit_leap_when_unaware(self):
        buckets = get_legal_buckets("hal", "checker", 61, hal_leap_deduced=False)
        assert LEAP_BUCKET not in buckets

    def test_hal_checker_buckets_omit_leap_when_amnesia(self):
        buckets = get_legal_buckets(
            "hal", "checker", 61, hal_leap_deduced=True, hal_memory_impaired=True,
        )
        assert LEAP_BUCKET not in buckets

    def test_baku_dropper_buckets_include_leap_on_leap_turn(self):
        buckets = get_legal_buckets("baku", "dropper", 61)
        assert LEAP_BUCKET in buckets

    def test_baku_checker_buckets_omit_leap_on_leap_turn(self):
        buckets = get_legal_buckets("baku", "checker", 61)
        assert LEAP_BUCKET not in buckets

    def test_normal_turn_no_leap_for_anyone(self):
        for actor in ("hal", "baku"):
            for role in ("dropper", "checker"):
                buckets = get_legal_buckets(actor, role, 60)
                assert LEAP_BUCKET not in buckets


class TestSearchSignAndPruning:
    """Regression for the immediate-payoff sign and dominated-pruning polarity bugs."""

    def test_signed_immediate_hal_dropper_negates_checker_perspective(self):
        from hal.search import _signed_immediate

        m = np.array([[-30.0]])
        val = _signed_immediate(m, 0, 0, hal_is_dropper=True)
        assert val > 0  # checker hurt → good for hal-as-dropper

    def test_signed_immediate_hal_checker_passes_through(self):
        from hal.search import _signed_immediate

        m = np.array([[-30.0]])
        val = _signed_immediate(m, 0, 0, hal_is_dropper=False)
        assert val < 0  # checker hurt → bad for hal-as-checker

    def test_pruning_keeps_at_least_one_hal_action(self):
        from hal.search import _prune_dominated

        hal_keep, baku_keep = _prune_dominated(
            STANDARD_BUCKETS, STANDARD_BUCKETS, checker_cylinder=100.0, hal_is_dropper=True,
        )
        assert len(hal_keep) >= 1
        assert len(baku_keep) >= 1

    def test_pruning_does_not_remove_dominant_action(self):
        from hal.search import _prune_dominated

        # The instant bucket is strong for hal-as-dropper at high checker cylinder
        # because it forces the checker to either fail or accept maximum ST.
        hal_keep, _ = _prune_dominated(
            STANDARD_BUCKETS, STANDARD_BUCKETS, checker_cylinder=250.0, hal_is_dropper=True,
        )
        # The instant bucket should not be pruned in this clear-cut state.
        assert 0 in hal_keep
