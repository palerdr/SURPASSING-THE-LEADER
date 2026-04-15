"""Tests for CFR game state abstraction."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from environment.cfr.game_state import (
    AbstractState,
    CYL_BUCKET_SIZE,
    CYL_NUM_BUCKETS,
    MAX_DEATH_BUCKET,
    bucket_cylinder,
    bucket_deaths,
    bucket_clock,
    make_abstract_state,
)


class TestBucketCylinder:
    def test_zero(self):
        assert bucket_cylinder(0.0) == 0

    def test_below_first_boundary(self):
        assert bucket_cylinder(9.0) == 0

    def test_at_first_boundary(self):
        assert bucket_cylinder(10.0) == 1

    def test_mid_range(self):
        assert bucket_cylinder(150.0) == 15  # 150 / 10

    def test_at_max(self):
        assert bucket_cylinder(300.0) == CYL_NUM_BUCKETS - 1

    def test_above_max(self):
        assert bucket_cylinder(999.0) == CYL_NUM_BUCKETS - 1

    def test_just_below_boundary(self):
        assert bucket_cylinder(19.9) == 1  # 19.9 / 10 = 1.99 → 1

    def test_returns_int(self):
        assert isinstance(bucket_cylinder(45.0), int)


class TestBucketDeaths:
    def test_zero(self):
        assert bucket_deaths(0) == 0

    def test_one(self):
        assert bucket_deaths(1) == 1

    def test_at_cap(self):
        assert bucket_deaths(3) == MAX_DEATH_BUCKET

    def test_above_cap(self):
        assert bucket_deaths(10) == MAX_DEATH_BUCKET

    def test_returns_int(self):
        assert isinstance(bucket_deaths(2), int)


class TestBucketClock:
    def test_opening(self):
        assert bucket_clock(720.0) == 12

    def test_near_leap(self):
        assert bucket_clock(3540.0) == 59

    def test_zero(self):
        assert bucket_clock(0.0) == 0


class TestMakeAbstractState:
    def test_basic_construction(self):
        state = make_abstract_state(
            round_num=3, half=1,
            my_cylinder=90.0, opp_cylinder=150.0,
            my_deaths=1, opp_deaths=2,
            game_clock=720.0,
        )
        assert state.round_num == 3
        assert state.half == 1
        assert state.my_cyl == 9       # 90 / 10 = 9
        assert state.opp_cyl == 15     # 150 / 10 = 15
        assert state.my_deaths == 1
        assert state.opp_deaths == 2
        assert state.clock == 12       # 720 / 60

    def test_zero_state(self):
        state = make_abstract_state(0, 0, 0.0, 0.0, 0, 0, 720.0)
        assert state == AbstractState(0, 0, 0, 0, 0, 0, 12)

    def test_max_values(self):
        state = make_abstract_state(9, 1, 300.0, 300.0, 5, 5, 3600.0)
        assert state.my_cyl == CYL_NUM_BUCKETS - 1
        assert state.opp_cyl == CYL_NUM_BUCKETS - 1
        assert state.my_deaths == MAX_DEATH_BUCKET
        assert state.opp_deaths == MAX_DEATH_BUCKET

    def test_hashable(self):
        s1 = make_abstract_state(0, 0, 0.0, 0.0, 0, 0, 720.0)
        s2 = make_abstract_state(0, 0, 0.0, 0.0, 0, 0, 720.0)
        d = {s1: "test"}
        assert d[s2] == "test"

    def test_different_states_not_equal(self):
        s1 = make_abstract_state(0, 0, 0.0, 0.0, 0, 0, 720.0)
        s2 = make_abstract_state(0, 0, 60.0, 0.0, 0, 0, 720.0)
        assert s1 != s2

    def test_different_clocks_not_equal(self):
        s1 = make_abstract_state(0, 0, 0.0, 0.0, 0, 0, 720.0)
        s2 = make_abstract_state(0, 0, 0.0, 0.0, 0, 0, 3540.0)
        assert s1 != s2

    def test_same_bucket_equal(self):
        """Two cylinder values in the same 10s bucket produce the same state."""
        s1 = make_abstract_state(0, 0, 11.0, 0.0, 0, 0, 720.0)
        s2 = make_abstract_state(0, 0, 19.0, 0.0, 0, 0, 720.0)
        assert s1 == s2
