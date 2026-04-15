"""Tests for multi-round CFR tree traversal."""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from environment.cfr.tree import (
    get_turn_duration,
    snap_to_next_minute,
    compute_successors,
    solve_game,
    REPRESENTATIVE_STS,
)
from environment.cfr.game_state import (
    AbstractState,
    make_abstract_state,
    CYL_BUCKET_SIZE,
    CLOCK_BUCKET_SIZE,
)
from src.Constants import (
    LS_WINDOW_START,
    LS_WINDOW_END,
    OPENING_START_CLOCK,
    TURN_DURATION_NORMAL,
    TURN_DURATION_LEAP,
    FAILED_CHECK_PENALTY,
    CYLINDER_MAX,
)


class TestGetTurnDuration:
    def test_normal_turn(self):
        assert get_turn_duration(720.0) == TURN_DURATION_NORMAL

    def test_leap_window_start(self):
        assert get_turn_duration(float(LS_WINDOW_START)) == TURN_DURATION_LEAP

    def test_leap_window_middle(self):
        assert get_turn_duration(3570.0) == TURN_DURATION_LEAP

    def test_after_leap_window(self):
        assert get_turn_duration(3601.0) == TURN_DURATION_NORMAL


class TestSnapToNextMinute:
    def test_mid_minute(self):
        assert snap_to_next_minute(730.0) == 780.0

    def test_on_boundary_advances(self):
        assert snap_to_next_minute(780.0) == 840.0

    def test_pre_leap_no_land_on_3600(self):
        """3600 is the leap second, not a minute boundary. Snap skips to 3601."""
        assert snap_to_next_minute(3540.0) == 3601.0  # would be 3600, pushed to 3601
        assert snap_to_next_minute(3580.0) == 3601.0

    def test_post_leap(self):
        """Post-leap minutes are at 3601 + n*60."""
        assert snap_to_next_minute(3610.0) == 3661.0


class TestRepresentativeSTs:
    def test_derived_from_bucket_size(self):
        """STs should be 1 plus bucket boundaries up to 59."""
        assert REPRESENTATIVE_STS[0] == 1
        for st in REPRESENTATIVE_STS[1:]:
            assert st % CYL_BUCKET_SIZE == 0
        assert all(st < 60 for st in REPRESENTATIVE_STS)

    def test_10s_buckets(self):
        """With 10s buckets: [1, 10, 20, 30, 40, 50]."""
        assert REPRESENTATIVE_STS == [1, 10, 20, 30, 40, 50]


class TestComputeSuccessors:
    def _opening_state(self):
        return make_abstract_state(0, 0, 0.0, 0.0, 0, 0, OPENING_START_CLOCK)

    def test_successor_count(self):
        """Should produce len(REPRESENTATIVE_STS) + 1 (failed check) successors."""
        state = self._opening_state()
        succs = compute_successors(state, TURN_DURATION_NORMAL)
        assert len(succs) == len(REPRESENTATIVE_STS) + 1

    def test_all_successors_are_half_1(self):
        """From half=0, all successors should be half=1 (same round)."""
        state = self._opening_state()
        succs = compute_successors(state, TURN_DURATION_NORMAL)
        for s, _dd in succs:
            assert s.half == 1
            assert s.round_num == 0

    def test_half_1_successors_advance_round(self):
        """From half=1, successors should be half=0 of next round."""
        state = make_abstract_state(0, 1, 0.0, 0.0, 0, 0, OPENING_START_CLOCK + 60)
        succs = compute_successors(state, TURN_DURATION_NORMAL)
        for s, _dd in succs:
            assert s.half == 0
            assert s.round_num == 1

    def test_failed_check_increments_deaths(self):
        """The failed-check successor should have one more death for the checker."""
        state = self._opening_state()
        succs = compute_successors(state, TURN_DURATION_NORMAL)
        failed = succs[-1][0]  # last one is the failed check
        # half=0 → opp is checker → opp_deaths should increment
        assert failed.opp_deaths == 1

    def test_failed_check_resets_cylinder(self):
        """After a failed check death (survived), checker's cylinder resets to 0."""
        state = make_abstract_state(0, 0, 0.0, 90.0, 0, 0, OPENING_START_CLOCK)
        succs = compute_successors(state, TURN_DURATION_NORMAL)
        failed = succs[-1][0]
        # half=0 → opp is checker → opp_cyl should be 0 after death
        assert failed.opp_cyl == 0

    def test_successful_check_increases_cylinder(self):
        """ST > 0 should increase the checker's cylinder bucket."""
        state = self._opening_state()
        succs = compute_successors(state, TURN_DURATION_NORMAL)
        # First successor: ST=1, checker cyl goes from 0 to 1 (still bucket 0)
        assert succs[0][0].opp_cyl == 0  # 1 < 10, stays in bucket 0
        # Second successor: ST=10, checker cyl goes from 0 to 10 (bucket 1)
        assert succs[1][0].opp_cyl == 1

    def test_overflow_from_st_causes_death(self):
        """If checker_cyl + ST >= 300, should cause death and reset cylinder."""
        # Put checker at cyl=290, so ST=10 (second representative) overflows: 290+10=300
        state = make_abstract_state(0, 0, 0.0, 290.0, 0, 0, OPENING_START_CLOCK)
        succs = compute_successors(state, TURN_DURATION_NORMAL)
        # ST=10: 290+10=300 >= 300 → overflow death
        st10_succ = succs[1][0]
        assert st10_succ.opp_cyl == 0     # cylinder resets
        assert st10_succ.opp_deaths == 1  # death incremented

    def test_clock_advances_more_on_death(self):
        """Death outcomes should have a later clock than no-death outcomes."""
        state = self._opening_state()
        succs = compute_successors(state, TURN_DURATION_NORMAL)
        no_death_clock = succs[0][0].clock    # ST=1, no death
        failed_clock = succs[-1][0].clock     # failed check, death
        assert failed_clock > no_death_clock


class TestSolveGame:
    def test_returns_strategy_table(self):
        table = solve_game(max_rounds=1, iterations_per_state=200)
        assert isinstance(table, dict)
        assert len(table) > 0

    def test_all_entries_are_valid(self):
        table = solve_game(max_rounds=1, iterations_per_state=200)
        for state, (d_strat, c_strat, gv) in table.items():
            assert isinstance(state, AbstractState)
            assert d_strat.shape[0] >= 60
            assert c_strat.shape[0] >= 60
            np.testing.assert_almost_equal(d_strat.sum(), 1.0)
            np.testing.assert_almost_equal(c_strat.sum(), 1.0)
            assert np.all(d_strat >= 0)
            assert np.all(c_strat >= 0)

    def test_initial_state_always_solved(self):
        table = solve_game(max_rounds=1, iterations_per_state=200)
        init = make_abstract_state(0, 0, 0.0, 0.0, 0, 0, OPENING_START_CLOCK)
        assert init in table

    def test_more_rounds_more_states(self):
        t1 = solve_game(max_rounds=1, iterations_per_state=200)
        t2 = solve_game(max_rounds=2, iterations_per_state=200)
        assert len(t2) > len(t1)

    def test_deterministic(self):
        t1 = solve_game(max_rounds=1, iterations_per_state=500)
        t2 = solve_game(max_rounds=1, iterations_per_state=500)
        assert set(t1.keys()) == set(t2.keys())
        for key in t1:
            np.testing.assert_array_equal(t1[key][0], t2[key][0])
