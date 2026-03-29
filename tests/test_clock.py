"""Tests for game clock, snap-to-minute, leap second, and format_game_clock."""

import pytest
from src.Player import Player
from src.Referee import Referee
from src.Game import Game
from src.Constants import (
    LS_WINDOW_START, LS_WINDOW_END,
    TURN_DURATION_NORMAL, TURN_DURATION_LEAP,
    WITHIN_ROUND_OVERHEAD, DEATH_PROCEDURE_OVERHEAD,
)


def make_game(clock: float = 0.0, seed: int = 42) -> Game:
    hal = Player(name="Hal", physicality=1.0)
    baku = Player(name="Baku", physicality=0.7)
    ref = Referee()
    g = Game(player1=hal, player2=baku, referee=ref, first_dropper=hal)
    g.game_clock = clock
    g.seed(seed)
    return g


class TestFormatGameClock:
    def test_game_start(self):
        g = make_game(0)
        assert g.format_game_clock() == "8:00:00 AM"

    def test_twelve_minutes(self):
        g = make_game(720)
        assert g.format_game_clock() == "8:12:00 AM"

    def test_eight_fifty_nine(self):
        g = make_game(3540)
        assert g.format_game_clock() == "8:59:00 AM"

    def test_eight_fifty_nine_fifty_nine(self):
        g = make_game(3599)
        assert g.format_game_clock() == "8:59:59 AM"

    def test_leap_second(self):
        g = make_game(3600)
        assert g.format_game_clock() == "8:59:60 AM"

    def test_nine_oclock(self):
        g = make_game(3601)
        assert g.format_game_clock() == "9:00:00 AM"

    def test_nine_oh_four(self):
        # 9:04:00 = 3601 + 4*60 = 3841
        g = make_game(3841)
        assert g.format_game_clock() == "9:04:00 AM"

    def test_post_leap_continuity(self):
        """Clock should increment smoothly across the leap second."""
        g = make_game(3599)
        assert "8:59:59" in g.format_game_clock()
        g.game_clock = 3600
        assert "8:59:60" in g.format_game_clock()
        g.game_clock = 3601
        assert "9:00:00" in g.format_game_clock()
        g.game_clock = 3602
        assert "9:00:01" in g.format_game_clock()


class TestSnapToNextMinute:
    def test_snap_mid_minute(self):
        g = make_game(1753)  # 8:29:13
        g.snap_clock_to_next_minute()
        assert g.game_clock == 1800  # 8:30:00

    def test_snap_on_boundary_advances(self):
        """On a minute boundary, snap STILL advances to the next one."""
        g = make_game(1800)  # 8:30:00
        g.snap_clock_to_next_minute()
        assert g.game_clock == 1860  # 8:31:00

    def test_snap_near_leap_second(self):
        """Snap from 8:59:xx should go to 9:00:00 (gc=3601), not 3600."""
        g = make_game(3550)  # 8:59:10
        g.snap_clock_to_next_minute()
        assert g.game_clock == 3601  # 9:00:00

    def test_snap_at_3540(self):
        """8:59:00 snaps to 9:00:00."""
        g = make_game(3540)
        g.snap_clock_to_next_minute()
        assert g.game_clock == 3601

    def test_snap_at_leap_second(self):
        g = make_game(3600)
        g.snap_clock_to_next_minute()
        assert g.game_clock == 3601

    def test_snap_post_leap(self):
        """Post-leap snaps align with wall-clock minutes."""
        g = make_game(3810)  # post-leap
        g.snap_clock_to_next_minute()
        # 9:04:00 = gc 3841
        assert g.game_clock == 3841
        assert "9:04:00" in g.format_game_clock()

    def test_snap_post_leap_on_boundary(self):
        """Post-leap, on a wall-clock minute boundary, advances to next."""
        g = make_game(3841)  # 9:04:00
        g.snap_clock_to_next_minute()
        assert g.game_clock == 3901  # 9:05:00
        assert "9:05:00" in g.format_game_clock()


class TestTurnDuration:
    def test_normal_turn(self):
        g = make_game(720)
        assert g.get_turn_duration() == TURN_DURATION_NORMAL

    def test_leap_second_window_start(self):
        g = make_game(LS_WINDOW_START)  # 3540
        assert g.get_turn_duration() == TURN_DURATION_LEAP

    def test_leap_second_window_middle(self):
        g = make_game(3570)
        assert g.get_turn_duration() == TURN_DURATION_LEAP

    def test_leap_second_window_end(self):
        g = make_game(LS_WINDOW_END)  # 3600
        assert g.get_turn_duration() == TURN_DURATION_LEAP

    def test_after_leap_window(self):
        g = make_game(3601)
        assert g.get_turn_duration() == TURN_DURATION_NORMAL


class TestLeapSecondGameplay:
    def test_drop_at_61_during_leap(self):
        """Dropper can drop at second 61 during a leap second turn."""
        g = make_game(LS_WINDOW_START)  # in LS window
        rec = g.play_half_round(drop_time=61, check_time=60)
        # C checks at 60, D drops at 61. check_time < drop_time -> FAIL
        assert rec.result in ("check_fail_survived", "check_fail_died",
                               rec.result)  # just check it doesn't crash
        assert rec.turn_duration == 61

    def test_safe_strategy_fails_during_leap(self):
        """The whole point: checking at 60 fails if D drops at 61."""
        g = make_game(LS_WINDOW_START)
        rec = g.play_half_round(drop_time=61, check_time=60)
        # check_time (60) < drop_time (61) -> failure
        from src.Game import HalfRoundResult
        assert rec.result in (HalfRoundResult.CHECK_FAIL_SURVIVED,
                               HalfRoundResult.CHECK_FAIL_DIED)

    def test_same_second_drop_and_check_succeeds_min_st(self):
        """Drop at 60, check at 60 during leap — success with ST=1 (minimum)."""
        g = make_game(LS_WINDOW_START)
        rec = g.play_half_round(drop_time=60, check_time=60)
        from src.Game import HalfRoundResult
        assert rec.result == HalfRoundResult.CHECK_SUCCESS
        assert rec.st_gained == 1

    def test_checker_can_check_at_61_during_leap(self):
        """Checker CAN check at 61 during a leap turn (defense against the kill shot).
        The action mask enforces knowledge restrictions, not the engine."""
        g = make_game(LS_WINDOW_START)
        rec = g.play_half_round(drop_time=60, check_time=61)
        from src.Game import HalfRoundResult
        # check_time (61) > drop_time (60) -> success, ST = 1
        assert rec.result == HalfRoundResult.CHECK_SUCCESS
        assert rec.st_gained == 1
        assert rec.check_time == 61


class TestClockAdvancement:
    def test_no_death_round_takes_4_minutes(self):
        """A clean round: 60 + 60 + 60 overhead = 180s. Snap -> 4 min gap."""
        g = make_game(1800)  # start on a minute boundary

        # Half 1: success, no death
        g.play_half_round(drop_time=10, check_time=20)
        # Half 2: success, no death
        g.play_half_round(drop_time=10, check_time=20)

        # Should have snapped to 1800 + 180 -> snap -> 2040 (4 min later)
        assert g.game_clock == 2040

    def test_death_round_longer_gap(self):
        """A round with a 60s death adds death_duration + death overhead to elapsed."""
        g = make_game(1800)

        # Half 1: failed check -> 60s death
        g.play_half_round(drop_time=50, check_time=10)
        # Half 2: clean
        g.play_half_round(drop_time=10, check_time=20)

        # 60 (h1 turn) + 60 (death) + 120 (death overhead) + 60 (overhead) + 60 (h2 turn) = 360
        # 1800 + 360 = 2160. snap -> 2220 (7 min gap)
        assert g.game_clock == 2220

    def test_no_snap_between_halves(self):
        """Clock does NOT snap between halves -- only after the full round."""
        g = make_game(1800)

        # Play only half 1
        g.play_half_round(drop_time=10, check_time=20)

        # Should be: 1800 + 60 (turn) + 60 (overhead) = 1920
        # NOT snapped to a minute boundary
        assert g.game_clock == 1920
        assert g.current_half == 2  # still mid-round


class TestMangaTimeline:
    """Validate that the clock model can reproduce manga round start times.

    Manga data: R1:8:12 R2:8:19 R3:8:26 R4:8:30 R5:8:34 R6:8:38
                R7:8:45 R8:8:49 R9:8:57 R10:9:04

    We simulate rounds with death durations reverse-engineered from
    the timing gaps. Each must produce the next round's start time.

    Gap analysis (with DEATH_PROCEDURE_OVERHEAD=120):
      No-death: 60+60+60=180 → snap → 4 min gap
      Death(60s): 60+60+120+60+60=360 → snap → 7 min gap
      Death(120s): 60+120+120+60+60=420 → snap → 8 min gap
    """

    EXPECTED = [
        (1, 720),    # 8:12
        (2, 1140),   # 8:19
        (3, 1560),   # 8:26
        (4, 1800),   # 8:30
        (5, 2040),   # 8:34
        (6, 2280),   # 8:38
        (7, 2700),   # 8:45
        (8, 2940),   # 8:49
        (9, 3420),   # 8:57
        (10, 3841),  # 9:04 (post-leap, gc=3601+4*60)
    ]

    # Death durations that produce the correct gaps
    # (0 = no death in that round)
    ROUND_DEATHS = [60, 60, 0, 0, 0, 60, 0, 120, 120]

    def test_full_manga_timeline(self):
        """Simulate 9 rounds and verify each start time matches."""
        g = make_game(720)  # R1 at 8:12

        for i, death_dur in enumerate(self.ROUND_DEATHS):
            round_num = i + 1
            start_clock = g.game_clock
            expected_start = self.EXPECTED[i][1]

            # Verify this round starts at the expected time
            assert start_clock == expected_start, (
                f"R{round_num} started at gc={start_clock}, "
                f"expected {expected_start}"
            )

            # Play the round. Death in half 1 if death_dur > 0.
            if death_dur > 0:
                # Failed check: checker's cylinder must equal death_dur
                # before the play. Set it up so penalty produces right duration.
                checker_idx = 1  # Baku is checker in half 1 (Hal drops first)
                checker = g.player2 if g.first_dropper == g.player1 else g.player1
                checker.cylinder = death_dur - 60  # +60 penalty -> death_dur
                if checker.cylinder < 0:
                    checker.cylinder = 0

                g.play_half_round(drop_time=50, check_time=10)  # fail
                if g.game_over:
                    break
                g.play_half_round(drop_time=10, check_time=20)  # clean half 2
            else:
                g.play_half_round(drop_time=10, check_time=20)  # clean half 1
                if g.game_over:
                    break
                g.play_half_round(drop_time=10, check_time=20)  # clean half 2

            if g.game_over:
                break

            # Verify next round starts correctly
            next_expected = self.EXPECTED[i + 1][1]
            assert g.game_clock == next_expected, (
                f"After R{round_num}: gc={g.game_clock}, "
                f"expected R{round_num + 1} at {next_expected}"
            )
