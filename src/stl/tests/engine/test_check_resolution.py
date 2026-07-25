"""Tests for check resolution logic."""

from stl.engine.game import HalfRoundResult


class TestCheckSuccess:
    """check_time >= drop_time = handkerchief is on the ground."""

    def test_check_after_drop(self, game):
        rec = game.play_half_round(drop_time=20, check_time=40)
        assert rec.result == HalfRoundResult.CHECK_SUCCESS
        assert rec.st_gained == 21

    def test_check_same_second_as_drop_succeeds_with_one_st(self, game):
        """D drops and C checks in second 30; inclusive ST is one."""
        rec = game.play_half_round(drop_time=30, check_time=30)
        assert rec.result == HalfRoundResult.CHECK_SUCCESS
        assert rec.st_gained == 1

    def test_check_one_second_after_drop(self, game):
        """One second after the drop produces two inclusive seconds of ST."""
        rec = game.play_half_round(drop_time=30, check_time=31)
        assert rec.result == HalfRoundResult.CHECK_SUCCESS
        assert rec.st_gained == 2

    def test_st_accumulates_across_halves(self, game):
        """ST from successful checks stacks in the checker's cylinder."""
        # Half 1: Baku checks, gains 21 inclusive ST
        game.play_half_round(drop_time=10, check_time=30)
        assert game.player2.cylinder == 21  # Baku is checker in half 1

        # Half 2: Hal checks, gains 16 inclusive ST
        game.play_half_round(drop_time=20, check_time=35)
        assert game.player1.cylinder == 16  # Hal is checker in half 2

    def test_max_st_drop_at_1_check_at_60(self, game):
        """Worst case: D drops at 1, C checks at 60. ST = 60."""
        rec = game.play_half_round(drop_time=1, check_time=60)
        assert rec.st_gained == 60


class TestCheckFailure:
    """check_time < drop_time = handkerchief NOT on the ground yet."""

    def test_check_before_drop(self, game):
        rec = game.play_half_round(drop_time=40, check_time=20)
        assert rec.result in (HalfRoundResult.CHECK_FAIL_SURVIVED,
                               HalfRoundResult.CHECK_FAIL_DIED)
        assert rec.st_gained == 0
        assert rec.death_duration > 0

    def test_check_one_second_before_drop(self, game):
        """D drops at 31, C checks at 30. Handkerchief not yet dropped."""
        rec = game.play_half_round(drop_time=31, check_time=30)
        assert rec.result in (HalfRoundResult.CHECK_FAIL_SURVIVED,
                               HalfRoundResult.CHECK_FAIL_DIED)

    def test_failed_check_adds_60s_penalty(self, game):
        """Failed check: +60s penalty, then entire cylinder injected."""
        rec = game.play_half_round(drop_time=50, check_time=10)
        # Checker had 0 cylinder, +60 penalty = 60s death
        assert rec.death_duration == 60

    def test_failed_check_stacks_with_existing_cylinder(self, game):
        """If checker already has ST in cylinder, penalty stacks."""
        # Half 1: Baku gains 31 inclusive ST (success)
        game.play_half_round(drop_time=10, check_time=40)
        assert game.player2.cylinder == 31

        # Half 2: Hal checks successfully, no death
        game.play_half_round(drop_time=10, check_time=20)

        # Round 2, Half 1: Baku fails check. cylinder = 31 + 60 = 91
        rec = game.play_half_round(drop_time=50, check_time=10)
        assert rec.death_duration == 91


class TestInputValidation:
    def test_drop_time_zero_rejected(self, game):
        import pytest
        with pytest.raises(ValueError):
            game.play_half_round(drop_time=0, check_time=30)

    def test_drop_time_over_60_rejected_normal(self, game):
        import pytest
        with pytest.raises(ValueError):
            game.play_half_round(drop_time=61, check_time=30)

    def test_check_time_zero_rejected(self, game):
        import pytest
        with pytest.raises(ValueError):
            game.play_half_round(drop_time=30, check_time=0)

    def test_check_time_over_60_rejected(self, game):
        import pytest
        with pytest.raises(ValueError):
            game.play_half_round(drop_time=30, check_time=61)

    def test_drop_at_60_is_valid(self, game):
        """D can drop at the last second."""
        rec = game.play_half_round(drop_time=60, check_time=30)
        assert rec.drop_time == 60

    def test_check_at_60_is_valid(self, game):
        """C can check at the last second (safe strategy)."""
        rec = game.play_half_round(drop_time=1, check_time=60)
        assert rec.check_time == 60
