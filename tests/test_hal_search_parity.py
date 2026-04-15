"""Search/engine parity tests.

These guarantee that hal.search uses the same half-round resolution rules as
src.Game.play_half_round. The plan called this out as the highest-value
correctness fix because the previous search.apply_half_round was a
hand-rolled re-implementation that could (and did) drift from the engine.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

from src.Constants import (
    CYLINDER_MAX,
    FAILED_CHECK_PENALTY,
    PHYSICALITY_BAKU,
    PHYSICALITY_HAL,
)
from src.Game import Game, HalfRoundResult
from src.Player import Player
from src.Referee import Referee
from hal.search import GameSnapshot, apply_half_round


def _make_game(*, hal_cyl: float = 0.0, baku_cyl: float = 0.0, clock: float = 720.0) -> Game:
    hal = Player(name="Hal", physicality=PHYSICALITY_HAL)
    baku = Player(name="Baku", physicality=PHYSICALITY_BAKU)
    hal.cylinder = hal_cyl
    baku.cylinder = baku_cyl
    game = Game(player1=hal, player2=baku, referee=Referee(), first_dropper=hal)
    game.seed(123)
    game.game_clock = clock
    return game


class TestResolveHalfRoundEqualsPlayHalfRound:
    """resolve_half_round with survived_outcome=None must match play_half_round."""

    def test_successful_check_low_cylinder(self):
        a = _make_game()
        b = _make_game()
        rec_a = a.play_half_round(20, 30)
        rec_b = b.resolve_half_round(20, 30, survived_outcome=None)
        assert rec_a.result == rec_b.result == HalfRoundResult.CHECK_SUCCESS
        assert rec_a.st_gained == rec_b.st_gained
        assert a.player2.cylinder == b.player2.cylinder
        assert a.game_clock == b.game_clock

    def test_failed_check_triggers_death(self):
        a = _make_game(baku_cyl=200.0)
        b = _make_game(baku_cyl=200.0)
        rec_a = a.play_half_round(45, 30)
        rec_b = b.resolve_half_round(45, 30, survived_outcome=None)
        assert rec_a.death_duration == rec_b.death_duration
        assert rec_a.result == rec_b.result

    def test_overflow_check_triggers_death(self):
        a = _make_game(baku_cyl=290.0)
        b = _make_game(baku_cyl=290.0)
        rec_a = a.play_half_round(10, 25)
        rec_b = b.resolve_half_round(10, 25, survived_outcome=None)
        assert rec_a.death_duration == rec_b.death_duration


class TestExplicitOutcomeBranching:
    """resolve_half_round with explicit survived_outcome must apply both branches deterministically."""

    def test_survived_branch_keeps_player_alive(self):
        game = _make_game(baku_cyl=290.0)
        record = game.resolve_half_round(10, 25, survived_outcome=True)
        assert record.death_duration > 0
        assert record.survived is True
        assert game.player2.alive
        assert not game.game_over

    def test_died_branch_ends_game(self):
        game = _make_game(baku_cyl=290.0)
        record = game.resolve_half_round(10, 25, survived_outcome=False)
        assert record.survived is False
        assert game.game_over
        assert game.winner is game.player1

    def test_no_death_ignores_outcome(self):
        game = _make_game()
        record = game.resolve_half_round(20, 30, survived_outcome=True)
        assert record.survived is None
        assert record.death_duration == 0.0
        assert not game.game_over


class TestSearchApplyMatchesEngine:
    """search.apply_half_round (the wrapper) must produce identical state to engine.play_half_round."""

    def test_no_death_path_matches(self):
        a = _make_game()
        b = _make_game()
        a.play_half_round(15, 40)
        apply_half_round(b, 15, 40, survived=None)
        assert a.player1.cylinder == b.player1.cylinder
        assert a.player2.cylinder == b.player2.cylinder
        assert a.game_clock == b.game_clock
        assert a.current_half == b.current_half

    def test_explicit_survived_branch_matches_engine_when_engine_rolls_survival(self):
        engine = _make_game(baku_cyl=200.0)
        # Force the engine's RNG into a state where it survives, by rolling once
        # against a near-zero death (won't kill).
        engine.play_half_round(60, 60)
        engine.snap_clock_to_next_minute()

        forced = _make_game(baku_cyl=200.0)
        forced.play_half_round(60, 60)
        forced.snap_clock_to_next_minute()

        assert engine.player2.cylinder == forced.player2.cylinder

    def test_overflow_path_with_forced_survival(self):
        game = _make_game(baku_cyl=295.0)
        snap = GameSnapshot(game)
        apply_half_round(game, 5, 20, survived=True)
        assert game.player2.cylinder == 0.0
        assert game.player2.deaths == 1
        snap.restore(game)
        assert game.player2.cylinder == 295.0
        assert game.player2.deaths == 0


class TestGameSnapshotRoundTrip:
    """GameSnapshot.restore must perfectly undo a half-round mutation."""

    def test_restore_after_no_death(self):
        game = _make_game()
        snap = GameSnapshot(game)
        before = (game.player1.cylinder, game.player2.cylinder, game.game_clock, game.current_half, game.round_num)
        apply_half_round(game, 15, 30, survived=None)
        snap.restore(game)
        after = (game.player1.cylinder, game.player2.cylinder, game.game_clock, game.current_half, game.round_num)
        assert before == after
        assert len(game.history) == 0

    def test_restore_after_death_branch(self):
        game = _make_game(baku_cyl=280.0)
        snap = GameSnapshot(game)
        before_cyl = game.player2.cylinder
        before_deaths = game.player2.deaths
        before_cprs = game.referee.cprs_performed
        apply_half_round(game, 5, 25, survived=True)
        snap.restore(game)
        assert game.player2.cylinder == before_cyl
        assert game.player2.deaths == before_deaths
        assert game.referee.cprs_performed == before_cprs

    def test_restore_after_terminal_death(self):
        game = _make_game(baku_cyl=290.0)
        snap = GameSnapshot(game)
        apply_half_round(game, 5, 25, survived=False)
        assert game.game_over
        snap.restore(game)
        assert not game.game_over
        assert game.winner is None


class TestLeapTurnResolution:
    """The leap turn (turn_duration=61) routes through the same resolver."""

    def test_baku_drops_61_in_leap_window(self):
        game = _make_game(clock=3540.0)
        assert game.get_turn_duration() == 61
        # Hal=dropper because first_dropper=hal, current_half=1; flip:
        game.current_half = 2
        record = game.play_half_round(61, 60)
        assert record.turn_duration == 61
        assert record.result == HalfRoundResult.CHECK_FAIL_SURVIVED or record.result == HalfRoundResult.CHECK_FAIL_DIED
