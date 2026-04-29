"""Integration tests for CanonicalHal — Phase 1 tickets 1.6-1.9."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

from src.Player import Player
from src.Referee import Referee
from src.Game import Game
from src.Constants import (
    PHYSICALITY_HAL, PHYSICALITY_BAKU, TURN_DURATION_NORMAL, TURN_DURATION_LEAP,
)
from hal.state import BeliefState, HalState, MemoryMode
from hal.state import update_memory
from hal.state import update_belief
from hal.hal_opponent import CanonicalHal


def make_game(seed=42, clock=720.0):
    hal = Player(name="Hal", physicality=PHYSICALITY_HAL)
    baku = Player(name="Baku", physicality=PHYSICALITY_BAKU)
    ref = Referee()
    g = Game(player1=hal, player2=baku, referee=ref, first_dropper=hal)
    g.seed(seed)
    g.game_clock = clock
    return g


# ── Memory model ──

class TestMemory:
    def test_normal_stays_normal_early(self):
        game = make_game(clock=720.0)
        assert update_memory(MemoryMode.NORMAL, game, False) == MemoryMode.NORMAL

    def test_normal_to_pre_amnesia_near_leap(self):
        game = make_game(clock=3300.0)
        assert update_memory(MemoryMode.NORMAL, game, False) == MemoryMode.PRE_AMNESIA

    def test_pre_amnesia_to_amnesia_on_leap_turn(self):
        game = make_game(clock=3540.0)
        assert game.is_leap_second_turn()
        assert update_memory(MemoryMode.PRE_AMNESIA, game, False) == MemoryMode.AMNESIA

    def test_pre_amnesia_stays_before_leap_turn(self):
        game = make_game(clock=3400.0)
        assert not game.is_leap_second_turn()
        assert update_memory(MemoryMode.PRE_AMNESIA, game, False) == MemoryMode.PRE_AMNESIA

    def test_amnesia_to_recovered_on_death(self):
        assert update_memory(MemoryMode.AMNESIA, make_game(), True) == MemoryMode.RECOVERED

    def test_amnesia_stays_without_death(self):
        assert update_memory(MemoryMode.AMNESIA, make_game(), False) == MemoryMode.AMNESIA

    def test_recovered_is_terminal(self):
        assert update_memory(MemoryMode.RECOVERED, make_game(), False) == MemoryMode.RECOVERED
        assert update_memory(MemoryMode.RECOVERED, make_game(), True) == MemoryMode.RECOVERED


# ── Belief tracker ──

class TestBelief:
    def _make_record(self, dropper="Hal", checker="Baku", drop_time=30, check_time=55):
        from src.Game import HalfRoundRecord, HalfRoundResult
        return HalfRoundRecord(
            round_num=0, half=1, dropper=dropper, checker=checker,
            drop_time=drop_time, check_time=check_time, turn_duration=60,
            result=HalfRoundResult.CHECK_SUCCESS, st_gained=check_time - drop_time,
            death_duration=0.0, survived=None, game_clock_at_start=720.0,
            survival_probability=None,
        )

    def test_initial_not_exploiting(self):
        b = BeliefState()
        assert not b.check_exploit
        assert not b.drop_exploit

    def test_tracks_baku_checks(self):
        b = BeliefState()
        r = self._make_record(checker="Baku", check_time=59)
        b = update_belief(b, r)
        assert b.baku_check_history == (59,)

    def test_tracks_baku_drops(self):
        b = BeliefState()
        r = self._make_record(dropper="Baku", drop_time=5)
        b = update_belief(b, r)
        assert b.baku_drop_history == (5,)

    def test_ignores_hal_actions(self):
        b = BeliefState()
        r = self._make_record(dropper="Hal", checker="Hal", drop_time=30, check_time=55)
        b = update_belief(b, r)
        assert b.baku_check_history == ()
        assert b.baku_drop_history == ()

    def test_exploitation_triggers_on_repetition(self):
        b = BeliefState()
        for _ in range(6):
            r = self._make_record(checker="Baku", check_time=59)
            b = update_belief(b, r)
        assert b.check_exploit
        assert b.baku_check_probs is not None

    def test_no_exploitation_with_varied_play(self):
        b = BeliefState()
        seconds = [5, 15, 30, 45, 55, 59]
        for s in seconds:
            r = self._make_record(checker="Baku", check_time=s)
            b = update_belief(b, r)
        assert not b.check_exploit

    def test_history_capped_at_10(self):
        b = BeliefState()
        for i in range(15):
            r = self._make_record(checker="Baku", check_time=50 + (i % 10))
            b = update_belief(b, r)
        assert len(b.baku_check_history) == 10


# ── CanonicalHal opponent ──

class TestCanonicalHal:
    def test_returns_valid_second(self):
        game = make_game()
        hal = CanonicalHal(seed=0, depth=1)
        second = hal.choose_action(game, "dropper", TURN_DURATION_NORMAL)
        assert 1 <= second <= TURN_DURATION_NORMAL

    def test_reset_clears_state(self):
        hal = CanonicalHal(seed=0)
        hal._state = HalState(memory=MemoryMode.AMNESIA, leap_deduced=True)
        hal.reset()
        assert hal._state.memory == MemoryMode.NORMAL
        assert not hal._state.leap_deduced

    def test_checker_action_valid(self):
        game = make_game()
        game.current_half = 2
        hal = CanonicalHal(seed=0, depth=1)
        second = hal.choose_action(game, "checker", TURN_DURATION_NORMAL)
        assert 1 <= second <= TURN_DURATION_NORMAL

    def test_plays_full_game_without_crash(self):
        game = make_game(seed=7)
        hal_ai = CanonicalHal(seed=7, depth=1)
        baku_actions = [35, 55, 30, 58, 25, 59, 40, 57, 20, 56]
        turn = 0
        while not game.game_over and turn < 20:
            dropper, checker = game.get_roles_for_half(game.current_half)
            td = game.get_turn_duration()
            if dropper.name.lower() == "hal":
                hal_action = hal_ai.choose_action(game, "dropper", td)
                baku_action = baku_actions[turn % len(baku_actions)]
                game.play_half_round(hal_action, baku_action)
            else:
                baku_action = baku_actions[turn % len(baku_actions)]
                hal_action = hal_ai.choose_action(game, "checker", td)
                game.play_half_round(baku_action, hal_action)
            turn += 1
        assert turn > 0


class TestCanonicalHalLeapLegality:
    """Item 20: regression for Hal's actor-aware leap-second legality."""

    def _leap_game(self, *, current_half: int = 1) -> "Game":
        from src.Game import Game
        from src.Player import Player
        from src.Referee import Referee
        from src.Constants import PHYSICALITY_HAL, PHYSICALITY_BAKU

        hal = Player(name="Hal", physicality=PHYSICALITY_HAL)
        baku = Player(name="Baku", physicality=PHYSICALITY_BAKU)
        ref = Referee()
        game = Game(player1=hal, player2=baku, referee=ref, first_dropper=hal)
        game.seed(42)
        game.game_clock = 3540.0
        game.current_half = current_half
        return game

    def test_hal_dropper_never_chooses_61_on_leap_turn(self):
        game = self._leap_game(current_half=1)
        hal_ai = CanonicalHal(seed=0, depth=1, use_adaptive=False)
        hal_ai._state = HalState(memory=MemoryMode.NORMAL, leap_deduced=True)

        for _ in range(10):
            action = hal_ai.choose_action(game, "dropper", 61)
            assert 1 <= action <= 60, f"Hal dropped {action} on leap turn"

    def test_hal_checker_can_pick_61_when_deduced(self):
        game = self._leap_game(current_half=2)
        hal_ai = CanonicalHal(seed=42, depth=1, use_adaptive=False)
        hal_ai._state = HalState(memory=MemoryMode.NORMAL, leap_deduced=True)

        for _ in range(20):
            action = hal_ai.choose_action(game, "checker", 61)
            assert 1 <= action <= 61

    def test_hal_checker_capped_at_60_when_amnesia(self):
        game = self._leap_game(current_half=2)
        hal_ai = CanonicalHal(seed=0, depth=1, use_adaptive=False)
        hal_ai._state = HalState(memory=MemoryMode.AMNESIA, leap_deduced=True)

        for _ in range(20):
            action = hal_ai.choose_action(game, "checker", 61)
            assert 1 <= action <= 60, f"Hal checked {action} despite amnesia"

    def test_hal_checker_capped_at_60_when_undeduced(self):
        game = self._leap_game(current_half=2)
        hal_ai = CanonicalHal(seed=0, depth=1, use_adaptive=False)
        hal_ai._state = HalState(memory=MemoryMode.NORMAL, leap_deduced=False)

        for _ in range(20):
            action = hal_ai.choose_action(game, "checker", 61)
            assert 1 <= action <= 60, f"Hal checked {action} without deducing leap"


class TestSearchMemoryRecursion:
    """Item 20: search must propagate memory updates recursively as the clock advances."""

    def test_memory_can_advance_when_simulating_into_leap_window(self):
        from src.Game import Game
        from src.Player import Player
        from src.Referee import Referee
        from src.Constants import PHYSICALITY_HAL, PHYSICALITY_BAKU
        from hal.search import search

        hal = Player(name="Hal", physicality=PHYSICALITY_HAL)
        baku = Player(name="Baku", physicality=PHYSICALITY_BAKU)
        game = Game(player1=hal, player2=baku, referee=Referee(), first_dropper=hal)
        game.seed(0)
        game.game_clock = 3000.0  # before PRE_AMNESIA threshold

        belief = BeliefState()
        strategy, value = search(game, depth=1, belief=belief, memory=MemoryMode.NORMAL, leap_deduced=True)
        assert strategy is not None
        assert -1.0 <= value <= 1.0
