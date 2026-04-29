"""Tests for candidate exact-second generation."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from environment.cfr.selective import (
    CRITICAL_SECONDS,
    generate_candidates,
    overflow_st_threshold,
    safe_st_budget,
)
from environment.cfr.exact import ExactSearchConfig
from environment.cfr.tactical_scenarios import (
    leap_second_check_61_probe,
    safe_budget_pressure_at_cylinder_240,
    safe_budget_pressure_at_cylinder_241,
)
from src.Constants import PHYSICALITY_BAKU, PHYSICALITY_HAL
from src.Game import Game
from src.Player import Player
from src.Referee import Referee


def make_game(*, clock=720.0, current_half=1, baku_cyl=0.0):
    hal = Player(name="Hal", physicality=PHYSICALITY_HAL)
    baku = Player(name="Baku", physicality=PHYSICALITY_BAKU)
    baku.cylinder = baku_cyl
    game = Game(player1=hal, player2=baku, referee=Referee(), first_dropper=hal)
    game.seed(0)
    game.game_clock = clock
    game.current_half = current_half
    return game


def test_critical_seconds_constant_lists_documented_values():
    assert CRITICAL_SECONDS == (1, 2, 58, 59, 60, 61)


def test_overflow_threshold_at_low_cylinder_uses_full_distance():
    assert overflow_st_threshold(0) == 300


def test_overflow_threshold_at_or_past_overflow_clamps_to_one():
    assert overflow_st_threshold(299) == 1
    assert overflow_st_threshold(300) == 1
    assert overflow_st_threshold(310) == 1


def test_safe_st_budget_at_threshold_boundaries():
    assert safe_st_budget(240) == 59
    assert safe_st_budget(241) == 58
    assert safe_st_budget(300) == 0


def test_candidates_include_legal_critical_seconds_at_normal_turn():
    game = make_game()
    candidates = generate_candidates(game)

    assert {1, 2, 58, 59, 60}.issubset(set(candidates.drop_seconds))
    assert {1, 2, 58, 59, 60}.issubset(set(candidates.check_seconds))


def test_candidates_exclude_61_at_normal_turn_even_when_deduced():
    # Outside the leap window, 61 is illegal regardless of deduction state.
    game = make_game(clock=720.0)
    candidates = generate_candidates(game, ExactSearchConfig(hal_leap_deduced=True))
    assert 61 not in candidates.drop_seconds
    assert 61 not in candidates.check_seconds


def test_candidates_include_drop_61_for_baku_dropper_in_leap_window():
    # At clock=3540, current_half=2 and first_dropper=hal, Baku is the dropper
    # and Baku always knows 61 is legal in the leap window.
    game = make_game(clock=3540.0, current_half=2)
    candidates = generate_candidates(game)
    assert 61 in candidates.drop_seconds


def test_candidates_include_check_61_only_when_hal_deduced():
    scenario = leap_second_check_61_probe()
    candidates = generate_candidates(scenario.game, scenario.config)
    assert 61 in candidates.check_seconds


def test_candidates_omit_check_61_when_hal_unaware():
    # Same leap-window state but with hal_leap_deduced=False:
    base = leap_second_check_61_probe()
    candidates = generate_candidates(base.game, ExactSearchConfig(hal_leap_deduced=False))
    assert 61 not in candidates.check_seconds


def test_candidates_include_overflow_threshold_seconds_at_high_cylinder():
    # At cyl=241 (overflow_st=59) drop=1 plus check=60 forces a terminal cell.
    scenario = safe_budget_pressure_at_cylinder_241()
    candidates = generate_candidates(scenario.game, scenario.config)
    assert 1 in candidates.drop_seconds
    assert 60 in candidates.check_seconds


def test_candidates_include_safe_check_boundary_check_at_low_drop():
    # safe_st = 59 at cyl=240; drop=1 paired with check=1+59=60 sits on the boundary.
    scenario = safe_budget_pressure_at_cylinder_240()
    candidates = generate_candidates(scenario.game, scenario.config)
    assert 1 in candidates.drop_seconds
    assert 60 in candidates.check_seconds


def test_candidates_smaller_than_full_width_at_normal_turn():
    game = make_game()
    candidates = generate_candidates(game)
    assert candidates.joint_count < 60 * 60


def test_candidates_return_sorted_distinct_seconds():
    game = make_game()
    candidates = generate_candidates(game)
    assert list(candidates.drop_seconds) == sorted(set(candidates.drop_seconds))
    assert list(candidates.check_seconds) == sorted(set(candidates.check_seconds))
    assert candidates.drop_seconds[0] >= 1
    assert candidates.check_seconds[0] >= 1
