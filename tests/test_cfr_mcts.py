"""Tests for MCTS Step 1: data structures and node initialization."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from environment.cfr.exact_transition import ExactGameSnapshot
from environment.cfr.mcts import MCTSNode, make_node
from environment.cfr.tactical_scenarios import (
    forced_baku_overflow_death,
    leap_second_check_61_probe,
)
from src.Constants import PHYSICALITY_BAKU, PHYSICALITY_HAL
from src.Game import Game
from src.Player import Player
from src.Referee import Referee


def _make_fresh_game() -> Game:
    hal = Player(name="Hal", physicality=PHYSICALITY_HAL)
    baku = Player(name="Baku", physicality=PHYSICALITY_BAKU)
    game = Game(player1=hal, player2=baku, referee=Referee(), first_dropper=hal)
    game.seed(0)
    return game


def test_make_node_non_terminal_has_zero_initialized_matrices():
    scenario = forced_baku_overflow_death()
    node = make_node(scenario.game, scenario.config)

    D = len(node.drop_seconds)
    C = len(node.check_seconds)

    assert isinstance(node, MCTSNode)
    assert isinstance(node.drop_seconds, tuple)
    assert isinstance(node.check_seconds, tuple)
    assert D > 0
    assert C > 0

    assert node.Q.shape == (D, C)
    assert node.Q.dtype == np.float64
    assert node.Q.sum() == 0.0

    assert node.N_cell.shape == (D, C)
    assert node.N_cell.dtype == np.int64
    assert node.N_cell.sum() == 0

    assert node.N_node == 0
    assert node.is_expanded is False
    assert node.terminal_value is None
    assert node.children == {}
    assert isinstance(node.game_snapshot, ExactGameSnapshot)


def test_make_node_non_terminal_has_uniform_prior_summing_to_one():
    scenario = forced_baku_overflow_death()
    node = make_node(scenario.game, scenario.config)

    D = len(node.drop_seconds)
    C = len(node.check_seconds)

    assert node.prior.shape == (D, C)
    assert node.prior.dtype == np.float64
    assert node.prior.sum() == pytest.approx(1.0)
    assert np.allclose(node.prior, 1.0 / (D * C))


def test_make_node_stores_actual_seconds_not_indices():
    # The candidate generator returns specific exact seconds; the node must
    # carry them through verbatim, not synthesize a 0..D-1 index range.
    scenario = forced_baku_overflow_death()
    node = make_node(scenario.game, scenario.config)

    # All stored seconds must be in the legal 1..61 range.
    for s in node.drop_seconds + node.check_seconds:
        assert 1 <= s <= 61

    # None of the candidate seconds should be 0 (would indicate index leak).
    assert 0 not in node.drop_seconds
    assert 0 not in node.check_seconds


def test_make_node_uses_default_config_when_none_passed():
    # Should not raise AttributeError — config=None must be normalized internally.
    scenario = forced_baku_overflow_death()
    node = make_node(scenario.game)
    assert node.terminal_value is None
    assert len(node.drop_seconds) > 0


def test_make_node_at_terminal_hal_win_returns_plus_one_no_actions():
    game = _make_fresh_game()
    game.game_over = True
    game.winner = game.player1  # Hal
    game.loser = game.player2

    node = make_node(game)

    assert node.terminal_value == 1.0
    assert node.drop_seconds == ()
    assert node.check_seconds == ()
    assert node.Q.shape == (0, 0)
    assert node.N_cell.shape == (0, 0)
    assert node.prior.shape == (0, 0)
    assert node.N_node == 0
    assert node.is_expanded is False
    assert node.children == {}


def test_make_node_at_terminal_baku_win_returns_minus_one():
    game = _make_fresh_game()
    game.game_over = True
    game.winner = game.player2  # Baku
    game.loser = game.player1

    node = make_node(game)

    assert node.terminal_value == -1.0
    assert node.drop_seconds == ()
    assert node.check_seconds == ()


def test_make_node_snapshot_round_trips_position_after_mutation():
    # The snapshot must capture enough state that restore() returns the engine
    # to the exact position the node was constructed at.
    scenario = forced_baku_overflow_death()
    original_cyl = scenario.game.player2.cylinder
    original_clock = scenario.game.game_clock

    node = make_node(scenario.game, scenario.config)

    # Mutate the live game.
    scenario.game.player2.cylinder = 50.0
    scenario.game.game_clock = 9999.0

    # Restore from the snapshot stored on the node.
    node.game_snapshot.restore(scenario.game)

    assert scenario.game.player2.cylinder == original_cyl
    assert scenario.game.game_clock == original_clock


def test_make_node_in_leap_window_with_deduced_hal_includes_check_61():
    scenario = leap_second_check_61_probe()
    node = make_node(scenario.game, scenario.config)

    assert 61 in node.check_seconds


def test_make_node_children_dict_is_empty_and_independent_per_node():
    # Each node must own its own children dict — not a shared default mutable.
    scenario = forced_baku_overflow_death()
    node_a = make_node(scenario.game, scenario.config)
    node_b = make_node(scenario.game, scenario.config)

    assert node_a.children is not node_b.children
    node_a.children[(1, 2, None)] = node_b
    assert node_b.children == {}
