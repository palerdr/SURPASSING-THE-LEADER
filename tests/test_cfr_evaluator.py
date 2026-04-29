"""Tests for MCTS Step 2: leaf evaluator interface and terminal-only evaluator."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from environment.cfr.evaluator import LeafEvaluator, TerminalOnlyEvaluator
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


def _make_terminal_game(*, hal_wins: bool) -> Game:
    game = _make_fresh_game()
    game.game_over = True
    if hal_wins:
        game.winner = game.player1
        game.loser = game.player2
    else:
        game.winner = game.player2
        game.loser = game.player1
    return game


def test_terminal_only_evaluator_returns_zero_on_non_terminal_game():
    evaluator = TerminalOnlyEvaluator()
    game = _make_fresh_game()
    assert evaluator(game) == 0.0


def test_terminal_only_evaluator_returns_plus_one_on_hal_win_terminal():
    evaluator = TerminalOnlyEvaluator()
    game = _make_terminal_game(hal_wins=True)
    assert evaluator(game) == 1.0


def test_terminal_only_evaluator_returns_minus_one_on_baku_win_terminal():
    evaluator = TerminalOnlyEvaluator()
    game = _make_terminal_game(hal_wins=False)
    assert evaluator(game) == -1.0


def test_terminal_only_evaluator_perspective_name_flips_sign():
    # Same terminal Hal-win game, but the evaluator is built from Baku's
    # perspective. From Baku's POV, Hal winning is a loss → return -1.0.
    hal_evaluator = TerminalOnlyEvaluator(perspective_name="Hal")
    baku_evaluator = TerminalOnlyEvaluator(perspective_name="Baku")
    game = _make_terminal_game(hal_wins=True)
    assert hal_evaluator(game) == 1.0
    assert baku_evaluator(game) == -1.0


def test_terminal_only_evaluator_default_perspective_is_hal():
    evaluator = TerminalOnlyEvaluator()
    game = _make_terminal_game(hal_wins=True)
    # No perspective_name passed → default is "Hal" → Hal-win returns +1.
    assert evaluator(game) == 1.0


def test_terminal_only_evaluator_satisfies_leaf_evaluator_protocol_structurally():
    # Protocol conformance is structural in Python's typing system: any object
    # whose __call__ has the right signature satisfies LeafEvaluator. We verify
    # the contract by exercising a function that takes a LeafEvaluator-typed
    # argument with our concrete class.
    def call_with_protocol(evaluator: LeafEvaluator, game: Game) -> float:
        return evaluator(game)

    evaluator = TerminalOnlyEvaluator()
    game = _make_terminal_game(hal_wins=True)
    assert call_with_protocol(evaluator, game) == 1.0


def test_terminal_only_evaluator_returns_float_type():
    # The protocol promises a float. Callers (MCTS backup) rely on this for
    # numpy arithmetic, so make sure the return is actually a float, not None
    # leaking through, not int.
    evaluator = TerminalOnlyEvaluator()
    assert isinstance(evaluator(_make_fresh_game()), float)
    assert isinstance(evaluator(_make_terminal_game(hal_wins=True)), float)
    assert isinstance(evaluator(_make_terminal_game(hal_wins=False)), float)
