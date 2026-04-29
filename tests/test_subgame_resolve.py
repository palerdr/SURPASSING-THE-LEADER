"""Tests for Phase 6: subgame re-solve at critical states.

Verifies the ``is_critical`` heuristic, the ``resolve_subgame`` deeper-horizon
fresh selective solve, and the optional ``subgame_resolve_at_critical`` MCTS
hook. The hook is opt-in: default behavior must match the existing MCTS
output exactly when the kwarg is False.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from environment.cfr.evaluator import TerminalOnlyEvaluator
from environment.cfr.mcts import MCTSConfig, mcts_search
from environment.cfr.subgame_resolve import is_critical, resolve_subgame
from environment.cfr.tactical_scenarios import (
    forced_baku_overflow_death,
)
from src.Constants import (
    PHYSICALITY_BAKU,
    PHYSICALITY_HAL,
    OPENING_START_CLOCK,
)
from src.Game import Game
from src.Player import Player
from src.Referee import Referee


def _config(iterations: int = 100, exploration_c: float = 1.0) -> MCTSConfig:
    return MCTSConfig(
        iterations=iterations,
        exploration_c=exploration_c,
        evaluator=None,
        use_tablebase=False,
    )


def _fresh_game_far_from_leap() -> Game:
    """Game at canonical R1 start, both cylinders empty, far from leap window."""
    hal = Player(name="Hal", physicality=PHYSICALITY_HAL)
    baku = Player(name="Baku", physicality=PHYSICALITY_BAKU)
    game = Game(player1=hal, player2=baku, referee=Referee(), first_dropper=hal)
    game.seed(0)
    game.game_clock = float(OPENING_START_CLOCK)
    game.current_half = 1
    return game


# ── 1. is_critical ────────────────────────────────────────────────────────


def test_is_critical_true_for_forced_baku_overflow_death():
    scenario = forced_baku_overflow_death()
    assert is_critical(scenario.game) is True


def test_is_critical_false_for_fresh_game_far_from_leap():
    game = _fresh_game_far_from_leap()
    assert is_critical(game) is False


# ── 2. resolve_subgame ────────────────────────────────────────────────────


def test_resolve_subgame_pins_forced_baku_overflow_to_plus_one():
    scenario = forced_baku_overflow_death()
    result = resolve_subgame(scenario.game, horizon=4, config=scenario.config)
    assert result.value_for_hal == pytest.approx(1.0, abs=1e-6)


def test_resolve_subgame_default_horizon_runs_without_error():
    scenario = forced_baku_overflow_death()
    result = resolve_subgame(scenario.game)
    assert result.value_for_hal == pytest.approx(1.0, abs=1e-6)


# ── 3. mcts_search subgame_resolve_at_critical hook ───────────────────────


def test_mcts_search_with_resolve_at_critical_state_returns_result():
    scenario = forced_baku_overflow_death()
    rng = np.random.default_rng(0)
    result = mcts_search(
        scenario.game,
        _config(iterations=50),
        TerminalOnlyEvaluator(),
        rng,
        scenario.config,
        subgame_resolve_at_critical=True,
    )
    assert result.root_value_for_hal == pytest.approx(1.0, abs=0.05)
    assert result.root_strategy_dropper.sum() == pytest.approx(1.0)
    assert result.root_strategy_checker.sum() == pytest.approx(1.0)


def test_mcts_search_default_no_resolve_still_succeeds():
    scenario = forced_baku_overflow_death()
    rng = np.random.default_rng(0)
    result = mcts_search(
        scenario.game,
        _config(iterations=50),
        TerminalOnlyEvaluator(),
        rng,
        scenario.config,
    )
    assert result.root_value_for_hal == pytest.approx(1.0, abs=0.05)


def test_mcts_search_resolve_false_matches_default_behavior():
    scenario_a = forced_baku_overflow_death()
    rng_a = np.random.default_rng(7)
    result_a = mcts_search(
        scenario_a.game,
        _config(iterations=50),
        TerminalOnlyEvaluator(),
        rng_a,
        scenario_a.config,
        subgame_resolve_at_critical=False,
    )

    scenario_b = forced_baku_overflow_death()
    rng_b = np.random.default_rng(7)
    result_b = mcts_search(
        scenario_b.game,
        _config(iterations=50),
        TerminalOnlyEvaluator(),
        rng_b,
        scenario_b.config,
    )

    assert result_a.root_value_for_hal == result_b.root_value_for_hal
    np.testing.assert_array_equal(
        result_a.root_strategy_dropper, result_b.root_strategy_dropper
    )
    np.testing.assert_array_equal(
        result_a.root_strategy_checker, result_b.root_strategy_checker
    )


def test_mcts_search_resolve_at_non_critical_state_matches_default():
    """At a non-critical state, the resolve hook should not fire so behavior
    must equal the default subgame_resolve_at_critical=False path bit-for-bit
    under the same RNG seed."""
    game_a = _fresh_game_far_from_leap()
    rng_a = np.random.default_rng(123)
    result_a = mcts_search(
        game_a,
        _config(iterations=40),
        TerminalOnlyEvaluator(),
        rng_a,
        subgame_resolve_at_critical=True,
    )

    game_b = _fresh_game_far_from_leap()
    rng_b = np.random.default_rng(123)
    result_b = mcts_search(
        game_b,
        _config(iterations=40),
        TerminalOnlyEvaluator(),
        rng_b,
        subgame_resolve_at_critical=False,
    )

    assert is_critical(_fresh_game_far_from_leap()) is False
    assert result_a.root_value_for_hal == result_b.root_value_for_hal
    np.testing.assert_array_equal(
        result_a.root_strategy_dropper, result_b.root_strategy_dropper
    )
    np.testing.assert_array_equal(
        result_a.root_strategy_checker, result_b.root_strategy_checker
    )
