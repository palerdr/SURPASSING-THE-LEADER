"""Tests for MCTS Step 6: the mcts_search loop end-to-end."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from environment.cfr.evaluator import TerminalOnlyEvaluator
from environment.cfr.mcts import MCTSConfig, mcts_search
from environment.cfr.tactical_scenarios import (
    forced_baku_overflow_death,
    forced_hal_overflow_death,
    safe_budget_pressure_at_cylinder_241,
)


def _config(iterations: int, exploration_c: float = 1.0) -> MCTSConfig:
    # MCTSConfig still has evaluator and use_tablebase fields from earlier
    # stubs; they're unused by mcts_search. Pass dummies.
    return MCTSConfig(
        iterations=iterations,
        exploration_c=exploration_c,
        evaluator=None,
        use_tablebase=False,
    )


def test_mcts_search_on_forced_baku_overflow_converges_to_plus_one():
    # Every joint action at cyl=299 is terminal Hal-win. After enough
    # iterations to populate all candidate cells, the Nash value is +1.0.
    scenario = forced_baku_overflow_death()
    rng = np.random.default_rng(0)
    evaluator = TerminalOnlyEvaluator()

    result = mcts_search(
        scenario.game,
        _config(iterations=500),
        evaluator,
        rng,
        scenario.config,
    )

    assert result.root_value_for_hal == pytest.approx(1.0, abs=0.05)


def test_mcts_search_on_forced_hal_overflow_converges_to_minus_one():
    # Symmetric: every cell terminal Baku-win.
    scenario = forced_hal_overflow_death()
    rng = np.random.default_rng(0)
    evaluator = TerminalOnlyEvaluator()

    result = mcts_search(
        scenario.game,
        _config(iterations=500),
        evaluator,
        rng,
        scenario.config,
    )

    assert result.root_value_for_hal == pytest.approx(-1.0, abs=0.05)


def test_mcts_search_returns_proper_probability_distributions():
    scenario = forced_baku_overflow_death()
    rng = np.random.default_rng(7)
    evaluator = TerminalOnlyEvaluator()

    result = mcts_search(
        scenario.game,
        _config(iterations=100),
        evaluator,
        rng,
        scenario.config,
    )

    assert result.root_strategy_dropper.sum() == pytest.approx(1.0)
    assert result.root_strategy_checker.sum() == pytest.approx(1.0)
    assert (result.root_strategy_dropper >= 0).all()
    assert (result.root_strategy_checker >= 0).all()


def test_mcts_search_is_deterministic_under_same_rng_seed():
    # Two runs with seeded RNGs (and fresh game state per run) must produce
    # identical strategies and value. Catches the iter=1 return bug because
    # determinism is only meaningful if all iterations actually run.
    cfg = _config(iterations=100)
    evaluator = TerminalOnlyEvaluator()

    scenario_a = forced_baku_overflow_death()
    rng_a = np.random.default_rng(42)
    result_a = mcts_search(scenario_a.game, cfg, evaluator, rng_a, scenario_a.config)

    scenario_b = forced_baku_overflow_death()
    rng_b = np.random.default_rng(42)
    result_b = mcts_search(scenario_b.game, cfg, evaluator, rng_b, scenario_b.config)

    assert result_a.root_value_for_hal == result_b.root_value_for_hal
    np.testing.assert_array_equal(result_a.root_strategy_dropper, result_b.root_strategy_dropper)
    np.testing.assert_array_equal(result_a.root_strategy_checker, result_b.root_strategy_checker)


def test_mcts_search_visit_count_grows_with_iterations():
    # If iterations actually run (and the loop doesn't return early), root
    # visit count should grow with the iteration budget. Catches the
    # `return inside loop` bug.
    cfg_small = _config(iterations=20)
    cfg_large = _config(iterations=200)
    evaluator = TerminalOnlyEvaluator()

    scenario_a = forced_baku_overflow_death()
    rng_a = np.random.default_rng(0)
    result_small = mcts_search(scenario_a.game, cfg_small, evaluator, rng_a, scenario_a.config)

    scenario_b = forced_baku_overflow_death()
    rng_b = np.random.default_rng(0)
    result_large = mcts_search(scenario_b.game, cfg_large, evaluator, rng_b, scenario_b.config)

    assert result_large.root_visits > result_small.root_visits
    # Specifically: small ~ 20 - 1 (first iter is the eval-root no-path step),
    # large ~ 200 - 1.  Use loose bounds.
    assert result_small.root_visits >= 15
    assert result_large.root_visits >= 150


def test_mcts_search_converges_to_half_on_safe_budget_pressure_241():
    # The full-width LP value at this position is +0.5 (from earlier analysis).
    # MCTS at high iteration count should approach that.
    scenario = safe_budget_pressure_at_cylinder_241()
    rng = np.random.default_rng(123)
    evaluator = TerminalOnlyEvaluator()

    result = mcts_search(
        scenario.game,
        _config(iterations=1500),
        evaluator,
        rng,
        scenario.config,
    )

    # 0.5 is the analytic Nash value. With TerminalOnlyEvaluator and no
    # transposition cache, MCTS at 1500 iterations gets close but not tight;
    # loosen the tolerance to document "converges" without overclaiming.
    assert result.root_value_for_hal == pytest.approx(0.5, abs=0.15)
