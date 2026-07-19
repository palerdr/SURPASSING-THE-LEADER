"""Tests for Phase 6: subgame re-solve at critical states.

Verifies the ``is_critical`` heuristic, the bounded ``resolve_subgame`` fresh
selective solve, and the optional ``subgame_resolve_at_critical`` MCTS
hook. The hook is opt-in: default behavior must match the existing MCTS
output exactly when the kwarg is False.
"""

from __future__ import annotations

import importlib.util
import os
import sys
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, os.getcwd())

from stl.solver.search import TerminalOnlyEvaluator
from stl.solver.search import MCTSConfig, mcts_search, selective_solve
from stl.engine.actions import ACTION_SIZE
from stl.solver.search import is_critical, resolve_subgame
from stl.solver.exact import CFRPlusConfig
from stl.solver.tablebase import (
    forced_baku_overflow_death,
)
from stl.engine.game import (
    PHYSICALITY_BAKU,
    PHYSICALITY_HAL,
    OPENING_START_CLOCK,
)
from stl.engine.game import Game
from stl.engine.game import Player
from stl.engine.game import Referee


def _config(iterations: int = 100, exploration_c: float = 1.0) -> MCTSConfig:
    return MCTSConfig(
        iterations=iterations,
        exploration_c=exploration_c,
    )


needs_rust_cfr = pytest.mark.skipif(
    importlib.util.find_spec("stl_solver_rs") is None,
    reason="optional stl_solver_rs extension is not installed",
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


def test_is_critical_false_for_active_lsr_hal_checker_state():
    game = _fresh_game_far_from_leap()
    game.game_clock = 29 * 60.0
    game.current_half = 2

    assert is_critical(game) is False


# ── 2. resolve_subgame ────────────────────────────────────────────────────


def test_resolve_subgame_pins_forced_baku_overflow_to_plus_one():
    scenario = forced_baku_overflow_death()
    result = resolve_subgame(
        scenario.game,
        horizon=4,
        config=scenario.config,
        solver="lp",
    )
    assert result.value_for_hal == pytest.approx(1.0, abs=1e-6)


def test_resolve_subgame_default_is_bounded_current_turn():
    scenario = forced_baku_overflow_death()
    result = resolve_subgame(scenario.game)
    assert result.half_round_horizon == 1
    assert result.value_for_hal == pytest.approx(1.0, abs=1e-6)


@needs_rust_cfr
def test_resolve_subgame_accepts_rust_cfr_plus_solver():
    scenario = forced_baku_overflow_death()
    result = resolve_subgame(
        scenario.game,
        horizon=1,
        config=scenario.config,
        solver="rust_cfr_plus",
        cfr_plus_config=CFRPlusConfig(iterations=200, average_delay=20),
    )

    assert result.value_for_hal == pytest.approx(1.0, abs=1e-6)
    assert result.dropper_strategy.sum() == pytest.approx(1.0)
    assert result.checker_strategy.sum() == pytest.approx(1.0)


def test_anchored_resolve_value_strictly_closer_to_deep_mcts_value_than_unanchored():
    """Phase D's central claim: at a critical state, anchoring the boundary at
    a value-net evaluator pulls the local solve closer to deep ground truth
    than letting the boundary return unresolved=0.

    On ``forced_baku_overflow_death``, the deep (horizon=4) LP selective solve
    converges to +1.0 (Baku's cylinder is at threshold; every action injects).
    A horizon=0 unanchored solve returns the unresolved frontier value (~0.0,
    the pessimistic stand-in). A horizon=0 solve anchored to an evaluator that
    returns 0.9 — close to deep truth — must produce a value strictly closer
    to 1.0 than the unanchored 0.0. If anchoring is silently a no-op (the
    evaluator's value is discarded), this assertion fails.
    """
    scenario = forced_baku_overflow_death()

    deep_result = resolve_subgame(
        scenario.game,
        horizon=4,
        config=scenario.config,
        solver="lp",
    )
    deep_value = deep_result.value_for_hal
    assert deep_value == pytest.approx(1.0, abs=1e-6), (
        f"Setup violated: forced overflow should resolve to +1.0 deep, got {deep_value}"
    )

    unanchored = resolve_subgame(scenario.game, horizon=0, config=scenario.config)
    unanchored_diff = abs(unanchored.value_for_hal - deep_value)
    assert unanchored_diff > 0.1, (
        f"Setup violated: horizon=0 unanchored should be far from deep truth at "
        f"a critical state. Got unanchored={unanchored.value_for_hal}, deep={deep_value}."
    )

    class NearTruthEvaluator:
        def __call__(self, game):
            del game
            return 0.9, np.zeros(ACTION_SIZE), np.zeros(ACTION_SIZE)

    anchored = resolve_subgame(
        scenario.game,
        horizon=0,
        config=scenario.config,
        evaluator=NearTruthEvaluator(),
    )
    anchored_diff = abs(anchored.value_for_hal - deep_value)

    assert anchored_diff < unanchored_diff, (
        f"Anchored value not strictly closer to deep truth than unanchored: "
        f"|anchored {anchored.value_for_hal} - deep {deep_value}| = {anchored_diff:.4f}, "
        f"|unanchored {unanchored.value_for_hal} - deep {deep_value}| = {unanchored_diff:.4f}."
    )
    assert anchored.unresolved_probability == pytest.approx(0.0), (
        "Anchored boundary must mark unresolved_probability=0 (DeepStack pattern)."
    )


def test_resolve_subgame_passes_evaluator_through_to_selective():
    game = _fresh_game_far_from_leap()

    class FixedEvaluator:
        def __call__(self, state):
            del state
            return 0.33, np.zeros(ACTION_SIZE), np.zeros(ACTION_SIZE)

    result = resolve_subgame(game, horizon=0, evaluator=FixedEvaluator())

    assert result.value_for_hal == pytest.approx(0.33)
    assert result.unresolved_probability == pytest.approx(0.0)


def test_selective_solve_cache_reuses_same_frontier_state():
    game = _fresh_game_far_from_leap()
    calls = 0

    class CountingEvaluator:
        def __call__(self, game):
            nonlocal calls
            calls += 1
            return 0.25

    cache = {}
    first = selective_solve(game, 0, evaluator=CountingEvaluator(), _cache=cache)
    second = selective_solve(game, 0, evaluator=CountingEvaluator(), _cache=cache)

    assert first.value_for_hal == pytest.approx(0.25)
    assert second.value_for_hal == pytest.approx(0.25)
    assert calls == 1


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


@needs_rust_cfr
def test_mcts_search_with_rust_resolve_at_critical_state_returns_result():
    scenario = forced_baku_overflow_death()
    rng = np.random.default_rng(0)
    result = mcts_search(
        scenario.game,
        _config(iterations=50),
        TerminalOnlyEvaluator(),
        rng,
        scenario.config,
        subgame_resolve_at_critical=True,
        subgame_resolve_solver="rust_cfr_plus",
        subgame_resolve_cfr_plus_config=CFRPlusConfig(iterations=200, average_delay=20),
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
