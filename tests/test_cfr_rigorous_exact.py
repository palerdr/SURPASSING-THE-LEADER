"""Tests for the rigorous exact-second CFR foundation."""

import os
import pathlib
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from environment.cfr.exact import (
    clear_solve_cache,
    evaluate_joint_action,
    exact_immediate_checker_payoff_matrix,
    solve_cache_size,
    solve_exact_finite_horizon,
)
from environment.cfr.exact import exact_public_state
from environment.cfr.exact import (
    ExactJointAction,
    ExactSearchConfig,
    enumerate_joint_actions,
    expand_joint_action,
)
from environment.cfr.half_round import compute_payoff_matrix
from environment.cfr.diagnostics import diagnose_exact_strategy
from environment.cfr.exact import solve_minimax
from environment.cfr.tactical_scenarios import forced_baku_overflow_death
from environment.cfr.exact import terminal_value
from src.Constants import PHYSICALITY_BAKU, PHYSICALITY_HAL, TURN_DURATION_NORMAL
from src.Game import Game
from src.Player import Player
from src.Referee import Referee


def make_game(*, clock: float = 720.0, current_half: int = 1, baku_cyl: float = 0.0) -> Game:
    hal = Player(name="Hal", physicality=PHYSICALITY_HAL)
    baku = Player(name="Baku", physicality=PHYSICALITY_BAKU)
    baku.cylinder = baku_cyl
    game = Game(player1=hal, player2=baku, referee=Referee(), first_dropper=hal)
    game.seed(123)
    game.game_clock = clock
    game.current_half = current_half
    return game


def test_exact_immediate_matrix_matches_existing_exact_half_round_matrix():
    game = make_game()

    exact = exact_immediate_checker_payoff_matrix(game)
    legacy_exact = compute_payoff_matrix(0.0, turn_duration=TURN_DURATION_NORMAL)

    assert exact.drop_actions == tuple(range(1, 61))
    assert exact.check_actions == tuple(range(1, 61))
    np.testing.assert_array_equal(exact.payoff, legacy_exact)


def test_cfr_owned_minimax_solver_solves_matching_pennies():
    payoff = np.array([[1.0, -1.0], [-1.0, 1.0]])

    strategy, value = solve_minimax(payoff)

    np.testing.assert_allclose(strategy, [0.5, 0.5], atol=1e-4)
    assert value == pytest.approx(0.0, abs=1e-4)


def test_exact_joint_actions_use_actor_aware_leap_legality():
    # Hal can never check at 61 (hard-coded in legal_actions). Baku-dropper
    # at clock=3540 inside the leap window can drop at 61. Result: 61 × 60
    # joint actions, with (61, 61) not in the legal set.
    game = make_game(clock=3540.0, current_half=2)

    actions = enumerate_joint_actions(game, ExactSearchConfig())

    assert len(actions) == 61 * 60
    assert ExactJointAction(61, 61) not in actions
    assert ExactJointAction(61, 60) in actions


def test_expand_joint_action_branches_death_chance_and_restores_game():
    game = make_game()
    before = exact_public_state(game)

    transitions = expand_joint_action(game, ExactJointAction(drop_time=45, check_time=30))

    assert exact_public_state(game) == before
    assert len(transitions) == 2
    assert sum(t.probability for t in transitions) == pytest.approx(1.0)
    assert {t.record.survived for t in transitions} == {True, False}
    assert any(t.terminal_value == 1.0 for t in transitions)


def test_exact_state_distinguishes_values_legacy_buckets_collapse():
    a = make_game(baku_cyl=59.0)
    b = make_game(baku_cyl=60.0)
    c = make_game(clock=3539.0)
    d = make_game(clock=3540.0)

    assert exact_public_state(a) != exact_public_state(b)
    assert exact_public_state(c) != exact_public_state(d)


def test_evaluate_joint_action_reports_unresolved_without_frontier_heuristic():
    game = make_game()

    value = evaluate_joint_action(
        game,
        ExactJointAction(drop_time=1, check_time=1),
        half_round_horizon=1,
    )

    assert value.value == 0.0
    assert value.unresolved_probability == 1.0
    assert terminal_value(game) is None


def test_evaluate_joint_action_weights_terminal_death_branch():
    game = make_game()

    value = evaluate_joint_action(
        game,
        ExactJointAction(drop_time=45, check_time=30),
        half_round_horizon=1,
    )

    assert -1.0 <= value.value <= 1.0
    assert value.hal_win_probability > 0.0
    assert value.unresolved_probability > 0.0
    assert value.hal_win_probability + value.unresolved_probability == pytest.approx(1.0)


def test_finite_horizon_solver_returns_exact_seconds_and_unresolved_mass():
    game = make_game()

    result = solve_exact_finite_horizon(game, half_round_horizon=1)

    assert result.dropper_strategy.shape == (60,)
    assert result.checker_strategy.shape == (60,)
    assert result.dropper_strategy.sum() == pytest.approx(1.0)
    assert result.checker_strategy.sum() == pytest.approx(1.0)
    assert result.unresolved_probability >= 0.0
    assert result.breakdown.hal_win_probability + result.breakdown.baku_win_probability + result.breakdown.unresolved_probability == pytest.approx(1.0)


def test_diagnostics_report_zero_gap_for_exact_minimax_solution():
    scenario = forced_baku_overflow_death()

    result = solve_exact_finite_horizon(
        scenario.game,
        half_round_horizon=scenario.half_round_horizon,
        config=scenario.config,
    )
    diagnostics = diagnose_exact_strategy(scenario.game, result)

    assert diagnostics.expected_value == pytest.approx(1.0)
    assert diagnostics.nash_gap == pytest.approx(0.0)
    assert diagnostics.dropper_exploitability == pytest.approx(0.0)
    assert diagnostics.checker_exploitability == pytest.approx(0.0)


def test_forced_overflow_tactical_scenario_is_exact_terminal_tablebase():
    scenario = forced_baku_overflow_death()

    result = solve_exact_finite_horizon(
        scenario.game,
        half_round_horizon=scenario.half_round_horizon,
        config=scenario.config,
    )

    assert result.value_for_hal == pytest.approx(1.0)
    assert result.breakdown.hal_win_probability == pytest.approx(1.0)
    assert result.unresolved_probability == pytest.approx(0.0)
    assert result.payoff_for_hal is not None
    np.testing.assert_array_equal(result.payoff_for_hal, np.ones((60, 60)))


def test_rigorous_cfr_modules_do_not_import_reward_or_value_heuristics():
    root = pathlib.Path(__file__).resolve().parents[1]
    rigorous_files = [
        root / "environment/cfr/exact.py",
        root / "environment/cfr/diagnostics.py",
        root / "environment/cfr/tactical_scenarios.py",
        root / "environment/cfr/tablebase.py",
        root / "environment/cfr/timing_features.py",
        root / "environment/cfr/selective.py",
    ]
    forbidden = (
        "environment.reward",
        "ROUTE_SHAPING",
        "shaped_reward",
        "hal.evaluate",
        "value_net",
        "_handcrafted_evaluate",
        "0.3 *",
        "0.7 *",
    )

    for path in rigorous_files:
        source = path.read_text()
        for marker in forbidden:
            assert marker not in source, f"{path} imports or references heuristic marker {marker!r}"


# ── Memoization (cache for solve_exact_finite_horizon) ───────────────────


def test_solve_cache_returns_bit_identical_results_with_and_without_cache():
    """Cached value/strategies/breakdown must match the uncached recursion
    bit-for-bit. This is correctness — if the cache key misses any state
    component, two different subproblems collide and corrupt the corpus."""
    scenario = forced_baku_overflow_death()
    game1 = scenario.game
    config = scenario.config

    clear_solve_cache()
    uncached = solve_exact_finite_horizon(game1, half_round_horizon=4, config=config)
    cached_size_after_first = solve_cache_size()
    assert cached_size_after_first > 0, (
        "Recursion at horizon=4 must populate the cache for non-terminal states."
    )

    # Re-build a fresh game so we are not fooled by snapshot state-restore quirks.
    scenario2 = forced_baku_overflow_death()
    game2 = scenario2.game
    cached = solve_exact_finite_horizon(game2, half_round_horizon=4, config=config)

    assert uncached.value_for_hal == pytest.approx(cached.value_for_hal)
    assert uncached.unresolved_probability == pytest.approx(cached.unresolved_probability)
    np.testing.assert_array_equal(uncached.dropper_strategy, cached.dropper_strategy)
    np.testing.assert_array_equal(uncached.checker_strategy, cached.checker_strategy)
    assert uncached.drop_actions == cached.drop_actions
    assert uncached.check_actions == cached.check_actions


def test_solve_cache_is_populated_during_recursion():
    """A horizon=3 solve at a near-overflow state must hit the cache at least
    once during its recursion. If size after one solve == size during cold,
    no recursive subproblems are sharing — the cache is dead."""
    clear_solve_cache()
    assert solve_cache_size() == 0
    scenario = forced_baku_overflow_death()
    solve_exact_finite_horizon(scenario.game, half_round_horizon=3, config=scenario.config)
    assert solve_cache_size() > 0


def test_clear_solve_cache_resets_to_empty():
    clear_solve_cache()
    scenario = forced_baku_overflow_death()
    solve_exact_finite_horizon(scenario.game, half_round_horizon=3, config=scenario.config)
    assert solve_cache_size() > 0
    clear_solve_cache()
    assert solve_cache_size() == 0


def test_solve_cache_respects_lru_maxsize_bound():
    """The cache must never exceed its LRU ceiling. An unbounded cache
    exhausts RAM during wide-grid corpus generation (each worker caches
    the shared deep-substate space, ×N workers). This test patches the
    maxsize down to a tiny value and verifies the bound holds after a
    horizon-3 solve that would otherwise cache far more entries."""
    import environment.cfr.exact as exact_mod

    clear_solve_cache()
    original_maxsize = exact_mod._SOLVE_CACHE_MAXSIZE
    try:
        exact_mod._SOLVE_CACHE_MAXSIZE = 10
        scenario = forced_baku_overflow_death()
        # A horizon=4 solve at a near-overflow state recurses through many
        # substates — without the bound the cache would hold dozens.
        solve_exact_finite_horizon(
            scenario.game, half_round_horizon=4, config=scenario.config
        )
        assert solve_cache_size() <= 10, (
            f"cache exceeded LRU bound: {solve_cache_size()} > 10"
        )
    finally:
        exact_mod._SOLVE_CACHE_MAXSIZE = original_maxsize
        clear_solve_cache()


def test_solve_cache_bit_identical_under_eviction_pressure():
    """Even when the LRU cap forces evictions mid-recursion, the final
    result must match an unbounded solve bit-for-bit. Eviction is only
    safe if recompute is identical — this guards that invariant."""
    import numpy as np
    import environment.cfr.exact as exact_mod

    scenario = forced_baku_overflow_death()

    clear_solve_cache()
    unbounded = solve_exact_finite_horizon(
        scenario.game, half_round_horizon=4, config=scenario.config
    )

    clear_solve_cache()
    original_maxsize = exact_mod._SOLVE_CACHE_MAXSIZE
    try:
        exact_mod._SOLVE_CACHE_MAXSIZE = 3  # aggressive eviction
        scenario2 = forced_baku_overflow_death()
        bounded = solve_exact_finite_horizon(
            scenario2.game, half_round_horizon=4, config=scenario2.config
        )
    finally:
        exact_mod._SOLVE_CACHE_MAXSIZE = original_maxsize
        clear_solve_cache()

    assert bounded.value_for_hal == pytest.approx(unbounded.value_for_hal)
    assert bounded.unresolved_probability == pytest.approx(unbounded.unresolved_probability)
    np.testing.assert_array_equal(bounded.dropper_strategy, unbounded.dropper_strategy)
    np.testing.assert_array_equal(bounded.checker_strategy, unbounded.checker_strategy)
