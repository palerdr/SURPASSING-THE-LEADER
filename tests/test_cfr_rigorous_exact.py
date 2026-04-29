"""Tests for the rigorous exact-second CFR foundation."""

import os
import pathlib
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from environment.cfr.exact import (
    evaluate_joint_action,
    exact_immediate_checker_payoff_matrix,
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
