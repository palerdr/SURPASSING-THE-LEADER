from __future__ import annotations

import copy

import numpy as np
import pytest

from stl.engine.actions import ACTION_SIZE, legal_mask, legal_seconds
from stl.engine.game import Game, PHYSICALITY_BAKU, PHYSICALITY_HAL, Player, Referee
from stl.solver.conformance import audit_cfr_plus_matrix
from stl.solver.exact import (
    CFRPlusConfig,
    ExactJointAction,
    ExactSearchConfig,
    enumerate_joint_actions,
    exact_public_state,
    expand_joint_action,
    survival_probability,
    solve_exact_finite_horizon,
    solve_minimax,
)
from stl.solver.search import _step_into_child, audit_against_full_width, make_node
from stl.solver.tablebase import (
    forced_baku_overflow_death,
    forced_hal_overflow_death,
    materialize_all,
    safe_budget_pressure_at_cylinder_240,
)


def _game(*, clock: float = 720.0, half: int = 1, first: str = "Hal") -> Game:
    hal = Player("Hal", PHYSICALITY_HAL)
    baku = Player("Baku", PHYSICALITY_BAKU)
    first_dropper = hal if first == "Hal" else baku
    game = Game(hal, baku, Referee(), first_dropper=first_dropper)
    game.seed(123)
    game.game_clock = clock
    game.current_half = half
    return game


def test_scalar_survival_helper_preserves_strict_total_ttd_boundary():
    assert survival_probability(60, 240, 0, 1.0) > 0.0
    assert survival_probability(60, 241, 0, 1.0) == 0.0


@pytest.mark.parametrize("clock", [720.0, 3540.0, 3600.0, 3601.0])
@pytest.mark.parametrize("half", [1, 2])
@pytest.mark.parametrize("first", ["Hal", "Baku"])
def test_joint_enumeration_masks_and_engine_legality_agree(clock, half, first):
    game = _game(clock=clock, half=half, first=first)
    dropper, checker = game.get_roles_for_half(half)
    duration = game.get_turn_duration()
    expected_drop = legal_seconds(dropper.name, "dropper", duration)
    expected_check = legal_seconds(checker.name, "checker", duration)
    actions = enumerate_joint_actions(game)

    assert {action.drop_time for action in actions} == set(expected_drop)
    assert {action.check_time for action in actions} == set(expected_check)
    assert len(actions) == len(expected_drop) * len(expected_check)
    np.testing.assert_array_equal(
        legal_mask(dropper.name, "dropper", duration).nonzero()[0],
        np.asarray(expected_drop),
    )
    np.testing.assert_array_equal(
        legal_mask(checker.name, "checker", duration).nonzero()[0],
        np.asarray(expected_check),
    )
    assert not legal_mask(dropper.name, "dropper", duration)[0]
    assert not legal_mask(checker.name, "checker", duration)[0]
    assert not legal_mask(checker.name, "checker", duration)[ACTION_SIZE - 1]


def test_normal_and_leap_full_width_shapes_are_literal_seconds():
    normal = _game(clock=720.0, half=1)
    leap = _game(clock=3540.0, half=2)  # Baku is the dropper.
    assert len(enumerate_joint_actions(normal)) == 60 * 60
    assert len(enumerate_joint_actions(leap)) == 61 * 60


@pytest.mark.parametrize(
    ("game", "action"),
    [
        (_game(), ExactJointAction(1, 1)),
        (_game(), ExactJointAction(1, 2)),
        (_game(), ExactJointAction(2, 1)),
        (_game(clock=3540.0, half=2), ExactJointAction(61, 60)),
    ],
)
def test_transition_expansion_matches_direct_engine_resolution(game, action):
    before = exact_public_state(game)
    rng_before = game.rng.getstate()
    transitions = expand_joint_action(game, action)
    assert exact_public_state(game) == before
    assert game.rng.getstate() == rng_before
    assert sum(branch.probability for branch in transitions) == pytest.approx(1.0, abs=1e-12)

    for branch in transitions:
        direct = copy.deepcopy(game)
        direct_record = direct.resolve_half_round(
            action.drop_time,
            action.check_time,
            survived_outcome=branch.record.survived,
        )
        assert direct_record == branch.record
        assert exact_public_state(direct) == branch.state


@pytest.mark.parametrize(
    "scenario_factory",
    [forced_baku_overflow_death, forced_hal_overflow_death],
)
def test_exact_matrix_orientation_matches_explicit_hal_row_form(scenario_factory):
    scenario = scenario_factory()
    result = solve_exact_finite_horizon(
        scenario.game,
        scenario.half_round_horizon,
        scenario.config,
    )
    assert result.payoff_for_hal is not None
    payoff = result.payoff_for_hal
    dropper, _ = scenario.game.get_roles_for_half(scenario.game.current_half)
    if dropper.name == "Hal":
        expected_drop, value = solve_minimax(payoff)
        expected_check, neg_value = solve_minimax((-payoff).T)
    else:
        expected_drop, neg_value = solve_minimax(-payoff)
        expected_check, value = solve_minimax(payoff.T)
    np.testing.assert_allclose(result.dropper_strategy, expected_drop, atol=1e-9)
    np.testing.assert_allclose(result.checker_strategy, expected_check, atol=1e-9)
    assert result.value_for_hal == pytest.approx(value, abs=1e-9)
    assert value == pytest.approx(-neg_value, abs=1e-9)


def test_lp_primal_dual_values_agree_on_frozen_matrices():
    matrices = (
        np.array([[1.0, -1.0], [-1.0, 1.0]]),
        np.array([[0.3, -0.4, 0.8], [-0.2, 0.6, -0.1]]),
        np.random.default_rng(7).uniform(-1.0, 1.0, size=(5, 4)),
    )
    for matrix in matrices:
        _row, primal = solve_minimax(matrix)
        _column, dual_negative = solve_minimax((-matrix).T)
        assert primal == pytest.approx(-dual_negative, abs=1e-9)


def test_python_cfr_plus_meets_frozen_lp_error_and_saddle_gap_gates():
    matrices = (
        np.array([[1.0, -1.0], [-1.0, 1.0]]),
        np.array([[0.3, -0.4, 0.8], [-0.2, 0.6, -0.1]]),
        np.random.default_rng(7).uniform(-1.0, 1.0, size=(5, 4)),
    )
    config = CFRPlusConfig(iterations=10_000, average_delay=100)
    for matrix in matrices:
        record = audit_cfr_plus_matrix(matrix, config)
        assert record.value_error <= 0.01
        assert record.saddle_gap <= 0.02


def test_candidate_audit_reports_omitted_action_best_response_gain():
    scenario = safe_budget_pressure_at_cylinder_240()
    audit = audit_against_full_width(
        scenario.game,
        scenario.half_round_horizon,
        scenario.config,
    )
    assert audit.dropper_omitted_action_gain >= 0.0
    assert audit.checker_omitted_action_gain >= 0.0
    assert audit.max_omitted_action_gain == max(
        audit.dropper_omitted_action_gain,
        audit.checker_omitted_action_gain,
    )


def test_frozen_tactical_and_stratified_candidate_pack_clears_gain_gate():
    fixtures = [scenario.game for scenario in materialize_all()]
    rng = np.random.default_rng(20260710)
    for _ in range(8):
        game = _game(
            clock=float(rng.choice([720, 1800, 3450, 3540, 3580, 3601])),
            half=int(rng.choice([1, 2])),
        )
        game.player1.cylinder = float(rng.integers(0, 301))
        game.player2.cylinder = float(rng.integers(0, 301))
        game.player1.ttd = float(rng.integers(0, 301))
        game.player2.ttd = float(rng.integers(0, 301))
        game.referee.cprs_performed = int(rng.integers(0, 13))
        game.round_num = int(rng.integers(0, 10))
        fixtures.append(game)

    for game in fixtures:
        audit = audit_against_full_width(game, 1)
        assert audit.max_omitted_action_gain <= 0.02


def test_mcts_chance_sampler_tracks_engine_probability_inside_99_percent_interval():
    game = _game()
    node = make_node(game, ExactSearchConfig())
    d_idx = node.drop_seconds.index(2)
    c_idx = node.check_seconds.index(1)
    branches = expand_joint_action(game, ExactJointAction(2, 1))
    survival_probability = sum(
        branch.probability for branch in branches if branch.record.survived is True
    )

    rng = np.random.default_rng(20260710)
    samples = 5_000
    survived = 0
    for _ in range(samples):
        node.game_snapshot.restore(game)
        _child, outcome = _step_into_child(
            node,
            game,
            d_idx,
            c_idx,
            rng,
            ExactSearchConfig(),
        )
        survived += int(outcome is True)
    node.game_snapshot.restore(game)

    observed = survived / samples
    standard_error = np.sqrt(survival_probability * (1.0 - survival_probability) / samples)
    assert abs(observed - survival_probability) <= 2.576 * standard_error + 1.0 / samples
