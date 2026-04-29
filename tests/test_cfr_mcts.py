"""End-to-end MCTS tests covering all of Phase 4.

Sections:
    1. make_node + data structures (Step 1)
    2. exploration bonus + selection (Step 3)
    3. chance-branch expansion via _step_into_child (Step 4)
    4. backup running-mean (Step 5)
    5. mcts_search end-to-end (Step 6)
    6. transposition cache (Slice 4c.1)
    7. principal line (Slice 4c.2)

The Step 2 tests (LeafEvaluator / TerminalOnlyEvaluator / TablebaseEvaluator
/ ValueNetEvaluator) live in test_cfr_evaluator.py.
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from environment.cfr.evaluator import TerminalOnlyEvaluator
from environment.cfr.exact import (
    ExactGameSnapshot,
    ExactJointAction,
    ExactPublicState,
    ExactSearchConfig,
    exact_public_state,
)
from environment.cfr.mcts import (
    MCTSConfig,
    MCTSNode,
    _backup,
    _exploration_augmented_matrix,
    _principal_line,
    _select_joint_action,
    _step_into_child,
    make_node,
    mcts_search,
)
from environment.cfr.tactical_scenarios import (
    forced_baku_overflow_death,
    forced_hal_overflow_death,
    leap_second_check_61_probe,
    safe_budget_pressure_at_cylinder_241,
)
from src.Constants import PHYSICALITY_BAKU, PHYSICALITY_HAL
from src.Game import Game
from src.Player import Player
from src.Referee import Referee


# ── Shared helpers ────────────────────────────────────────────────────────


def _baku_checker_at_cylinder(cyl: float = 0.0) -> Game:
    hal = Player(name="Hal", physicality=PHYSICALITY_HAL)
    baku = Player(name="Baku", physicality=PHYSICALITY_BAKU)
    game = Game(player1=hal, player2=baku, referee=Referee(), first_dropper=hal)
    game.seed(0)
    game.player2.cylinder = cyl
    return game


def _terminal_game(*, hal_wins: bool) -> Game:
    game = _baku_checker_at_cylinder(0.0)
    game.game_over = True
    if hal_wins:
        game.winner = game.player1
        game.loser = game.player2
    else:
        game.winner = game.player2
        game.loser = game.player1
    return game


def _idx(seconds: tuple[int, ...], target: int) -> int:
    return seconds.index(target)


def _fresh_node() -> MCTSNode:
    scenario = forced_baku_overflow_death()
    return make_node(scenario.game, scenario.config)


def _config(iterations: int, exploration_c: float = 1.0) -> MCTSConfig:
    return MCTSConfig(
        iterations=iterations,
        exploration_c=exploration_c,
        evaluator=None,
        use_tablebase=False,
    )


# ── 1. make_node + data structures (Step 1) ───────────────────────────────


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
    scenario = forced_baku_overflow_death()
    node = make_node(scenario.game, scenario.config)

    for s in node.drop_seconds + node.check_seconds:
        assert 1 <= s <= 61
    assert 0 not in node.drop_seconds
    assert 0 not in node.check_seconds


def test_make_node_uses_default_config_when_none_passed():
    scenario = forced_baku_overflow_death()
    node = make_node(scenario.game)
    assert node.terminal_value is None
    assert len(node.drop_seconds) > 0


def test_make_node_at_terminal_hal_win_returns_plus_one_no_actions():
    game = _terminal_game(hal_wins=True)
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
    game = _terminal_game(hal_wins=False)
    node = make_node(game)
    assert node.terminal_value == -1.0
    assert node.drop_seconds == ()
    assert node.check_seconds == ()


def test_make_node_snapshot_round_trips_position_after_mutation():
    scenario = forced_baku_overflow_death()
    original_cyl = scenario.game.player2.cylinder
    original_clock = scenario.game.game_clock

    node = make_node(scenario.game, scenario.config)

    scenario.game.player2.cylinder = 50.0
    scenario.game.game_clock = 9999.0
    node.game_snapshot.restore(scenario.game)

    assert scenario.game.player2.cylinder == original_cyl
    assert scenario.game.game_clock == original_clock


def test_make_node_in_leap_window_with_deduced_hal_includes_check_61():
    scenario = leap_second_check_61_probe()
    node = make_node(scenario.game, scenario.config)
    assert 61 in node.check_seconds


def test_make_node_children_dict_is_empty_and_independent_per_node():
    scenario = forced_baku_overflow_death()
    node_a = make_node(scenario.game, scenario.config)
    node_b = make_node(scenario.game, scenario.config)

    assert node_a.children is not node_b.children
    node_a.children[(1, 2, None)] = node_b
    assert node_b.children == {}


# ── 2. Exploration bonus + selection (Step 3) ─────────────────────────────


def test_exploration_bonus_at_fresh_node_is_zero_so_q_explore_equals_q():
    scenario = forced_baku_overflow_death()
    node = make_node(scenario.game, scenario.config)
    q_explore = _exploration_augmented_matrix(node, exploration_c=1.0)
    assert q_explore.shape == node.Q.shape
    np.testing.assert_array_equal(q_explore, node.Q)


def test_exploration_bonus_shape_matches_node_matrix_shape():
    scenario = forced_baku_overflow_death()
    node = make_node(scenario.game, scenario.config)
    D = len(node.drop_seconds)
    C = len(node.check_seconds)
    node.N_node = 100
    q_explore = _exploration_augmented_matrix(node, exploration_c=1.0)
    assert q_explore.shape == (D, C)


def test_exploration_bonus_shrinks_for_well_visited_cells():
    # With N_node=100 and N_cell[0,0]=99 vs N_cell[0,1]=0, ratio 100x.
    scenario = forced_baku_overflow_death()
    node = make_node(scenario.game, scenario.config)
    assert len(node.check_seconds) >= 2

    node.N_node = 100
    node.N_cell[0, 0] = 99
    node.N_cell[0, 1] = 0

    q_explore = _exploration_augmented_matrix(node, exploration_c=1.0)
    assert q_explore[0, 1] > q_explore[0, 0]
    assert q_explore[0, 1] / q_explore[0, 0] == pytest.approx(100.0)


def test_exploration_constant_scales_bonus_linearly():
    scenario = forced_baku_overflow_death()
    node = make_node(scenario.game, scenario.config)
    node.N_node = 50

    q_c1 = _exploration_augmented_matrix(node, exploration_c=1.0)
    q_c2 = _exploration_augmented_matrix(node, exploration_c=2.0)
    np.testing.assert_allclose(q_c2, 2.0 * q_c1)


def test_select_joint_action_returns_indices_within_matrix_bounds():
    scenario = forced_baku_overflow_death()
    node = make_node(scenario.game, scenario.config)
    D = len(node.drop_seconds)
    C = len(node.check_seconds)
    rng = np.random.default_rng(42)

    for _ in range(50):
        d_idx, c_idx = _select_joint_action(node, exploration_c=1.0, rng=rng)
        assert isinstance(d_idx, int)
        assert isinstance(c_idx, int)
        assert 0 <= d_idx < D
        assert 0 <= c_idx < C


def test_select_joint_action_is_deterministic_under_same_seed():
    scenario = forced_baku_overflow_death()
    node = make_node(scenario.game, scenario.config)

    rng_a = np.random.default_rng(123)
    rng_b = np.random.default_rng(123)

    samples_a = [_select_joint_action(node, exploration_c=1.0, rng=rng_a) for _ in range(20)]
    samples_b = [_select_joint_action(node, exploration_c=1.0, rng=rng_b) for _ in range(20)]
    assert samples_a == samples_b


def test_select_joint_action_avoids_dropper_rows_with_strongly_negative_q():
    scenario = forced_baku_overflow_death()
    node = make_node(scenario.game, scenario.config)
    assert node.hal_is_dropper is True

    D = len(node.drop_seconds)
    node.Q[0, :] = -1.0
    node.N_node = 10_000

    rng = np.random.default_rng(0)
    counts = np.zeros(D, dtype=np.int64)
    n_samples = 500
    for _ in range(n_samples):
        d_idx, _ = _select_joint_action(node, exploration_c=0.01, rng=rng)
        counts[d_idx] += 1

    assert counts[0] < n_samples * 0.05


def test_select_joint_action_works_when_hal_is_checker():
    scenario = forced_hal_overflow_death()
    node = make_node(scenario.game, scenario.config)
    assert node.hal_is_dropper is False
    D = len(node.drop_seconds)
    C = len(node.check_seconds)
    rng = np.random.default_rng(0)

    for _ in range(50):
        d_idx, c_idx = _select_joint_action(node, exploration_c=1.0, rng=rng)
        assert 0 <= d_idx < D
        assert 0 <= c_idx < C


def test_select_joint_action_strategies_at_fresh_node_are_valid_distributions():
    scenario = forced_baku_overflow_death()
    node = make_node(scenario.game, scenario.config)
    rng = np.random.default_rng(99)
    d_idx, c_idx = _select_joint_action(node, exploration_c=1.0, rng=rng)
    assert d_idx >= 0 and c_idx >= 0


# ── 3. Chance-branch expansion via _step_into_child (Step 4) ──────────────


def test_step_into_child_no_death_creates_one_child_with_none_key():
    game = _baku_checker_at_cylinder(0.0)
    config = ExactSearchConfig()
    node = make_node(game, config)

    d_idx = _idx(node.drop_seconds, 1)
    c_idx = _idx(node.check_seconds, 60)
    rng = np.random.default_rng(42)

    child, survived = _step_into_child(node, game, d_idx, c_idx, rng, config)

    assert survived is None
    assert isinstance(child, MCTSNode)
    assert len(node.children) == 1
    assert (1, 60, None) in node.children
    assert node.children[(1, 60, None)] is child


def test_step_into_child_with_seeded_rng_picks_deterministic_branch():
    config = ExactSearchConfig()

    game_a = _baku_checker_at_cylinder(180.0)
    node_a = make_node(game_a, config)
    d_a = _idx(node_a.drop_seconds, 60)
    c_a = _idx(node_a.check_seconds, 1)
    rng_a = np.random.default_rng(7)
    _, survived_a = _step_into_child(node_a, game_a, d_a, c_a, rng_a, config)

    game_b = _baku_checker_at_cylinder(180.0)
    node_b = make_node(game_b, config)
    d_b = _idx(node_b.drop_seconds, 60)
    c_b = _idx(node_b.check_seconds, 1)
    rng_b = np.random.default_rng(7)
    _, survived_b = _step_into_child(node_b, game_b, d_b, c_b, rng_b, config)

    assert survived_a is not None
    assert survived_a == survived_b


def test_step_into_child_cache_hit_does_not_double_create():
    config = ExactSearchConfig()
    game = _baku_checker_at_cylinder(0.0)
    node = make_node(game, config)
    d_idx = _idx(node.drop_seconds, 1)
    c_idx = _idx(node.check_seconds, 60)

    rng = np.random.default_rng(0)
    child_first, _ = _step_into_child(node, game, d_idx, c_idx, rng, config)

    node.game_snapshot.restore(game)
    rng2 = np.random.default_rng(0)
    child_second, _ = _step_into_child(node, game, d_idx, c_idx, rng2, config)

    assert len(node.children) == 1
    assert child_first is child_second


def test_step_into_child_leaves_engine_at_post_action_position():
    config = ExactSearchConfig()
    game = _baku_checker_at_cylinder(0.0)
    node = make_node(game, config)
    d_idx = _idx(node.drop_seconds, 1)
    c_idx = _idx(node.check_seconds, 60)
    rng = np.random.default_rng(0)

    _step_into_child(node, game, d_idx, c_idx, rng, config)

    assert game.current_half == 2
    assert game.player2.cylinder == 59
    assert game.player1.cylinder == 0


def test_step_into_child_at_zero_survival_probability_forces_died_branch():
    scenario = forced_baku_overflow_death()
    config = scenario.config
    node = make_node(scenario.game, config)

    d_idx = _idx(node.drop_seconds, 1)
    c_idx = _idx(node.check_seconds, 60)
    rng = np.random.default_rng(42)

    child, survived = _step_into_child(node, scenario.game, d_idx, c_idx, rng, config)
    assert survived is False
    assert child.terminal_value == 1.0


# ── 4. Backup running-mean (Step 5) ───────────────────────────────────────


def test_backup_single_step_updates_one_cell_only():
    node = _fresh_node()
    _backup([(node, 0, 0)], value=1.0)

    assert node.N_node == 1
    assert node.N_cell[0, 0] == 1
    assert node.N_cell.sum() == 1
    assert node.Q[0, 0] == 1.0

    other_visits = node.N_cell.copy()
    other_visits[0, 0] = 0
    assert other_visits.sum() == 0
    other_q = node.Q.copy()
    other_q[0, 0] = 0
    assert other_q.sum() == 0


def test_backup_running_mean_averages_two_visits_to_same_cell():
    node = _fresh_node()
    _backup([(node, 0, 0)], value=1.0)
    _backup([(node, 0, 0)], value=-1.0)
    assert node.N_cell[0, 0] == 2
    assert node.Q[0, 0] == pytest.approx(0.0)


def test_backup_running_mean_handles_three_values_correctly():
    node = _fresh_node()
    for v in (1.0, -1.0, 0.5):
        _backup([(node, 0, 0)], value=v)
    assert node.N_cell[0, 0] == 3
    assert node.Q[0, 0] == pytest.approx((1.0 - 1.0 + 0.5) / 3)


def test_backup_preserves_n_node_equals_n_cell_sum_invariant():
    node = _fresh_node()
    cells = [(0, 0), (0, 1), (1, 0), (0, 0), (1, 1), (0, 0)]
    for d_idx, c_idx in cells:
        _backup([(node, d_idx, c_idx)], value=0.5)
        assert node.N_node == node.N_cell.sum()
    assert node.N_node == len(cells)


def test_backup_multi_node_path_updates_every_node_once():
    scenario_a = forced_baku_overflow_death()
    node_a = make_node(scenario_a.game, scenario_a.config)
    scenario_b = forced_baku_overflow_death()
    node_b = make_node(scenario_b.game, scenario_b.config)

    _backup([(node_a, 0, 0), (node_b, 1, 1)], value=0.5)

    assert node_a.N_node == 1
    assert node_a.N_cell[0, 0] == 1
    assert node_a.Q[0, 0] == pytest.approx(0.5)
    assert node_b.N_node == 1
    assert node_b.N_cell[1, 1] == 1
    assert node_b.Q[1, 1] == pytest.approx(0.5)


def test_backup_with_empty_path_is_a_noop():
    node = _fresh_node()
    pre_q = node.Q.copy()
    pre_n = node.N_node
    _backup([], value=0.5)
    assert node.N_node == pre_n
    np.testing.assert_array_equal(node.Q, pre_q)


def test_backup_negative_leaf_value_propagates_correctly():
    node = _fresh_node()
    _backup([(node, 0, 0)], value=-1.0)
    assert node.Q[0, 0] == -1.0
    assert node.N_cell[0, 0] == 1


# ── 5. mcts_search end-to-end (Step 6) ────────────────────────────────────


def test_mcts_search_on_forced_baku_overflow_converges_to_plus_one():
    scenario = forced_baku_overflow_death()
    rng = np.random.default_rng(0)
    result = mcts_search(scenario.game, _config(500), TerminalOnlyEvaluator(), rng, scenario.config)
    assert result.root_value_for_hal == pytest.approx(1.0, abs=0.05)


def test_mcts_search_on_forced_hal_overflow_converges_to_minus_one():
    scenario = forced_hal_overflow_death()
    rng = np.random.default_rng(0)
    result = mcts_search(scenario.game, _config(500), TerminalOnlyEvaluator(), rng, scenario.config)
    assert result.root_value_for_hal == pytest.approx(-1.0, abs=0.05)


def test_mcts_search_returns_proper_probability_distributions():
    scenario = forced_baku_overflow_death()
    rng = np.random.default_rng(7)
    result = mcts_search(scenario.game, _config(100), TerminalOnlyEvaluator(), rng, scenario.config)
    assert result.root_strategy_dropper.sum() == pytest.approx(1.0)
    assert result.root_strategy_checker.sum() == pytest.approx(1.0)
    assert (result.root_strategy_dropper >= 0).all()
    assert (result.root_strategy_checker >= 0).all()


def test_mcts_search_is_deterministic_under_same_rng_seed():
    cfg = _config(100)
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
    cfg_small = _config(20)
    cfg_large = _config(200)
    evaluator = TerminalOnlyEvaluator()

    scenario_a = forced_baku_overflow_death()
    rng_a = np.random.default_rng(0)
    result_small = mcts_search(scenario_a.game, cfg_small, evaluator, rng_a, scenario_a.config)

    scenario_b = forced_baku_overflow_death()
    rng_b = np.random.default_rng(0)
    result_large = mcts_search(scenario_b.game, cfg_large, evaluator, rng_b, scenario_b.config)

    assert result_large.root_visits > result_small.root_visits
    assert result_small.root_visits >= 15
    assert result_large.root_visits >= 150


def test_mcts_search_value_above_horizon_one_baseline_on_safe_budget():
    # MCTS deepens past horizon=1 LP value (0.5) on safe_budget_pressure_at_241
    # because deep search reaches the forced-overflow position. Value sits in
    # [0.5, 1.0], strictly above the unresolved 0.0.
    scenario = safe_budget_pressure_at_cylinder_241()
    rng = np.random.default_rng(123)
    result = mcts_search(scenario.game, _config(1500), TerminalOnlyEvaluator(), rng, scenario.config)
    assert 0.5 <= result.root_value_for_hal <= 1.0


# ── 6. Transposition cache (Slice 4c.1) ───────────────────────────────────


def test_step_into_child_default_no_transposition_preserves_behavior():
    game = _baku_checker_at_cylinder(0.0)
    config = ExactSearchConfig()
    node = make_node(game, config)
    rng = np.random.default_rng(0)

    d = _idx(node.drop_seconds, 1)
    c = _idx(node.check_seconds, 2)
    child, survived = _step_into_child(node, game, d, c, rng, config)

    assert isinstance(child, MCTSNode)
    assert survived is None
    assert (1, 2, None) in node.children
    assert node.children[(1, 2, None)] is child


def test_step_into_child_writes_to_transposition_cache_on_miss():
    game = _baku_checker_at_cylinder(0.0)
    config = ExactSearchConfig()
    node = make_node(game, config)
    transposition: dict[ExactPublicState, MCTSNode] = {}
    rng = np.random.default_rng(0)

    d = _idx(node.drop_seconds, 1)
    c = _idx(node.check_seconds, 2)
    child, _ = _step_into_child(node, game, d, c, rng, config, transposition=transposition)

    assert len(transposition) == 1
    state = exact_public_state(game)
    assert state in transposition
    assert transposition[state] is child


def test_two_edges_to_same_state_share_a_single_node():
    config = ExactSearchConfig()
    transposition: dict[ExactPublicState, MCTSNode] = {}

    game_a = _baku_checker_at_cylinder(0.0)
    node_a_root = make_node(game_a, config)
    transposition[exact_public_state(game_a)] = node_a_root
    rng = np.random.default_rng(0)
    d_a = _idx(node_a_root.drop_seconds, 1)
    c_a = _idx(node_a_root.check_seconds, 2)
    child_a, _ = _step_into_child(node_a_root, game_a, d_a, c_a, rng, config, transposition=transposition)

    node_a_root.game_snapshot.restore(game_a)
    d_b = _idx(node_a_root.drop_seconds, 2)
    c_b = _idx(node_a_root.check_seconds, 3)
    child_b, _ = _step_into_child(node_a_root, game_a, d_b, c_b, rng, config, transposition=transposition)

    assert child_a is child_b
    assert node_a_root.children[(1, 2, None)] is node_a_root.children[(2, 3, None)]
    assert len(transposition) == 2


def test_visit_counts_accumulate_across_paths_through_shared_node():
    config = ExactSearchConfig()
    transposition: dict[ExactPublicState, MCTSNode] = {}
    game = _baku_checker_at_cylinder(0.0)
    root = make_node(game, config)
    transposition[exact_public_state(game)] = root
    rng = np.random.default_rng(0)

    d1 = _idx(root.drop_seconds, 1)
    c1 = _idx(root.check_seconds, 2)
    shared_child, _ = _step_into_child(root, game, d1, c1, rng, config, transposition=transposition)
    _backup([(root, d1, c1), (shared_child, 0, 0)], value=0.5)

    root.game_snapshot.restore(game)
    d2 = _idx(root.drop_seconds, 2)
    c2 = _idx(root.check_seconds, 3)
    shared_child_again, _ = _step_into_child(root, game, d2, c2, rng, config, transposition=transposition)
    assert shared_child_again is shared_child
    _backup([(root, d2, c2), (shared_child, 0, 0)], value=-0.5)

    assert shared_child.N_node == 2
    assert shared_child.N_cell[0, 0] == 2
    assert shared_child.Q[0, 0] == pytest.approx(0.0)


def test_mcts_search_with_transposition_still_converges_on_forced_overflow():
    scenario = forced_baku_overflow_death()
    rng = np.random.default_rng(0)
    result = mcts_search(scenario.game, _config(200), TerminalOnlyEvaluator(), rng, scenario.config)
    assert result.root_value_for_hal == pytest.approx(1.0, abs=0.05)


def test_step_into_child_transposition_none_does_not_populate_anything():
    game = _baku_checker_at_cylinder(0.0)
    config = ExactSearchConfig()
    node = make_node(game, config)
    rng = np.random.default_rng(0)

    d = _idx(node.drop_seconds, 1)
    c = _idx(node.check_seconds, 2)
    child, _ = _step_into_child(node, game, d, c, rng, config, transposition=None)
    assert isinstance(child, MCTSNode)
    assert (1, 2, None) in node.children


# ── 7. Principal line (Slice 4c.2) ────────────────────────────────────────


def test_principal_line_on_unsearched_root_is_empty():
    scenario = forced_baku_overflow_death()
    root = make_node(scenario.game, scenario.config)
    assert _principal_line(root) == []


def test_principal_line_after_search_is_non_empty():
    scenario = forced_baku_overflow_death()
    rng = np.random.default_rng(0)
    result = mcts_search(scenario.game, _config(50), TerminalOnlyEvaluator(), rng, scenario.config)
    assert len(result.principal_line) >= 1
    assert all(isinstance(action, ExactJointAction) for action in result.principal_line)


def test_principal_line_is_deterministic_under_same_seed():
    scenario = safe_budget_pressure_at_cylinder_241()
    rng = np.random.default_rng(7)
    result = mcts_search(scenario.game, _config(200), TerminalOnlyEvaluator(), rng, scenario.config)

    scenario2 = safe_budget_pressure_at_cylinder_241()
    rng2 = np.random.default_rng(7)
    result2 = mcts_search(scenario2.game, _config(200), TerminalOnlyEvaluator(), rng2, scenario2.config)

    assert result.principal_line == result2.principal_line


def test_principal_line_stops_at_terminal_node_after_one_step():
    scenario = forced_baku_overflow_death()
    rng = np.random.default_rng(42)
    result = mcts_search(scenario.game, _config(200), TerminalOnlyEvaluator(), rng, scenario.config)

    assert len(result.principal_line) == 1
    first = result.principal_line[0]
    assert 1 <= first.drop_time <= 60
    assert 1 <= first.check_time <= 60


def test_principal_line_handles_zero_n_cell_gracefully():
    scenario = forced_baku_overflow_death()
    root = make_node(scenario.game, scenario.config)
    root.N_node = 5
    line = _principal_line(root)
    assert line == []


def test_principal_line_returns_legal_seconds_in_actions():
    scenario = forced_baku_overflow_death()
    rng = np.random.default_rng(0)
    result = mcts_search(scenario.game, _config(100), TerminalOnlyEvaluator(), rng, scenario.config)

    first = result.principal_line[0]
    assert 1 <= first.drop_time <= 61
    assert 1 <= first.check_time <= 61
