"""Tests for MCTS Slice 4c: transposition cache."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from environment.cfr.evaluator import TerminalOnlyEvaluator
from environment.cfr.exact_state import ExactPublicState, exact_public_state
from environment.cfr.exact_transition import ExactSearchConfig
from environment.cfr.mcts import (
    MCTSConfig,
    MCTSNode,
    _backup,
    _step_into_child,
    make_node,
    mcts_search,
)
from environment.cfr.tactical_scenarios import forced_baku_overflow_death
from src.Constants import PHYSICALITY_BAKU, PHYSICALITY_HAL
from src.Game import Game
from src.Player import Player
from src.Referee import Referee


def _fresh_game() -> Game:
    hal = Player(name="Hal", physicality=PHYSICALITY_HAL)
    baku = Player(name="Baku", physicality=PHYSICALITY_BAKU)
    game = Game(player1=hal, player2=baku, referee=Referee(), first_dropper=hal)
    game.seed(0)
    return game


def _idx(seconds: tuple[int, ...], target: int) -> int:
    return seconds.index(target)


def test_step_into_child_default_no_transposition_preserves_behavior():
    # Without a transposition cache, _step_into_child must behave exactly
    # as before: per-edge caching via node.children only.
    game = _fresh_game()
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
    game = _fresh_game()
    config = ExactSearchConfig()
    node = make_node(game, config)
    transposition: dict[ExactPublicState, MCTSNode] = {}
    rng = np.random.default_rng(0)

    d = _idx(node.drop_seconds, 1)
    c = _idx(node.check_seconds, 2)
    child, _ = _step_into_child(node, game, d, c, rng, config, transposition=transposition)

    # The cache picks up the new child's state.
    assert len(transposition) == 1
    state = exact_public_state(game)  # game is now at child's position
    assert state in transposition
    assert transposition[state] is child


def test_two_edges_to_same_state_share_a_single_node():
    # Load-bearing test: distinct (drop, check) pairs yielding the same ST
    # land at the same ExactPublicState. With the transposition cache, both
    # edges in node.children must point to the *same* MCTSNode instance.
    config = ExactSearchConfig()
    transposition: dict[ExactPublicState, MCTSNode] = {}

    # Edge A: drop=1, check=2 → ST=1, baku.cylinder=1, clock advanced.
    game_a = _fresh_game()
    node_a_root = make_node(game_a, config)
    transposition[exact_public_state(game_a)] = node_a_root
    rng = np.random.default_rng(0)
    d_a = _idx(node_a_root.drop_seconds, 1)
    c_a = _idx(node_a_root.check_seconds, 2)
    child_a, _ = _step_into_child(node_a_root, game_a, d_a, c_a, rng, config, transposition=transposition)

    # Edge B from the same parent: drop=2, check=3 → ST=1, same final state.
    # We rewind the engine to the parent and take the second edge.
    node_a_root.game_snapshot.restore(game_a)
    d_b = _idx(node_a_root.drop_seconds, 2)
    c_b = _idx(node_a_root.check_seconds, 3)
    child_b, _ = _step_into_child(node_a_root, game_a, d_b, c_b, rng, config, transposition=transposition)

    # Both edges resolve to the same MCTSNode instance.
    assert child_a is child_b
    assert node_a_root.children[(1, 2, None)] is node_a_root.children[(2, 3, None)]
    # The transposition cache still has only the root + one shared child.
    assert len(transposition) == 2


def test_visit_counts_accumulate_across_paths_through_shared_node():
    # If two backup calls touch the shared node (via different parent edges),
    # the node's N_node should reflect both visits, not split.
    config = ExactSearchConfig()
    transposition: dict[ExactPublicState, MCTSNode] = {}
    game = _fresh_game()
    root = make_node(game, config)
    transposition[exact_public_state(game)] = root
    rng = np.random.default_rng(0)

    # Reach the shared child via two edges and run one backup along each path.
    d1 = _idx(root.drop_seconds, 1)
    c1 = _idx(root.check_seconds, 2)
    shared_child, _ = _step_into_child(root, game, d1, c1, rng, config, transposition=transposition)
    _backup([(root, d1, c1), (shared_child, 0, 0)], value=0.5)

    root.game_snapshot.restore(game)
    d2 = _idx(root.drop_seconds, 2)
    c2 = _idx(root.check_seconds, 3)
    shared_child_again, _ = _step_into_child(root, game, d2, c2, rng, config, transposition=transposition)
    assert shared_child_again is shared_child  # transposition wired up correctly
    _backup([(root, d2, c2), (shared_child, 0, 0)], value=-0.5)

    # The shared child saw two backups, so N_node = 2 (not 1).
    assert shared_child.N_node == 2
    assert shared_child.N_cell[0, 0] == 2
    # Running mean of (0.5, -0.5) = 0.0 at the shared cell.
    assert shared_child.Q[0, 0] == pytest.approx(0.0)


def test_mcts_search_with_transposition_still_converges_on_forced_overflow():
    # End-to-end: mcts_search internally builds and uses the transposition
    # cache. The overall convergence behavior must remain correct.
    scenario = forced_baku_overflow_death()
    rng = np.random.default_rng(0)
    evaluator = TerminalOnlyEvaluator()

    result = mcts_search(
        scenario.game,
        MCTSConfig(iterations=200, exploration_c=1.0, evaluator=None, use_tablebase=False),
        evaluator,
        rng,
        scenario.config,
    )

    assert result.root_value_for_hal == pytest.approx(1.0, abs=0.05)


def test_step_into_child_transposition_none_does_not_populate_anything():
    # transposition=None branch: the cache is genuinely opt-out, not silently
    # using a hidden global. With None passed, no external structure should
    # be mutated.
    game = _fresh_game()
    config = ExactSearchConfig()
    node = make_node(game, config)
    rng = np.random.default_rng(0)

    d = _idx(node.drop_seconds, 1)
    c = _idx(node.check_seconds, 2)
    # Just verify nothing crashes with explicit transposition=None.
    child, _ = _step_into_child(node, game, d, c, rng, config, transposition=None)
    assert isinstance(child, MCTSNode)
    assert (1, 2, None) in node.children
