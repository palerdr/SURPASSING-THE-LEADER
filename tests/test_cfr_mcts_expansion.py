"""Tests for MCTS Step 4: chance-branch expansion via _step_into_child."""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from environment.cfr.exact_transition import ExactSearchConfig
from environment.cfr.mcts import MCTSNode, _step_into_child, make_node
from environment.cfr.tactical_scenarios import forced_baku_overflow_death
from src.Constants import PHYSICALITY_BAKU, PHYSICALITY_HAL
from src.Game import Game
from src.Player import Player
from src.Referee import Referee


def _baku_checker_at_cylinder(cyl: float) -> Game:
    hal = Player(name="Hal", physicality=PHYSICALITY_HAL)
    baku = Player(name="Baku", physicality=PHYSICALITY_BAKU)
    game = Game(player1=hal, player2=baku, referee=Referee(), first_dropper=hal)
    game.seed(0)
    game.player2.cylinder = cyl
    return game


def _idx(seconds: tuple[int, ...], target: int) -> int:
    return seconds.index(target)


def test_step_into_child_no_death_creates_one_child_with_none_key():
    # Fresh game (cyl=0); drop=1, check=60 → success, ST=59, cyl=59, no death.
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
    # Baku checker at cyl=180; failed check (drop=60, check=1) → death event
    # with intermediate survival probability. Same seed → same outcome.
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

    assert survived_a is not None  # death occurred
    assert survived_a == survived_b


def test_step_into_child_cache_hit_does_not_double_create():
    # Two consecutive calls with the same rng seed should produce the same key,
    # and the children dict must not grow on the second call.
    config = ExactSearchConfig()
    game = _baku_checker_at_cylinder(0.0)
    node = make_node(game, config)
    d_idx = _idx(node.drop_seconds, 1)
    c_idx = _idx(node.check_seconds, 60)

    rng = np.random.default_rng(0)
    child_first, _ = _step_into_child(node, game, d_idx, c_idx, rng, config)

    # Rewind the engine to the parent's position before the second call.
    node.game_snapshot.restore(game)

    rng2 = np.random.default_rng(0)
    child_second, _ = _step_into_child(node, game, d_idx, c_idx, rng2, config)

    assert len(node.children) == 1
    assert child_first is child_second


def test_step_into_child_leaves_engine_at_post_action_position():
    # After stepping a no-death success (drop=1, check=60), engine state should
    # reflect post-half-round: half flipped to 2, Baku cylinder = 59,
    # Hal cylinder unchanged at 0.
    config = ExactSearchConfig()
    game = _baku_checker_at_cylinder(0.0)
    node = make_node(game, config)
    d_idx = _idx(node.drop_seconds, 1)
    c_idx = _idx(node.check_seconds, 60)
    rng = np.random.default_rng(0)

    _step_into_child(node, game, d_idx, c_idx, rng, config)

    assert game.current_half == 2
    assert game.player2.cylinder == 59  # Baku gained ST=59
    assert game.player1.cylinder == 0   # Hal (dropper) unchanged


def test_step_into_child_at_zero_survival_probability_forces_died_branch():
    # forced_baku_overflow_death: cyl=299. Any action's death_duration hits 300,
    # survival_probability = 0, so sampling must produce survived=False every
    # time, and the resulting child must be a terminal Hal-win node.
    scenario = forced_baku_overflow_death()
    config = scenario.config
    node = make_node(scenario.game, config)

    d_idx = _idx(node.drop_seconds, 1)
    c_idx = _idx(node.check_seconds, 60)
    rng = np.random.default_rng(42)

    child, survived = _step_into_child(node, scenario.game, d_idx, c_idx, rng, config)

    assert survived is False
    assert child.terminal_value == 1.0
