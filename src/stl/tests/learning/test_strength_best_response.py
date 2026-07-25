"""Best-response exploitability probe (plan ticket 8).

Pins:
- exactness on a forced-terminal scenario (interval collapses to the pin),
- interval semantics (frontier bracketing, lo <= hi, containment),
- determinism,
- the headline discrimination: a deterministic policy is measurably more
  exploitable than a mixing one at the same state — the property that
  value-MSE can never see and the reason this probe exists.
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.getcwd())

from stl.solver.tablebase import forced_baku_overflow_death
from stl.engine.game import PHYSICALITY_BAKU, PHYSICALITY_HAL
from stl.engine.game import Game
from stl.engine.game import Player
from stl.engine.game import Referee
from stl.learning.strength import (
    best_response_interval,
    fixed_second_policy,
    uniform_policy,
)


def make_game(*, clock=720.0, half=1, hal_cyl=0.0, baku_cyl=0.0) -> Game:
    hal = Player(name="Hal", physicality=PHYSICALITY_HAL)
    baku = Player(name="Baku", physicality=PHYSICALITY_BAKU)
    game = Game(player1=hal, player2=baku, referee=Referee(), first_dropper=hal)
    game.seed(0)
    game.game_clock = clock
    game.current_half = half
    hal.cylinder = hal_cyl
    baku.cylinder = baku_cyl
    return game


def test_forced_overflow_scenario_collapses_to_pinned_value():
    """Where the outcome is forced regardless of the frozen seat's play,
    the BR interval must equal the tablebase pin exactly."""
    scenario = forced_baku_overflow_death()
    result = best_response_interval(
        scenario.game, uniform_policy(), depth=3, frozen_name="Hal"
    )
    assert result.lo == pytest.approx(scenario.expected_value, abs=1e-9)
    assert result.hi == pytest.approx(scenario.expected_value, abs=1e-9)


def test_zero_depth_returns_frontier_bracket():
    result = best_response_interval(make_game(), uniform_policy(), depth=0)
    assert (result.lo, result.hi) == (-1.0, 1.0)
    assert result.frontier_hits == 1


def test_interval_is_ordered_and_contained():
    result = best_response_interval(make_game(), uniform_policy(), depth=2)
    assert -1.0 <= result.lo <= result.hi <= 1.0
    assert result.states_solved > 0


def test_deeper_search_never_widens_the_interval():
    game = make_game(hal_cyl=120.0, baku_cyl=120.0, clock=3300.0)
    shallow = best_response_interval(game, uniform_policy(), depth=1)
    deep = best_response_interval(game, uniform_policy(), depth=3)
    assert deep.lo >= shallow.lo - 1e-12
    assert deep.hi <= shallow.hi + 1e-12


def test_deterministic_given_same_inputs():
    a = best_response_interval(make_game(), uniform_policy(), depth=2)
    b = best_response_interval(make_game(), uniform_policy(), depth=2)
    assert (a.lo, a.hi, a.states_solved) == (b.lo, b.hi, b.states_solved)


def test_deterministic_checker_is_more_exploitable_than_safe_checker():
    """The probe's reason to exist: Hal checking at a FIXED second 30 with
    cylinder 180 hands Baku a drop at 31 — a guaranteed failed check and a
    240s death roll. The safe-check policy (always 60) cannot be forced
    into a depth-1 death at all. The fixed policy's certified ceiling must
    sit far below the safe policy's."""
    game = make_game(half=2, hal_cyl=180.0)  # Baku drops, Hal checks

    fixed = best_response_interval(game, fixed_second_policy(30), depth=1)
    safe = best_response_interval(game, fixed_second_policy(60), depth=1)

    # Fixed-30: BR drops at 31 => fail => death duration 240 => survive prob
    # p; value <= 2p - 1 < 0. Safe-60: no death reachable in one half-round.
    assert fixed.hi < safe.hi - 0.5
    assert fixed.hi < 0.0


def test_frozen_baku_seat_works_too():
    game = make_game(half=1)  # Hal drops, Baku checks
    result = best_response_interval(
        game, uniform_policy(), depth=1, frozen_name="Baku"
    )
    assert result.adversary_name == "hal"
    assert -1.0 <= result.lo <= result.hi <= 1.0
