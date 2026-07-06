"""Analytic stage-transition map vs the live engine (plan ticket 13).

The tablebase's cost model rests on assembling payoff matrices from ~60
outcome classes instead of ~9,000 engine calls per node. That is only
defensible if the analytic map reproduces the engine EXACTLY — these
tests sweep randomized states (all clock regimes, both halves, the leap
turn, post-death epochs) and assert outcome-class-by-outcome-class
equality of successor public states, death durations, and survival
probabilities (the probability is the engine referee's own value, so
equality is bitwise).
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.getcwd())

from stl.solver.backward import (
    MAX_ST,
    analytic_stage_outcomes,
    stage_matrix_from_values,
    verify_stage_outcomes_against_engine,
)
from stl.engine.game import LS_WINDOW_START, PHYSICALITY_BAKU, PHYSICALITY_HAL
from stl.engine.game import Game
from stl.engine.game import Player
from stl.engine.game import Referee


def make_game(*, clock=720.0, half=1, hal_cyl=0.0, baku_cyl=0.0,
              hal_deaths=0, baku_deaths=0, hal_ttd=0.0, baku_ttd=0.0) -> Game:
    hal = Player(name="Hal", physicality=PHYSICALITY_HAL)
    baku = Player(name="Baku", physicality=PHYSICALITY_BAKU)
    game = Game(player1=hal, player2=baku, referee=Referee(), first_dropper=hal)
    game.seed(0)
    game.game_clock = clock
    game.current_half = half
    hal.cylinder, baku.cylinder = hal_cyl, baku_cyl
    hal.deaths, baku.deaths = hal_deaths, baku_deaths
    hal.ttd, baku.ttd = hal_ttd, baku_ttd
    game.referee.cprs_performed = hal_deaths + baku_deaths  # engine invariant
    return game


def random_reachable_game(rng: np.random.Generator) -> Game:
    """A state respecting engine reachability (ttd >= 60*deaths, cprs = sum)."""
    region = rng.choice(["preleap", "leap", "postleap"])
    if region == "preleap":
        clock = float(rng.integers(12, 59) * 60)
    elif region == "leap":
        clock = float(rng.integers(3540, 3601))
    else:
        clock = float(3601 + rng.integers(0, 40) * 60)
    deaths = (int(rng.integers(0, 3)), int(rng.integers(0, 3)))
    ttds = tuple(
        float(sum(rng.integers(60, 300) for _ in range(d))) for d in deaths
    )
    return make_game(
        clock=clock,
        half=int(rng.integers(1, 3)),
        hal_cyl=float(rng.integers(0, 300)),
        baku_cyl=float(rng.integers(0, 300)),
        hal_deaths=deaths[0],
        baku_deaths=deaths[1],
        hal_ttd=ttds[0],
        baku_ttd=ttds[1],
    )


def test_engine_equivalence_random_sweep():
    rng = np.random.default_rng(42)
    for _ in range(300):
        game = random_reachable_game(rng)
        verify_stage_outcomes_against_engine(game)


def test_engine_equivalence_handpicked_edges():
    edge_states = [
        make_game(clock=720.0, half=1),                                  # opening
        make_game(clock=float(LS_WINDOW_START), half=2),                 # leap, Baku drops 61
        make_game(clock=3600.0, half=2),                                 # closed-boundary leap
        make_game(clock=3601.0, half=1),                                 # first post-leap
        make_game(clock=720.0, half=1, baku_cyl=239.0),                  # last survivable fail
        make_game(clock=720.0, half=1, baku_cyl=240.0),                  # fail always fatal
        make_game(clock=720.0, half=1, baku_cyl=299.0),                  # every ST overflows
        make_game(clock=2100.0, half=2, hal_cyl=250.0, hal_deaths=2,
                  baku_deaths=1, hal_ttd=200.0, baku_ttd=90.0),          # deep epoch
    ]
    for game in edge_states:
        verify_stage_outcomes_against_engine(game)


def test_leap_turn_gives_baku_dropper_61_rows():
    game = make_game(clock=float(LS_WINDOW_START), half=2)  # Baku drops
    outcomes = analytic_stage_outcomes(game)
    assert outcomes.turn_duration == 61
    assert outcomes.drop_seconds[-1] == 61
    assert outcomes.check_seconds[-1] == 60  # checker capped — drop 61 beats all


def test_fail_probability_is_engine_referee_value():
    game = make_game(clock=720.0, half=1, baku_cyl=120.0, baku_deaths=1, baku_ttd=90.0)
    outcomes = analytic_stage_outcomes(game)
    _, checker = game.get_roles_for_half(game.current_half)
    expected = game.referee.compute_survival_probability(checker, death_duration=180.0)
    assert outcomes.fail.death_duration == 180.0
    assert outcomes.fail.survival_probability == expected


def test_stage_matrix_assembly_semantics():
    game = make_game(clock=720.0, half=1, baku_cyl=297.0)  # ST >= 3 overflows
    outcomes = analytic_stage_outcomes(game)

    success_values = np.zeros(MAX_ST)
    for s in outcomes.successes:
        success_values[s.st - 1] = -1.0 if s.overflow else 0.5  # checker=Baku dies => +1? see below
    # Baku is checker: overflow means Baku dies => Hal wins => +1.
    success_values = np.where(success_values == -1.0, 1.0, success_values)
    fail_value = 1.0  # fail at cyl 297 is fatal for Baku

    matrix = stage_matrix_from_values(outcomes, success_values, fail_value)

    assert matrix.shape == (len(outcomes.drop_seconds), len(outcomes.check_seconds))
    # fail triangle: check < drop
    assert matrix[5, 0] == 1.0
    # tie cell (drop 1, check 1): ST = 1 -> 297+1 = 298 < 300 -> continuation
    assert matrix[0, 0] == 0.5
    # ST=2 continuation (297+2=299), ST=3 overflow (300)
    assert matrix[0, 2] == 0.5
    assert matrix[0, 3] == 1.0


def test_st_never_exceeds_59_even_in_leap_turn():
    game = make_game(clock=float(LS_WINDOW_START), half=2)
    outcomes = analytic_stage_outcomes(game)
    assert len(outcomes.successes) == MAX_ST == 59
