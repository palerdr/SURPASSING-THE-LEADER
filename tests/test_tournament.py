"""Tests for Phase 5 tournament harness.

Verifies that ``play_match`` runs every game to completion, returns a
well-formed ``MatchResult``, and is reproducible under seeded RNG. We
use deterministic action callables (``always-1``, ``always-min``) and a
small RNG-driven double rather than ``RandomBot`` so the tests do not
depend on the global ``random`` module.
"""

from __future__ import annotations

import os
import random
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from environment.cfr.tactical_scenarios import forced_baku_overflow_death
from src.Constants import TURN_DURATION_NORMAL
from src.Game import Game
from training.tournament import MatchResult, play_match


def _seeded_random_action(seed: int):
    """Stateful callable that picks a uniformly random legal second.

    Mirrors ``environment.opponents.random_bot.RandomBot`` but uses a
    private ``random.Random`` so it can be re-seeded per test without
    leaking into the global RNG.
    """
    rng = random.Random(seed)

    def choose(game: Game, role: str, turn_duration: int) -> int:
        del game
        if role == "dropper":
            return rng.randint(1, turn_duration)
        return rng.randint(1, TURN_DURATION_NORMAL)

    return choose


def _always_one(game: Game, role: str, turn_duration: int) -> int:
    del game, role, turn_duration
    return 1


def _always_60(game: Game, role: str, turn_duration: int) -> int:
    del game, turn_duration
    if role == "checker":
        return 60
    return 1


def test_play_match_runs_all_games_and_returns_match_result():
    hal_action = _seeded_random_action(seed=11)
    baku_action = _seeded_random_action(seed=23)
    result = play_match(
        hal_choose_action=hal_action,
        baku_choose_action=baku_action,
        n_games=5,
        seed=42,
    )
    assert isinstance(result, MatchResult)
    assert result.games_played == 5
    assert result.hal_wins + result.baku_wins + result.draws == 5


def test_play_match_avg_length_is_positive():
    hal_action = _seeded_random_action(seed=11)
    baku_action = _seeded_random_action(seed=23)
    result = play_match(hal_action, baku_action, n_games=3, seed=7)
    assert result.avg_game_length_half_rounds > 0.0


def test_play_match_termination_causes_recorded():
    hal_action = _seeded_random_action(seed=11)
    baku_action = _seeded_random_action(seed=23)
    result = play_match(hal_action, baku_action, n_games=5, seed=42)
    assert sum(result.cause_of_termination.values()) == result.games_played


def test_play_match_is_deterministic_under_seed():
    # Same seed AND same per-callable seeds → identical match.
    a1 = _seeded_random_action(seed=11)
    b1 = _seeded_random_action(seed=23)
    r1 = play_match(a1, b1, n_games=4, seed=99)

    a2 = _seeded_random_action(seed=11)
    b2 = _seeded_random_action(seed=23)
    r2 = play_match(a2, b2, n_games=4, seed=99)

    assert r1 == r2


def test_play_match_changes_with_seed():
    # Different match seeds (but identical callables) produce different
    # game-state RNG draws and therefore generally different outcomes.
    a1 = _seeded_random_action(seed=11)
    b1 = _seeded_random_action(seed=23)
    r1 = play_match(a1, b1, n_games=5, seed=1)

    a2 = _seeded_random_action(seed=11)
    b2 = _seeded_random_action(seed=23)
    r2 = play_match(a2, b2, n_games=5, seed=2)

    # We only assert *some* signal differs; the totals could coincidentally
    # match on rare seeds, but the per-game-seed RNG draws differ so the
    # average half-round length almost always differs as well.
    assert (r1.hal_wins, r1.avg_game_length_half_rounds) != (
        r2.hal_wins,
        r2.avg_game_length_half_rounds,
    )


def test_play_match_with_starting_scenario_finishes_immediately():
    # ``forced_baku_overflow_death`` is a pinned tablebase entry where any
    # legal half-round terminates with a Baku-side death. Hal should win
    # 100% of games even when both players act "blindly".
    scenario = forced_baku_overflow_death()
    result = play_match(
        hal_choose_action=_always_one,
        baku_choose_action=_always_60,
        n_games=5,
        seed=0,
        starting_scenario=scenario,
    )
    assert result.games_played == 5
    assert result.hal_wins == 5
    # All games finish in their first half-round.
    assert result.avg_game_length_half_rounds == pytest.approx(1.0)


def test_play_match_zero_games_returns_empty_result():
    result = play_match(_always_one, _always_60, n_games=0, seed=0)
    assert result.games_played == 0
    assert result.hal_wins == 0
    assert result.baku_wins == 0
    assert result.draws == 0
    assert result.avg_game_length_half_rounds == 0.0
    assert result.cause_of_termination == {}


def test_play_match_all_games_run_to_completion():
    # Random vs random always terminates because every game must
    # eventually overflow a cylinder or fail a check; we verify that the
    # match harness reports the expected total without timing out.
    hal_action = _seeded_random_action(seed=11)
    baku_action = _seeded_random_action(seed=23)
    result = play_match(hal_action, baku_action, n_games=5, seed=42)
    assert result.games_played == result.hal_wins + result.baku_wins + result.draws
    # No "unfinished" causes — the safety limit was never hit.
    assert "unfinished" not in result.cause_of_termination
