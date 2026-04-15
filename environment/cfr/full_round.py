"""Two half-round (one full round) CFR with state transitions.

A full round is:
  Half 1: Player A drops, Player B checks → resolve → state updates
  Half 2: Player B drops, Player A checks → resolve → state updates

Between halves, the game state changes:
  - Cylinder may increase (from ST on successful check)
  - A death may occur (failed check or cylinder overflow)
  - If death: cylinder resets to 0, TTD increases, death count increments
  - Roles swap

This module chains two calls to solve_half_round with state transitions
between them, using AbstractState as the info set key.
"""

from __future__ import annotations

import numpy as np

from src.Constants import (
    CYLINDER_MAX,
    FAILED_CHECK_PENALTY,
    TURN_DURATION_NORMAL,
)
from src.Referee import Referee
from src.Player import Player

from .half_round import solve_half_round, compute_payoff_matrix
from .game_state import AbstractState, make_abstract_state


def resolve_half_round(
    checker_cylinder: float,
    drop_time: int,
    check_time: int,
) -> dict:
    """Compute the outcome of a single half-round.

    Given concrete actions from both players, determine what happens.

    Args:
        checker_cylinder: Checker's cylinder before this half-round.
        drop_time: Second the dropper drops (1-60).
        check_time: Second the checker checks (1-60).

    Returns:
        dict with:
            "success": bool — did the checker find the handkerchief?
            "st": int — squandered time (0 if failed check)
            "new_cylinder": float — checker's cylinder after resolution
            "injection": bool — was the cylinder injected?
            "injection_amount": float — amount injected (0 if no injection)
"""
    if check_time >= drop_time:
        success = True
        st = max(1, check_time - drop_time)
        new_cyl = checker_cylinder + st
        if new_cyl >= CYLINDER_MAX:
            injection = True
            injection_amount = CYLINDER_MAX
        else:
            injection = False
            injection_amount = 0.0
    else:
        success = False
        new_cyl = 0
        st = 0
        injection = True
        injection_amount = min(checker_cylinder + FAILED_CHECK_PENALTY, CYLINDER_MAX)

    return {
            "success" : success,
            "st" : st,
            "new_cylinder" : new_cyl,
            "injection" : injection,
            "injection_amount" : injection_amount,
            }

def simulate_round(
    p1_cylinder: float,
    p2_cylinder: float,
    p1_deaths: int,
    p2_deaths: int,
    round_num: int,
    turn_duration: int = TURN_DURATION_NORMAL,
    iterations: int = 5_000,
) -> dict:
    """Solve one full round (2 half-rounds) and return expected outcomes.

    Half 1: P1 drops, P2 checks.
    Half 2: P2 drops, P1 checks.

    This function:
      1. Solves half 1 using solve_half_round (P2 is checker, P2's cylinder matters)
      2. Computes the expected state after half 1 (weighted by both strategies)
      3. Solves half 2 using solve_half_round (P1 is checker, P1's cylinder matters)
      4. Returns both strategies and expected state transitions

    Args:
        p1_cylinder: Player 1's cylinder at round start.
        p2_cylinder: Player 2's cylinder at round start.
        p1_deaths: Player 1's death count.
        p2_deaths: Player 2's death count.
        round_num: Current round number (0-indexed).
        turn_duration: Seconds per half-round (60 or 61).
        iterations: CFR iterations per half-round solve.

    Returns:
        dict with:
            "h1_dropper_strat": np.ndarray — P1's drop strategy for half 1
            "h1_checker_strat": np.ndarray — P2's check strategy for half 1
            "h2_dropper_strat": np.ndarray — P2's drop strategy for half 2
            "h2_checker_strat": np.ndarray — P1's check strategy for half 2
            "h1_game_value": float — half 1 expected Checker payoff
            "h2_game_value": float — half 2 expected Checker payoff
    """
    h1_dropper_strat, h1_checker_strat, h1_value = solve_half_round(p2_cylinder, turn_duration=turn_duration, iterations=iterations)
    h2_dropper_strat, h2_checker_strat, h2_value = solve_half_round(p1_cylinder, turn_duration=turn_duration, iterations=iterations)
    return {
        "h1_dropper_strat" : h1_dropper_strat,
        "h1_checker_strat" : h1_checker_strat,
        "h1_game_value" : h1_value,
        "h2_dropper_strat" : h2_dropper_strat,
        "h2_checker_strat" : h2_checker_strat,
        "h2_game_value" : h2_value,
        }