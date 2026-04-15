"""Single half-round CFR solver for Drop The Handkerchief.

CFR finds Nash equilibrium strategies by iterating regret matching.
The payoff matrix can be immediate-only or augmented with continuation values.
"""

from __future__ import annotations

import numpy as np

from src.Constants import (
    FAILED_CHECK_PENALTY,
    CYLINDER_MAX,
    BASE_CURVE_K,
    CARDIAC_DECAY,
    REFEREE_DECAY,
    REFEREE_FLOOR,
)


def regret_match(cumulative_regret: np.ndarray) -> np.ndarray:
    """Convert cumulative regret vector into a strategy (probability distribution)."""
    positive = np.maximum(0, cumulative_regret)
    total = np.sum(positive)
    if total > 0:
        return positive / total
    return np.ones(len(cumulative_regret)) / len(cumulative_regret)


def survival_probability(
    death_duration: float,
    player_ttd: float,
    cprs_performed: int,
    physicality: float,
) -> float:
    """Compute survival probability using the engine's exact formula."""
    if death_duration >= CYLINDER_MAX:
        return 0.0
    base = max(0.0, 1.0 - (death_duration / CYLINDER_MAX) ** BASE_CURVE_K)
    cardiac = CARDIAC_DECAY ** (player_ttd / 60.0)
    referee = max(REFEREE_FLOOR, REFEREE_DECAY ** cprs_performed)
    return base * cardiac * referee * physicality


def compute_payoff_matrix(
    checker_cylinder: float,
    turn_duration: int = 60,
) -> np.ndarray:
    """Build the Checker's immediate payoff matrix (no continuation values)."""
    n = turn_duration
    payoff = np.zeros((n, n), dtype=np.float64)

    for d in range(n):
        drop_time = d + 1
        for c in range(n):
            check_time = c + 1

            if check_time >= drop_time:
                st = max(1, check_time - drop_time)
                if checker_cylinder + st >= CYLINDER_MAX:
                    payoff[d][c] = -CYLINDER_MAX
                else:
                    payoff[d][c] = -st
            else:
                injection = min(checker_cylinder + FAILED_CHECK_PENALTY, CYLINDER_MAX)
                payoff[d][c] = -injection

    return payoff


def build_augmented_payoff_matrix(
    st_to_cont_val: dict[int, float],
    fail_cont_val: float,
    fail_surv_prob: float,
    turn_duration: int = 60,
    lose_value: float = -1.0,
) -> np.ndarray:
    """Build augmented payoff matrix from precomputed continuation values.

    Args:
        st_to_cont_val: Maps ST (1..59) → checker continuation value for that outcome.
            For overflow STs, the value already includes survival probability weighting.
        fail_cont_val: Checker continuation value for failed check (survived).
        fail_surv_prob: Survival probability for failed check.
    """
    n = turn_duration
    payoff = np.zeros((n, n), dtype=np.float64)

    fail_payoff = fail_surv_prob * fail_cont_val + (1 - fail_surv_prob) * lose_value

    for d in range(n):
        drop_time = d + 1
        for c in range(n):
            check_time = c + 1

            if check_time >= drop_time:
                st = max(1, check_time - drop_time)
                payoff[d][c] = st_to_cont_val.get(st, 0.0)
            else:
                payoff[d][c] = fail_payoff

    return payoff


def solve_half_round(
    checker_cylinder: float,
    turn_duration: int = 60,
    iterations: int = 10_000,
    payoff_matrix: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Solve a single half-round for Nash equilibrium strategies."""
    n = turn_duration
    if payoff_matrix is not None:
        payoff = payoff_matrix
    else:
        payoff = compute_payoff_matrix(checker_cylinder, turn_duration)

    dropper_regret = np.zeros(n)
    checker_regret = np.zeros(n)
    dropper_strategy_sum = np.zeros(n)
    checker_strategy_sum = np.zeros(n)

    for _ in range(iterations):
        dropper_strat = regret_match(dropper_regret)
        checker_strat = regret_match(checker_regret)

        dropper_strategy_sum += dropper_strat
        checker_strategy_sum += checker_strat

        dropper_action_values = -payoff @ checker_strat
        checker_action_values = payoff.T @ dropper_strat

        dropper_EV = dropper_strat @ dropper_action_values
        checker_EV = checker_strat @ checker_action_values

        dropper_regret += dropper_action_values - dropper_EV
        checker_regret += checker_action_values - checker_EV

    avg_dropper = dropper_strategy_sum / dropper_strategy_sum.sum()
    avg_checker = checker_strategy_sum / checker_strategy_sum.sum()
    game_value = avg_dropper @ payoff @ avg_checker

    return (avg_dropper, avg_checker, game_value)
