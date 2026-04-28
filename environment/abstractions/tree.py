"""Multi-round CFR with backward induction over abstract game states.

Solves backward from the last round: terminal states get win/loss values,
earlier states fold in continuation values via augmented payoff matrices.
"""

from __future__ import annotations

import numpy as np

from src.Constants import (
    CYLINDER_MAX,
    FAILED_CHECK_PENALTY,
    TURN_DURATION_NORMAL,
    TURN_DURATION_LEAP,
    LS_WINDOW_START,
    LS_WINDOW_END,
    OPENING_START_CLOCK,
    WITHIN_ROUND_OVERHEAD,
    DEATH_PROCEDURE_OVERHEAD,
    PHYSICALITY_HAL,
    PHYSICALITY_BAKU,
)
from environment.cfr.half_round import (
    solve_half_round,
    build_augmented_payoff_matrix,
    survival_probability,
)
from .game_state import (
    AbstractState,
    make_abstract_state,
    bucket_cylinder,
    representative_ttd,
    total_cprs,
    CYL_BUCKET_SIZE,
    CLOCK_BUCKET_SIZE,
)


StrategyTable = dict[AbstractState, tuple[np.ndarray, np.ndarray, float]]

REPRESENTATIVE_STS = [1] + list(range(CYL_BUCKET_SIZE, 60, CYL_BUCKET_SIZE))


def get_turn_duration(game_clock: float) -> int:
    if LS_WINDOW_START <= game_clock <= LS_WINDOW_END:
        return TURN_DURATION_LEAP
    return TURN_DURATION_NORMAL


def snap_to_next_minute(clock: float) -> float:
    gc = int(clock)
    if gc < 3600:
        snapped = ((gc // 60) + 1) * 60
        if snapped == 3600:
            snapped = 3601
        return float(snapped)
    elif gc <= 3600:
        return 3601.0
    else:
        elapsed = gc - 3601
        return float(3601 + ((elapsed // 60) + 1) * 60)


def compute_successors(
    state: AbstractState,
    turn_duration: int,
) -> list[tuple[AbstractState, float]]:
    """Compute successor states with death duration for each outcome."""
    half = state.half
    clock = state.clock * CLOCK_BUCKET_SIZE

    if half == 0:
        checker_cyl = state.opp_cyl * CYL_BUCKET_SIZE
        checker_deaths = state.opp_deaths
        dropper_cyl = state.my_cyl * CYL_BUCKET_SIZE
        dropper_deaths = state.my_deaths
    else:
        checker_cyl = state.my_cyl * CYL_BUCKET_SIZE
        checker_deaths = state.my_deaths
        dropper_cyl = state.opp_cyl * CYL_BUCKET_SIZE
        dropper_deaths = state.opp_deaths

    nc_base = clock + turn_duration
    successors = []

    for st in REPRESENTATIVE_STS:
        new_cyl = checker_cyl + st
        if new_cyl >= CYLINDER_MAX:
            dd = float(CYLINDER_MAX)
            death_clock = nc_base + dd + DEATH_PROCEDURE_OVERHEAD
            if half == 0:
                s = make_abstract_state(state.round_num, 1,
                                        dropper_cyl, 0.0,
                                        dropper_deaths, checker_deaths + 1,
                                        death_clock)
            else:
                death_clock = snap_to_next_minute(death_clock + WITHIN_ROUND_OVERHEAD)
                s = make_abstract_state(state.round_num + 1, 0,
                                        0.0, dropper_cyl,
                                        checker_deaths + 1, dropper_deaths,
                                        death_clock)
            successors.append((s, dd))
        else:
            if half == 0:
                s = make_abstract_state(state.round_num, 1,
                                        dropper_cyl, new_cyl,
                                        dropper_deaths, checker_deaths,
                                        nc_base)
            else:
                sc = snap_to_next_minute(nc_base + WITHIN_ROUND_OVERHEAD)
                s = make_abstract_state(state.round_num + 1, 0,
                                        new_cyl, dropper_cyl,
                                        checker_deaths, dropper_deaths,
                                        sc)
            successors.append((s, 0.0))

    # Failed check
    dd = min(checker_cyl + FAILED_CHECK_PENALTY, CYLINDER_MAX)
    fail_clock = nc_base + dd + DEATH_PROCEDURE_OVERHEAD
    if half == 0:
        s = make_abstract_state(state.round_num, 1,
                                dropper_cyl, 0.0,
                                dropper_deaths, checker_deaths + 1,
                                fail_clock)
    else:
        fail_clock = snap_to_next_minute(fail_clock + WITHIN_ROUND_OVERHEAD)
        s = make_abstract_state(state.round_num + 1, 0,
                                0.0, dropper_cyl,
                                checker_deaths + 1, dropper_deaths,
                                fail_clock)
    successors.append((s, dd))

    return successors


def solve_game(
    max_rounds: int = 10,
    iterations_per_state: int = 5_000,
) -> StrategyTable:
    """Solve the full game with backward induction."""
    strategy_table: StrategyTable = {}

    init_state = make_abstract_state(
        round_num=0, half=0,
        my_cylinder=0.0, opp_cylinder=0.0,
        my_deaths=0, opp_deaths=0,
        game_clock=OPENING_START_CLOCK,
    )

    # ── Phase 1: Forward pass ────────────────────────────────────────
    total_steps = max_rounds * 2
    states_by_step: list[set[AbstractState]] = [set() for _ in range(total_steps)]
    states_by_step[0].add(init_state)

    for step in range(total_steps):
        for state in states_by_step[step]:
            clock = state.clock * CLOCK_BUCKET_SIZE
            td = get_turn_duration(clock)
            for succ, _ in compute_successors(state, td):
                target = succ.round_num * 2 + succ.half
                if target < total_steps:
                    states_by_step[target].add(succ)

    total_states = sum(len(s) for s in states_by_step)
    print(f"  Forward pass: {total_states} reachable states across {total_steps} steps")

    # ── Phase 2: Backward pass ───────────────────────────────────────
    continuation_values: dict[AbstractState, float] = {}

    for step in range(total_steps - 1, -1, -1):
        half = step % 2

        for state in states_by_step[step]:
            clock = state.clock * CLOCK_BUCKET_SIZE
            turn_duration = get_turn_duration(clock)

            if half == 0:
                checker_cyl = state.opp_cyl * CYL_BUCKET_SIZE
                checker_deaths = state.opp_deaths
                checker_phys = PHYSICALITY_BAKU
                dropper_cyl = state.my_cyl * CYL_BUCKET_SIZE
                dropper_deaths = state.my_deaths
            else:
                checker_cyl = state.my_cyl * CYL_BUCKET_SIZE
                checker_deaths = state.my_deaths
                checker_phys = PHYSICALITY_HAL
                dropper_cyl = state.opp_cyl * CYL_BUCKET_SIZE
                dropper_deaths = state.opp_deaths

            # Derive TTD and CPR from death counts
            checker_ttd = representative_ttd(checker_deaths)
            cprs = total_cprs(state.my_deaths, state.opp_deaths)

            sign = -1.0 if half == 0 else 1.0
            nc_base = clock + turn_duration

            # ── Build ST → continuation value mapping ────────────
            st_to_cont: dict[int, float] = {}
            _bkt_cache: dict[int, float] = {}

            overflow_st = max(1, int(CYLINDER_MAX - checker_cyl))

            # All overflow STs → same successor
            if overflow_st <= turn_duration:
                dd = float(CYLINDER_MAX)
                surv = survival_probability(dd, checker_ttd, cprs, checker_phys)
                death_clock = nc_base + dd + DEATH_PROCEDURE_OVERHEAD
                if half == 0:
                    s = make_abstract_state(state.round_num, 1,
                                            dropper_cyl, 0.0,
                                            dropper_deaths, checker_deaths + 1,
                                            death_clock)
                else:
                    death_clock = snap_to_next_minute(death_clock + WITHIN_ROUND_OVERHEAD)
                    s = make_abstract_state(state.round_num + 1, 0,
                                            0.0, dropper_cyl,
                                            checker_deaths + 1, dropper_deaths,
                                            death_clock)
                ov_val = surv * sign * continuation_values.get(s, 0.0) + (1 - surv) * (-1.0)
                for st in range(overflow_st, turn_duration + 1):
                    st_to_cont[st] = ov_val

            # Non-overflow STs — cache by bucketed cylinder
            for st in range(1, min(overflow_st, turn_duration + 1)):
                bkt = bucket_cylinder(checker_cyl + st)
                if bkt not in _bkt_cache:
                    if half == 0:
                        s = make_abstract_state(state.round_num, 1,
                                                dropper_cyl, checker_cyl + st,
                                                dropper_deaths, checker_deaths,
                                                nc_base)
                    else:
                        sc = snap_to_next_minute(nc_base + WITHIN_ROUND_OVERHEAD)
                        s = make_abstract_state(state.round_num + 1, 0,
                                                checker_cyl + st, dropper_cyl,
                                                checker_deaths, dropper_deaths,
                                                sc)
                    _bkt_cache[bkt] = sign * continuation_values.get(s, 0.0)
                st_to_cont[st] = _bkt_cache[bkt]

            # Failed check
            fail_dd = min(checker_cyl + FAILED_CHECK_PENALTY, CYLINDER_MAX)
            fail_surv = survival_probability(fail_dd, checker_ttd, cprs, checker_phys)
            fail_clock = nc_base + fail_dd + DEATH_PROCEDURE_OVERHEAD
            if half == 0:
                fail_succ = make_abstract_state(state.round_num, 1,
                                                dropper_cyl, 0.0,
                                                dropper_deaths, checker_deaths + 1,
                                                fail_clock)
            else:
                fail_clock = snap_to_next_minute(fail_clock + WITHIN_ROUND_OVERHEAD)
                fail_succ = make_abstract_state(state.round_num + 1, 0,
                                                0.0, dropper_cyl,
                                                checker_deaths + 1, dropper_deaths,
                                                fail_clock)
            fail_cv = sign * continuation_values.get(fail_succ, 0.0)

            # Build payoff matrix and solve
            aug_payoff = build_augmented_payoff_matrix(
                st_to_cont_val=st_to_cont,
                fail_cont_val=fail_cv,
                fail_surv_prob=fail_surv,
                turn_duration=turn_duration,
            )

            d_strat, c_strat, game_value = solve_half_round(
                checker_cylinder=checker_cyl,
                turn_duration=turn_duration,
                iterations=iterations_per_state,
                payoff_matrix=aug_payoff,
            )

            strategy_table[state] = (d_strat, c_strat, game_value)

            if half == 0:
                continuation_values[state] = -game_value
            else:
                continuation_values[state] = game_value

    return strategy_table
