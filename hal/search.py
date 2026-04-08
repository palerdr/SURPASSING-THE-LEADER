from __future__ import annotations

import numpy as np

from src.Game import Game
from src.Constants import (
    CYLINDER_MAX,
    FAILED_CHECK_PENALTY,
    DEATH_PROCEDURE_OVERHEAD,
    WITHIN_ROUND_OVERHEAD,
    TURN_DURATION_LEAP,
    TURN_DURATION_NORMAL,
)
from cfr.half_round import survival_probability

from .types import Bucket, BeliefState, MemoryMode
from .buckets import bucket_pair_payoff, get_buckets
from .solver import solve_minimax, best_response
from .evaluate import evaluate
from .memory import update_memory


class GameSnapshot:
    __slots__ = (
        "p1_cylinder", "p1_ttd", "p1_deaths", "p1_alive", "p1_death_history_len",
        "p2_cylinder", "p2_ttd", "p2_deaths", "p2_alive", "p2_death_history_len",
        "cprs", "clock", "current_half", "round_num", "game_over",
        "winner", "loser", "history_len",
    )

    def __init__(self, game: Game):
        self.p1_cylinder = game.player1.cylinder
        self.p1_ttd = game.player1.ttd
        self.p1_deaths = game.player1.deaths
        self.p1_alive = game.player1.alive
        self.p1_death_history_len = len(game.player1.death_history)
        self.p2_cylinder = game.player2.cylinder
        self.p2_ttd = game.player2.ttd
        self.p2_deaths = game.player2.deaths
        self.p2_alive = game.player2.alive
        self.p2_death_history_len = len(game.player2.death_history)
        self.cprs = game.referee.cprs_performed
        self.clock = game.game_clock
        self.current_half = game.current_half
        self.round_num = game.round_num
        self.game_over = game.game_over
        self.winner = game.winner
        self.loser = game.loser
        self.history_len = len(game.history)

    def restore(self, game: Game) -> None:
        game.player1.cylinder = self.p1_cylinder
        game.player1.ttd = self.p1_ttd
        game.player1.deaths = self.p1_deaths
        game.player1.alive = self.p1_alive
        del game.player1.death_history[self.p1_death_history_len:]
        game.player2.cylinder = self.p2_cylinder
        game.player2.ttd = self.p2_ttd
        game.player2.deaths = self.p2_deaths
        game.player2.alive = self.p2_alive
        del game.player2.death_history[self.p2_death_history_len:]
        game.referee.cprs_performed = self.cprs
        game.game_clock = self.clock
        game.current_half = self.current_half
        game.round_num = self.round_num
        game.game_over = self.game_over
        game.winner = self.winner
        game.loser = self.loser
        del game.history[self.history_len:]


def apply_half_round(game: Game, drop_time: int, check_time: int, survived: bool | None) -> float:
    dropper, checker = game.get_roles_for_half(game.current_half)
    turn_duration = game.get_turn_duration()

    success = check_time >= drop_time
    death_occurred = False
    death_duration = 0.0
    surv_prob = 0.0

    if success:
        st = max(1, check_time - drop_time)
        checker.add_to_cylinder(st)
        if checker.cylinder >= CYLINDER_MAX:
            death_occurred = True
            death_duration = min(checker.cylinder, CYLINDER_MAX)
    else:
        checker.add_to_cylinder(FAILED_CHECK_PENALTY)
        death_occurred = True
        death_duration = min(checker.cylinder, CYLINDER_MAX)

    if death_occurred:
        surv_prob = survival_probability(
            death_duration, checker.ttd, game.referee.cprs_performed, checker.physicality,
        )
        game.referee.cprs_performed += 1
        checker.on_death(death_duration)

        if survived:
            checker.on_revival()
        else:
            checker.on_permanent_death()
            game.game_over = True
            game.winner = dropper
            game.loser = checker

    game.advance_clock(turn_duration)
    if death_occurred:
        game.advance_clock(death_duration + DEATH_PROCEDURE_OVERHEAD)

    if not game.game_over:
        if game.current_half == 1:
            game.advance_clock(WITHIN_ROUND_OVERHEAD)
            game.current_half = 2
        else:
            game.snap_clock_to_next_minute()
            game.current_half = 1
            game.round_num += 1

    return surv_prob


def _simulate_and_recurse(
    game: Game, d_bucket: Bucket, c_bucket: Bucket,
    depth: int, belief: BeliefState, memory: MemoryMode,
) -> float:
    drop_time = (d_bucket.lo + d_bucket.hi) // 2
    check_time = (c_bucket.lo + c_bucket.hi) // 2

    _, checker = game.get_roles_for_half(game.current_half)
    success = check_time >= drop_time

    if success:
        st = max(1, check_time - drop_time)
        causes_death = checker.cylinder + st >= CYLINDER_MAX
    else:
        causes_death = True

    snap = GameSnapshot(game)

    if not causes_death:
        apply_half_round(game, drop_time, check_time, survived=None)
        _, cont_val = search(game, depth - 1, belief, memory)
        snap.restore(game)
        return cont_val

    death_duration = min(
        checker.cylinder + (max(1, check_time - drop_time) if success else FAILED_CHECK_PENALTY),
        CYLINDER_MAX,
    )
    surv_prob = survival_probability(
        death_duration, checker.ttd, game.referee.cprs_performed, checker.physicality,
    )

    apply_half_round(game, drop_time, check_time, survived=True)
    if game.game_over:
        survived_val = 1.0 if game.winner.name.lower() == "hal" else -1.0
    else:
        _, survived_val = search(game, depth - 1, belief, memory)
    snap.restore(game)

    apply_half_round(game, drop_time, check_time, survived=False)
    if game.game_over:
        died_val = 1.0 if game.winner.name.lower() == "hal" else -1.0
    else:
        _, died_val = search(game, depth - 1, belief, memory)
    snap.restore(game)

    return surv_prob * survived_val + (1.0 - surv_prob) * died_val


def search(
    game: Game, depth: int, belief: BeliefState, memory: MemoryMode
) -> tuple[np.ndarray | None, float]:
    if depth == 0 or game.game_over:
        return None, evaluate(game)

    dropper, _ = game.get_roles_for_half(game.current_half)
    hal_is_dropper = dropper.name.lower() == "hal"
    turn_duration = game.get_turn_duration()
    effective_td = (
        TURN_DURATION_NORMAL
        if (memory == MemoryMode.AMNESIA and turn_duration == TURN_DURATION_LEAP)
        else turn_duration
    )

    hal_knows_leap = memory != MemoryMode.AMNESIA
    hal_buckets = get_buckets(effective_td, knows_leap=hal_knows_leap)
    baku_buckets = get_buckets(turn_duration, knows_leap=True)

    n_hal = len(hal_buckets)
    n_baku = len(baku_buckets)
    _, checker = game.get_roles_for_half(game.current_half)

    payoff = np.zeros((n_hal, n_baku))
    for i, h_bucket in enumerate(hal_buckets):
        for j, b_bucket in enumerate(baku_buckets):
            d_bucket = h_bucket if hal_is_dropper else b_bucket
            c_bucket = b_bucket if hal_is_dropper else h_bucket

            cont_val = _simulate_and_recurse(
                game, d_bucket, c_bucket, depth, belief, memory,
            )

            immediate = bucket_pair_payoff(d_bucket, c_bucket, checker.cylinder)
            immediate_norm = -immediate / CYLINDER_MAX
            if hal_is_dropper:
                immediate_norm = -immediate_norm

            payoff[i][j] = 0.3 * immediate_norm + 0.7 * cont_val

    can_exploit = (
        belief.exploitation_mode
        and belief.baku_predicted_bucket_probs is not None
        and len(belief.baku_predicted_bucket_probs) == n_baku
    )
    if can_exploit:
        strategy, value = best_response(belief, payoff)
    else:
        strategy, value = solve_minimax(payoff)

    return strategy, value
