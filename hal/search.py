from __future__ import annotations

import numpy as np
import torch

from src.Game import Game
from src.Constants import (
    CYLINDER_MAX,
    FAILED_CHECK_PENALTY,
    LS_WINDOW_START,
)

from .state import Bucket, BeliefState, MemoryMode, update_memory
from .action_model import best_response, build_payoff_matrix, get_legal_buckets
from environment.cfr.minimax import solve_minimax
from .evaluate import evaluate, get_nn_model


def _exploit_or_nash(
    belief: BeliefState,
    payoff: np.ndarray,
    hal_is_dropper: bool,
    baku_buckets: tuple[Bucket, ...],
    baku_keep: list[int],
) -> tuple[np.ndarray, float]:
    baku_probs = belief.baku_check_probs if hal_is_dropper else belief.baku_drop_probs
    can_exploit_role = belief.check_exploit if hal_is_dropper else belief.drop_exploit
    can_exploit = (
        can_exploit_role
        and baku_probs is not None
        and len(baku_probs) == len(baku_buckets)
    )
    if can_exploit:
        pruned_probs = np.array([baku_probs[j] for j in baku_keep])
        s = pruned_probs.sum()
        if s > 1e-9:
            pruned_probs /= s
        pruned_belief = BeliefState(
            baku_check_history=belief.baku_check_history,
            baku_drop_history=belief.baku_drop_history,
            check_exploit=True,
            drop_exploit=True,
            baku_check_probs=tuple(pruned_probs) if hal_is_dropper else None,
            baku_drop_probs=tuple(pruned_probs) if not hal_is_dropper else None,
        )
        return best_response(pruned_belief, payoff, hal_is_dropper)
    return solve_minimax(payoff)


class GameSnapshot:
    __slots__ = (
        "p1_cylinder", "p1_ttd", "p1_deaths", "p1_alive", "p1_dh_len",
        "p2_cylinder", "p2_ttd", "p2_deaths", "p2_alive", "p2_dh_len",
        "cprs", "clock", "current_half", "round_num", "game_over",
        "winner", "loser", "hist_len",
    )

    def __init__(self, game: Game):
        self.p1_cylinder = game.player1.cylinder
        self.p1_ttd = game.player1.ttd
        self.p1_deaths = game.player1.deaths
        self.p1_alive = game.player1.alive
        self.p1_dh_len = len(game.player1.death_history)
        self.p2_cylinder = game.player2.cylinder
        self.p2_ttd = game.player2.ttd
        self.p2_deaths = game.player2.deaths
        self.p2_alive = game.player2.alive
        self.p2_dh_len = len(game.player2.death_history)
        self.cprs = game.referee.cprs_performed
        self.clock = game.game_clock
        self.current_half = game.current_half
        self.round_num = game.round_num
        self.game_over = game.game_over
        self.winner = game.winner
        self.loser = game.loser
        self.hist_len = len(game.history)

    def restore(self, game: Game) -> None:
        game.player1.cylinder = self.p1_cylinder
        game.player1.ttd = self.p1_ttd
        game.player1.deaths = self.p1_deaths
        game.player1.alive = self.p1_alive
        del game.player1.death_history[self.p1_dh_len:]
        game.player2.cylinder = self.p2_cylinder
        game.player2.ttd = self.p2_ttd
        game.player2.deaths = self.p2_deaths
        game.player2.alive = self.p2_alive
        del game.player2.death_history[self.p2_dh_len:]
        game.referee.cprs_performed = self.cprs
        game.game_clock = self.clock
        game.current_half = self.current_half
        game.round_num = self.round_num
        game.game_over = self.game_over
        game.winner = self.winner
        game.loser = self.loser
        del game.history[self.hist_len:]


def apply_half_round(game: Game, drop_time: int, check_time: int, survived: bool | None) -> float:
    """Thin wrapper around game.resolve_half_round.

    Returns the survival probability that was computed (0.0 if no death).
    Search uses the shared resolver to guarantee parity with engine semantics.
    """
    record = game.resolve_half_round(drop_time, check_time, survived_outcome=survived)
    return record.survival_probability or 0.0


def _hal_immediate_view(
    hal_buckets: tuple[Bucket, ...],
    baku_buckets: tuple[Bucket, ...],
    checker_cylinder: float,
    hal_is_dropper: bool,
) -> np.ndarray:
    """Return a payoff matrix from Hal's perspective (high = good for Hal)."""
    raw = build_payoff_matrix(hal_buckets, baku_buckets, checker_cylinder, hal_is_dropper)
    return -raw if hal_is_dropper else raw


def _prune_dominated(
    hal_buckets: tuple[Bucket, ...],
    baku_buckets: tuple[Bucket, ...],
    checker_cylinder: float,
    hal_is_dropper: bool,
) -> tuple[list[int], list[int]]:
    """No-op pruning. Keeps all bucket indices.

    Why: dominance based purely on the bucket-mean immediate payoff is
    misleading. The matrix averages over all (drop_second, check_second)
    pairs in the buckets, including pairs where the checker fails — which
    inflates rows where the dropper plays late and depresses rows where the
    dropper plays instant. At high checker cylinders, this incorrectly makes
    the safe-drop row look strictly dominant, even though instant drop is
    the correct strategy when the checker is one ST away from overflow.

    Item 27 (threshold-aware bucket splitting) will make the immediate matrix
    trustworthy enough to prune. Until then, the 7x7 LP is fast enough that
    unconditional pruning is the safest behavior.
    """
    n_hal = len(hal_buckets)
    n_baku = len(baku_buckets)
    return list(range(n_hal)), list(range(n_baku))


def adaptive_depth(base_depth: int, game: Game, max_depth: int = 3) -> int:
    remaining = max(0.0, LS_WINDOW_START - game.game_clock)
    if remaining < 300:
        return min(base_depth + 2, max_depth)
    if remaining < 600:
        return min(base_depth + 1, max_depth)
    return base_depth


def _hal_role(game: Game) -> tuple[bool, str, str]:
    """Return (hal_is_dropper, hal_role, baku_role) for the current half-round."""
    dropper, _ = game.get_roles_for_half(game.current_half)
    hal_is_dropper = dropper.name.lower() == "hal"
    if hal_is_dropper:
        return True, "dropper", "checker"
    return False, "checker", "dropper"


def _bucket_pair_for_state(
    game: Game,
    memory: MemoryMode,
    leap_deduced: bool,
) -> tuple[bool, tuple[Bucket, ...], tuple[Bucket, ...]]:
    """Get (hal_is_dropper, hal_buckets, baku_buckets) using actor-aware legality."""
    hal_is_dropper, hal_role, baku_role = _hal_role(game)
    turn_duration = game.get_turn_duration()
    hal_memory_impaired = memory == MemoryMode.AMNESIA

    hal_buckets = get_legal_buckets(
        "hal", hal_role, turn_duration,
        hal_leap_deduced=leap_deduced,
        hal_memory_impaired=hal_memory_impaired,
    )
    baku_buckets = get_legal_buckets("baku", baku_role, turn_duration)
    return hal_is_dropper, hal_buckets, baku_buckets


def _signed_immediate(immediate_matrix: np.ndarray, i: int, j: int, hal_is_dropper: bool) -> float:
    """Convert checker-perspective immediate payoff to Hal's perspective.

    build_payoff_matrix returns checker-perspective values. When Hal is the
    dropper, Hal's perspective is the negation. When Hal is the checker, Hal
    IS the checker, so the value passes through unchanged.
    """
    raw = float(immediate_matrix[i, j]) / CYLINDER_MAX
    return -raw if hal_is_dropper else raw


def _collect_and_evaluate_leaves(
    game: Game, depth: int, belief: BeliefState, memory: MemoryMode, leap_deduced: bool,
) -> float:
    if depth == 0 or game.game_over:
        return evaluate(game)

    nn_model = get_nn_model()
    if depth == 1 and nn_model is not None:
        return _search_depth1_batched(game, belief, memory, leap_deduced)

    return _search_recursive(game, depth, belief, memory, leap_deduced)


def _search_depth1_batched(
    game: Game, belief: BeliefState, memory: MemoryMode, leap_deduced: bool,
) -> float:
    from .value_net import extract_features, DEVICE

    nn_model = get_nn_model()
    if nn_model is None:
        return _search_recursive(game, 1, belief, memory, leap_deduced)

    _, chk_player = game.get_roles_for_half(game.current_half)
    hal_is_dropper, hal_buckets, baku_buckets = _bucket_pair_for_state(game, memory, leap_deduced)

    hal_keep, baku_keep = _prune_dominated(
        hal_buckets, baku_buckets, chk_player.cylinder, hal_is_dropper,
    )
    pruned_hal = tuple(hal_buckets[i] for i in hal_keep)
    pruned_baku = tuple(baku_buckets[j] for j in baku_keep)
    n_hal = len(pruned_hal)
    n_baku = len(pruned_baku)

    immediate_matrix = build_payoff_matrix(
        pruned_hal, pruned_baku, chk_player.cylinder, hal_is_dropper,
    )

    leaf_features: list[np.ndarray] = []
    leaf_slots: list[tuple] = []

    for i, h_bucket in enumerate(pruned_hal):
        for j, b_bucket in enumerate(pruned_baku):
            d_bucket = h_bucket if hal_is_dropper else b_bucket
            c_bucket = b_bucket if hal_is_dropper else h_bucket
            drop_time = (d_bucket.lo + d_bucket.hi) // 2
            check_time = (c_bucket.lo + c_bucket.hi) // 2

            _, chk = game.get_roles_for_half(game.current_half)
            success = check_time >= drop_time
            if success:
                st = max(1, check_time - drop_time)
                causes_death = chk.cylinder + st >= CYLINDER_MAX
            else:
                causes_death = True

            snap = GameSnapshot(game)

            if not causes_death:
                game.resolve_half_round(drop_time, check_time, survived_outcome=None)
                if game.game_over:
                    leaf_slots.append(("term_only", i, j, _hal_terminal_value(game)))
                else:
                    leaf_features.append(extract_features(game))
                    leaf_slots.append(("nn_only", i, j, len(leaf_features) - 1))
                snap.restore(game)
            else:
                sp = game.referee.compute_survival_probability(chk, death_duration=min(
                    chk.cylinder + (max(1, check_time - drop_time) if success else FAILED_CHECK_PENALTY),
                    CYLINDER_MAX,
                ))

                game.resolve_half_round(drop_time, check_time, survived_outcome=True)
                if game.game_over:
                    surv_ref: tuple = ("term", _hal_terminal_value(game))
                else:
                    leaf_features.append(extract_features(game))
                    surv_ref = ("nn", len(leaf_features) - 1)
                snap.restore(game)

                game.resolve_half_round(drop_time, check_time, survived_outcome=False)
                if game.game_over:
                    died_ref: tuple = ("term", _hal_terminal_value(game))
                else:
                    leaf_features.append(extract_features(game))
                    died_ref = ("nn", len(leaf_features) - 1)
                snap.restore(game)

                leaf_slots.append(("branch", i, j, sp, surv_ref, died_ref))

    if leaf_features:
        batch = torch.tensor(np.stack(leaf_features), device=DEVICE)
        with torch.no_grad():
            nn_values = nn_model(batch).squeeze(-1).cpu().numpy()
    else:
        nn_values = np.zeros(0, dtype=np.float64)

    def resolve(ref: tuple) -> float:
        kind, payload = ref
        if kind == "term":
            return float(payload)
        return float(nn_values[payload])

    payoff = np.zeros((n_hal, n_baku))
    for slot in leaf_slots:
        kind = slot[0]
        if kind == "term_only":
            _, i, j, val = slot
            cont_val = float(val)
        elif kind == "nn_only":
            _, i, j, idx = slot
            cont_val = float(nn_values[idx])
        else:
            _, i, j, sp, surv_ref, died_ref = slot
            cont_val = sp * resolve(surv_ref) + (1.0 - sp) * resolve(died_ref)

        imm = _signed_immediate(immediate_matrix, i, j, hal_is_dropper)
        payoff[i, j] = 0.3 * imm + 0.7 * cont_val

    _, value = _exploit_or_nash(belief, payoff, hal_is_dropper, baku_buckets, baku_keep)
    return value


def _search_recursive(
    game: Game, depth: int, belief: BeliefState, memory: MemoryMode, leap_deduced: bool,
) -> float:
    _, chk_player = game.get_roles_for_half(game.current_half)
    hal_is_dropper, hal_buckets, baku_buckets = _bucket_pair_for_state(game, memory, leap_deduced)

    hal_keep, baku_keep = _prune_dominated(
        hal_buckets, baku_buckets, chk_player.cylinder, hal_is_dropper,
    )
    pruned_hal = tuple(hal_buckets[i] for i in hal_keep)
    pruned_baku = tuple(baku_buckets[j] for j in baku_keep)
    n_hal = len(pruned_hal)
    n_baku = len(pruned_baku)

    immediate_matrix = build_payoff_matrix(
        pruned_hal, pruned_baku, chk_player.cylinder, hal_is_dropper,
    )

    payoff = np.zeros((n_hal, n_baku))
    for i, h_bucket in enumerate(pruned_hal):
        for j, b_bucket in enumerate(pruned_baku):
            d_bucket = h_bucket if hal_is_dropper else b_bucket
            c_bucket = b_bucket if hal_is_dropper else h_bucket

            cont_val = _simulate_and_recurse(
                game, d_bucket, c_bucket, depth, belief, memory, leap_deduced,
            )

            imm = _signed_immediate(immediate_matrix, i, j, hal_is_dropper)
            payoff[i, j] = 0.3 * imm + 0.7 * cont_val

    _, value = _exploit_or_nash(belief, payoff, hal_is_dropper, baku_buckets, baku_keep)
    return value


def _hal_terminal_value(game: Game) -> float:
    if game.winner is None:
        return 0.0
    return 1.0 if game.winner.name.lower() == "hal" else -1.0


def _simulate_and_recurse(
    game: Game, d_bucket: Bucket, c_bucket: Bucket,
    depth: int, belief: BeliefState, memory: MemoryMode, leap_deduced: bool,
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
        game.resolve_half_round(drop_time, check_time, survived_outcome=None)
        new_memory = update_memory(memory, game, death_occurred=False)
        cont_val = _collect_and_evaluate_leaves(game, depth - 1, belief, new_memory, leap_deduced)
        snap.restore(game)
        return cont_val

    sp = game.referee.compute_survival_probability(checker, death_duration=min(
        checker.cylinder + (max(1, check_time - drop_time) if success else FAILED_CHECK_PENALTY),
        CYLINDER_MAX,
    ))

    game.resolve_half_round(drop_time, check_time, survived_outcome=True)
    survived_memory = update_memory(memory, game, death_occurred=True)
    survived_leap = leap_deduced or True
    if game.game_over:
        survived_val = _hal_terminal_value(game)
    else:
        survived_val = _collect_and_evaluate_leaves(game, depth - 1, belief, survived_memory, survived_leap)
    snap.restore(game)

    game.resolve_half_round(drop_time, check_time, survived_outcome=False)
    died_memory = update_memory(memory, game, death_occurred=True)
    died_leap = leap_deduced or True
    if game.game_over:
        died_val = _hal_terminal_value(game)
    else:
        died_val = _collect_and_evaluate_leaves(game, depth - 1, belief, died_memory, died_leap)
    snap.restore(game)

    return sp * survived_val + (1.0 - sp) * died_val


def search(
    game: Game,
    depth: int,
    belief: BeliefState,
    memory: MemoryMode,
    leap_deduced: bool = False,
) -> tuple[np.ndarray | None, float]:
    if depth == 0 or game.game_over:
        return None, evaluate(game)

    _, chk_player = game.get_roles_for_half(game.current_half)
    hal_is_dropper, hal_buckets, baku_buckets = _bucket_pair_for_state(game, memory, leap_deduced)

    hal_keep, baku_keep = _prune_dominated(
        hal_buckets, baku_buckets, chk_player.cylinder, hal_is_dropper,
    )
    pruned_hal = tuple(hal_buckets[i] for i in hal_keep)
    pruned_baku = tuple(baku_buckets[j] for j in baku_keep)
    n_hal = len(pruned_hal)
    n_baku = len(pruned_baku)

    immediate_matrix = build_payoff_matrix(
        pruned_hal, pruned_baku, chk_player.cylinder, hal_is_dropper,
    )

    payoff = np.zeros((n_hal, n_baku))
    for i, h_bucket in enumerate(pruned_hal):
        for j, b_bucket in enumerate(pruned_baku):
            d_bucket = h_bucket if hal_is_dropper else b_bucket
            c_bucket = b_bucket if hal_is_dropper else h_bucket

            cont_val = _simulate_and_recurse(
                game, d_bucket, c_bucket, depth, belief, memory, leap_deduced,
            )

            imm = _signed_immediate(immediate_matrix, i, j, hal_is_dropper)
            payoff[i, j] = 0.3 * imm + 0.7 * cont_val

    pruned_strategy, value = _exploit_or_nash(belief, payoff, hal_is_dropper, baku_buckets, baku_keep)

    full_strategy = np.zeros(len(hal_buckets))
    for idx, kept_idx in enumerate(hal_keep):
        full_strategy[kept_idx] = pruned_strategy[idx]

    return full_strategy, value
