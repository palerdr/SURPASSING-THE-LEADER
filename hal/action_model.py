"""Legacy bucketed action abstraction for CanonicalHal.

This module owns the intentionally non-rigorous bucket/action model used by
the playable CanonicalHal prototype. Rigorous exact-second CFR lives under
``environment.cfr`` and should not import this module.
"""

from __future__ import annotations

from random import Random

import numpy as np

from environment.cfr.minimax import solve_minimax
from src.Constants import CYLINDER_MAX, FAILED_CHECK_PENALTY

from .state import BeliefState, Bucket


STANDARD_BUCKETS: tuple[Bucket, ...] = (
    Bucket(1, 1, "instant"),
    Bucket(2, 10, "early"),
    Bucket(11, 25, "mid_early"),
    Bucket(26, 40, "mid"),
    Bucket(41, 52, "mid_late"),
    Bucket(53, 58, "late"),
    Bucket(59, 60, "safe"),
)

LEAP_BUCKET = Bucket(61, 61, "leap")


def bucket_pair_payoff(drop_bucket: Bucket, check_bucket: Bucket, checker_cylinder: float) -> float:
    d_range = np.arange(drop_bucket.lo, drop_bucket.hi + 1)
    c_range = np.arange(check_bucket.lo, check_bucket.hi + 1)
    drops, checks = np.meshgrid(d_range, c_range, indexing="ij")
    success = checks >= drops
    st = np.maximum(1, checks - drops)
    overflow = (checker_cylinder + st) >= CYLINDER_MAX
    injection = min(checker_cylinder + FAILED_CHECK_PENALTY, CYLINDER_MAX)

    payoff = np.where(
        success,
        np.where(overflow, -CYLINDER_MAX, -st),
        -injection,
    )
    return float(payoff.mean())


def build_payoff_matrix(
    hal_buckets: tuple[Bucket, ...],
    baku_buckets: tuple[Bucket, ...],
    checker_cylinder: float,
    hal_is_dropper: bool,
) -> np.ndarray:
    matrix = np.empty((len(hal_buckets), len(baku_buckets)))
    for i, hal_bucket in enumerate(hal_buckets):
        for j, baku_bucket in enumerate(baku_buckets):
            drop_bucket = hal_bucket if hal_is_dropper else baku_bucket
            check_bucket = baku_bucket if hal_is_dropper else hal_bucket
            matrix[i, j] = bucket_pair_payoff(drop_bucket, check_bucket, checker_cylinder)
    return matrix


def get_buckets(turn_duration: int, knows_leap: bool) -> tuple[Bucket, ...]:
    if turn_duration == 61 and knows_leap:
        return STANDARD_BUCKETS + (LEAP_BUCKET,)
    return STANDARD_BUCKETS


def get_legal_buckets(
    actor: str,
    role: str,
    turn_duration: int,
    *,
    hal_leap_deduced: bool = False,
    hal_memory_impaired: bool = False,
) -> tuple[Bucket, ...]:
    from environment.legal_actions import legal_max_second

    max_sec = legal_max_second(
        actor,
        role,
        turn_duration,
        hal_leap_deduced=hal_leap_deduced,
        hal_memory_impaired=hal_memory_impaired,
    )
    if max_sec >= 61:
        return STANDARD_BUCKETS + (LEAP_BUCKET,)
    return STANDARD_BUCKETS


def resolve_bucket(bucket: Bucket, rng: Random) -> int:
    return rng.randint(bucket.lo, bucket.hi)


def best_response(belief: BeliefState, payoff: np.ndarray, hal_is_dropper: bool = True) -> tuple[np.ndarray, float]:
    probs = belief.baku_check_probs if hal_is_dropper else belief.baku_drop_probs
    baku_strategy = np.array(probs)
    ev_per_hal_action = payoff @ baku_strategy
    best_idx = int(np.argmax(ev_per_hal_action))
    strategy = np.zeros(payoff.shape[0])
    strategy[best_idx] = 1.0
    return strategy, float(ev_per_hal_action[best_idx])

