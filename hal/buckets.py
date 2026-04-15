from random import Random

import numpy as np

from src.Constants import FAILED_CHECK_PENALTY, CYLINDER_MAX
from .types import Bucket

STANDARD_BUCKETS: tuple[Bucket, ...] = (
    Bucket(1,   1,  "instant"),
    Bucket(2,  10,  "early"),
    Bucket(11, 25,  "mid_early"),
    Bucket(26, 40,  "mid"),
    Bucket(41, 52,  "mid_late"),
    Bucket(53, 58,  "late"),
    Bucket(59, 60,  "safe"),
)

LEAP_BUCKET = Bucket(61, 61, "leap")


def bucket_pair_payoff(D: Bucket, C: Bucket, checker_cylinder: float) -> float:
    d_range = np.arange(D.lo, D.hi + 1)
    c_range = np.arange(C.lo, C.hi + 1)
    drops, checks = np.meshgrid(d_range, c_range, indexing='ij')
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
    n_hal = len(hal_buckets)
    n_baku = len(baku_buckets)
    matrix = np.empty((n_hal, n_baku))
    for i, hb in enumerate(hal_buckets):
        for j, bb in enumerate(baku_buckets):
            d_bucket = hb if hal_is_dropper else bb
            c_bucket = bb if hal_is_dropper else hb
            matrix[i, j] = bucket_pair_payoff(d_bucket, c_bucket, checker_cylinder)
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
        actor, role, turn_duration,
        hal_leap_deduced=hal_leap_deduced,
        hal_memory_impaired=hal_memory_impaired,
    )
    base = STANDARD_BUCKETS
    if max_sec >= 61:
        return base + (LEAP_BUCKET,)
    return base


def resolve_bucket(bucket: Bucket, rng: Random) -> int:
    return rng.randint(bucket.lo, bucket.hi)
