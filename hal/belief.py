from __future__ import annotations

import math
from dataclasses import replace

from src.Game import HalfRoundRecord
from .types import BeliefState
from .buckets import STANDARD_BUCKETS


EXPLOITATION_THRESHOLD = 1.2


def _bucket_index(second: int) -> int:
    for i, b in enumerate(STANDARD_BUCKETS):
        if b.lo <= second <= b.hi:
            return i
    return len(STANDARD_BUCKETS) - 1


def _entropy(counts: list[int]) -> float:
    total = sum(counts)
    if total == 0:
        return 2.0
    h = 0.0
    for c in counts:
        if c > 0:
            p = c / total
            h -= p * math.log(p)
    return h


def _empirical_bucket_probs(history: tuple[int, ...]) -> tuple[float, ...]:
    n = len(STANDARD_BUCKETS)
    counts = [0] * n
    for s in history:
        counts[_bucket_index(s)] += 1
    total = sum(counts)
    if total == 0:
        return tuple(1.0 / n for _ in range(n))
    return tuple(c / total for c in counts)


def _role_exploit(history: tuple[int, ...]) -> tuple[bool, float, tuple[float, ...] | None]:
    counts = [0] * len(STANDARD_BUCKETS)
    for s in history:
        counts[_bucket_index(s)] += 1
    ent = _entropy(counts)
    exploit = ent < EXPLOITATION_THRESHOLD and len(history) >= 4
    probs = _empirical_bucket_probs(history) if exploit else None
    return exploit, ent, probs


def update_belief(belief: BeliefState, record: HalfRoundRecord) -> BeliefState:
    new_checks = belief.baku_check_history
    new_drops = belief.baku_drop_history

    if record.checker.lower() == "baku":
        new_checks = (new_checks + (record.check_time,))[-10:]
    if record.dropper.lower() == "baku":
        new_drops = (new_drops + (record.drop_time,))[-10:]

    chk_exploit, chk_ent, chk_probs = _role_exploit(new_checks)
    drp_exploit, drp_ent, drp_probs = _role_exploit(new_drops)

    return replace(
        belief,
        baku_check_history=new_checks,
        baku_drop_history=new_drops,
        check_exploit=chk_exploit,
        drop_exploit=drp_exploit,
        baku_check_probs=chk_probs,
        baku_drop_probs=drp_probs,
        check_entropy=chk_ent,
        drop_entropy=drp_ent,
    )
