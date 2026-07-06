"""Hal state, belief, and memory primitives.

This module is the single home for Hal's non-search state. It deliberately
does not know about bucket payoffs, neural value estimates, or engine rules.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from enum import Enum, auto

from src.Game import Game, HalfRoundRecord


@dataclass(frozen=True)
class Bucket:
    lo: int
    hi: int
    label: str


class MemoryMode(Enum):
    NORMAL = auto()
    PRE_AMNESIA = auto()
    AMNESIA = auto()
    RECOVERED = auto()


@dataclass(frozen=True)
class BeliefState:
    baku_check_history: tuple[int, ...] = ()
    baku_drop_history: tuple[int, ...] = ()
    check_exploit: bool = False
    drop_exploit: bool = False
    baku_check_probs: tuple[float, ...] | None = None
    baku_drop_probs: tuple[float, ...] | None = None
    check_entropy: float = 2.0
    drop_entropy: float = 2.0


@dataclass(frozen=True)
class HalState:
    memory: MemoryMode = MemoryMode.NORMAL
    belief: BeliefState = field(default_factory=BeliefState)
    leap_deduced: bool = False


EXPLOITATION_THRESHOLD = 1.2


def _standard_buckets():
    from .action_model import STANDARD_BUCKETS

    return STANDARD_BUCKETS


def _bucket_index(second: int) -> int:
    buckets = _standard_buckets()
    for i, bucket in enumerate(buckets):
        if bucket.lo <= second <= bucket.hi:
            return i
    return len(buckets) - 1


def _entropy(counts: list[int]) -> float:
    total = sum(counts)
    if total == 0:
        return 2.0
    h = 0.0
    for count in counts:
        if count > 0:
            p = count / total
            h -= p * math.log(p)
    return h


def _empirical_bucket_probs(history: tuple[int, ...]) -> tuple[float, ...]:
    buckets = _standard_buckets()
    counts = [0] * len(buckets)
    for second in history:
        counts[_bucket_index(second)] += 1
    total = sum(counts)
    if total == 0:
        return tuple(1.0 / len(buckets) for _ in buckets)
    return tuple(count / total for count in counts)


def _role_exploit(history: tuple[int, ...]) -> tuple[bool, float, tuple[float, ...] | None]:
    buckets = _standard_buckets()
    counts = [0] * len(buckets)
    for second in history:
        counts[_bucket_index(second)] += 1
    entropy = _entropy(counts)
    exploit = entropy < EXPLOITATION_THRESHOLD and len(history) >= 4
    probs = _empirical_bucket_probs(history) if exploit else None
    return exploit, entropy, probs


def update_belief(belief: BeliefState, record: HalfRoundRecord) -> BeliefState:
    new_checks = belief.baku_check_history
    new_drops = belief.baku_drop_history

    if record.checker.lower() == "baku":
        new_checks = (new_checks + (record.check_time,))[-10:]
    if record.dropper.lower() == "baku":
        new_drops = (new_drops + (record.drop_time,))[-10:]

    check_exploit, check_entropy, check_probs = _role_exploit(new_checks)
    drop_exploit, drop_entropy, drop_probs = _role_exploit(new_drops)

    return replace(
        belief,
        baku_check_history=new_checks,
        baku_drop_history=new_drops,
        check_exploit=check_exploit,
        drop_exploit=drop_exploit,
        baku_check_probs=check_probs,
        baku_drop_probs=drop_probs,
        check_entropy=check_entropy,
        drop_entropy=drop_entropy,
    )


def hal_can_check_leap(memory: MemoryMode, leap_deduced: bool) -> bool:
    return leap_deduced and memory != MemoryMode.AMNESIA


def update_memory(memory: MemoryMode, game: Game, death_occurred: bool) -> MemoryMode:
    if memory == MemoryMode.NORMAL:
        if game.game_clock >= 3300 and not game.is_leap_second_turn():
            return MemoryMode.PRE_AMNESIA
        return MemoryMode.NORMAL

    if memory == MemoryMode.PRE_AMNESIA:
        if game.is_leap_second_turn():
            return MemoryMode.AMNESIA
        return MemoryMode.PRE_AMNESIA

    if memory == MemoryMode.AMNESIA:
        if death_occurred:
            return MemoryMode.RECOVERED
        return MemoryMode.AMNESIA

    return MemoryMode.RECOVERED

