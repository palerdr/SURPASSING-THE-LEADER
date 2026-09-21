"""STL-owned quotient profiles and clocks for the public leap game."""
from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from stl.engine.game import Player, Referee
from stl.solver.canonical import turn_duration

N_ALIVE = 16_711
N = 17_011
WIN = N


def snap(clock: int) -> int:
    if clock < 3600:
        next_minute = (clock // 60 + 1) * 60
        return 3601 if next_minute == 3600 else next_minute
    return 3601 if clock == 3600 else 3601 + ((clock - 3601) // 60 + 1) * 60


def child_clock(half: int, clock: int, q: int = 0) -> int:
    if half not in (1, 2) or q < 0:
        raise ValueError('invalid half or dose')
    end = clock + turn_duration(clock) + (q + 120 if q else 0)
    return end + 60 if half == 1 else snap(end)


def eligible(s: int, t: int) -> bool:
    return s + 60 < 300 and s + t + 60 <= 300


@dataclass(frozen=True)
class Profiles:
    st: np.ndarray
    ttd: np.ndarray
    succ: np.ndarray
    fail: np.ndarray
    rev: np.ndarray
    s0: np.ndarray
    idx0: np.ndarray
    blocks: tuple


@lru_cache(None)
def profiles() -> Profiles:
    ids = {}; blocks = []; st = []; ttd = []
    for t in [0, *range(60, 301)]:
        start = len(st)
        for s in range(300):
            if eligible(s, t):
                ids[s, t] = len(st); st.append(s); ttd.append(t)
        if len(st) > start:
            blocks.append((start, len(st)))
    assert len(st) == N_ALIVE
    st.extend(range(300)); ttd.extend([301]*300)
    assert len(st) == N
    succ = np.empty((N, 60), np.int32)
    fail = np.full(N, WIN, np.int32)
    rev = np.zeros(N)
    referee = Referee()
    for pc, (s, t) in enumerate(zip(st, ttd)):
        for lag in range(1, 61):
            grown = s + lag
            succ[pc, lag-1] = WIN if grown >= 300 else ids.get((grown, t), N_ALIVE+grown)
        if pc < N_ALIVE:
            fail[pc] = ids.get((0, s+t+60), N_ALIVE)
            rev[pc] = referee.compute_survival_probability(Player('Checker', ttd=t), s+60)
    s0 = np.array([ids[0, t] for t in [0, *range(60, 241)]]+[N_ALIVE], np.int32)
    assert len(s0) == 183
    idx0 = np.full(N+1, -1, np.int32); idx0[s0] = np.arange(183); idx0[WIN] = 183
    return Profiles(np.array(st, np.int32), np.array(ttd, np.int32), succ, fail, rev,
                    s0, idx0, tuple(blocks))
