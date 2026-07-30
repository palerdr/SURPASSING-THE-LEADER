"""Packed class codec for the per-player TTD-dead quotient of complete DTH.

The complete game reads a player's TTD in exactly one place: the revival
probability of a failed check.  A profile that fails ``survive_injection`` has
revival probability identically zero, and no transition can make it survivable
again, so its TTD is never read and collapses to a single dead sentinel per ST
value.  ``tests/test_dead_ttd_quotient.py`` locks that argument; the class
space it induces is what this module makes addressable:

- a *profile* is one player's ``(ST, TTD)`` pair, quotiented to an id in
  ``[0, 17_011)``: 16,711 alive profiles ordered by (TTD ascending over
  ``{0} | [60, 240]``, ST ascending), then 300 dead sentinels ordered by ST;
- a *class* is ``checker_profile * 17_011 + dropper_profile``, one of
  289,374,121 indices, each the address of one float64 in the backup
  tablebase's value array.

The addressable domain is the transition-closed set of states whose alive
TTDs lie in ``{0} | [60, 300]``.  An *alive* profile with TTD in 1..59 is a
valid live state but unreachable from any root this artifact serves, and the
codec fails closed on it.  A *dead* profile is accepted with any TTD, because
the quotient discards it exactly.

The topological schedule uses the potential ``phi(profile) = ST + rho`` with
``rho = TTD`` when alive and ``rho = 301`` when dead, summed over both
profiles.  Every live transition strictly increases the class potential (see
``docs/EXACTNESS_PROOF.md``), so a sweep in decreasing potential order sees
every child before its parents.  ``packed_class_children`` re-checks that
invariant on every call and refuses to return a non-increasing edge.

Cross-backend behavior of everything built on these tables is governed by
``docs/DTH_BACKUP_PARITY.md``.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cache

import numpy as np

from dth.solver import NTState, revival_model, survive_injection, validate_live_state

__all__ = [
    "PACKED_CLASS_ENCODING",
    "PROFILE_COUNT",
    "ALIVE_PROFILE_COUNT",
    "DEAD_PROFILE_BASE",
    "CLASS_COUNT",
    "DEAD_TTD_REPRESENTATIVE",
    "DEAD_POTENTIAL_OFFSET",
    "MAX_PROFILE_POTENTIAL",
    "MAX_CLASS_POTENTIAL",
    "ALIVE_TTD_DOMAIN",
    "QuotientProfileTable",
    "build_profile_table",
    "profile_id",
    "profile_representative",
    "encode_class",
    "decode_class",
    "class_potential",
    "packed_class_children",
    "layer_rectangles",
]

PACKED_CLASS_ENCODING = "dth-packed-class-v1"
ALIVE_PROFILE_COUNT = 16_711
DEAD_PROFILE_BASE = 16_711
PROFILE_COUNT = 17_011
CLASS_COUNT = PROFILE_COUNT * PROFILE_COUNT
DEAD_TTD_REPRESENTATIVE = 300
DEAD_POTENTIAL_OFFSET = 301
MAX_PROFILE_POTENTIAL = 600
MAX_CLASS_POTENTIAL = 1_200
ALIVE_TTD_DOMAIN = (0, *range(60, 301))


@dataclass(frozen=True)
class QuotientProfileTable:
    """Precomputed per-profile rule tables, the bit-level solver authority.

    Every transcendental quantity the backup sweep needs is evaluated here,
    once, in Python: kernels downstream (numpy or Rust) only gather from these
    arrays, which is what makes their arithmetic reproducible bit-for-bit.
    """

    alive_id_by_st_ttd: np.ndarray  # (300, 301) int32; -1 off the alive domain
    st_by_profile: np.ndarray  # (17011,) int16
    ttd_by_profile: np.ndarray  # (17011,) int16; -1 marks the dead sentinel
    potential_by_profile: np.ndarray  # (17011,) int16; ST + TTD or ST + 301
    revival_by_profile: np.ndarray  # (17011,) float64; 0.0 for dead
    success_child_by_profile: np.ndarray  # (17011, 60) int32; -1 = overflow W
    failure_child_by_profile: np.ndarray  # (17011,) int32; -1 = dead checker W
    bucket_profiles: tuple[np.ndarray, ...]  # 601 uint32 arrays by potential


@cache
def build_profile_table() -> QuotientProfileTable:
    """Enumerate the quotient in its normative order and derive rule tables.

    The enumeration order is part of ``PACKED_CLASS_ENCODING`` and must never
    change without a version bump: TTD ascending over the alive domain with ST
    ascending inside each TTD, then the 300 dead sentinels by ST.
    """

    alive_id = np.full((300, 301), -1, dtype=np.int32)
    st_by = np.zeros(PROFILE_COUNT, dtype=np.int16)
    ttd_by = np.full(PROFILE_COUNT, -1, dtype=np.int16)
    next_id = 0
    for ttd in ALIVE_TTD_DOMAIN:
        for st in range(300):
            if survive_injection(st, ttd):
                alive_id[st, ttd] = next_id
                st_by[next_id] = st
                ttd_by[next_id] = ttd
                next_id += 1
    if next_id != ALIVE_PROFILE_COUNT:
        raise RuntimeError(f"expected {ALIVE_PROFILE_COUNT} alive profiles, got {next_id}")
    for st in range(300):
        st_by[DEAD_PROFILE_BASE + st] = st

    potential = np.where(
        ttd_by >= 0,
        st_by.astype(np.int32) + ttd_by.astype(np.int32),
        st_by.astype(np.int32) + DEAD_POTENTIAL_OFFSET,
    ).astype(np.int16)

    success = np.full((PROFILE_COUNT, 60), -1, dtype=np.int32)
    failure = np.full(PROFILE_COUNT, -1, dtype=np.int32)
    revival = np.zeros(PROFILE_COUNT, dtype=np.float64)
    for pid in range(PROFILE_COUNT):
        st = int(st_by[pid])
        ttd = int(ttd_by[pid])
        alive = ttd >= 0
        for lag in range(1, 61):
            grown = st + lag
            if grown >= 300:
                continue  # cylinder overflow: terminal W for the mover
            if alive:
                child = int(alive_id[grown, ttd])
                success[pid, lag - 1] = child if child >= 0 else DEAD_PROFILE_BASE + grown
            else:
                success[pid, lag - 1] = DEAD_PROFILE_BASE + grown
        if alive:
            revival[pid] = revival_model(st, ttd)
            revived_ttd = ttd + st + 60
            child = int(alive_id[0, revived_ttd])
            failure[pid] = child if child >= 0 else DEAD_PROFILE_BASE

    buckets = tuple(
        np.flatnonzero(potential == value).astype(np.uint32)
        for value in range(MAX_PROFILE_POTENTIAL + 1)
    )
    for array in (alive_id, st_by, ttd_by, potential, success, failure, revival):
        array.setflags(write=False)
    for bucket in buckets:
        bucket.setflags(write=False)
    return QuotientProfileTable(
        alive_id_by_st_ttd=alive_id,
        st_by_profile=st_by,
        ttd_by_profile=ttd_by,
        potential_by_profile=potential,
        revival_by_profile=revival,
        success_child_by_profile=success,
        failure_child_by_profile=failure,
        bucket_profiles=buckets,
    )


def _validate_profile(st: int, ttd: int) -> tuple[int, int]:
    if any(
        isinstance(value, bool) or not isinstance(value, (int, np.integer))
        for value in (st, ttd)
    ):
        raise ValueError(f"profile coordinates must be literal integers, got {(st, ttd)!r}")
    st, ttd = int(st), int(ttd)
    if not 0 <= st < 300:
        raise ValueError(f"profile ST must be in 0..299, got {st}")
    if not 0 <= ttd <= 300:
        raise ValueError(f"profile TTD must be in 0..300, got {ttd}")
    return st, ttd


def profile_id(st: int, ttd: int) -> int:
    """Quotient a single player's ``(ST, TTD)`` profile to its packed id.

    Fails closed on an alive profile whose TTD is in 1..59: such a state is
    live but off this artifact's transition-closed domain.  A dead profile is
    accepted with any TTD, because the quotient discards it exactly.
    """

    st, ttd = _validate_profile(st, ttd)
    if not survive_injection(st, ttd):
        return DEAD_PROFILE_BASE + st
    packed = int(build_profile_table().alive_id_by_st_ttd[st, ttd])
    if packed < 0:
        raise ValueError(
            f"alive profile ({st}, {ttd}) has an off-domain TTD in 1..59; "
            "the backup tablebase does not address it"
        )
    return packed


def profile_representative(packed: int) -> tuple[int, int]:
    """Return the canonical ``(ST, TTD)`` representative of a profile id."""

    if isinstance(packed, bool) or not isinstance(packed, (int, np.integer)):
        raise ValueError("profile id must be an integer")
    packed = int(packed)
    if not 0 <= packed < PROFILE_COUNT:
        raise ValueError(f"profile id must be in 0..{PROFILE_COUNT - 1}, got {packed}")
    table = build_profile_table()
    ttd = int(table.ttd_by_profile[packed])
    return int(table.st_by_profile[packed]), (
        ttd if ttd >= 0 else DEAD_TTD_REPRESENTATIVE
    )


def encode_class(state: NTState) -> int:
    """Pack one role-canonical live state into its quotient class index."""

    checker_st, checker_ttd, dropper_st, dropper_ttd = validate_live_state(state)
    return profile_id(checker_st, checker_ttd) * PROFILE_COUNT + profile_id(
        dropper_st, dropper_ttd
    )


def decode_class(index: int) -> NTState:
    """Return the canonical representative state of a class index."""

    if isinstance(index, bool) or not isinstance(index, (int, np.integer)):
        raise ValueError("class index must be an integer")
    index = int(index)
    if not 0 <= index < CLASS_COUNT:
        raise ValueError(f"class index must be in 0..{CLASS_COUNT - 1}, got {index}")
    checker, dropper = divmod(index, PROFILE_COUNT)
    checker_st, checker_ttd = profile_representative(checker)
    dropper_st, dropper_ttd = profile_representative(dropper)
    return (checker_st, checker_ttd, dropper_st, dropper_ttd)


def class_potential(index: int) -> int:
    """Return the topological potential of a class index."""

    if isinstance(index, bool) or not isinstance(index, (int, np.integer)):
        raise ValueError("class index must be an integer")
    index = int(index)
    if not 0 <= index < CLASS_COUNT:
        raise ValueError(f"class index must be in 0..{CLASS_COUNT - 1}, got {index}")
    table = build_profile_table()
    checker, dropper = divmod(index, PROFILE_COUNT)
    return int(table.potential_by_profile[checker]) + int(
        table.potential_by_profile[dropper]
    )


def packed_class_children(index: int) -> tuple[int, ...]:
    """Enumerate the distinct live child classes of one class index.

    Every returned child strictly increases the class potential; a violation
    raises rather than returning an unsortable edge, mirroring the raw-state
    guard in ``solver.complete_game_dependencies``.
    """

    table = build_profile_table()
    parent_potential = class_potential(index)
    checker, dropper = divmod(int(index), PROFILE_COUNT)
    children: set[int] = set()
    for child_profile in table.success_child_by_profile[checker]:
        if child_profile >= 0:
            children.add(dropper * PROFILE_COUNT + int(child_profile))
    failure_profile = int(table.failure_child_by_profile[checker])
    if failure_profile >= 0:
        children.add(dropper * PROFILE_COUNT + failure_profile)
    for child in children:
        if class_potential(child) <= parent_potential:
            raise RuntimeError(
                f"class transition does not increase potential: "
                f"{index} (phi={parent_potential}) -> {child}"
            )
    return tuple(sorted(children, key=lambda child: (class_potential(child), child)))


def layer_rectangles(potential: int) -> tuple[tuple[np.ndarray, np.ndarray], ...]:
    """Return the (checker-bucket, dropper-bucket) rectangles of one layer.

    The classes of state potential ``P`` are exactly the union of the
    rectangles ``bucket(a) x bucket(P - a)``; the ascending-``a`` order here is
    normative for the sweep's deterministic work partition.
    """

    if isinstance(potential, bool) or not isinstance(potential, (int, np.integer)):
        raise ValueError("potential must be an integer")
    potential = int(potential)
    if not 0 <= potential <= MAX_CLASS_POTENTIAL:
        raise ValueError(
            f"potential must be in 0..{MAX_CLASS_POTENTIAL}, got {potential}"
        )
    table = build_profile_table()
    rectangles = []
    lowest = max(0, potential - MAX_PROFILE_POTENTIAL)
    highest = min(MAX_PROFILE_POTENTIAL, potential)
    for checker_potential in range(lowest, highest + 1):
        checker_bucket = table.bucket_profiles[checker_potential]
        dropper_bucket = table.bucket_profiles[potential - checker_potential]
        if len(checker_bucket) and len(dropper_bucket):
            rectangles.append((checker_bucket, dropper_bucket))
    return tuple(rectangles)
