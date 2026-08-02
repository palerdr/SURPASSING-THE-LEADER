"""The packed quotient codec is a bijection onto the addressable class domain.

These tests lock the codec against the rule authority in ``dth.solver``: the
profile enumeration realizes exactly the quotient that
``tests/test_dead_ttd_quotient.py`` proves sound, and the child tables agree
with ``transition`` through ``complete_game_dependencies``.
"""

import numpy as np
import pytest

from dth.packed import (
    ALIVE_PROFILE_COUNT,
    ALIVE_TTD_DOMAIN,
    CLASS_COUNT,
    DEAD_PROFILE_BASE,
    DEAD_TTD_REPRESENTATIVE,
    PROFILE_COUNT,
    build_profile_table,
    class_potential,
    decode_class,
    encode_class,
    layer_rectangles,
    packed_class_children,
    profile_id,
    profile_representative,
)
from dth.solver import (
    complete_game_dependencies,
    revival_model,
    survive_injection,
)


def test_profile_and_class_counts_match_the_locked_quotient() -> None:
    # The same 17,011 / 289,374,121 that test_dead_ttd_quotient derives from
    # first principles over all 72,600 per-player profiles.
    assert ALIVE_PROFILE_COUNT == 16_711
    assert PROFILE_COUNT == 17_011
    assert CLASS_COUNT == PROFILE_COUNT**2 == 289_374_121
    table = build_profile_table()
    assert int((table.ttd_by_profile >= 0).sum()) == ALIVE_PROFILE_COUNT
    assert int((table.alive_id_by_st_ttd >= 0).sum()) == ALIVE_PROFILE_COUNT


def test_profile_round_trip_is_exhaustive() -> None:
    for packed in range(PROFILE_COUNT):
        st, ttd = profile_representative(packed)
        assert profile_id(st, ttd) == packed
        if packed < DEAD_PROFILE_BASE:
            assert survive_injection(st, ttd)
            assert ttd in ALIVE_TTD_DOMAIN
        else:
            assert ttd == DEAD_TTD_REPRESENTATIVE
            assert not survive_injection(st, ttd)


def test_profile_id_matches_the_quotient_definition() -> None:
    # Alive domain profiles map to distinct ids below the dead base; every
    # profile that fails survive_injection collapses to its per-ST sentinel.
    seen_alive = set()
    for ttd in ALIVE_TTD_DOMAIN:
        for st in range(300):
            packed = profile_id(st, ttd)
            if survive_injection(st, ttd):
                assert packed < DEAD_PROFILE_BASE
                seen_alive.add(packed)
            else:
                assert packed == DEAD_PROFILE_BASE + st
    assert seen_alive == set(range(ALIVE_PROFILE_COUNT))


def test_alive_off_domain_ttd_fails_closed() -> None:
    # (0, 30) is a valid live profile, but no state this artifact addresses
    # can reach an alive TTD in 1..59; the codec must refuse, not approximate.
    with pytest.raises(ValueError, match="off-domain"):
        profile_id(0, 30)
    with pytest.raises(ValueError, match="off-domain"):
        encode_class((0, 30, 0, 0))
    # A dead profile's TTD is discarded by the quotient, so any TTD is exact.
    assert profile_id(299, 30) == DEAD_PROFILE_BASE + 299
    assert encode_class((299, 30, 0, 0)) == (DEAD_PROFILE_BASE + 299) * PROFILE_COUNT


def test_class_round_trip_and_dead_sentinel_representative() -> None:
    samples = [
        0,
        CLASS_COUNT - 1,
        encode_class((0, 0, 0, 0)),
        encode_class((240, 0, 240, 0)),  # independent dead-band anchor
        encode_class((239, 0, 0, 240)),  # the h4/h5 anchor state
        encode_class((180, 60, 299, 300)),
    ]
    for index in samples:
        state = decode_class(index)
        assert encode_class(state) == index
        checker, dropper = divmod(index, PROFILE_COUNT)
        if checker >= DEAD_PROFILE_BASE:
            assert state[1] == DEAD_TTD_REPRESENTATIVE
        if dropper >= DEAD_PROFILE_BASE:
            assert state[3] == DEAD_TTD_REPRESENTATIVE


def test_rule_tables_match_the_solver_authority() -> None:
    table = build_profile_table()
    rng = np.random.default_rng(20260730)
    for packed in rng.choice(PROFILE_COUNT, size=400, replace=False):
        packed = int(packed)
        st, ttd = profile_representative(packed)
        alive = packed < DEAD_PROFILE_BASE
        # Revival is evaluated here once, bit-for-bit, and nowhere downstream.
        expected_revival = revival_model(st, ttd) if alive else 0.0
        assert float(table.revival_by_profile[packed]) == expected_revival
        for lag in range(1, 61):
            child = int(table.success_child_by_profile[packed, lag - 1])
            if st + lag >= 300:
                assert child == -1  # cylinder overflow, terminal W
            else:
                assert child == profile_id(st + lag, ttd)
        failure = int(table.failure_child_by_profile[packed])
        if alive:
            assert failure == profile_id(0, ttd + st + 60)
        else:
            assert failure == -1


def test_children_match_complete_game_dependencies() -> None:
    # The packed child enumeration must factor the raw dependency set through
    # the quotient map exactly, for alive-alive, alive-dead, and dead-dead
    # parents alike.
    states = [
        (0, 0, 0, 0),
        (240, 0, 240, 0),
        (239, 0, 0, 240),
        (0, 240, 239, 0),
        (100, 100, 250, 300),
        (299, 300, 299, 300),
        (180, 60, 20, 200),
        (250, 300, 0, 0),
    ]
    for state in states:
        index = encode_class(state)
        packed_children = set(packed_class_children(index))
        raw_children = {
            encode_class(child)
            for child in complete_game_dependencies(decode_class(index))
        }
        assert packed_children == raw_children


def test_children_strictly_increase_potential_in_sorted_order() -> None:
    index = encode_class((0, 0, 0, 0))
    children = packed_class_children(index)
    assert children
    parent = class_potential(index)
    potentials = [class_potential(child) for child in children]
    assert min(potentials) > parent
    assert potentials == sorted(potentials)


def test_layer_rectangles_partition_the_class_space() -> None:
    total = 0
    for potential in range(1201):
        rectangles = layer_rectangles(potential)
        checker_potentials = []
        for checker_bucket, dropper_bucket in rectangles:
            table = build_profile_table()
            checker_potentials.append(
                int(table.potential_by_profile[checker_bucket[0]])
            )
            total += len(checker_bucket) * len(dropper_bucket)
        # Ascending checker-potential order is normative for the sweep.
        assert checker_potentials == sorted(checker_potentials)
    assert total == CLASS_COUNT
    assert len(layer_rectangles(1200)) == 1
    top_checker, top_dropper = layer_rectangles(1200)[0]
    assert len(top_checker) == len(top_dropper) == 1
