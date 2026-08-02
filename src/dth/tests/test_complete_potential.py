"""The complete sweep potential strictly increases along every live transition.

``phi(profile) = ST + TTD`` when alive and ``ST + 301`` when dead; a class's
potential is the sum over both profiles.  Descending-potential dynamic
programming is valid iff every live edge strictly increases it, so that claim
is checked exhaustively here rather than assumed — this is the machine half of
the proof recorded in ``docs/EXACTNESS_PROOF.md``.
"""

import numpy as np

from dth.packed import (
    CLASS_COUNT,
    DEAD_POTENTIAL_OFFSET,
    MAX_CLASS_POTENTIAL,
    MAX_PROFILE_POTENTIAL,
    build_profile_table,
    class_potential,
    decode_class,
    encode_class,
)
from dth.solver import survive_injection


def test_every_profile_transition_strictly_increases_potential() -> None:
    # A class transition (p_c, p_d) -> (p_d, child(p_c)) increases the class
    # potential iff phi(child(p_c)) > phi(p_c), so the exhaustive check runs
    # at profile level.  Live success edges: every alive profile has ST <= 239
    # so all 60 lags stay under the cap (16,711 x 60 = 1,002,660); dead
    # sentinels contribute min(60, 299 - ST) each (240 x 60 + sum 0..59 =
    # 16,170).  Failure edges exist only for alive profiles (16,711).
    table = build_profile_table()
    phi = table.potential_by_profile.astype(np.int64)
    success = table.success_child_by_profile
    failure = table.failure_child_by_profile

    source = np.repeat(phi[:, None], 60, axis=1)
    live_success = success >= 0
    assert int(live_success.sum()) == 1_018_830
    assert int((phi[success[live_success]] <= source[live_success]).sum()) == 0

    live_failure = failure >= 0
    assert int(live_failure.sum()) == 16_711
    assert int((phi[failure[live_failure]] <= phi[live_failure]).sum()) == 0


def test_failure_edges_increase_potential_by_exactly_sixty_or_more() -> None:
    # A revived checker moves phi by (0 + ttd + st + 60) - (st + ttd) = 60
    # when the revived profile is alive, and by 301 - (st + ttd) >= 61 when it
    # is dead; both floors are load-bearing for the no-same-layer-edge claim.
    table = build_profile_table()
    phi = table.potential_by_profile.astype(np.int64)
    live = table.failure_child_by_profile >= 0
    deltas = phi[table.failure_child_by_profile[live]] - phi[live]
    assert int(deltas.min()) == 60


def test_layer_sizes_partition_the_class_space() -> None:
    table = build_profile_table()
    bucket_sizes = np.array([len(b) for b in table.bucket_profiles], dtype=np.int64)
    layer_sizes = np.convolve(bucket_sizes, bucket_sizes)
    assert len(layer_sizes) == MAX_CLASS_POTENTIAL + 1
    assert int(layer_sizes.sum()) == CLASS_COUNT
    assert int((layer_sizes > 0).sum()) == 1_201  # every state layer is real
    assert int(layer_sizes.max()) == 1_678_715
    assert int(layer_sizes.argmax()) == 374


def test_empty_profile_buckets_are_exactly_the_unachievable_band() -> None:
    # Alive profiles top out at phi = ST + TTD <= 240; dead sentinels start at
    # phi = 0 + 301.  The gap 241..300 is structurally empty.
    table = build_profile_table()
    bucket_sizes = np.array([len(b) for b in table.bucket_profiles])
    assert list(np.flatnonzero(bucket_sizes == 0)) == list(range(241, 301))
    assert len(bucket_sizes) == MAX_PROFILE_POTENTIAL + 1


def test_class_potential_matches_its_definition_on_samples() -> None:
    samples = [
        (0, 0, 0, 0),
        (240, 0, 240, 0),
        (239, 0, 0, 240),
        (299, 300, 299, 300),
        (180, 60, 250, 0),
    ]
    for state in samples:
        checker_st, checker_ttd, dropper_st, dropper_ttd = state
        expected = sum(
            st + (ttd if survive_injection(st, ttd) else DEAD_POTENTIAL_OFFSET)
            for st, ttd in ((checker_st, checker_ttd), (dropper_st, dropper_ttd))
        )
        index = encode_class(state)
        assert class_potential(index) == expected
        assert class_potential(index) <= MAX_CLASS_POTENTIAL
        assert decode_class(index) == state or not all(
            survive_injection(st, ttd)
            for st, ttd in ((checker_st, checker_ttd), (dropper_st, dropper_ttd))
        )
    # The unique top of the order: both players dead at the ST ceiling.
    assert class_potential(encode_class((299, 300, 299, 300))) == MAX_CLASS_POTENTIAL
