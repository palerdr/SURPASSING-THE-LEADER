from __future__ import annotations

from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from arena.contracts import (
    CanonicalDecision,
    PublicDecisionState,
    PublicGameOutcome,
    PublicHalfRound,
    PublicPlayerState,
)
from arena.policies.opponent_league import (
    ACTIONS,
    AUDIT_FAMILY_MANIFEST,
    BAIT_THEN_REVERSE,
    COPY_RECENT,
    COUNTER_RECENT,
    DETERMINISTIC,
    EARLY,
    FAMILY_MANIFESTS,
    LATE,
    MULTIMODAL,
    NARROW,
    PERIODIC,
    RETREAT_AFTER_DETECTED_EXPLOITATION,
    STATE_THRESHOLD,
    SUPPORTED_FAMILIES,
    SWITCH,
    TEST_FAMILY_MANIFEST,
    TRAIN_FAMILY_MANIFEST,
    VALIDATION_FAMILY_MANIFEST,
    WIN_STAY_LOSE_SHIFT,
    iter_manifest_opponents,
    make_opponent,
)


class _ExplodingNativeState:
    def __getattribute__(self, name: str):
        raise AssertionError(
            f"opponent inspected private native state attribute {name!r}"
        )


def _decision(
    *,
    role: str = "dropper",
    actor_name: str = "Opponent",
    checker_cylinder: float = 0.0,
    checker_ttd: float = 0.0,
    dropper_cylinder: float = 0.0,
    dropper_ttd: float = 0.0,
    native_state: object | None = None,
    legal_seconds: tuple[int, ...] = ACTIONS,
    turn_duration: int = 60,
) -> CanonicalDecision:
    return CanonicalDecision(
        role=role,
        actor_name=actor_name,
        turn_duration=turn_duration,
        legal_seconds=legal_seconds,
        checker_cylinder_seconds=checker_cylinder,
        checker_ttd_seconds=checker_ttd,
        dropper_cylinder_seconds=dropper_cylinder,
        dropper_ttd_seconds=dropper_ttd,
        native_state=_ExplodingNativeState() if native_state is None else native_state,
    )


def _record(
    *,
    provider_role: str,
    own_action: int,
    opponent_action: int,
    provider_name: str = "Opponent",
    opponent_name: str = "Hal",
    game_index: int = 0,
    half_round_index: int = 0,
    game_over: bool = False,
) -> PublicHalfRound:
    if provider_role == "dropper":
        dropper_name, checker_name = provider_name, opponent_name
        drop_time, check_time = own_action, opponent_action
    else:
        dropper_name, checker_name = opponent_name, provider_name
        drop_time, check_time = opponent_action, own_action
    return PublicHalfRound(
        game_index=game_index,
        half_round_index=half_round_index,
        pre_decision_state=PublicDecisionState(
            game_clock_seconds=720.0,
            round_index=half_round_index // 2,
            half_index=half_round_index % 2,
            turn_duration=60,
            players=(
                PublicPlayerState(opponent_name, 0.0, 0.0),
                PublicPlayerState(provider_name, 0.0, 0.0),
            ),
        ),
        dropper_name=dropper_name,
        checker_name=checker_name,
        drop_time=drop_time,
        check_time=check_time,
        outcome="check_success" if check_time >= drop_time else "check_fail_survived",
        game_over=game_over,
        winner_name=None,
    )


def _argmax_action(distribution: np.ndarray) -> int:
    return int(np.argmax(distribution)) + 1


def _play_local_result(provider, *, won: bool) -> None:
    current = _argmax_action(provider.true_distribution(_decision()))
    if won and current > 1:
        role, opponent_action = "dropper", current - 1
    elif won:
        role, opponent_action = "checker", current
    elif current < 60:
        role, opponent_action = "checker", current + 1
    else:
        role, opponent_action = "dropper", current
    decision = _decision(role=role)
    provider.policy(decision)
    provider.observe(
        _record(
            provider_role=role,
            own_action=current,
            opponent_action=opponent_action,
            half_round_index=provider.observation_count,
        )
    )


def test_manifests_are_immutable_disjoint_and_cover_every_family() -> None:
    manifests = (
        TRAIN_FAMILY_MANIFEST,
        VALIDATION_FAMILY_MANIFEST,
        TEST_FAMILY_MANIFEST,
    )
    family_sets = [set(manifest.families) for manifest in manifests]
    assert family_sets[0].isdisjoint(family_sets[1])
    assert family_sets[0].isdisjoint(family_sets[2])
    assert family_sets[1].isdisjoint(family_sets[2])
    assert set.union(*family_sets) == set(SUPPORTED_FAMILIES)
    assert FAMILY_MANIFESTS["train"] is TRAIN_FAMILY_MANIFEST
    assert FAMILY_MANIFESTS["audit"] is AUDIT_FAMILY_MANIFEST
    assert set(AUDIT_FAMILY_MANIFEST.families) == {
        RETREAT_AFTER_DETECTED_EXPLOITATION,
        BAIT_THEN_REVERSE,
    }
    with pytest.raises(TypeError):
        FAMILY_MANIFESTS["new"] = TRAIN_FAMILY_MANIFEST  # type: ignore[index]
    with pytest.raises(FrozenInstanceError):
        TRAIN_FAMILY_MANIFEST.split = "test"  # type: ignore[misc]

    expected_variants = sum(len(entry.seeds) for entry in TRAIN_FAMILY_MANIFEST.entries)
    assert (
        len(tuple(iter_manifest_opponents(TRAIN_FAMILY_MANIFEST))) == expected_variants
    )


@pytest.mark.parametrize("family", SUPPORTED_FAMILIES)
def test_every_family_is_seed_reproducible_and_returns_a_distribution(
    family: str,
) -> None:
    left = make_opponent(family, seed=12345)
    right = make_opponent(family, seed=12345)
    decision = _decision(role="dropper")

    first = left.true_distribution(decision)
    np.testing.assert_array_equal(first, left.true_distribution(decision))
    np.testing.assert_array_equal(first, right.true_distribution(decision))
    assert first.shape == (60,)
    assert np.all(first >= 0.0)
    assert float(first.sum()) == pytest.approx(1.0)

    left_policy = left.policy(decision)
    right_policy = right.policy(decision)
    assert left_policy == right_policy
    assert set(left_policy).issubset(ACTIONS)
    action = _argmax_action(first)
    reveal = _record(
        provider_role="dropper",
        own_action=action,
        opponent_action=max(1, action - 1),
    )
    left.observe(reveal)
    right.observe(reveal)
    np.testing.assert_array_equal(
        left.true_distribution(decision),
        right.true_distribution(decision),
    )


def test_stationary_shape_families_have_distinct_expected_geometry() -> None:
    decision = _decision()
    early = make_opponent(EARLY, seed=4).true_distribution(decision)
    late = make_opponent(LATE, seed=4).true_distribution(decision)
    narrow = make_opponent(NARROW, seed=4).true_distribution(decision)
    multimodal = make_opponent(MULTIMODAL, seed=4).true_distribution(decision)
    deterministic = make_opponent(DETERMINISTIC, seed=4).true_distribution(decision)
    positions = np.arange(1, 61, dtype=np.float64)

    assert float(early @ positions) < 20.0
    assert float(late @ positions) > 42.0
    assert float(narrow.max()) > 0.10
    local_peaks = np.flatnonzero(
        (multimodal[1:-1] > multimodal[:-2]) & (multimodal[1:-1] > multimodal[2:])
    )
    assert len(local_peaks) >= 3
    assert np.count_nonzero(deterministic) == 1


def test_state_threshold_uses_public_coordinates_and_never_native_state() -> None:
    provider = make_opponent(STATE_THRESHOLD, seed=9)
    low = _decision(
        role="checker",
        checker_cylinder=0.0,
        checker_ttd=0.0,
        native_state=_ExplodingNativeState(),
    )
    high = _decision(
        role="checker",
        checker_cylinder=299.0,
        checker_ttd=299.0,
        native_state=_ExplodingNativeState(),
    )
    low_distribution = provider.true_distribution(low)
    high_distribution = provider.true_distribution(high)
    assert not np.array_equal(low_distribution, high_distribution)
    assert _argmax_action(low_distribution) < _argmax_action(high_distribution)

    equivalent = _decision(
        role="checker",
        checker_cylinder=0.0,
        checker_ttd=0.0,
        native_state=object(),
    )
    np.testing.assert_array_equal(
        low_distribution,
        provider.true_distribution(equivalent),
    )


def test_periodic_family_advances_only_when_policy_is_committed() -> None:
    provider = make_opponent(PERIODIC, seed=12)
    decision = _decision()
    initial = provider.true_distribution(decision)
    np.testing.assert_array_equal(initial, provider.true_distribution(decision))

    observed = [initial]
    for index in range(6):
        distribution = provider.true_distribution(decision)
        action = _argmax_action(distribution)
        provider.policy(decision)
        provider.observe(
            _record(
                provider_role="dropper",
                own_action=action,
                opponent_action=max(1, action - 1),
                half_round_index=index,
            )
        )
        observed.append(provider.true_distribution(decision))
    assert any(not np.array_equal(initial, item) for item in observed[1:])


def test_win_stay_lose_shift_responds_to_public_local_result() -> None:
    provider = make_opponent(WIN_STAY_LOSE_SHIFT, seed=21)
    initial = _argmax_action(provider.true_distribution(_decision()))
    _play_local_result(provider, won=True)
    assert _argmax_action(provider.true_distribution(_decision())) == initial

    _play_local_result(provider, won=False)
    assert _argmax_action(provider.true_distribution(_decision())) != initial


@pytest.mark.parametrize(
    ("family", "expected"),
    ((COPY_RECENT, 17), (COUNTER_RECENT, 44)),
)
def test_recent_reveal_families_use_only_public_opponent_action(
    family: str,
    expected: int,
) -> None:
    provider = make_opponent(family, seed=31)
    decision = _decision(role="checker")
    own_action = _argmax_action(provider.true_distribution(decision))
    provider.policy(decision)
    provider.observe(
        _record(
            provider_role="checker",
            own_action=own_action,
            opponent_action=17,
        )
    )
    assert _argmax_action(provider.true_distribution(decision)) == expected


def test_switch_family_persists_mode_across_game_boundaries() -> None:
    provider = make_opponent(SWITCH, seed=41)
    decision = _decision()
    provider.reset_game()
    before = provider.true_distribution(decision)
    provider.end_game(PublicGameOutcome(0, None, 0))
    for game_index in range(1, 5):
        provider.reset_game()
        provider.end_game(PublicGameOutcome(game_index, None, 0))
    after = provider.true_distribution(decision)
    assert not np.array_equal(before, after)
    assert _argmax_action(before) < _argmax_action(after)


def test_session_reset_restores_the_seeded_opponent_initial_state() -> None:
    provider = make_opponent(SWITCH, seed=41)
    decision = _decision()
    initial = provider.true_distribution(decision)
    provider.reset_game()
    provider.end_game(PublicGameOutcome(0, None, 0))
    for game_index in range(1, 5):
        provider.reset_game()
        provider.end_game(PublicGameOutcome(game_index, None, 0))
    assert not np.array_equal(initial, provider.true_distribution(decision))

    provider.reset_session()

    np.testing.assert_array_equal(initial, provider.true_distribution(decision))
    assert provider.game_index == 0
    assert provider.decision_count == 0


def test_game_boundary_rebinds_actor_seat_without_erasing_session_memory() -> None:
    provider = make_opponent(COPY_RECENT, seed=45)
    provider.reset_game()
    baku_decision = _decision(role="checker", actor_name="Baku")
    baku_action = _argmax_action(provider.true_distribution(baku_decision))
    provider.policy(baku_decision)
    provider.observe(
        _record(
            provider_role="checker",
            provider_name="Baku",
            opponent_name="Hal",
            own_action=baku_action,
            opponent_action=17,
        )
    )
    provider.end_game(PublicGameOutcome(0, None, 1))

    provider.reset_game()
    hal_decision = _decision(role="dropper", actor_name="Hal")
    assert _argmax_action(provider.true_distribution(hal_decision)) == 17
    provider.policy(hal_decision)
    provider.observe(
        _record(
            provider_role="dropper",
            provider_name="Hal",
            opponent_name="Baku",
            own_action=17,
            opponent_action=16,
            game_index=1,
        )
    )
    assert provider.game_index == 1
    assert provider.observation_count == 2


def test_retreat_family_broadens_after_repeated_public_losses() -> None:
    provider = make_opponent(RETREAT_AFTER_DETECTED_EXPLOITATION, seed=51)
    initial = provider.true_distribution(_decision())
    assert float(initial.max()) > 0.10
    for _ in range(3):
        _play_local_result(provider, won=False)
    retreated = provider.true_distribution(_decision())
    np.testing.assert_allclose(retreated, np.full(60, 1.0 / 60.0), atol=1e-12)


def test_bait_then_reverse_changes_after_committed_decisions() -> None:
    provider = make_opponent(BAIT_THEN_REVERSE, seed=61)
    decision = _decision()
    initial_action = _argmax_action(provider.true_distribution(decision))
    for index in range(8):
        action = _argmax_action(provider.true_distribution(decision))
        provider.policy(decision)
        provider.observe(
            _record(
                provider_role="dropper",
                own_action=action,
                opponent_action=max(1, action - 1),
                half_round_index=index,
            )
        )
    assert _argmax_action(provider.true_distribution(decision)) == 61 - initial_action


def test_pure_dth_boundary_rejects_leap_actions_and_bad_lifecycle() -> None:
    provider = make_opponent(EARLY, seed=71)
    leap = _decision(legal_seconds=tuple(range(1, 62)), turn_duration=61)
    with pytest.raises(ValueError, match="1..60"):
        provider.true_distribution(leap)

    decision = _decision(role="checker")
    provider.policy(decision)
    with pytest.raises(RuntimeError, match="twice"):
        provider.policy(decision)
    with pytest.raises(ValueError, match="1..60"):
        provider.observe(
            _record(
                provider_role="checker",
                own_action=30,
                opponent_action=61,
            )
        )
