import pytest

from abstract.rules import (
    FROZEN_REVIVAL_MODEL,
    Bucket6Frozen95Rules,
    Bucket12Frozen95Rules,
    ruleset_for_name,
)
from abstract.state import AbstractState


def test_ten_second_bucket_actions_and_inclusive_st() -> None:
    rules = Bucket6Frozen95Rules()
    assert rules.action_values == (1, 2, 3, 4, 5, 6)
    assert [rules.action_seconds(action) for action in rules.action_values] == [
        10,
        20,
        30,
        40,
        50,
        60,
    ]
    branch = rules.expand_joint_action(
        AbstractState(checker_load=7, checker_ttd=2, dropper_load=11, dropper_ttd=4),
        2,
        5,
    )[0]
    assert branch.event == "check_success"
    assert branch.squandered_units == 4
    assert branch.state == AbstractState(
        checker_load=11,
        checker_ttd=4,
        dropper_load=11,
        dropper_ttd=2,
    )


def test_failed_check_uses_frozen_revival_surface() -> None:
    rules = Bucket6Frozen95Rules()
    branches = rules.expand_joint_action(AbstractState(), 2, 1)
    assert [branch.probability for branch in branches] == pytest.approx([0.95, 0.05])
    assert branches[0].state == AbstractState(
        checker_load=0,
        checker_ttd=0,
        dropper_load=0,
        dropper_ttd=6,
    )
    assert branches[0].death_dose_units == 6
    assert branches[1].terminal_value == 1.0
    assert rules.revival_model_metadata["kind"] == FROZEN_REVIVAL_MODEL
    assert rules.revival_probability(24, 6) == pytest.approx(0.95 * 0.75**4)
    assert rules.revival_probability(25, 6) == 0.0
    assert rules.revival_probability(0, 30) == 0.0


def test_five_and_ten_second_curves_are_seconds_equivalent() -> None:
    ten = Bucket6Frozen95Rules()
    five = Bucket12Frozen95Rules()
    for st_seconds, ttd_seconds in (
        (0, 0),
        (60, 0),
        (0, 120),
        (120, 120),
        (230, 0),
    ):
        assert ten.revival_probability(
            ttd_seconds // 10,
            (st_seconds + 60) // 10,
        ) == pytest.approx(
            five.revival_probability(
                ttd_seconds // 5,
                (st_seconds + 60) // 5,
            ),
            abs=1e-15,
        )


def test_overflow_is_terminal_and_nonterminal_children_increase_potential() -> None:
    rules = Bucket6Frozen95Rules()
    state = AbstractState(checker_load=29, checker_ttd=0, dropper_load=0, dropper_ttd=0)
    overflow = rules.expand_joint_action(state, 1, 1)
    assert overflow[0].terminal_value == 1.0
    assert overflow[0].death_dose_units == 30

    for drop in rules.action_values:
        for check in rules.action_values:
            for branch in rules.expand_joint_action(AbstractState(), drop, check):
                if branch.state is not None:
                    assert branch.state.potential > 0


def test_ruleset_name_surface_contains_only_frozen_models() -> None:
    rules = ruleset_for_name("bucket6_frozen95")
    assert rules.physical_state_upper_bound == 30 * 31 * 30 * 31
    assert ruleset_for_name("bucket12_frozen95") == Bucket12Frozen95Rules()
    with pytest.raises(ValueError, match="bucket6_frozen95.*bucket12_frozen95"):
        ruleset_for_name("unsupported_pre_freeze_ruleset")
