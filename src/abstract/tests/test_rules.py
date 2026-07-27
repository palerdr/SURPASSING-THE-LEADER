import numpy as np
import pytest

from abstract.rules import (
    FROZEN_REVIVAL_MODEL,
    UNIFIED_REVIVAL_MODEL,
    Bucket6Frozen95Rules,
    Bucket6TTDCurve95Rules,
    Bucket6Unified80Rules,
    Bucket12Frozen95Rules,
    Bucket12TTDCurve95Rules,
    Bucket12Unified80Rules,
    ruleset_for_name,
)
from abstract.state import AbstractState


def test_ten_second_bucket_actions_and_inclusive_st() -> None:
    rules = Bucket6TTDCurve95Rules()
    assert rules.action_values == (1, 2, 3, 4, 5, 6)
    assert [rules.action_seconds(action) for action in rules.action_values] == [10, 20, 30, 40, 50, 60]
    branch = rules.expand_joint_action(AbstractState(checker_load=7, checker_ttd=2, dropper_load=11, dropper_ttd=4), 2, 5)[0]
    assert branch.event == "check_success"
    assert branch.squandered_units == 4
    assert branch.state == AbstractState(checker_load=11, checker_ttd=4, dropper_load=11, dropper_ttd=2)


def test_failed_check_uses_no_cpr_dose_and_ttd_curve() -> None:
    rules = Bucket6TTDCurve95Rules()
    branches = rules.expand_joint_action(AbstractState(), 2, 1)
    assert [branch.probability for branch in branches] == pytest.approx([0.9424, 0.0576])
    assert branches[0].state == AbstractState(checker_load=0, checker_ttd=0, dropper_load=0, dropper_ttd=6)
    assert branches[0].death_dose_units == 6
    assert branches[1].terminal_value == 1.0

    assert rules.revival_probability(24, 6) == pytest.approx(
        0.95 * (1.0 - (6 / 30) ** 3) * 2.0 ** (-((24 / 12) ** 1.3))
    )
    assert rules.revival_probability(25, 6) == 0.0
    assert rules.revival_probability(0, 30) == 0.0


def test_unified_revival_curve_is_linear_in_st_and_uses_only_st_and_ttd() -> None:
    rules = Bucket6Unified80Rules()
    branches = rules.expand_joint_action(AbstractState(), 2, 1)
    assert [branch.probability for branch in branches] == pytest.approx([0.8, 0.2])
    assert rules.revival_model_kind == UNIFIED_REVIVAL_MODEL
    assert [rules.revival_probability(0, dose) for dose in (6, 12, 18, 24)] == pytest.approx(
        [0.8, 0.6, 0.4, 0.2]
    )
    assert rules.revival_probability(24, 6) == pytest.approx(
        0.8 * 2.0 ** (-((24 / 12) ** 1.3)) * max(0.4, 0.88 ** (24 / 6))
    )
    assert rules.revival_probability(24, 6) > 0.0
    assert rules.revival_probability(25, 6) == 0.0
    assert rules.revival_probability(0, 30) == 0.0


def test_five_and_ten_second_unified_curves_are_seconds_equivalent() -> None:
    ten = Bucket6Unified80Rules()
    five = Bucket12Unified80Rules()
    for st_seconds, ttd_seconds in ((0, 0), (60, 0), (0, 120), (120, 120), (230, 0)):
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
    rules = Bucket6TTDCurve95Rules()
    state = AbstractState(checker_load=29, checker_ttd=0, dropper_load=0, dropper_ttd=0)
    overflow = rules.expand_joint_action(state, 1, 1)
    assert overflow[0].terminal_value == 1.0
    assert overflow[0].death_dose_units == 30

    for drop in rules.action_values:
        for check in rules.action_values:
            for branch in rules.expand_joint_action(AbstractState(), drop, check):
                if branch.state is not None:
                    assert branch.state.potential > 0


def test_frozen_revival_surface_is_geometric_and_seconds_invariant() -> None:
    ten = Bucket6Frozen95Rules()
    five = Bucket12Frozen95Rules()
    assert ten.revival_model_kind == FROZEN_REVIVAL_MODEL
    assert ten.expand_joint_action(AbstractState(), 2, 1)[0].probability == pytest.approx(0.95)
    assert ten.revival_probability(24, 6) == pytest.approx(0.95 * 0.75**4)
    assert ten.revival_probability(25, 6) == 0.0
    assert ten.revival_probability(0, 30) == 0.0

    for st_seconds, ttd_seconds in ((0, 0), (60, 0), (0, 120), (120, 120), (230, 0)):
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


def test_ruleset_name_is_closed_and_role_relative_domain_is_known() -> None:
    rules = ruleset_for_name("bucket6_unified80")
    assert rules.physical_state_upper_bound == 30 * 31 * 30 * 31
    assert ruleset_for_name("bucket12_unified80") == Bucket12Unified80Rules()
    assert ruleset_for_name("bucket6_ttd_curve95") == Bucket6TTDCurve95Rules()
    assert ruleset_for_name("bucket12_ttd_curve95") == Bucket12TTDCurve95Rules()
    assert ruleset_for_name("bucket6_frozen95") == Bucket6Frozen95Rules()
    assert ruleset_for_name("bucket12_frozen95") == Bucket12Frozen95Rules()
    with pytest.raises(ValueError, match="bucket6_unified80.*bucket12_unified80"):
        ruleset_for_name("bucket12_fixed50")
