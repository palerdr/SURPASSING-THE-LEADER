import pytest

from toy.rules import Bucket12Fixed50Rules, ruleset_for_name
from toy.state import ToyState


def test_action_mapping_and_same_bucket_transition():
    rules = Bucket12Fixed50Rules()
    assert rules.action_values == tuple(range(1, 13))
    assert [rules.action_seconds(action) for action in rules.action_values] == list(range(5, 61, 5))
    state = ToyState(hal_load=7, baku_load=11, role_phase=0)
    branch = rules.expand_joint_action(state, 4, 4)[0]
    assert branch.probability == 1.0
    assert branch.squandered_units == 1
    assert branch.state == ToyState(hal_load=7, baku_load=12, role_phase=1)


def test_successful_overcheck_accumulates_checker_load():
    rules = Bucket12Fixed50Rules()
    branch = rules.expand_joint_action(ToyState(baku_load=10), 2, 5)[0]
    assert branch.event == "check_success"
    assert branch.squandered_units == 4
    assert branch.state is not None
    assert branch.state.baku_load == 14


def test_failed_and_overflow_death_have_fixed_half_branches():
    rules = Bucket12Fixed50Rules()
    for branches in (
        rules.expand_joint_action(ToyState(), 5, 4),
        rules.expand_joint_action(ToyState(baku_load=59), 1, 2),
    ):
        assert [branch.probability for branch in branches] == [0.5, 0.5]
        assert branches[0].state is not None
        assert branches[0].state.baku_load == 0
        assert branches[1].terminal_value == 1.0


def test_canonical_toy_has_no_leap_or_alternate_rulesets():
    rules = ruleset_for_name("bucket12_fixed50")
    state = ToyState(role_phase=1, game_clock=3600)
    assert rules.action_size == 12
    assert 61 not in rules.legal_drop_actions(state)
    assert 61 not in rules.legal_check_actions(state)
    with pytest.raises(ValueError, match="expected 'bucket12_fixed50'"):
        ruleset_for_name("seconds60_leap")

