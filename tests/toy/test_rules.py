import math

from stl.toy.rules import (
    Bucket12Fixed50Rules,
    FullSecondLeapRules,
    FullSecondTTDCPRPhysicalityRules,
    FullSecondVariableRevivalRules,
)
from stl.toy.state import ToyState


def test_v0_action_mapping_and_same_bucket_transition():
    rules = Bucket12Fixed50Rules()
    assert rules.action_values == tuple(range(1, 13))
    assert [rules.action_seconds(action) for action in rules.action_values] == list(range(5, 61, 5))
    state = ToyState(hal_load=7, baku_load=11, role_phase=0)
    branch = rules.expand_joint_action(state, 4, 4)[0]
    assert branch.probability == 1.0
    assert branch.squandered_units == 1
    assert branch.state == ToyState(hal_load=7, baku_load=12, role_phase=1)
    assert state == ToyState(hal_load=7, baku_load=11, role_phase=0)


def test_successful_overcheck_accumulates_checker_load():
    rules = Bucket12Fixed50Rules()
    branch = rules.expand_joint_action(ToyState(baku_load=10), 2, 5)[0]
    assert branch.event == "check_success"
    assert branch.squandered_units == 4
    assert branch.state is not None
    assert branch.state.baku_load == 14
    assert branch.state.role_phase == 1


def test_failed_and_overflow_death_have_fixed_half_branches():
    rules = Bucket12Fixed50Rules()
    failed = rules.expand_joint_action(ToyState(), 5, 4)
    overflow = rules.expand_joint_action(ToyState(baku_load=59), 1, 2)
    for branches in (failed, overflow):
        assert [branch.probability for branch in branches] == [0.5, 0.5]
        assert branches[0].state is not None
        assert branches[0].state.role_phase == 1
        assert branches[0].state.baku_load == 0
        assert branches[1].terminal_value == 1.0


def test_v0_has_no_leap_second_or_dynamic_action():
    rules = Bucket12Fixed50Rules()
    state = ToyState(role_phase=1, game_clock=3600)
    assert rules.action_size == 12
    assert 61 not in rules.legal_drop_actions(state)
    assert 61 not in rules.legal_check_actions(state)


def test_variable_revival_endpoints_and_history_modifiers():
    variable = FullSecondVariableRevivalRules()
    state = ToyState()
    assert variable.survival_probability(state, checker_is_hal=True, dose_units=0) == 1.0
    assert variable.survival_probability(state, checker_is_hal=True, dose_units=300) == 0.0
    branches = variable.expand_joint_action(ToyState(baku_load=299), 1, 60)
    assert math.isclose(sum(branch.probability for branch in branches), 1.0)
    assert branches[0].death_dose_units == 300
    assert branches[0].event == "overflow_died"
    assert branches[0].probability == 1.0

    history = FullSecondTTDCPRPhysicalityRules()
    history_state = ToyState(hal_ttd=60, baku_ttd=0, cprs_performed=2)
    expected = 0.85 * max(0.4, 0.88**2) * 1.0
    assert math.isclose(
        history.survival_probability(history_state, checker_is_hal=True, dose_units=0),
        expected,
        rel_tol=1e-12,
    )
    exact_boundary = ToyState(hal_ttd=240)
    over_boundary = ToyState(hal_ttd=241)
    assert history.survival_probability(exact_boundary, checker_is_hal=True, dose_units=60) > 0.0
    assert history.survival_probability(over_boundary, checker_is_hal=True, dose_units=60) == 0.0
    assert history.encode_state(history_state, 8).shape == (9,)


def test_leap_rules_are_deferred_and_role_oriented():
    rules = FullSecondLeapRules()
    leap_state = ToyState(role_phase=1, game_clock=3540)
    hal_state = leap_state.with_updates(role_phase=0)
    assert rules.action_size == 61
    assert 61 in rules.legal_drop_actions(leap_state)
    assert 61 not in rules.legal_drop_actions(hal_state)
    assert 61 not in rules.legal_check_actions(leap_state)
