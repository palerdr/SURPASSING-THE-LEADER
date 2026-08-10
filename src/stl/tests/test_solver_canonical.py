from __future__ import annotations

from dataclasses import FrozenInstanceError, replace

import pytest

from stl.engine.game import (
    LS_WINDOW_END,
    LS_WINDOW_START,
    OPENING_START_CLOCK,
    TURN_DURATION_LEAP,
    TURN_DURATION_NORMAL,
)
from stl.solver.canonical import (
    GameState,
    PublicHalfRound,
    is_active_lsr,
    is_leap_window,
    leap_drop_available,
    lsr_variation,
    root_node,
    turn_duration,
)


def test_game_state_is_deeply_immutable_and_hashable() -> None:
    record = PublicHalfRound(drop_second=12, check_second=27, survived=None)
    state = GameState(world=root_node(), public_history=(record,), winner=None)

    assert state in {state}
    assert state.public_history == (record,)

    with pytest.raises(FrozenInstanceError):
        setattr(record, "drop_second", 13)
    with pytest.raises(FrozenInstanceError):
        setattr(state, "winner", "Hal")


def test_game_state_terminal_winner_is_a_fixed_identity() -> None:
    state = GameState(world=root_node(), public_history=(), winner="Hal")

    assert state.winner == "Hal"


def test_leap_window_is_closed_and_turn_duration_agrees() -> None:
    expected = {
        LS_WINDOW_START - 1: False,
        LS_WINDOW_START: True,
        LS_WINDOW_END: True,
        LS_WINDOW_END + 1: False,
    }

    for clock, inside in expected.items():
        assert is_leap_window(clock) is inside
        assert turn_duration(clock) == (
            TURN_DURATION_LEAP if inside else TURN_DURATION_NORMAL
        )


def test_lsr_variation_uses_round_start_minute_congruence() -> None:
    assert [lsr_variation(minute) for minute in range(12, 16)] == [1, 2, 3, 4]
    assert lsr_variation(57) == 2
    assert is_active_lsr(57)
    assert not is_active_lsr(56)


def test_leap_drop_is_only_available_to_baku_in_canonical_half_two() -> None:
    opening = root_node()
    leap_half_one = replace(opening, clock=LS_WINDOW_START, half=1)
    leap_half_two = replace(opening, clock=LS_WINDOW_START, half=2)

    assert not leap_drop_available(opening)
    assert not leap_drop_available(leap_half_one)
    assert leap_drop_available(leap_half_two)
    assert not leap_drop_available(replace(leap_half_two, clock=LS_WINDOW_END + 1))


def test_root_node_is_the_exact_opening() -> None:
    root = root_node()

    assert root.clock == OPENING_START_CLOCK
    assert root.half == 1
    assert (root.baku_load, root.baku_ttd, root.hal_load, root.hal_ttd) == (0, 0, 0, 0)
    assert root.hal_leap_memory is False
