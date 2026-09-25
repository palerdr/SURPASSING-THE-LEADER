"""Phase-machine invariants for :mod:`arena.session`.

``python -m terminal play`` parity lives in ``src/terminal/tests/test_cli.py``.
"""

from __future__ import annotations

import pytest

from arena.session import Phase, PlaySession, SessionPhaseError
from arena.testing import RecordingHal as _RecordingHal, make_session as _session
from stl.engine.game import (
    LS_WINDOW_START,
    PHYSICALITY_BAKU,
    Player,
)


def test_session_starts_on_the_rules_phase_and_begins_into_action() -> None:
    session = _session()
    assert session.phase is Phase.RULES
    assert session.stopped is False
    session.begin()
    assert session.phase is Phase.AWAITING_ACTION
    assert session.stopped is False


def test_submit_outside_the_action_phase_is_rejected() -> None:
    session = _session()
    with pytest.raises(SessionPhaseError, match="AWAITING_ACTION"):
        session.submit(30)
    session.begin()
    session.submit(30)
    assert session.phase is Phase.AWAITING_ACK
    with pytest.raises(SessionPhaseError, match="AWAITING_ACTION"):
        session.submit(30)


def test_acknowledge_outside_the_reveal_phase_is_rejected() -> None:
    session = _session()
    session.begin()
    with pytest.raises(SessionPhaseError, match="AWAITING_ACK"):
        session.acknowledge()


def test_begin_is_not_repeatable() -> None:
    session = _session()
    session.begin()
    with pytest.raises(SessionPhaseError, match="RULES"):
        session.begin()


def test_hal_is_not_consulted_until_the_human_has_committed() -> None:
    """The structural basis of the hidden-information guarantee.

    While the session waits for the human, Hal's action does not exist, so no
    snapshot taken in that phase can leak it.
    """

    hal_agent = _RecordingHal()
    session = _session(hal_agent=hal_agent)
    session.begin()

    # Everything a client may read during AWAITING_ACTION.
    session.roles()
    session.legal_actions()
    session.pre_decision_state()
    assert hal_agent.calls == []

    session.submit(session.legal_actions()[0])
    assert len(hal_agent.calls) == 1


def test_hal_is_consulted_exactly_once_per_half_round() -> None:
    hal_agent = _RecordingHal()
    session = _session(hal_agent=hal_agent, max_half_rounds=4)
    session.begin()
    while session.phase is Phase.AWAITING_ACTION:
        session.submit(session.legal_actions()[0])
        session.acknowledge()
    assert len(hal_agent.calls) == session.half_rounds


def test_sequence_advances_on_every_transition() -> None:
    session = _session()
    seen = [session.sequence]
    session.begin()
    seen.append(session.sequence)
    session.submit(30)
    seen.append(session.sequence)
    session.acknowledge()
    seen.append(session.sequence)
    assert seen == sorted(set(seen))


def test_the_half_round_cap_ends_the_session_without_a_winner() -> None:
    session = _session(max_half_rounds=1)
    session.begin()
    session.submit(30)
    session.acknowledge()
    assert session.phase is Phase.GAME_OVER
    assert session.stopped is True
    assert session.finish()["half_rounds"] == 1


def test_finish_notifies_the_provider_only_once() -> None:
    class _CountingProvider:
        def __init__(self) -> None:
            self.outcomes = []

        def end_game(self, outcome) -> None:
            self.outcomes.append(outcome)

    hal_agent = _RecordingHal()
    hal_agent.provider = _CountingProvider()
    session = _session(hal_agent=hal_agent, max_half_rounds=1)
    session.begin()
    session.submit(30)
    session.acknowledge()
    session.finish()
    session.finish()
    assert len(hal_agent.provider.outcomes) == 1


def test_finish_rejects_a_nonterminal_session() -> None:
    session = _session()
    with pytest.raises(SessionPhaseError, match="GAME_OVER"):
        session.finish()
    session.begin()
    with pytest.raises(SessionPhaseError, match="GAME_OVER"):
        session.finish()


@pytest.mark.parametrize("second", [True, 2.0])
def test_submit_rejects_non_integer_action_types_before_consulting_hal(second) -> None:
    hal_agent = _RecordingHal()
    session = _session(hal_agent=hal_agent)
    session.begin()
    with pytest.raises(ValueError, match="Illegal action"):
        session.submit(second)
    assert hal_agent.calls == []


def test_display_label_never_replaces_bakus_rule_identity() -> None:
    session = _session(
        start_clock=LS_WINDOW_START,
        human_display_name="Alice <the challenger>",
    )
    session.game.current_half = 2

    assert session.human.name == "Baku"
    assert session.human_display_name == "Alice <the challenger>"
    assert session.legal_actions()[-1] == 61

    session.game.current_half = 1
    session.max_half_rounds = 1
    session.begin()
    session.submit(30)
    session.acknowledge()
    transcript = session.finish()
    first = transcript["public_history"][0]
    assert first["checker"] == "Alice <the challenger>"
    assert first["public_state_before"]["players"][1]["name"] == (
        "Alice <the challenger>"
    )


def test_session_rejects_reserved_display_and_foreign_players() -> None:
    with pytest.raises(ValueError, match="reserved"):
        _session(human_display_name="hal")

    session = _session()
    outsider = Player(name="Baku", physicality=PHYSICALITY_BAKU)
    with pytest.raises(ValueError, match="owned by the game"):
        PlaySession(
            game=session.game,
            hal_agent=_RecordingHal(),
            hal=session.hal,
            human=outsider,
        )


def test_only_a_human_dropper_may_use_the_leap_second() -> None:
    """Legality is the engine's call; the session must not widen it."""

    session = _session(start_clock=LS_WINDOW_START)
    assert session.turn_duration() == 61
    for half in (1, 2):
        session.game.current_half = half
        legal = session.legal_actions()
        if session.human_role() == "dropper":
            assert legal[-1] == 61
        else:
            assert legal[-1] == 60
