"""Phase-machine invariants and CLI parity for :mod:`arena.session`."""

from __future__ import annotations

import json
import random
from pathlib import Path

import pytest

from arena import cli
from arena.session import Phase, PlaySession, SessionPhaseError
from arena.web.schema import snapshot_from_session
from stl.engine.game import (
    LS_WINDOW_START,
    OPENING_START_CLOCK,
    PHYSICALITY_BAKU,
    PHYSICALITY_HAL,
    TOTAL_TTD_MAX,
    Game,
    Player,
    Referee,
)


class _RecordingHal:
    """Hal stand-in that records exactly when it is consulted."""

    def __init__(self, second: int = 30) -> None:
        self.second = second
        self.calls: list[str] = []
        self.provider = object()

    def choose_action(self, game, role, turn_duration):
        del game
        self.calls.append(role)
        return min(self.second, turn_duration)


def _session(
    *,
    hal_agent=None,
    seed: int | None = 41,
    start_clock: int = OPENING_START_CLOCK,
    max_half_rounds: int | None = None,
    human_display_name: str = "Baku",
) -> PlaySession:
    hal = Player(name="Hal", physicality=PHYSICALITY_HAL)
    human = Player(name="Baku", physicality=PHYSICALITY_BAKU)
    game = Game(
        player1=hal,
        player2=human,
        referee=Referee(),
        rng=random.Random(seed),
    )
    game.game_clock = start_clock
    return PlaySession(
        game=game,
        hal_agent=hal_agent if hal_agent is not None else _RecordingHal(),
        hal=hal,
        human=human,
        human_display_name=human_display_name,
        game_seed=seed,
        start_clock=start_clock,
        max_half_rounds=max_half_rounds,
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


@pytest.mark.parametrize(
    ("half", "human_second", "fatal_player", "winner_is_human"),
    [
        (1, 1, "human", False),
        (2, 60, "hal", True),
    ],
)
def test_snapshot_carries_authoritative_winner_seat(
    half: int,
    human_second: int,
    fatal_player: str,
    winner_is_human: bool,
) -> None:
    session = _session(human_display_name="A display label")
    session.game.current_half = half
    player = session.human if fatal_player == "human" else session.hal
    player.ttd = TOTAL_TTD_MAX
    session.begin()
    session.submit(human_second)

    snapshot = snapshot_from_session(session)
    assert snapshot.last_outcome is not None
    assert snapshot.last_outcome.game_over is True
    assert snapshot.last_outcome.session_ending is True
    assert snapshot.winner_is_human is winner_is_human


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


def test_session_reproduces_the_cli_public_history(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """The refactor guard: driving the session by hand matches the CLI."""

    def _fresh_hal(*_args, **_kwargs):
        return _RecordingHal()

    monkeypatch.setattr(cli, "_make_hal", _fresh_hal)
    monkeypatch.setattr(cli, "_human_action", lambda *, actor, role, legal: legal[-1])
    transcript = tmp_path / "session.json"
    args = cli.build_parser().parse_args(
        [
            "play",
            "--seed",
            "41",
            "--max-half-rounds",
            "4",
            "--transcript",
            str(transcript),
        ]
    )
    assert cli.command_play(args) == 0
    from_cli = json.loads(transcript.read_text(encoding="utf-8"))["games"][0]

    session = _session(max_half_rounds=4)
    session.begin()
    while session.phase is Phase.AWAITING_ACTION:
        session.submit(session.legal_actions()[-1])
        session.acknowledge()

    assert session.finish()["public_history"] == from_cli["public_history"]
