"""The seat-scoped browser snapshot of a :mod:`arena.session` play session."""

from __future__ import annotations

import pytest

from arena.testing import make_session as _session
from browser.schema import snapshot_from_session
from stl.engine.game import TOTAL_TTD_MAX


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
