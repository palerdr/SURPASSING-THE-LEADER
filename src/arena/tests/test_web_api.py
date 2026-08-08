"""Contract tests for the browser API.

The load-bearing one is :func:`test_no_unrevealed_action_reaches_the_client`.
Everything else guards the sequence protocol and legality delegation.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from arena.web.app import SessionConfig, create_app
from stl.engine.game import LS_WINDOW_START


class _StubHal:
    """Deterministic Hal. Avoids memory-mapping the 2.4 GB DTH artifact."""

    def __init__(self, second: int = 37) -> None:
        self.second = second
        self.provider = object()

    def choose_action(self, game, role, turn_duration):
        del game, role
        return min(self.second, turn_duration)


def _client(**config) -> TestClient:
    app = create_app(hal_factory=_StubHal, config=SessionConfig(seed=41, **config))
    return TestClient(app)


def _walk(payload) -> list:
    """Every scalar anywhere in a JSON payload, at any depth."""

    if isinstance(payload, dict):
        out = []
        for key, value in payload.items():
            out.append(key)
            out.extend(_walk(value))
        return out
    if isinstance(payload, list):
        return [item for value in payload for item in _walk(value)]
    return [payload]


def test_no_unrevealed_action_reaches_the_client() -> None:
    """While the human is deciding, no action exists to leak.

    Hal's second is chosen inside ``PlaySession.submit``, after the human has
    committed, so a snapshot taken during AWAITING_ACTION cannot carry it. This
    walks the whole payload rather than checking named fields, so a future
    nested addition cannot reintroduce the leak unnoticed.
    """

    client = _client()
    snapshot = client.get("/api/session").json()
    begun = client.post("/api/session/begin", json={"sequence": snapshot["sequence"]}).json()

    assert begun["phase"] == "awaiting_action"
    assert begun["last_outcome"] is None
    keys = _walk(begun)
    assert "drop_time" not in keys
    assert "check_time" not in keys

    revealed = client.post(
        "/api/session/action",
        json={"sequence": begun["sequence"], "second": begun["legal_seconds"][0]},
    ).json()
    assert revealed["phase"] == "awaiting_ack"
    assert revealed["last_outcome"]["drop_time"] >= 1
    assert revealed["last_outcome"]["check_time"] >= 1
    # The same walk finds both keys once they are public, so the assertions
    # above are detecting absence rather than passing vacuously.
    revealed_keys = _walk(revealed)
    assert "drop_time" in revealed_keys
    assert "check_time" in revealed_keys


def test_the_reveal_is_dropped_again_once_acknowledged() -> None:
    client = _client()
    snapshot = client.get("/api/session").json()
    snapshot = client.post("/api/session/begin", json={"sequence": snapshot["sequence"]}).json()
    snapshot = client.post(
        "/api/session/action",
        json={"sequence": snapshot["sequence"], "second": snapshot["legal_seconds"][0]},
    ).json()
    snapshot = client.post("/api/session/ack", json={"sequence": snapshot["sequence"]}).json()
    if snapshot["phase"] == "awaiting_action":
        assert snapshot["last_outcome"] is None
        assert "drop_time" not in _walk(snapshot)


def test_legal_seconds_are_empty_until_the_human_is_on_the_clock() -> None:
    client = _client()
    assert client.get("/api/session").json()["legal_seconds"] == []


def test_a_stale_sequence_is_rejected() -> None:
    client = _client()
    snapshot = client.get("/api/session").json()
    client.post("/api/session/begin", json={"sequence": snapshot["sequence"]})
    replayed = client.post("/api/session/begin", json={"sequence": snapshot["sequence"]})
    assert replayed.status_code == 409


def test_an_illegal_second_is_refused_without_advancing_the_game() -> None:
    client = _client()
    snapshot = client.get("/api/session").json()
    snapshot = client.post("/api/session/begin", json={"sequence": snapshot["sequence"]}).json()
    refused = client.post(
        "/api/session/action", json={"sequence": snapshot["sequence"], "second": 99}
    )
    assert refused.status_code == 422
    assert client.get("/api/session").json()["sequence"] == snapshot["sequence"]


def test_only_a_human_dropper_is_offered_the_leap_second() -> None:
    """The 61 asymmetry is decided server-side and never in the client."""

    client = _client(start_clock=LS_WINDOW_START)
    snapshot = client.get("/api/session").json()
    snapshot = client.post("/api/session/begin", json={"sequence": snapshot["sequence"]}).json()
    assert snapshot["turn_duration"] == 61
    if snapshot["human_role"] == "dropper":
        assert snapshot["legal_seconds"][-1] == 61
    else:
        assert snapshot["legal_seconds"][-1] == 60


def test_a_full_game_reaches_a_terminal_phase() -> None:
    client = _client(max_half_rounds=6)
    snapshot = client.get("/api/session").json()
    snapshot = client.post("/api/session/begin", json={"sequence": snapshot["sequence"]}).json()
    while snapshot["phase"] == "awaiting_action":
        snapshot = client.post(
            "/api/session/action",
            json={"sequence": snapshot["sequence"], "second": snapshot["legal_seconds"][0]},
        ).json()
        snapshot = client.post(
            "/api/session/ack", json={"sequence": snapshot["sequence"]}
        ).json()
    assert snapshot["phase"] == "game_over"
    assert snapshot["half_rounds"] >= 1


def test_rules_text_is_served_for_the_opening_screen() -> None:
    payload = _client().get("/api/rules").json()
    assert payload["human_name"] == "Baku"
    assert len(payload["lines"]) > 0


def test_missing_art_is_a_404_rather_than_a_crash() -> None:
    class _EmptyArt:
        def frame(self, character, pose, index):
            del character, pose, index
            return None

    app = create_app(hal_factory=_StubHal, art_loader=_EmptyArt)
    assert TestClient(app).get("/art/baku/idle/0.png").status_code == 404


@pytest.mark.parametrize("phase_endpoint", ["ack", "action"])
def test_transitions_are_refused_from_the_rules_phase(phase_endpoint: str) -> None:
    client = _client()
    sequence = client.get("/api/session").json()["sequence"]
    body = {"sequence": sequence}
    if phase_endpoint == "action":
        body["second"] = 30
    assert client.post(f"/api/session/{phase_endpoint}", json=body).status_code == 409
