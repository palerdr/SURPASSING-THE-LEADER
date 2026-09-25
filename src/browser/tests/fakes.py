"""Fakes that more than one browser test module shares.

``tests/parity/test_terminal_browser.py`` at the repository root imports them
too. No runtime module imports this one.

- ``StubHal`` is a deterministic Hal that never opens the DTH artifact.
  ``make_client`` builds a local browser app around it, and ``play_out``
  drives one game of that app to game over.
- ``MemoryStore`` holds the hosted session records in memory with the
  compare-and-set of the Redis store, and ``begin`` opens the first decision
  of a hosted game.
"""

from __future__ import annotations

from fastapi.testclient import TestClient

from browser.app import SessionConfig, create_app


class StubHal:
    """Deterministic Hal. Avoids memory-mapping the 2.4 GB DTH artifact."""

    def __init__(self, second: int = 37) -> None:
        self.second = second
        self.provider = object()

    def choose_action(self, game, role, turn_duration):
        del game, role
        return min(self.second, turn_duration)


def make_client(**config) -> TestClient:
    app = create_app(
        hal_factory=StubHal, config=SessionConfig(seed=41, **config), webclient_dist=None
    )
    return TestClient(app)


def play_out(client: TestClient, snapshot: dict) -> dict:
    """Drive one game from the rules screen to game over with the first legal second."""

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
    return snapshot


class MemoryStore:
    def __init__(self):
        self.rows = {}

    async def get(self, key):
        return self.rows.get(key)

    async def compare_set(self, key, old, new):
        if self.rows.get(key) != old:
            return False
        self.rows[key] = new
        return True


def begin(client):
    state = client.get("/api/session").json()
    response = client.post("/api/session/begin", json={"sequence": state["sequence"]})
    assert response.status_code == 200
    return response.json()
