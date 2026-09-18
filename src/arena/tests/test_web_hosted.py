"""Exercise player isolation and commits across stateless HTTP workers."""

import asyncio
import json

import httpx
from fastapi.testclient import TestClient

from arena.web.app import SessionConfig, create_app
from arena.web.hosted import COOKIE, create_hosted_app


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


class Hal:
    provider = object()

    def choose_action(self, game, role, turn_duration):
        return 37


def hosted(store, version="test"):
    return create_hosted_app(
        store,
        lambda game_seed, policy_seed: create_app(
            hal_factory=Hal, config=SessionConfig(seed=game_seed), webclient_dist=None
        ),
        version=version,
        secure_cookie=False,
    )


def begin(client):
    state = client.get("/api/session").json()
    response = client.post("/api/session/begin", json={"sequence": state["sequence"]})
    assert response.status_code == 200
    return response.json()


def test_players_are_isolated_and_a_fresh_worker_recovers_the_game():
    store = MemoryStore()
    first, other = TestClient(hosted(store)), TestClient(hosted(store))
    state = begin(first)
    response = first.post(
        "/api/session/action", json={"sequence": state["sequence"], "second": 5}
    )
    assert response.status_code == 200
    resolved = response.json()
    assert resolved["last_outcome"] is not None
    assert other.get("/api/session").json()["phase"] == "rules"
    replacement = TestClient(hosted(store))
    replacement.cookies.update(first.cookies)
    assert replacement.get("/api/session").json() == resolved
    assert first.cookies[COOKIE] != other.cookies[COOKIE]
    transcript = replacement.get("/api/transcript").json()

    def check_seeds(value):
        if isinstance(value, dict):
            for key, child in value.items():
                if key in {"seed", "game_seed", "base_seed"}:
                    assert child is None
                check_seeds(child)
        elif isinstance(value, list):
            for child in value:
                check_seeds(child)

    check_seeds(transcript)


def test_conflicting_commits_reveal_only_the_winner():
    async def exercise():
        store = MemoryStore()
        transport = httpx.ASGITransport(app=hosted(store))
        async with httpx.AsyncClient(
            transport=transport, base_url="http://test"
        ) as client:
            state = (await client.get("/api/session")).json()
            state = (
                await client.post(
                    "/api/session/begin", json={"sequence": state["sequence"]}
                )
            ).json()
            responses = await asyncio.gather(
                *[
                    client.post(
                        "/api/session/action",
                        json={"sequence": state["sequence"], "second": second},
                    )
                    for second in (1, 60)
                ]
            )
            assert sorted(r.status_code for r in responses) == [200, 409]
            loser = next(r for r in responses if r.status_code == 409)
            assert set(loser.json()) == {"detail"}
            winner = next(r for r in responses if r.status_code == 200)
            assert (await client.get("/api/session")).json() == winner.json()
            assert len(json.loads(next(iter(store.rows.values())))["events"]) == 2

    asyncio.run(exercise())


def test_restart_survives_worker_recovery_and_keeps_other_players_isolated():
    store = MemoryStore()
    client, other = TestClient(hosted(store)), TestClient(hosted(store))
    other_state = begin(other)
    state = begin(client)
    state = client.post(
        "/api/session/action", json={"sequence": state["sequence"], "second": 60}
    ).json()
    response = client.post(
        "/api/session/restart", json={"sequence": state["sequence"]}
    )
    assert response.status_code == 200
    fresh = response.json()
    assert fresh["phase"] == "rules"
    assert fresh["sequence"] > state["sequence"]
    assert fresh["last_outcome"] is None
    assert fresh["half_rounds"] == 0
    assert other.get("/api/session").json() == other_state
    replacement = TestClient(hosted(store))
    replacement.cookies.update(client.cookies)
    assert replacement.get("/api/session").json() == fresh
    history = replacement.get("/api/transcript").json()
    assert history["current_game"]["public_history"] == []
    assert history["base_seed"] is None
    assert not any(history["tally"].values())
    state = begin(replacement)
    assert replacement.post(
        "/api/session/action", json={"sequence": state["sequence"], "second": 60}
    ).status_code == 200


def test_initial_reads_only_issue_one_cookie_and_reject_client_seeds():
    client = TestClient(hosted(MemoryStore()))
    for route in ("rules", "transcript"):
        assert "set-cookie" not in client.get(f"/api/{route}").headers
    response = client.get("/api/session")
    assert "HttpOnly" in response.headers["set-cookie"]
    assert "SameSite=strict" in response.headers["set-cookie"]
    response = client.post("/api/session", json={"sequence": 0, "seed": 12})
    assert response.status_code == 422
    response = client.post(
        "/api/session/begin",
        json={"sequence": 0},
        headers={"origin": "https://other.example"},
    )
    assert response.status_code == 403


def test_expired_sessions_require_reload_and_failed_storage_reveals_nothing():
    store = MemoryStore()
    client = TestClient(hosted(store))
    begin(client)
    replacement = TestClient(hosted(store, version="new-code"))
    replacement.cookies.update(client.cookies)
    assert (
        replacement.post(
            "/api/session/action", json={"sequence": 1, "second": 1}
        ).status_code
        == 409
    )
    assert replacement.get("/api/session").json()["phase"] == "rules"
    assert replacement.cookies[COOKIE] != client.cookies[COOKIE]

    async def fail(*args):
        raise RuntimeError("storage failed")

    store.compare_set = fail
    response = replacement.post("/api/session/begin", json={"sequence": 0})
    assert response.status_code == 503
    assert set(response.json()) == {"detail"}
