import base64
from dataclasses import replace
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import zlib

import numpy as np
import pytest
from fastapi.testclient import TestClient

from arena.agent import PolicyDrivenAgent
from arena.translated_hal_adapter import TranslatedHalPolicyProvider
from arena.web.app import SessionConfig, SeriesConfig, create_app
from arena.web.hosted import create_hosted_app
from arena.web.opponent_memory import OpponentMemory
from arena.testing import (
    StageAgent as _StageAgent,
    make_decision as _decision,
    make_forecast as _forecast,
    make_reveal as _reveal,
)
from arena.tests.test_web_hosted import MemoryStore, begin
from stl.engine.game import LS_WINDOW_START


def provider():
    return TranslatedHalPolicyProvider("unused", agent=_StageAgent())


def memory_app(p):
    return SimpleNamespace(state=SimpleNamespace(hal_provider=p))


def test_checkpoint_preserves_forecasts_without_sharing_mutable_evidence():
    first = provider()
    rng = np.random.default_rng(2291)
    for index in range(100):
        role = "checker" if index % 2 else "dropper"
        model = first.opponent_model
        model.observe(_forecast(model, role), opponent_action=int(rng.integers(1, 61)), self_action=int(rng.integers(1, 61)))
    memory = OpponentMemory()
    snapshot = memory.dump(memory_app(first))
    restored = provider()
    memory.restore(memory_app(restored), snapshot)
    for role in ("dropper", "checker"):
        np.testing.assert_array_equal(_forecast(first.opponent_model, role).policy, _forecast(restored.opponent_model, role).policy)
    restored.opponent_model.observe(_forecast(restored.opponent_model, "dropper"), opponent_action=12, self_action=3)
    assert first.opponent_model.total_observations == 100
    assert restored.opponent_model.total_observations == 101
    assert memory.dump(memory_app(first)) == snapshot


@pytest.mark.parametrize("damage", ["schema", "config", "nan", "shape", "action", "compressed"])
def test_checkpoint_rejects_corruption_without_replacing_the_model(damage):
    first = provider()
    memory = OpponentMemory()
    encoded = memory.dump(memory_app(first))
    data = json.loads(zlib.decompress(base64.b64decode(encoded)))
    if damage == "schema":
        data["schema"] = "unknown"
    elif damage == "config":
        data["config"]["offset_retention"] = 0.01
    elif damage == "nan":
        data["roles"]["checker"]["global_counts"][0] = float("nan")
    elif damage == "shape":
        data["offsets"]["dropper"] = []
    elif damage == "action":
        data["last_self"] = 61
    encoded = base64.b64encode(zlib.compress(json.dumps(data).encode())).decode()
    if damage == "compressed":
        encoded = encoded[:12]
    original = first.opponent_model
    with pytest.raises(ValueError):
        memory.restore(memory_app(first), encoded)
    assert first.opponent_model is original


def hosted(store, *, start_clock=720, max_half_rounds=6):
    def factory(game_seed, policy_seed, sequence_start):
        return create_app(hal_factory=lambda: PolicyDrivenAgent(provider(), seed=policy_seed),
            config=SessionConfig(seed=game_seed, start_clock=start_clock, max_half_rounds=max_half_rounds),
            series=SeriesConfig(conceal_hal_details=True), webclient_dist=None,
            sequence_start=sequence_start)
    return create_hosted_app(store, factory, version="translated-test", secure_cookie=False,
        memory=OpponentMemory(), policy_label="translated-hal-v1")


def move(client, state, action=12):
    response = client.post("/api/session/action", json={"sequence": state["sequence"], "second": action})
    assert response.status_code == 200, response.text
    return response.json()


def observations(store):
    record = json.loads(next(iter(store.rows.values())))
    p = provider()
    OpponentMemory().restore(memory_app(p), record["opponent_memory"])
    return p.opponent_model.total_observations


@pytest.mark.parametrize("boundary", ["restart", "next_game"])
def test_memory_survives_game_boundaries_and_worker_replacement(boundary):
    store = MemoryStore()
    client = TestClient(hosted(store, max_half_rounds=1))
    revealed = move(client, begin(client))
    if boundary == "restart":
        path, state = "/api/session/restart", revealed
    else:
        state = client.post("/api/session/ack", json={"sequence": revealed["sequence"]}).json()
        assert state["phase"] == "game_over"
        path = "/api/session"
    response = client.post(path, json={"sequence": state["sequence"]})
    assert response.status_code == 200, response.text
    assert observations(store) == 1
    recovered = TestClient(hosted(store, max_half_rounds=1))
    recovered.cookies.update(client.cookies)
    assert recovered.get("/api/session").json() == response.json()
    revealed = move(recovered, begin(recovered))
    restarted = recovered.post("/api/session/restart", json={"sequence": revealed["sequence"]})
    assert restarted.status_code == 200
    assert observations(store) == 2
    stranger = TestClient(hosted(store))
    assert stranger.get("/api/session").json()["sequence"] == 0
    records = [json.loads(r) for r in store.rows.values()]
    assert sum("opponent_memory" in r for r in records) == 1
    transcript = recovered.get("/api/transcript").text
    assert "offsets" not in transcript and "opponent_memory" not in transcript


def test_cold_replay_and_warm_worker_produce_the_same_next_reveal():
    store = MemoryStore()
    warm = TestClient(hosted(store))
    state = move(warm, begin(warm))
    warm.post("/api/session/restart", json={"sequence": state["sequence"]})
    state = move(warm, begin(warm), 17)
    state = warm.post("/api/session/ack", json={"sequence": state["sequence"]}).json()
    fork_store = MemoryStore()
    fork_store.rows = dict(store.rows)
    cold = TestClient(hosted(fork_store))
    cold.cookies.update(warm.cookies)
    assert cold.get("/api/session").json() == state
    assert move(cold, state, 21) == move(warm, state, 21)
    assert fork_store.rows == store.rows


def test_failed_commit_does_not_learn_or_reveal_a_speculative_move():
    class RejectStore(MemoryStore):
        reject = False

        async def compare_set(self, key, old, new):
            return False if self.reject else await super().compare_set(key, old, new)

    store = RejectStore()
    client = TestClient(hosted(store))
    state = begin(client)
    before = dict(store.rows)
    store.reject = True
    refused = client.post("/api/session/action", json={"sequence": state["sequence"], "second": 8})
    assert refused.status_code == 409
    assert "last_outcome" not in refused.text
    assert store.rows == before
    store.reject = False
    revealed = move(client, state, 8)
    client.post("/api/session/restart", json={"sequence": revealed["sequence"]})
    assert observations(store) == 1


def test_canonical_leap_61_resolves_and_does_not_enter_the_60_action_model():
    store = MemoryStore()
    client = TestClient(hosted(store, start_clock=LS_WINDOW_START - 60))
    state = begin(client)
    first = move(client, state, 60)
    state = client.post("/api/session/ack", json={"sequence": first["sequence"]}).json()
    assert state["human_role"] == "dropper"
    assert state["legal_seconds"][-1] == 61
    revealed = move(client, state, 61)
    assert revealed["last_outcome"] is not None
    client.post("/api/session/restart", json={"sequence": revealed["sequence"]})
    assert observations(store) == 1


def test_leap_fallback_keeps_evidence_and_clears_sequence_references():
    p = provider()
    p.policy(_decision())
    p.observe(_reveal())
    leap = replace(_decision(role="checker"), turn_duration=61)
    assert set(p.policy(leap)) == set(range(1, 61))
    reveal = _reveal(self_role="checker", opponent_action=61, half_round_index=1)
    reveal = replace(reveal, pre_decision_state=replace(reveal.pre_decision_state, turn_duration=61))
    p.observe(reveal)
    assert p.opponent_model.total_observations == 1
    assert p.opponent_model._last_self is None
    p.policy(_decision())
    p.observe(_reveal(half_round_index=2))
    assert p.opponent_model.total_observations == 2


def test_hosted_package_imports_the_candidate_without_training_dependencies(tmp_path):
    from arena.web.prepare_vercel import copy_runtime_sources

    root = Path(__file__).resolve().parents[3]
    copy_runtime_sources(root, tmp_path)
    runtime = tmp_path / "runtime/src"
    code = (
        "import sys; sys.path.insert(0, sys.argv[1]); "
        "from arena.translated_hal_adapter import TranslatedHalPolicyProvider; "
        "from arena.web.production import create_production_app; "
        "p = TranslatedHalPolicyProvider('unused', agent=object()); "
        "assert p.config.offset_retention == 0.97; "
        "assert 'torch' not in sys.modules; "
        "assert 'arena.policies.adaptive' not in sys.modules"
    )
    result = subprocess.run([sys.executable, "-I", "-c", code, str(runtime)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_corrupt_durable_memory_fails_closed():
    store = MemoryStore()
    client = TestClient(hosted(store))
    state = move(client, begin(client))
    client.post("/api/session/restart", json={"sequence": state["sequence"]})
    key = next(iter(store.rows))
    record = json.loads(store.rows[key])
    record["opponent_memory"] = "invalid"
    store.rows[key] = json.dumps(record)
    fresh = TestClient(hosted(store))
    fresh.cookies.update(client.cookies)
    response = fresh.get("/api/session")
    assert response.status_code == 503
    assert "last_outcome" not in response.text


@pytest.mark.parametrize("policy, label", [("exact", "certified-dth"), ("translated-v1", "translated-hal-v1")])
def test_production_requires_an_explicit_candidate_switch(monkeypatch, policy, label):
    from arena.web import production

    agent = _StageAgent()
    agent.tablebase = SimpleNamespace(metadata={"code_config_digest": "test"})
    agent.decide = lambda state: SimpleNamespace(drop_policy=np.ones(60) / 60, check_policy=np.ones(60) / 60)
    monkeypatch.setattr(production, "CompleteDTHAgent", lambda *args, **kwargs: agent)
    store = MemoryStore()
    monkeypatch.setattr(production, "RedisSessionStore", lambda *args: store)
    monkeypatch.setenv("KV_REST_API_URL", "https://example.invalid")
    monkeypatch.setenv("KV_REST_API_TOKEN", "test")
    monkeypatch.delenv("SUPABASE_URL", raising=False)
    if policy == "exact":
        monkeypatch.delenv("STL_HAL_POLICY", raising=False)
    else:
        monkeypatch.setenv("STL_HAL_POLICY", policy)
    with TestClient(production.create_production_app(Path("unused")), base_url="https://testserver") as client:
        response = client.get("/api/health")
        assert response.status_code == 200
        assert response.json()["policy"] == label
        state = move(client, begin(client))
        response = client.post("/api/session/restart", json={"sequence": state["sequence"]})
        assert response.status_code == 200
        record = json.loads(next(iter(store.rows.values())))
        assert ("opponent_memory" in record) == (policy == "translated-v1")
