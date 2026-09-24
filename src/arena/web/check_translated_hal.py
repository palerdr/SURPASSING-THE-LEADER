"""Measure local serving and replay with the real certified tablebase."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from fastapi.testclient import TestClient

from arena.agent import PolicyDrivenAgent
from arena.translated_hal_adapter import TranslatedHalPolicyProvider
from arena.web.app import SessionConfig, SeriesConfig, create_app
from arena.web.hosted import create_hosted_app
from arena.web.opponent_memory import OpponentMemory
from dth.agent import CompleteDTHAgent
from stl.engine.game import LS_WINDOW_START


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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("choose a new report path")
    start = time.perf_counter()
    agent = CompleteDTHAgent(args.artifact)
    artifact_open_seconds = time.perf_counter() - start
    decisions, requests, recoveries = [], [], []
    memory_sizes, record_sizes = [], []
    leap_actions = 0

    class MeasuredProvider(TranslatedHalPolicyProvider):
        def policy(self, decision):
            started = time.perf_counter()
            result = super().policy(decision)
            decisions.append((time.perf_counter() - started) * 1000)
            return result

    for clock in (720, LS_WINDOW_START - 60):
        store = MemoryStore()

        def factory(game_seed, policy_seed, sequence_start):
            return create_app(
                hal_factory=lambda: PolicyDrivenAgent(MeasuredProvider(args.artifact, agent=agent), seed=policy_seed),
                config=SessionConfig(seed=game_seed, start_clock=clock, max_half_rounds=80),
                series=SeriesConfig(conceal_hal_details=True), webclient_dist=None, sequence_start=sequence_start)

        def worker():
            return create_hosted_app(store, factory, version="runtime-check-v1", secure_cookie=False,
                memory=OpponentMemory(), policy_label="translated-hal-v1")

        with TestClient(worker()) as client:
            def post(path, state, **values):
                started = time.perf_counter()
                response = client.post(path, json={"sequence": state["sequence"], **values})
                requests.append((time.perf_counter() - started) * 1000)
                if response.status_code != 200:
                    raise RuntimeError(response.text)
                return response.json()

            state = client.get("/api/session").json()
            for game in range(16):
                state = post("/api/session/begin", state)
                while state["phase"] == "awaiting_action":
                    if 61 in state["legal_seconds"]:
                        action = 61
                        leap_actions += 1
                    elif state["human_role"] == "checker":
                        action = 60
                    else:
                        action = 1 + (game * 7) % 60
                    state = post("/api/session/action", state, second=action)
                    # Rebuild from Redis-shaped state with a fresh worker before the next action.
                    with TestClient(worker()) as cold:
                        cold.cookies.update(client.cookies)
                        started = time.perf_counter()
                        response = cold.get("/api/session")
                        recoveries.append((time.perf_counter() - started) * 1000)
                        if response.status_code != 200 or response.json() != state:
                            raise RuntimeError("worker recovery changed the revealed state")
                    state = post("/api/session/ack", state)
                state = post("/api/session", state)
                record = json.loads(next(iter(store.rows.values())))
                memory_sizes.append(len(record["opponent_memory"]))
                record_sizes.append(len(next(iter(store.rows.values()))))

    def stats(values):
        return {"count": len(values), "p50_ms": float(np.quantile(values, .5)),
            "p95_ms": float(np.quantile(values, .95)), "max_ms": max(values)}

    report = {"schema": "arena-translated-hal-runtime-check-v1",
        "scope": "Local ASGI with an in-memory CAS store and real artifact. Excludes Redis network latency, platform boot, and deployment cold starts.",
        "artifact_open_seconds": artifact_open_seconds,
        "decision": stats(decisions), "request": stats(requests), "worker_recovery": stats(recoveries),
        "leap_61_reveals": leap_actions, "games": 32,
        "max_checkpoint_bytes": max(memory_sizes), "max_boundary_record_bytes": max(record_sizes),
        "gates": {"decision_p95_ms": 100, "request_p95_ms": 300, "recovery_p95_ms": 2000},
        "source_sha256": {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in [Path(__file__), Path("src/arena/translated_hal_adapter.py"),
                Path("src/arena/web/opponent_memory.py"), Path("src/arena/web/hosted.py"),
                Path("src/arena/web/app.py"), args.artifact / "tablebase.json"]}}
    report["passed"] = (report["decision"]["p95_ms"] < 100
        and report["request"]["p95_ms"] < 300
        and report["worker_recovery"]["p95_ms"] < 2000 and leap_actions > 0)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")
    print(json.dumps(report, indent=2))
    if not report["passed"]:
        raise SystemExit("runtime gate failed")


if __name__ == "__main__":
    main()
