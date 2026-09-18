"""Certified DTH provider and durable session factory for Vercel."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

from arena.agent import PolicyDrivenAgent
from arena.dth_adapter import project_to_dth_state
from arena.web.app import SessionConfig, SeriesConfig, create_app
from arena.web.hosted import RedisSessionStore, create_hosted_app
from dth.agent import CompleteDTHAgent


def create_production_app(artifact: Path):
    agent = CompleteDTHAgent(artifact)

    class Policy:
        def policy(self, decision):
            move = agent.decide(project_to_dth_state(decision))
            row = move.drop_policy if decision.role == "dropper" else move.check_policy
            return {
                second: float(mass) for second, mass in enumerate(row, 1) if mass > 0
            }

    def factory(game_seed, policy_seed):
        return create_app(
            hal_factory=lambda: PolicyDrivenAgent(Policy(), seed=policy_seed),
            config=SessionConfig(seed=game_seed),
            series=SeriesConfig(conceal_hal_details=True),
            webclient_dist=None,
        )

    digest = hashlib.sha256()
    source = Path(__file__).resolve().parents[2]
    for path in [
        source / "arena/agent.py",
        source / "arena/contracts.py",
        source / "arena/dth_adapter.py",
        source / "arena/session.py",
        source / "arena/web/app.py",
        source / "arena/web/hosted.py",
        source / "arena/web/production.py",
        source / "arena/web/schema.py",
        source / "stl/engine/game.py",
        source / "stl/engine/actions.py",
    ]:
        digest.update(path.read_bytes())
    digest.update(agent.tablebase.metadata["code_config_digest"].encode())
    url = os.environ.get("KV_REST_API_URL") or os.environ.get(
        "UPSTASH_REDIS_REST_URL", ""
    )
    token = os.environ.get("KV_REST_API_TOKEN") or os.environ.get(
        "UPSTASH_REDIS_REST_TOKEN", ""
    )
    return create_hosted_app(
        RedisSessionStore(url, token), factory, version=digest.hexdigest()
    )
