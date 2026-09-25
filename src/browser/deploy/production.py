"""Certified DTH provider and durable session factory for Vercel."""

from __future__ import annotations

import hashlib
import logging
import os
from pathlib import Path

from arena.agent import PolicyDrivenAgent
from arena.dth_adapter import project_to_dth_state
from browser.app import SessionConfig, SeriesConfig, create_app
from browser.deploy.manifest import REPOSITORY_ROOT, version_entries
from browser.hosted import create_hosted_app
from browser.ledger import SupabaseLedger
from browser.redis_store import RedisSessionStore
from dth.agent import CompleteDTHAgent


def create_production_app(artifact: Path):
    # httpx logs one INFO line for each call, the in-process game calls
    # included. A move wrote up to forty lines to Vercel's log pipe.
    logging.getLogger("httpx").setLevel(logging.WARNING)
    # `prepare_vercel` hashes both arrays before it copies them into the
    # bundle, and the bundle is immutable once deployed. Rehashing 2.3 GB on
    # every cold start cost about ten seconds before the first answer.
    agent = CompleteDTHAgent(artifact, verify_hashes=False)
    policy_name = os.environ.get("STL_HAL_POLICY", "exact")
    if policy_name not in ("exact", "translated-v1"):
        raise ValueError("STL_HAL_POLICY must be exact or translated-v1")
    memory = None
    if policy_name == "translated-v1":
        from arena.translated_hal_adapter import TranslatedHalPolicyProvider
        from browser.opponent_memory import OpponentMemory

        memory = OpponentMemory()

    class Policy:
        def policy(self, decision):
            move = agent.decide(project_to_dth_state(decision))
            row = move.drop_policy if decision.role == "dropper" else move.check_policy
            return {
                second: float(mass) for second, mass in enumerate(row, 1) if mass > 0
            }

    def factory(game_seed, policy_seed, sequence_start=0):
        policy = (TranslatedHalPolicyProvider(artifact, agent=agent)
                  if policy_name == "translated-v1" else Policy())
        return create_app(
            hal_factory=lambda: PolicyDrivenAgent(policy, seed=policy_seed),
            config=SessionConfig(seed=game_seed),
            series=SeriesConfig(hal_agent=policy_name, conceal_hal_details=True),
            webclient_dist=None,
            sequence_start=sequence_start,
        )

    # The hosted code version. deploy/manifest.py flags the files that enter
    # it for each policy.
    digest = hashlib.sha256()
    digest.update(policy_name.encode())
    for entry in version_entries(policy_name):
        digest.update((REPOSITORY_ROOT / entry.path).read_bytes())
    digest.update(agent.tablebase.metadata["code_config_digest"].encode())
    url = os.environ.get("KV_REST_API_URL") or os.environ.get(
        "UPSTASH_REDIS_REST_URL", ""
    )
    token = os.environ.get("KV_REST_API_TOKEN") or os.environ.get(
        "UPSTASH_REDIS_REST_TOKEN", ""
    )
    # The ledger is optional: without its two settings the game plays as
    # before, records nothing, and serves no leaderboard.
    ledger_url = os.environ.get("SUPABASE_URL", "")
    ledger_key = os.environ.get("SUPABASE_SECRET_KEY") or os.environ.get(
        "SUPABASE_SERVICE_ROLE_KEY", ""
    )
    ledger = None
    if ledger_url and ledger_key:
        try:
            ledger = SupabaseLedger(ledger_url, ledger_key)
        except ValueError:
            # A bad ledger setting costs the leaderboard and never the game.
            logging.getLogger(__name__).exception("game ledger is misconfigured")
    return create_hosted_app(
        RedisSessionStore(url, token),
        factory,
        version=digest.hexdigest(),
        ledger=ledger,
        memory=memory,
        policy_label="translated-hal-v1" if memory is not None else "certified-dth",
    )
