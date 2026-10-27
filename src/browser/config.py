"""The one reader of environment variables in the browser app.

``deploy.production`` calls these functions when it builds the hosted app,
at cold start and never at import, so a test sets a variable before that
call. Each default is the value the hosted app used before: the exact policy,
and empty store and ledger settings.
"""

from __future__ import annotations

import os


def hal_policy() -> str:
    """``STL_HAL_POLICY``, the hosted Hal; ``exact`` when unset."""

    return os.environ.get("STL_HAL_POLICY", "exact")


def session_store() -> tuple[str, str]:
    """The Upstash Redis REST URL and token, under Vercel KV's names or Upstash's."""

    url = os.environ.get("KV_REST_API_URL") or os.environ.get("UPSTASH_REDIS_REST_URL", "")
    token = os.environ.get("KV_REST_API_TOKEN") or os.environ.get(
        "UPSTASH_REDIS_REST_TOKEN", ""
    )
    return url, token


def ledger() -> tuple[str, str]:
    """The Supabase URL and secret key. An empty value leaves the ledger off."""

    url = os.environ.get("SUPABASE_URL", "")
    key = os.environ.get("SUPABASE_SECRET_KEY") or os.environ.get(
        "SUPABASE_SERVICE_ROLE_KEY", ""
    )
    return url, key
