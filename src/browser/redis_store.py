"""The Upstash Redis session store that the hosted Vercel runtime uses."""

from __future__ import annotations

import httpx

from browser.hosted import TTL_SECONDS


class RedisSessionStore:
    """Use an atomic Redis compare-and-set to serialize competing mutations."""

    def __init__(self, url: str, token: str):
        if not url.startswith("https://") or not token:
            raise ValueError("session storage requires an HTTPS URL and token")
        self.url, self.token = url, token
        self._client: httpx.AsyncClient | None = None

    async def command(self, command: list):
        # One client per worker keeps the TLS connection to the store open, so
        # each command costs a round trip and not a fresh handshake.
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=15)
        try:
            response = await self._client.post(
                self.url,
                json=command,
                headers={"Authorization": f"Bearer {self.token}"},
            )
        except (httpx.TransportError, RuntimeError):
            # A suspended worker can wake with a dead connection or a closed
            # event loop; open a fresh client once and retry.
            self._client = httpx.AsyncClient(timeout=15)
            response = await self._client.post(
                self.url,
                json=command,
                headers={"Authorization": f"Bearer {self.token}"},
            )
        response.raise_for_status()
        body = response.json()
        if "error" in body:
            raise RuntimeError("session storage command failed")
        return body["result"]

    async def get(self, key):
        return await self.command(["GET", key])

    async def compare_set(self, key, old, new):
        script = """
local current = redis.call('GET', KEYS[1])
if (ARGV[1] == 'missing' and current) or
   (ARGV[1] == 'present' and current ~= ARGV[2]) then return 0 end
redis.call('SET', KEYS[1], ARGV[3], 'EX', ARGV[4])
return 1
"""
        result = await self.command(
            [
                "EVAL",
                script,
                1,
                key,
                "missing" if old is None else "present",
                old or "",
                new,
                TTL_SECONDS,
            ]
        )
        return result == 1
