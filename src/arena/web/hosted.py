"""Durable per-player hosting around the existing arena HTTP contract."""

from __future__ import annotations

import hashlib
import json
import re
import secrets
from typing import Callable, Protocol

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

COOKIE = "stl_session"
TTL_SECONDS = 7 * 24 * 60 * 60
_READ_PATHS = {"/api/rules", "/api/session", "/api/transcript"}
_WRITE_PATHS = {
    "/api/session",
    "/api/session/begin",
    "/api/session/action",
    "/api/session/ack",
    "/api/session/restart",
}


class SessionStore(Protocol):
    async def get(self, key: str) -> str | None: ...
    async def compare_set(self, key: str, old: str | None, new: str) -> bool: ...


class RedisSessionStore:
    """Use an atomic Redis compare-and-set to serialize competing mutations."""

    def __init__(self, url: str, token: str):
        if not url.startswith("https://") or not token:
            raise ValueError("session storage requires an HTTPS URL and token")
        self.url, self.token = url, token

    async def command(self, command: list):
        async with httpx.AsyncClient(timeout=15) as client:
            response = await client.post(
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


def _public(payload):
    if isinstance(payload, dict):
        return {
            key: None if key in {"seed", "game_seed", "base_seed"} else _public(value)
            for key, value in payload.items()
        }
    if isinstance(payload, list):
        return [_public(value) for value in payload]
    return payload


def create_hosted_app(
    store: SessionStore,
    factory: Callable[[int, int], FastAPI],
    *,
    version: str,
    secure_cookie: bool = True,
):
    """Replay accepted commands with private seeds; persist before revealing."""
    app = FastAPI(
        title="Surpassing The Leader", docs_url=None, redoc_url=None, openapi_url=None
    )

    @app.get("/api/health")
    async def health():
        return {"status": "ok", "policy": "certified-dth", "version": version}

    @app.api_route("/api/{path:path}", methods=["GET", "POST"])
    async def route(request: Request, path: str):
        route_path = f"/api/{path}"
        mutation = request.method == "POST"
        if route_path not in (_WRITE_PATHS if mutation else _READ_PATHS):
            return JSONResponse({"detail": "Unknown route"}, status_code=404)
        if mutation:
            origin = request.headers.get("origin")
            if origin and origin.rstrip("/") != str(request.base_url).rstrip("/"):
                return JSONResponse(
                    {"detail": "Cross-origin action refused"}, status_code=403
                )
            data = await request.body()
            if len(data) > 4096:
                return JSONResponse({"detail": "Request is too large"}, status_code=413)
            try:
                command = json.loads(data)
            except (ValueError, UnicodeDecodeError):
                return JSONResponse({"detail": "Invalid JSON"}, status_code=422)
            if not isinstance(command, dict):
                return JSONResponse({"detail": "Expected an object"}, status_code=422)
            if any(
                key in command for key in ("seed", "start_clock", "max_half_rounds")
            ):
                return JSONResponse(
                    {"detail": "Game settings are server-controlled"}, status_code=422
                )
        token = request.cookies.get(COOKIE, "")
        valid_token = re.fullmatch(r"[A-Za-z0-9_-]{43}", token) is not None
        key = (
            "stl:v1:" + hashlib.sha256(token.encode()).hexdigest()
            if valid_token
            else None
        )
        try:
            raw = await store.get(key) if key else None
            record = json.loads(raw) if raw else None
            new_cookie = False
            if record is not None and record["version"] != version:
                record, raw, key = None, None, None
            if record is None:
                if mutation:
                    return JSONResponse(
                        {"detail": "Session expired. Reload the page."}, status_code=409
                    )
                record = {
                    "version": version,
                    "game_seed": secrets.randbits(128),
                    "policy_seed": secrets.randbits(128),
                    "events": [],
                }
                # The client's initial reads run in parallel. Only this route issues a cookie.
                if route_path == "/api/session":
                    token = secrets.token_urlsafe(32)
                    key = "stl:v1:" + hashlib.sha256(token.encode()).hexdigest()
                    raw = json.dumps(record, separators=(",", ":"))
                    if not await store.compare_set(key, None, raw):
                        raise RuntimeError("session creation collision")
                    new_cookie = True
            inner = factory(record["game_seed"], record["policy_seed"])
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=inner), base_url="http://arena"
            ) as client:
                for event in record["events"]:
                    replay = await client.post(event["path"], json=event["body"])
                    if replay.status_code != 200:
                        raise RuntimeError("stored session could not be replayed")
                response = await client.request(
                    request.method, route_path, json=command if mutation else None
                )
            if mutation and response.status_code == 200:
                record["events"].append({"path": route_path, "body": command})
                updated = json.dumps(record, separators=(",", ":"))
                if not await store.compare_set(key, raw, updated):
                    # The losing request must not reveal its speculative Hal action.
                    return JSONResponse(
                        {"detail": "Stale sequence. Reload the current turn."},
                        status_code=409,
                        headers={"Cache-Control": "no-store"},
                    )
            result = JSONResponse(
                _public(response.json()),
                status_code=response.status_code,
                headers={"Cache-Control": "no-store"},
            )
            if new_cookie:
                result.set_cookie(
                    COOKIE,
                    token,
                    max_age=TTL_SECONDS,
                    secure=secure_cookie,
                    httponly=True,
                    samesite="strict",
                    path="/",
                )
            return result
        except (httpx.HTTPError, RuntimeError, ValueError, KeyError):
            return JSONResponse(
                {"detail": "Game storage is unavailable. Please retry."},
                status_code=503,
                headers={"Cache-Control": "no-store"},
            )

    return app
