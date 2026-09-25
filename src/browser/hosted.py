"""Durable per-player hosting around the existing arena HTTP contract."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import secrets
import time
from collections import OrderedDict
from typing import Awaitable, Callable, Protocol

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import ValidationError

from browser.ledger import GameLedger, game_row
from browser.schema import PlayerNameRequest, top_standings

COOKIE = "stl_session"
TTL_SECONDS = 7 * 24 * 60 * 60
# The session cookie dies with each code version. The player cookie outlives
# deployments, so a standing on the leaderboard follows the same browser.
PLAYER_COOKIE = "stl_player"
PLAYER_TTL_SECONDS = 365 * 24 * 60 * 60
_TOKEN = re.compile(r"[A-Za-z0-9_-]{43}")
_log = logging.getLogger(__name__)
# Latency diagnosis: one id and one boot time per process.
_INSTANCE = secrets.token_hex(4)
_BOOTED = time.monotonic()


def _process_age() -> str:
    """Seconds since the kernel started this process, where Linux reports it."""
    try:
        with open("/proc/self/stat") as file:
            ticks = int(file.read().rsplit(")", 1)[1].split()[19])
        with open("/proc/uptime") as file:
            uptime = float(file.read().split()[0])
        return f"{uptime - ticks / os.sysconf('SC_CLK_TCK'):.1f}"
    except (OSError, ValueError, IndexError):
        return "-"


_AGE_AT_IMPORT = _process_age()
# "yes" when this module loaded from shipped bytecode and not from source.
_BYTECODE = "yes" if __spec__ and __spec__.cached and os.path.exists(__spec__.cached) else "no"
_READ_PATHS = {"/api/rules", "/api/session", "/api/transcript"}
# Games one process keeps between requests. The store stays the authority.
HELD_GAMES = 256
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


class SessionMemory(Protocol):
    def dump(self, app: FastAPI) -> str: ...
    def restore(self, app: FastAPI, memory: str) -> None: ...


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


def _public(payload):
    if isinstance(payload, dict):
        return {
            key: None if key in {"seed", "game_seed", "base_seed"} else _public(value)
            for key, value in payload.items()
        }
    if isinstance(payload, list):
        return [_public(value) for value in payload]
    return payload


def _cross_origin(request: Request) -> bool:
    origin = request.headers.get("origin")
    return bool(origin) and origin.rstrip("/") != str(request.base_url).rstrip("/")


def _player_id(request: Request) -> str | None:
    token = request.cookies.get(PLAYER_COOKIE, "")
    if _TOKEN.fullmatch(token) is None:
        return None
    return hashlib.sha256(token.encode()).hexdigest()


async def _ledger_write(write: Callable[[], Awaitable[None]]) -> None:
    """A ledger failure must not hide a reveal the session store has committed.

    The caller passes a function, so building the row fails here as well.
    """
    try:
        await write()
    except Exception:
        _log.exception("game ledger write failed")


def create_hosted_app(
    store: SessionStore,
    factory: Callable[[int, int, int], FastAPI],
    *,
    version: str,
    secure_cookie: bool = True,
    ledger: GameLedger | None = None,
    memory: SessionMemory | None = None,
    policy_label: str = "certified-dth",
):
    """Replay accepted commands with private seeds; persist before revealing.

    A record is one game: its seeds, the sequence its session starts at, and
    the commands accepted since. A restart and a next game each replace the
    record with a fresh one and an empty command list, so replay cost follows
    the current game. An optional memory adapter checkpoints opponent evidence
    at each game boundary. A new worker restores that evidence before replay.
    The checkpoint and accepted commands share one compare-and-set record.

    A process keeps the game it last served beside the stored text that game
    matches. A request whose stored text still matches plays on that game and
    skips the rebuild. A held game that another process has moved past plays
    the commands it lacks. A held game from another record is rebuilt, so a
    process cannot answer from a stale game.
    """
    # key -> stored text, game, series, and the commands that game has played.
    held_games: OrderedDict[str, tuple[str, FastAPI, str, list]] = OrderedDict()
    unplayed: list[FastAPI] = []
    app = FastAPI(
        title="Surpassing The Leader", docs_url=None, redoc_url=None, openapi_url=None
    )

    served = 0

    @app.middleware("http")
    async def diagnose(request: Request, call_next):
        # Latency diagnosis: an answer names its process, that process's age
        # and request count, the handler's time, and the held-game result.
        nonlocal served
        served += 1
        started = time.monotonic()
        response = await call_next(request)
        response.headers["x-stl-diagnosis"] = (
            f"instance={_INSTANCE} up={started - _BOOTED:.1f} n={served} "
            f"ms={(time.monotonic() - started) * 1000:.0f} "
            f"held={getattr(request.state, 'held', '-')} "
            f"age={_process_age()} age_at_import={_AGE_AT_IMPORT} "
            f"bytecode={_BYTECODE}"
        )
        return response

    @app.get("/api/health")
    async def health():
        return {"status": "ok", "policy": policy_label, "version": version}

    async def board(player_id: str | None) -> JSONResponse:
        # An unknown player reads the same public standings with no row marked.
        try:
            payload = await ledger.leaderboard(player_id or "")
            body = top_standings(payload).model_dump()
        except (httpx.HTTPError, RuntimeError, ValueError):
            return JSONResponse(
                {"detail": "The leaderboard is unavailable."},
                status_code=503,
                headers={"Cache-Control": "no-store"},
            )
        return JSONResponse(body, headers={"Cache-Control": "no-store"})

    @app.get("/api/leaderboard")
    async def read_leaderboard(request: Request):
        if ledger is None:
            return JSONResponse({"detail": "Unknown route"}, status_code=404)
        return await board(_player_id(request))

    @app.post("/api/leaderboard/name")
    async def post_name(request: Request):
        if ledger is None:
            return JSONResponse({"detail": "Unknown route"}, status_code=404)
        if _cross_origin(request):
            return JSONResponse(
                {"detail": "Cross-origin action refused"}, status_code=403
            )
        data = await request.body()
        if len(data) > 4096:
            return JSONResponse({"detail": "Request is too large"}, status_code=413)
        try:
            name = PlayerNameRequest.model_validate_json(data).name
        except ValidationError as error:
            # The player reads this line, so drop pydantic's "Value error, " label.
            reason = error.errors()[0]["msg"].removeprefix("Value error, ")
            return JSONResponse(
                {"detail": reason[:1].upper() + reason[1:] + "."}, status_code=422
            )
        player_id = _player_id(request)
        if player_id is None:
            return JSONResponse(
                {"detail": "Session expired. Reload the page."}, status_code=409
            )
        try:
            posted = await ledger.set_player_name(player_id, name)
        except (httpx.HTTPError, RuntimeError, ValueError):
            return JSONResponse(
                {"detail": "The leaderboard is unavailable."}, status_code=503
            )
        if posted == "no_game":
            return JSONResponse(
                {"detail": "Play a game before you post a name."}, status_code=409
            )
        if posted == "too_soon":
            return JSONResponse(
                {"detail": "Wait a minute before you change the name."},
                status_code=429,
            )
        if posted != "ok":
            # An answer this server does not know is no proof the name was set.
            return JSONResponse(
                {"detail": "The leaderboard is unavailable."}, status_code=503
            )
        return await board(player_id)

    @app.api_route("/api/{path:path}", methods=["GET", "POST"])
    async def route(request: Request, path: str):
        route_path = f"/api/{path}"
        mutation = request.method == "POST"
        if route_path not in (_WRITE_PATHS if mutation else _READ_PATHS):
            return JSONResponse({"detail": "Unknown route"}, status_code=404)
        if mutation:
            if _cross_origin(request):
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
        valid_token = _TOKEN.fullmatch(token) is not None
        player_id = _player_id(request)
        player_token = None
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
                    "series_id": secrets.token_hex(16),
                    "game_seed": secrets.randbits(128),
                    "policy_seed": secrets.randbits(128),
                    "sequence_start": 0,
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
            # The page makes one session read, so one load mints one player.
            # Two tabs opened together can mint two; the browser keeps the last.
            if player_id is None and not mutation and route_path == "/api/session":
                player_token = secrets.token_urlsafe(32)
            series_id = record["series_id"]
            history = None
            # The held game leaves the table while this request uses it, so a
            # concurrent request for the same player rebuilds its own copy.
            held = held_games.pop(key, None) if key else None
            played = len(held[3]) if held else 0
            if held is not None and held[0] == raw:
                request.state.held = "hit"
                inner, pending = held[1], []
            elif (
                held is not None
                and held[2] == series_id
                and record["events"][:played] == held[3]
            ):
                request.state.held = "behind"
                inner, pending = held[1], record["events"][played:]
            elif raw is None and route_path == "/api/rules":
                # The client's warm-up read carries no cookie. The rules of an
                # unplayed game are the same for every seed, so one game answers.
                request.state.held = "rules"
                if not unplayed:
                    unplayed.append(factory(0, 0, 0))
                inner, pending = unplayed[0], []
            else:
                request.state.held = "miss"
                inner = factory(
                    record["game_seed"],
                    record["policy_seed"],
                    record.get("sequence_start", 0),
                )
                if memory is not None and record.get("opponent_memory") is not None:
                    memory.restore(inner, record["opponent_memory"])
                pending = record["events"]
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=inner), base_url="http://arena"
            ) as client:
                for event in pending:
                    replay = await client.post(event["path"], json=event["body"])
                    if replay.status_code != 200:
                        raise RuntimeError("stored session could not be replayed")
                response = await client.request(
                    request.method, route_path, json=command if mutation else None
                )
                snapshot = response.json() if response.status_code == 200 else None
                # A resolved half-round changes the ledger row. The closing
                # acknowledgement repeats the last write in case it failed.
                if (
                    ledger is not None
                    and player_id is not None
                    and snapshot is not None
                    and (
                        route_path == "/api/session/action"
                        or (
                            route_path == "/api/session/ack"
                            and snapshot["phase"] == "game_over"
                        )
                    )
                ):
                    transcript = await client.get("/api/transcript")
                    if transcript.status_code == 200:
                        history = transcript.json()
            restarted = snapshot is not None and route_path == "/api/session/restart"
            next_game = snapshot is not None and mutation and route_path == "/api/session"
            if restarted or next_game:
                checkpoint = memory.dump(inner) if memory is not None else None
                # The old record validated this request. Open the new game as
                # its own record: new seeds, so a reload cannot rehearse Hal's
                # samples, and the next sequence number, so a request from
                # before the change stays stale.
                record = {
                    "version": version,
                    "series_id": secrets.token_hex(16),
                    "game_seed": secrets.randbits(128),
                    "policy_seed": secrets.randbits(128),
                    "sequence_start": int(response.json()["sequence"]),
                    "events": [],
                }
                if checkpoint is not None:
                    record["opponent_memory"] = checkpoint
                fresh = factory(
                    record["game_seed"], record["policy_seed"], record["sequence_start"]
                )
                if checkpoint is not None:
                    memory.restore(fresh, checkpoint)
                async with httpx.AsyncClient(
                    transport=httpx.ASGITransport(app=fresh), base_url="http://arena"
                ) as client:
                    response = await client.get("/api/session")
                if response.status_code != 200:
                    raise RuntimeError("fresh game could not be opened")
                inner = fresh
            elif mutation and response.status_code == 200:
                record["events"].append({"path": route_path, "body": command})
            if mutation and response.status_code == 200:
                updated = json.dumps(record, separators=(",", ":"))
                if not await store.compare_set(key, raw, updated):
                    # The losing request must not reveal its speculative Hal action.
                    # Its game played a move the store refused, so it is dropped.
                    return JSONResponse(
                        {"detail": "Stale sequence. Reload the current turn."},
                        status_code=409,
                        headers={"Cache-Control": "no-store"},
                    )
                raw = updated
            # A refused command leaves the game unheld; the next request rebuilds.
            if key and raw and (not mutation or response.status_code == 200):
                held_games[key] = (
                    raw,
                    inner,
                    record["series_id"],
                    list(record["events"]),
                )
                while len(held_games) > HELD_GAMES:
                    held_games.popitem(last=False)
            # The ledger follows the committed record, so a losing request
            # writes nothing. A restart closes the old series' open game.
            if ledger is not None and restarted:
                await _ledger_write(lambda: ledger.abandon_series(series_id))
            elif history is not None:

                async def record_row():
                    row = game_row(
                        player_id=player_id,
                        series_id=series_id,
                        policy_seed=record["policy_seed"],
                        version=version,
                        snapshot=snapshot,
                        transcript=history,
                    )
                    if row is not None:
                        await ledger.record_game(row)

                await _ledger_write(record_row)
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
            if player_token is not None:
                result.set_cookie(
                    PLAYER_COOKIE,
                    player_token,
                    max_age=PLAYER_TTL_SECONDS,
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
