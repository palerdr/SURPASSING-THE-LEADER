"""Durable record of every hosted game, and the leaderboard read from it.

Redis holds a command log for seven days so a worker can replay a live game.
The ledger is the permanent copy: one row per game, rewritten after each
resolved half-round, so a closed tab still leaves every move behind. Rows hold
only what an outcome screen has already shown, plus the server's private seeds.
The seeds never reach a browser; they let an offline job replay a game exactly.
"""

from __future__ import annotations

from typing import Protocol

import httpx


class GameLedger(Protocol):
    async def record_game(self, row: dict[str, object]) -> None: ...
    async def abandon_series(self, series_id: str) -> None: ...
    async def set_player_name(self, player_id: str, name: str) -> str:
        """Return ``ok``, ``no_game`` for a player with no recorded game, or
        ``too_soon`` for a rename inside the hold period."""
        ...
    async def leaderboard(self, player_id: str) -> dict[str, object]: ...


class SupabaseLedger:
    """Call the ledger's Postgres functions through Supabase's REST gateway."""

    def __init__(self, url: str, key: str):
        if not url.startswith("https://") or not key:
            raise ValueError("the game ledger requires an HTTPS URL and a secret key")
        self.url, self.key = url.rstrip("/"), key
        self._client: httpx.AsyncClient | None = None

    async def _rpc(self, function: str, arguments: dict[str, object]):
        # One client per worker keeps the TLS connection open, as the session
        # store does; a suspended worker gets one fresh client and one retry.
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=5)
        endpoint = f"{self.url}/rest/v1/rpc/{function}"
        headers = {"apikey": self.key}
        try:
            response = await self._client.post(endpoint, json=arguments, headers=headers)
        except (httpx.TransportError, RuntimeError):
            self._client = httpx.AsyncClient(timeout=5)
            response = await self._client.post(endpoint, json=arguments, headers=headers)
        response.raise_for_status()
        return response.json() if response.content else None

    async def record_game(self, row):
        await self._rpc("record_game", {"p": row})

    async def abandon_series(self, series_id):
        await self._rpc("abandon_series", {"p_series": series_id})

    async def set_player_name(self, player_id, name):
        return str(
            await self._rpc("set_player_name", {"p_player": player_id, "p_name": name})
        )

    async def leaderboard(self, player_id):
        return await self._rpc("leaderboard", {"p_player": player_id})


def life_left(snapshot: dict) -> float | None:
    """The leaderboard score: a winning human's distance from permanent death.

    Capacity less accrued death time less the squandered time still waiting in
    the cylinder. A game the human did not win has no score.
    """

    if not snapshot["winner_is_human"]:
        return None
    human = next(player for player in snapshot["players"] if player["is_human"])
    return max(
        0.0,
        snapshot["ttd_max"] - human["ttd_seconds"] - human["cylinder_seconds"],
    )


def game_row(
    *,
    player_id: str,
    series_id: str,
    policy_seed: int,
    version: str,
    snapshot: dict,
    transcript: dict,
) -> dict[str, object] | None:
    """One ledger row from a post-reveal snapshot and the inner transcript."""

    outcome = snapshot["last_outcome"]
    game = transcript["current_game"]
    if outcome is None or not game["public_history"]:
        return None
    if outcome["game_over"]:
        status = "finished"
    elif outcome["session_ending"]:
        status = "stopped"
    else:
        status = "active"
    finished = status == "finished"
    human = next(player for player in snapshot["players"] if player["is_human"])
    hal = next(player for player in snapshot["players"] if not player["is_human"])
    return {
        "player_id": player_id,
        "series_id": series_id,
        "game_index": game["game_index"],
        "status": status,
        "human_won": bool(snapshot["winner_is_human"]) if finished else None,
        "score": life_left(snapshot) if finished else None,
        "half_rounds": snapshot["half_rounds"],
        "human_cylinder_seconds": human["cylinder_seconds"],
        "human_ttd_seconds": human["ttd_seconds"],
        "human_deaths": human["deaths"],
        "hal_cylinder_seconds": hal["cylinder_seconds"],
        "hal_ttd_seconds": hal["ttd_seconds"],
        "hal_deaths": hal["deaths"],
        "start_clock": game["start_clock"],
        "pure_dth": snapshot["pure_dth"],
        "hal_agent": transcript["hal_agent"],
        "code_version": version,
        # The seeds are 128-bit; text keeps them exact in JSON and Postgres.
        "game_seed": str(game["seed"]),
        "policy_seed": str(policy_seed),
        "public_history": game["public_history"],
    }
