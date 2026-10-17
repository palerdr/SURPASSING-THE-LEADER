"""Dagster resources: the hosted ledger, the local data paths, and dbt.

`SupabaseResource` reads ledger rows through Supabase's REST gateway.
`PathsResource` names every generated file under one data directory.
`make_dbt_resource` points the dagster-dbt CLI resource at the loop's dbt project.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import httpx
from dagster import ConfigurableResource
from dagster_dbt import DbtCliResource

PACKAGE_DIR = Path(__file__).resolve().parent
LOOP_DIR = PACKAGE_DIR.parent
REPOSITORY_ROOT = LOOP_DIR.parents[1]
DEFAULT_DATA_DIR = LOOP_DIR / "data"
DBT_DIR = LOOP_DIR / "dbt"
DEFAULT_ARENA_CONFIG_DIR = REPOSITORY_ROOT / "src" / "arena" / "config"
DEFAULT_DTH_ARTIFACT_DIR = REPOSITORY_ROOT / "src" / "dth" / "artifacts" / "complete_full_v1"
GAME_ORDER = "updated_at,series_id,game_index"


def content_range_total(response: httpx.Response) -> int | None:
    """Return the total of a `Content-Range: first-last/total` header, or None when the total is `*` or absent."""
    _, _, total = response.headers.get("Content-Range", "").partition("/")
    return int(total) if total.isdigit() else None


class SupabaseResource(ConfigurableResource):
    """Page the ledger's `games` table by an `updated_at` window.

    The gateway can cap a page below `page_size` (PostgREST's max_rows), so a short page
    does not end the read. Each request asks for the exact row count. The read stops when
    it holds that many rows, or on an empty page when the gateway omits the count.
    """

    url: str
    service_key: str
    page_size: int = 1000
    retries: int = 3
    backoff_seconds: float = 1.0
    timeout_seconds: float = 30.0

    def make_transport(self) -> httpx.BaseTransport | None:
        """Return the HTTP transport. Tests override this method with a mock transport."""
        return None

    def _get(self, client: httpx.Client, params: list[tuple[str, str]], first: int) -> httpx.Response:
        headers = {
            "apikey": self.service_key,
            "Authorization": f"Bearer {self.service_key}",
            "Range-Unit": "items",
            "Range": f"{first}-{first + self.page_size - 1}",
            "Prefer": "count=exact",
        }
        for attempt in range(self.retries + 1):
            try:
                response = client.get("/rest/v1/games", params=params, headers=headers)
            except httpx.TransportError:
                if attempt == self.retries:
                    raise
            else:
                if response.status_code < 500 or attempt == self.retries:
                    response.raise_for_status()
                    return response
            time.sleep(self.backoff_seconds * 2**attempt)
        raise AssertionError("the retry loop always returns or raises")

    def fetch_games(self, start: str, end: str) -> list[dict]:
        """Return every row with `start <= updated_at < end`, in ledger order."""
        params = [
            ("select", "*"),
            ("updated_at", f"gte.{start}"),
            ("updated_at", f"lt.{end}"),
            ("order", GAME_ORDER),
        ]
        rows: list[dict] = []
        with httpx.Client(
            base_url=self.url.rstrip("/"),
            transport=self.make_transport(),
            timeout=self.timeout_seconds,
        ) as client:
            while True:
                response = self._get(client, params, len(rows))
                page = response.json()
                rows.extend(page)
                total = content_range_total(response)
                if not page or (total is not None and len(rows) >= total):
                    return rows


class PathsResource(ConfigurableResource):
    """Resolve the loop's generated paths to absolute paths.

    `bronze_root` overrides the bronze directory, for example to read a fixture.
    The environment variable HAL_LOOP_WAREHOUSE overrides the warehouse file,
    because the dbt profile reads the same variable.
    """

    data_dir: str = str(DEFAULT_DATA_DIR)
    bronze_root: str | None = None
    arena_config_dir: str = str(DEFAULT_ARENA_CONFIG_DIR)

    @property
    def root(self) -> Path:
        return Path(self.data_dir).resolve()

    @property
    def bronze_dir(self) -> Path:
        return Path(self.bronze_root).resolve() if self.bronze_root else self.root / "bronze"

    @property
    def warehouse(self) -> Path:
        override = os.environ.get("HAL_LOOP_WAREHOUSE")
        return Path(override).resolve() if override else self.root / "warehouse.duckdb"

    @property
    def mlruns(self) -> Path:
        return self.root / "mlruns"

    @property
    def models_dir(self) -> Path:
        return self.root / "models"


def make_dbt_resource() -> DbtCliResource:
    """Point the dbt CLI resource at the loop's dbt project and its profiles.yml."""
    return DbtCliResource(project_dir=str(DBT_DIR), profiles_dir=str(DBT_DIR))
