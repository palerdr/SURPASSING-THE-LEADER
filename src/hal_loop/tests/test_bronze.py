"""bronze_games pages the ledger through Range windows and writes an idempotent daily extract."""

import hashlib
import json
from pathlib import Path

import httpx
import pytest
from dagster import materialize

from hal_loop.assets.bronze import bronze_games, day_window
from hal_loop.resources import PathsResource, SupabaseResource

FIXTURE = Path(__file__).parent / "fixtures" / "bronze" / "games" / "2026-10-13.jsonl"
DAY = "2026-10-13"
KEY = "test-service-key"


class Ledger:
    """A mock Supabase gateway that serves the fixture rows by Range window."""

    def __init__(self, rows, failures=0, max_rows=None, exact_count=True):
        self.rows = rows
        self.failures = failures
        # max_rows caps every page as PostgREST's db-max-rows setting does.
        self.max_rows = max_rows
        self.exact_count = exact_count
        self.requests: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if self.failures:
            self.failures -= 1
            return httpx.Response(503)
        assert request.url.path == "/rest/v1/games"
        assert request.headers["apikey"] == KEY
        assert request.headers["authorization"] == f"Bearer {KEY}"
        filters = request.url.params.get_list("updated_at")
        start, end = filters[0].removeprefix("gte."), filters[1].removeprefix("lt.")
        assert request.url.params["order"] == "updated_at,series_id,game_index"
        selected = [row for row in self.rows if start[:10] <= row["updated_at"][:10] < end[:10]]
        first, last = (int(value) for value in request.headers["range"].split("-"))
        if self.max_rows is not None:
            last = min(last, first + self.max_rows - 1)
        page = selected[first : last + 1]
        # PostgREST reports the total only when the request asks for an exact count.
        exact = self.exact_count and request.headers.get("prefer") == "count=exact"
        total = len(selected) if exact else "*"
        return httpx.Response(206, json=page, headers={"Content-Range": f"{first}-{first + len(page) - 1}/{total}"})


def fixture_rows():
    return [json.loads(line) for line in FIXTURE.read_text().splitlines()]


def ledger_resource(ledger: Ledger, page_size: int = 10) -> SupabaseResource:
    class MockSupabase(SupabaseResource):
        def make_transport(self):
            return httpx.MockTransport(ledger)

    return MockSupabase(url="https://ledger.test", service_key=KEY, page_size=page_size, backoff_seconds=0.0)


def run(ledger: Ledger, data_dir: Path, page_size: int = 10):
    return materialize(
        [bronze_games],
        partition_key=DAY,
        resources={"supabase": ledger_resource(ledger, page_size), "paths": PathsResource(data_dir=str(data_dir))},
    )


def test_day_window_is_half_open_utc():
    assert day_window(DAY) == ("2026-10-13T00:00:00+00:00", "2026-10-14T00:00:00+00:00")


def test_extract_pages_writes_jsonl_and_manifest(tmp_path):
    rows = fixture_rows()
    ledger = Ledger(rows)
    result = run(ledger, tmp_path)
    assert result.success
    assert [request.headers["range"] for request in ledger.requests] == ["0-9", "10-19"]

    games = tmp_path / "bronze" / "games"
    payload = (games / f"{DAY}.jsonl").read_bytes()
    assert payload == "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows).encode()
    assert {row["status"] for row in rows} == {"finished", "stopped", "active"}

    manifest = json.loads((games / f"{DAY}.manifest.json").read_text())
    assert manifest["row_count"] == 15
    assert manifest["sha256"] == hashlib.sha256(payload).hexdigest()
    assert manifest["min_updated_at"] == min(row["updated_at"] for row in rows)
    assert manifest["max_updated_at"] == max(row["updated_at"] for row in rows)
    assert "extracted_at" in manifest

    metadata = result.asset_materializations_for_node("bronze_games")[0].metadata
    assert metadata["row_count"].value == 15
    assert metadata["sha256"].value == manifest["sha256"]


def test_rerun_overwrites_with_identical_bytes(tmp_path):
    rows = fixture_rows()
    run(Ledger(rows), tmp_path)
    games = tmp_path / "bronze" / "games"
    first = (games / f"{DAY}.jsonl").read_bytes()
    first_manifest = json.loads((games / f"{DAY}.manifest.json").read_text())
    assert run(Ledger(rows), tmp_path).success
    assert (games / f"{DAY}.jsonl").read_bytes() == first
    second_manifest = json.loads((games / f"{DAY}.manifest.json").read_text())
    assert second_manifest["sha256"] == first_manifest["sha256"]
    assert sorted(path.name for path in games.iterdir()) == [f"{DAY}.jsonl", f"{DAY}.manifest.json"]


def test_retries_server_errors():
    ledger = Ledger(fixture_rows(), failures=2)
    rows = ledger_resource(ledger, page_size=1000).fetch_games(*day_window(DAY))
    assert len(rows) == 15
    assert len(ledger.requests) == 3


@pytest.mark.parametrize("exact_count", [True, False])
def test_pages_past_a_server_cap_below_the_page_size(exact_count):
    ledger = Ledger(fixture_rows(), max_rows=2, exact_count=exact_count)
    rows = ledger_resource(ledger, page_size=1000).fetch_games(*day_window(DAY))
    assert rows == fixture_rows()
    ranges = [request.headers["range"] for request in ledger.requests]
    # Each page holds 2 rows; without the count, one empty page ends the read.
    firsts = [*range(0, 16, 2), *([] if exact_count else [15])]
    assert ranges == [f"{first}-{first + 999}" for first in firsts]


def test_gives_up_after_three_retries():
    ledger = Ledger(fixture_rows(), failures=10)
    with pytest.raises(httpx.HTTPStatusError):
        ledger_resource(ledger).fetch_games(*day_window(DAY))
    assert len(ledger.requests) == 4


def test_empty_day_writes_empty_extract(tmp_path):
    result = materialize(
        [bronze_games],
        partition_key="2026-10-12",
        resources={"supabase": ledger_resource(Ledger(fixture_rows())), "paths": PathsResource(data_dir=str(tmp_path))},
    )
    assert result.success
    games = tmp_path / "bronze" / "games"
    assert (games / "2026-10-12.jsonl").read_bytes() == b""
    assert json.loads((games / "2026-10-12.manifest.json").read_text())["row_count"] == 0


def test_partition_range_writes_each_day_in_one_run(tmp_path):
    result = materialize(
        [bronze_games],
        tags={"dagster/asset_partition_range_start": "2026-10-12", "dagster/asset_partition_range_end": "2026-10-14"},
        resources={"supabase": ledger_resource(Ledger(fixture_rows())), "paths": PathsResource(data_dir=str(tmp_path))},
    )
    assert result.success
    games = tmp_path / "bronze" / "games"
    counts = {day: json.loads((games / f"{day}.manifest.json").read_text())["row_count"] for day in ("2026-10-12", "2026-10-13", "2026-10-14")}
    assert counts == {"2026-10-12": 0, "2026-10-13": 15, "2026-10-14": 0}
    metadata = result.asset_materializations_for_node("bronze_games")[0].metadata
    assert (metadata["row_count"].value, metadata["partitions"].value) == (15, 3)
