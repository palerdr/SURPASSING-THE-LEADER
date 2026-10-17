"""The bronze layer: one immutable JSONL extract of the ledger per UTC day.

Each partition holds the rows whose `updated_at` falls on that UTC day.
A day without rows gets an empty JSONL file and a manifest with row_count 0.
A rerun overwrites the day's files, so a rerun on unchanged rows writes the same bytes.
The extract keeps every status, including active games; dbt filters them downstream.
The single-run backfill policy lets one run extract a range of days,
as `dagster asset materialize --partition-range` requires.
"""


import hashlib
import json
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

from dagster import AssetExecutionContext, BackfillPolicy, DailyPartitionsDefinition, MaterializeResult, asset

from hal_loop.resources import PathsResource, SupabaseResource

# The hosted ledger recorded its first game on this UTC day.
LEDGER_START = "2026-09-18"
daily_partitions = DailyPartitionsDefinition(start_date=LEDGER_START, timezone="UTC")


def day_window(partition_key: str) -> tuple[str, str]:
    """Return the half-open UTC window [start, end) of one partition day."""
    day = date.fromisoformat(partition_key)
    start = datetime(day.year, day.month, day.day, tzinfo=timezone.utc)
    return start.isoformat(), (start + timedelta(days=1)).isoformat()


def _write_atomic(path: Path, data: bytes) -> None:
    staging = path.with_name(path.name + ".partial")
    staging.write_bytes(data)
    staging.replace(path)


def write_bronze_day(rows: list[dict], games_dir: Path, partition_key: str) -> dict:
    """Write one day's JSONL extract and its manifest, and return the manifest."""
    games_dir.mkdir(parents=True, exist_ok=True)
    payload = "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows).encode("utf-8")
    stamps = [str(row["updated_at"]) for row in rows]
    manifest = {
        "partition": partition_key,
        "row_count": len(rows),
        "min_updated_at": min(stamps) if stamps else None,
        "max_updated_at": max(stamps) if stamps else None,
        "sha256": hashlib.sha256(payload).hexdigest(),
        "extracted_at": datetime.now(timezone.utc).isoformat(),
    }
    _write_atomic(games_dir / f"{partition_key}.jsonl", payload)
    _write_atomic(
        games_dir / f"{partition_key}.manifest.json",
        (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode("utf-8"),
    )
    return manifest


@asset(
    partitions_def=daily_partitions,
    backfill_policy=BackfillPolicy.single_run(),
    group_name="bronze",
    compute_kind="python",
)
def bronze_games(
    context: AssetExecutionContext, supabase: SupabaseResource, paths: PathsResource
) -> MaterializeResult:
    """Extract each UTC day of the run's partitions to data/bronze/games/<day>.jsonl."""
    counts = {}
    for day in context.partition_keys:
        manifest = write_bronze_day(supabase.fetch_games(*day_window(day)), paths.bronze_dir / "games", day)
        counts[day] = manifest["row_count"]
        context.log.info(f"wrote {manifest['row_count']} rows for {day}")
    metadata = {"row_count": sum(counts.values()), "partitions": len(counts), "rows_per_day": counts}
    if len(counts) == 1:
        metadata["sha256"] = manifest["sha256"]
    return MaterializeResult(metadata=metadata)
