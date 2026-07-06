#!/usr/bin/env python3
"""Merge saved ValueTarget corpora with exact duplicate-state dedupe."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from training.target_merge import merge_duplicate_targets
from training.value_targets import load_targets_as_records, save_targets


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    parser.add_argument("--report", default=None)
    parser.add_argument(
        "--dedupe-source",
        action="append",
        default=None,
        help="Only merge duplicate groups whose sources are all in this set. Repeatable.",
    )
    parser.add_argument("inputs", nargs="+")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    records = []
    input_counts = {}
    for path in args.inputs:
        loaded = load_targets_as_records(path)
        input_counts[path] = len(loaded)
        records.extend(loaded)

    merge_sources = set(args.dedupe_source) if args.dedupe_source else None
    merged, summary = merge_duplicate_targets(records, merge_sources=merge_sources)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    save_targets(merged, out)

    report = {
        "inputs": input_counts,
        "out": str(out),
        "merge_sources": sorted(merge_sources) if merge_sources is not None else None,
        "summary": summary.to_json(),
    }
    report_path = Path(args.report) if args.report else out.with_suffix(".merge_report.json")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(
        f"Wrote {len(merged)} records to {out} from {len(records)} inputs; "
        f"merged_groups={summary.merged_groups} "
        f"conflicting_groups={summary.conflicting_groups} "
        f"report={report_path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
