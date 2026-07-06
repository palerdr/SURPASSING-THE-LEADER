#!/usr/bin/env python3
"""Combine compatible checkpoint ladder reports without rerunning games."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.compare_checkpoints_ladder import _aggregate_reports, _json_safe


IGNORED_CONFIG_KEYS = {"seed", "seeds", "out"}


def load_json(path: str | Path) -> dict:
    with Path(path).open(encoding="utf-8") as fh:
        return json.load(fh)


def _relevant_config(config: dict) -> dict:
    return {
        key: value
        for key, value in config.items()
        if key not in IGNORED_CONFIG_KEYS
    }


def combine_reports(paths: list[str | Path]) -> dict:
    if not paths:
        raise ValueError("at least one report is required")

    reports = [load_json(path) for path in paths]
    base = reports[0]
    base_config = _relevant_config(base.get("config", {}))
    opponents = list(base.get("opponents", []))
    if not opponents:
        raise ValueError("base report has no opponents")

    seed_reports: list[dict] = []
    seen_seeds: set[int] = set()
    elapsed = 0.0
    for path, report in zip(paths, reports):
        config = _relevant_config(report.get("config", {}))
        if config != base_config:
            raise ValueError(f"config mismatch in {path}")
        report_opponents = list(report.get("opponents", []))
        if report_opponents != opponents:
            raise ValueError(f"opponent mismatch in {path}")
        elapsed += float(report.get("elapsed_seconds", 0.0))
        for seed_report in report.get("seed_reports", []):
            seed = int(seed_report["seed"])
            if seed in seen_seeds:
                raise ValueError(f"duplicate seed {seed} in {path}")
            seen_seeds.add(seed)
            seed_reports.append(seed_report)

    seed_reports.sort(key=lambda row: int(row["seed"]))
    seeds = [int(row["seed"]) for row in seed_reports]
    config = dict(base.get("config", {}))
    config["seed"] = seeds[0] if seeds else config.get("seed")
    config["seeds"] = ",".join(str(seed) for seed in seeds)
    return {
        "config": config,
        "seeds": seeds,
        "opponents": opponents,
        "seed_reports": seed_reports,
        "aggregate": _aggregate_reports(seed_reports, opponents),
        "elapsed_seconds": round(elapsed, 2),
        "combined_from": [str(path) for path in paths],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", action="append", required=True)
    parser.add_argument("--out", required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    combined = combine_reports(args.report)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    combined["config"]["out"] = str(out)
    with out.open("w", encoding="utf-8") as fh:
        json.dump(_json_safe(combined), fh, indent=2, allow_nan=False)
    print(f"Combined {len(args.report)} reports into {out}")
    print(f"Aggregate delta: {combined['aggregate']['delta']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
