#!/usr/bin/env python3
"""Summarize first divergences in matched checkpoint trace games."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.compare_checkpoints_ladder import _json_safe
from training.trace_divergence import analyze_trace_divergences


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace-report", required=True)
    parser.add_argument("--out", default=None)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    with Path(args.trace_report).open(encoding="utf-8") as fh:
        trace_report = json.load(fh)

    report = analyze_trace_divergences(trace_report)
    report["trace_report"] = args.trace_report

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8") as fh:
            json.dump(_json_safe(report), fh, indent=2, allow_nan=False)
        print(f"Report: {out}")
    else:
        print(json.dumps(_json_safe(report), indent=2, allow_nan=False))

    summary = report["summary"]
    print(
        "Summary: "
        f"paired={summary['paired_games']} "
        f"score_delta={summary['candidate_score_delta_total']} "
        f"outcomes={summary['outcomes']} "
        f"target_hints={summary['target_hints']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
