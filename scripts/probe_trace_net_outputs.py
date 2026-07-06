#!/usr/bin/env python3
"""Probe raw ValueNet policy/value heads at audited trace states."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.compare_checkpoints_ladder import _json_safe
from training.trace_net_probe import probe_trace_net_outputs
from training.trace_policy_probe import parse_checkpoint_spec


def _split_csv(raw: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in raw.split(",") if item.strip())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace-report", required=True)
    parser.add_argument("--audit-report", required=True)
    parser.add_argument(
        "--checkpoint",
        action="append",
        required=True,
        metavar="LABEL:PATH",
        help="Checkpoint to probe. Repeatable; first checkpoint is baseline.",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--roles", default="trace")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--out", required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    with Path(args.trace_report).open(encoding="utf-8") as fh:
        trace_report = json.load(fh)
    with Path(args.audit_report).open(encoding="utf-8") as fh:
        audit_report = json.load(fh)

    report = probe_trace_net_outputs(
        trace_report=trace_report,
        audit_report=audit_report,
        checkpoints=[parse_checkpoint_spec(item) for item in args.checkpoint],
        device=args.device,
        roles=_split_csv(args.roles),
        top_k=args.top_k,
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as fh:
        json.dump(_json_safe(report), fh, indent=2, allow_nan=False)
    print(f"Report: {out}")
    print(f"Summary: {report['summary']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
