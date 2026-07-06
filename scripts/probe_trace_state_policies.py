#!/usr/bin/env python3
"""Probe deployed SolverAgent policy/value at audited trace states."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.compare_checkpoints_ladder import _json_safe
from training.trace_policy_probe import parse_checkpoint_spec, probe_trace_policies


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
        help="Checkpoint to probe. Repeatable; first checkpoint is the baseline.",
    )
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--policy-ensemble-size", type=int, default=1)
    parser.add_argument("--policy-uniform-mix", type=float, default=0.0)
    parser.add_argument("--search-prior-uniform-mix", type=float, default=0.0)
    parser.add_argument(
        "--resolve-at-critical",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--resolve-horizon", type=int, default=3)
    parser.add_argument(
        "--roles",
        default="trace",
        help="'trace', 'dropper', 'checker', or comma-separated roles.",
    )
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--out", required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    with Path(args.trace_report).open(encoding="utf-8") as fh:
        trace_report = json.load(fh)
    with Path(args.audit_report).open(encoding="utf-8") as fh:
        audit_report = json.load(fh)

    report = probe_trace_policies(
        trace_report=trace_report,
        audit_report=audit_report,
        checkpoints=[parse_checkpoint_spec(item) for item in args.checkpoint],
        iterations=args.iterations,
        seed=args.seed,
        policy_ensemble_size=args.policy_ensemble_size,
        policy_uniform_mix=args.policy_uniform_mix,
        search_prior_uniform_mix=args.search_prior_uniform_mix,
        resolve_at_critical=args.resolve_at_critical,
        resolve_horizon=args.resolve_horizon,
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
