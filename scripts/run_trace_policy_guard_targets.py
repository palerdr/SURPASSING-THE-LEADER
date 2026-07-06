#!/usr/bin/env python3
"""Generate policy_guard targets from audited checkpoint trace divergences."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hal.agent import SolverAgent
from training.policy_guard import save_policy_guard_records
from training.trace_policy_guard import generate_trace_policy_guard_records
from training.value_targets import SOURCE_POLICY_GUARD, VALID_SOURCES, source_breakdown


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace-report", required=True)
    parser.add_argument("--audit-report", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--policy-ensemble-size", type=int, default=1)
    parser.add_argument("--policy-uniform-mix", type=float, default=0.0)
    parser.add_argument(
        "--resolve-at-critical",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--resolve-horizon", type=int, default=3)
    parser.add_argument("--max-states", type=int, default=None)
    parser.add_argument(
        "--source",
        default=SOURCE_POLICY_GUARD,
        choices=VALID_SOURCES,
        help="Source label to write into the generated ValueTarget rows.",
    )
    parser.add_argument(
        "--critical-only",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    with Path(args.trace_report).open(encoding="utf-8") as fh:
        trace_report = json.load(fh)
    with Path(args.audit_report).open(encoding="utf-8") as fh:
        audit_report = json.load(fh)

    label_agent = SolverAgent(
        args.checkpoint,
        player_name="Hal",
        iterations=args.iterations,
        seed=args.seed,
        policy_ensemble_size=args.policy_ensemble_size,
        policy_uniform_mix=args.policy_uniform_mix,
        resolve_at_critical=args.resolve_at_critical,
        resolve_horizon=args.resolve_horizon,
    )
    records, summary = generate_trace_policy_guard_records(
        trace_report=trace_report,
        audit_report=audit_report,
        label_agent=label_agent,
        label_checkpoint=args.checkpoint,
        source=args.source,
        max_states=args.max_states,
        critical_only=args.critical_only,
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    save_policy_guard_records(records, out)
    print(
        f"Wrote {len(records)} trace policy_guard targets to {out}; "
        f"breakdown={source_breakdown([record.target for record in records])}; "
        f"summary={{'states_considered': {summary.states_considered}, "
        f"'skipped_duplicates': {summary.skipped_duplicates}, "
        f"'skipped_missing_games': {summary.skipped_missing_games}, "
        f"'skipped_noncritical': {summary.skipped_noncritical}}}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
