#!/usr/bin/env python3
"""Gate a trace-state policy probe report before expensive canaries."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from training.trace_policy_gate import (
    TracePolicyGateConfig,
    combine_trace_policy_gate_decisions,
    evaluate_trace_policy_probe,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--probe-report",
        action="append",
        required=True,
        help="Probe report to gate. Repeatable for rolling trace-gate sets.",
    )
    parser.add_argument("--out", default=None)
    parser.add_argument("--max-policy-tv", type=float, default=0.20)
    parser.add_argument("--max-abs-value-delta", type=float, default=0.15)
    parser.add_argument(
        "--max-trace-champion-action-prob-drop",
        type=float,
        default=0.20,
    )
    parser.add_argument(
        "--max-trace-candidate-action-prob-rise",
        type=float,
        default=0.06,
    )
    parser.add_argument(
        "--max-trace-candidate-action-probability",
        type=float,
        default=None,
        help=(
            "Optional absolute cap on the candidate checkpoint's probability "
            "for the action that the regressing trace actually took."
        ),
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    config = TracePolicyGateConfig(
        max_policy_tv=args.max_policy_tv,
        max_abs_value_delta=args.max_abs_value_delta,
        max_trace_champion_action_prob_drop=(
            args.max_trace_champion_action_prob_drop
        ),
        max_trace_candidate_action_prob_rise=(
            args.max_trace_candidate_action_prob_rise
        ),
        max_trace_candidate_action_probability=(
            args.max_trace_candidate_action_probability
        ),
    )
    per_report = []
    for probe_path in args.probe_report:
        with Path(probe_path).open(encoding="utf-8") as fh:
            probe_report = json.load(fh)
        report_decision = evaluate_trace_policy_probe(probe_report, config)
        per_report.append({"probe_report": probe_path, "decision": report_decision})
    decision = combine_trace_policy_gate_decisions(per_report, config)
    report = {
        "probe_reports": args.probe_report,
        "decision": decision,
        "per_report": per_report,
    }
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"Report: {out}")
    print(f"Trace policy gate: {decision['status'].upper()}")
    for reason in decision["reasons"]:
        print(f"  reject: {reason}")
    print(f"Metrics: {decision['metrics']}")
    return 0 if decision["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
