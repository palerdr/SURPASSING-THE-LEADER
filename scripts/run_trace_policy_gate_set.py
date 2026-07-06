#!/usr/bin/env python3
"""Probe and gate a rolling set of trace/audit policy checks for one candidate."""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.compare_checkpoints_ladder import _json_safe
from training.trace_policy_gate import (
    TracePolicyGateConfig,
    combine_trace_policy_gate_decisions,
    evaluate_trace_policy_probe,
)
from training.trace_policy_probe import (
    CheckpointProbeSpec,
    probe_trace_policies,
)


@dataclass(frozen=True)
class TraceAuditSpec:
    name: str
    trace_report: str
    audit_report: str


def parse_trace_audit_spec(raw: str) -> TraceAuditSpec:
    """Parse ``name|trace_report|audit_report`` specs.

    ``|`` avoids ambiguity with Windows drive-letter paths.
    """
    parts = raw.split("|")
    if len(parts) != 3 or not all(part.strip() for part in parts):
        raise ValueError(
            "--trace-audit must be formatted as 'name|trace_report|audit_report'"
        )
    return TraceAuditSpec(
        name=parts[0].strip(),
        trace_report=parts[1].strip(),
        audit_report=parts[2].strip(),
    )


def _split_csv(raw: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in raw.split(",") if item.strip())


def _safe_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("_") or "trace"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--trace-audit",
        action="append",
        required=True,
        metavar="NAME|TRACE|AUDIT",
        help="Rolling gate case. Repeatable. Quote this argument in PowerShell.",
    )
    parser.add_argument("--champion-checkpoint", required=True)
    parser.add_argument("--candidate-checkpoint", required=True)
    parser.add_argument("--candidate-label", default="candidate")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--policy-ensemble-size", type=int, default=1)
    parser.add_argument("--policy-uniform-mix", type=float, default=0.0)
    parser.add_argument("--search-prior-uniform-mix", type=float, default=0.0)
    parser.add_argument("--champion-search-prior-uniform-mix", type=float, default=None)
    parser.add_argument("--candidate-search-prior-uniform-mix", type=float, default=None)
    parser.add_argument(
        "--resolve-at-critical",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--resolve-horizon", type=int, default=3)
    parser.add_argument("--roles", default="trace")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--max-policy-tv", type=float, default=0.20)
    parser.add_argument("--max-abs-value-delta", type=float, default=0.15)
    parser.add_argument("--max-trace-champion-action-prob-drop", type=float, default=0.20)
    parser.add_argument("--max-trace-candidate-action-prob-rise", type=float, default=0.06)
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
    specs = [parse_trace_audit_spec(item) for item in args.trace_audit]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    gate_config = TracePolicyGateConfig(
        max_policy_tv=args.max_policy_tv,
        max_abs_value_delta=args.max_abs_value_delta,
        max_trace_champion_action_prob_drop=args.max_trace_champion_action_prob_drop,
        max_trace_candidate_action_prob_rise=args.max_trace_candidate_action_prob_rise,
        max_trace_candidate_action_probability=(
            args.max_trace_candidate_action_probability
        ),
    )
    checkpoints = [
        CheckpointProbeSpec(
            "champion",
            args.champion_checkpoint,
            search_prior_uniform_mix=(
                args.search_prior_uniform_mix
                if args.champion_search_prior_uniform_mix is None
                else args.champion_search_prior_uniform_mix
            ),
        ),
        CheckpointProbeSpec(
            args.candidate_label,
            args.candidate_checkpoint,
            search_prior_uniform_mix=(
                args.search_prior_uniform_mix
                if args.candidate_search_prior_uniform_mix is None
                else args.candidate_search_prior_uniform_mix
            ),
        ),
    ]

    per_report = []
    for spec in specs:
        with Path(spec.trace_report).open(encoding="utf-8") as fh:
            trace_report = json.load(fh)
        with Path(spec.audit_report).open(encoding="utf-8") as fh:
            audit_report = json.load(fh)
        probe = probe_trace_policies(
            trace_report=trace_report,
            audit_report=audit_report,
            checkpoints=checkpoints,
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
        probe_path = out_dir / f"{_safe_name(spec.name)}_probe.json"
        with probe_path.open("w", encoding="utf-8") as fh:
            json.dump(_json_safe(probe), fh, indent=2, allow_nan=False)
        decision = evaluate_trace_policy_probe(probe, gate_config)
        per_report.append(
            {
                "case": spec.name,
                "trace_report": spec.trace_report,
                "audit_report": spec.audit_report,
                "probe_report": str(probe_path),
                "decision": decision,
            }
        )
        print(f"{spec.name}: {decision['status']} {decision['metrics']}")

    decision = combine_trace_policy_gate_decisions(per_report, gate_config)
    report = {
        "config": vars(args),
        "trace_audits": [spec.__dict__ for spec in specs],
        "decision": decision,
        "per_report": per_report,
    }
    report_path = out_dir / "trace_policy_gate_set.json"
    with report_path.open("w", encoding="utf-8") as fh:
        json.dump(_json_safe(report), fh, indent=2, allow_nan=False)
    print(f"Trace policy gate set: {decision['status'].upper()}")
    for reason in decision["reasons"]:
        print(f"  reject: {reason}")
    print(f"Report: {report_path}")
    return 0 if decision["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
