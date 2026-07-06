#!/usr/bin/env python3
"""Probe bounded selective resolves at audited trace states."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from environment.cfr.evaluator import TablebaseEvaluator, TerminalOnlyEvaluator, ValueNetEvaluator
from scripts.compare_checkpoints_ladder import _json_safe
from training.trace_policy_probe import parse_checkpoint_spec
from training.trace_resolve_probe import ResolveProbeSpec, probe_trace_resolves
from training.train_value_net import load_checkpoint, make_predict_fn


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace-report", required=True)
    parser.add_argument("--audit-report", required=True)
    parser.add_argument(
        "--horizon",
        type=int,
        action="append",
        default=None,
        help="Resolve horizon to probe. Repeatable. Defaults to 3.",
    )
    parser.add_argument(
        "--leaf-checkpoint",
        action="append",
        default=[],
        metavar="LABEL:PATH",
        help="Checkpoint-backed leaf evaluator to use. Repeatable.",
    )
    parser.add_argument(
        "--terminal-only",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Include terminal-only leaves. Used by default if no leaf checkpoint is passed.",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--out", required=True)
    return parser


def _build_evaluators(args: argparse.Namespace) -> list[ResolveProbeSpec]:
    specs: list[ResolveProbeSpec] = []
    if args.terminal_only or not args.leaf_checkpoint:
        specs.append(ResolveProbeSpec("terminal_only", TerminalOnlyEvaluator()))

    for raw in args.leaf_checkpoint:
        spec = parse_checkpoint_spec(raw)
        model = load_checkpoint(spec.checkpoint, device=args.device)
        evaluator = TablebaseEvaluator(
            ValueNetEvaluator(make_predict_fn(model, device=args.device))
        )
        specs.append(ResolveProbeSpec(spec.label, evaluator))
    return specs


def main() -> int:
    args = build_parser().parse_args()
    with Path(args.trace_report).open(encoding="utf-8") as fh:
        trace_report = json.load(fh)
    with Path(args.audit_report).open(encoding="utf-8") as fh:
        audit_report = json.load(fh)

    report = probe_trace_resolves(
        trace_report=trace_report,
        audit_report=audit_report,
        evaluators=_build_evaluators(args),
        horizons=tuple(args.horizon or [3]),
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
