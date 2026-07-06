#!/usr/bin/env python3
"""Generate targeted MCTS-bootstrap targets from checkpoint trace regressions."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training.trace_bootstrap import (
    generate_trace_bootstrap_records,
    generate_trace_bootstrap_records_from_hints,
    save_trace_bootstrap_records,
)
from training.train_value_net import load_checkpoint, make_predict_fn
from training.value_targets import source_breakdown


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace-report", required=True)
    parser.add_argument(
        "--target-hints-report",
        default=None,
        help=(
            "Optional divergence-audit JSON. When set, labels the explicit "
            "target_hints instead of a trailing state window."
        ),
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--outcome-filter",
        choices=[
            "champion_win_candidate_loss",
            "candidate_win_champion_loss",
            "candidate_losses",
            "all",
        ],
        default="champion_win_candidate_loss",
    )
    parser.add_argument(
        "--trajectory",
        choices=["candidate", "champion", "both"],
        default="candidate",
    )
    parser.add_argument(
        "--state-window",
        type=int,
        default=3,
        help="Number of pre-decision states to label from the end of each selected game.",
    )
    parser.add_argument("--min-history", type=int, default=0)
    parser.add_argument("--max-states", type=int, default=None)
    parser.add_argument(
        "--critical-only",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Only label states flagged by environment.cfr.subgame_resolve.is_critical().",
    )
    parser.add_argument(
        "--subgame-resolve-at-critical",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use bounded exact subgame resolve inside MCTS at critical states.",
    )
    parser.add_argument(
        "--subgame-resolve-horizon",
        type=int,
        default=3,
        help="Selective-solve horizon for critical subgame resolve. Default 3.",
    )
    parser.add_argument("--exploration-c", type=float, default=1.0)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    with Path(args.trace_report).open(encoding="utf-8") as fh:
        trace_report = json.load(fh)

    model = load_checkpoint(args.checkpoint, device=args.device)
    predict_fn = make_predict_fn(model, device=args.device)
    if args.target_hints_report:
        with Path(args.target_hints_report).open(encoding="utf-8") as fh:
            hints_report = json.load(fh)
        records, summary = generate_trace_bootstrap_records_from_hints(
            trace_report=trace_report,
            target_hints=list(hints_report.get("target_hints", [])),
            predict_fn=predict_fn,
            checkpoint=args.checkpoint,
            iterations=args.iterations,
            seed=args.seed,
            max_states=args.max_states,
            critical_only=args.critical_only,
            subgame_resolve_at_critical=args.subgame_resolve_at_critical,
            subgame_resolve_horizon=args.subgame_resolve_horizon,
            exploration_c=args.exploration_c,
        )
    else:
        records, summary = generate_trace_bootstrap_records(
            trace_report=trace_report,
            predict_fn=predict_fn,
            checkpoint=args.checkpoint,
            iterations=args.iterations,
            seed=args.seed,
            outcome_filter=args.outcome_filter,
            trajectory=args.trajectory,
            state_window=args.state_window,
            min_history=args.min_history,
            max_states=args.max_states,
            critical_only=args.critical_only,
            subgame_resolve_at_critical=args.subgame_resolve_at_critical,
            subgame_resolve_horizon=args.subgame_resolve_horizon,
            exploration_c=args.exploration_c,
        )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    save_trace_bootstrap_records(records, out)
    print(
        f"Wrote {len(records)} trace bootstrap targets to {out}; "
        f"breakdown={source_breakdown([record.target for record in records])}; "
        f"summary={{'matched_games': {summary.matched_games}, "
        f"'states_considered': {summary.states_considered}, "
        f"'skipped_noncritical': {summary.skipped_noncritical}, "
        f"'skipped_duplicates': {summary.skipped_duplicates}, "
        f"'skipped_missing_games': {summary.skipped_missing_games}}}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
