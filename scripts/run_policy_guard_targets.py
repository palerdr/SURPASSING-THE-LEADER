#!/usr/bin/env python3
"""Generate champion policy_guard targets for drift-sensitive scenarios."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.run_exploitability import SCENARIOS
from scripts.run_tier_a_frontier_event import _split_csv
from training.policy_guard import (
    generate_policy_guard_records,
    save_policy_guard_records,
)
from training.value_targets import source_breakdown


DEFAULT_SCENARIOS = "opening,postleap_230,postleap_d1_hal_120_230"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--policy-ensemble-size", type=int, default=1)
    parser.add_argument("--policy-uniform-mix", type=float, default=0.0)
    parser.add_argument(
        "--resolve-at-critical",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use SolverAgent critical-state subgame resolve while labeling policies.",
    )
    parser.add_argument(
        "--resolve-horizon",
        type=int,
        default=3,
        help="Selective-solve horizon for --resolve-at-critical.",
    )
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--scenarios", default=DEFAULT_SCENARIOS)
    parser.add_argument("--player", default="Hal", choices=("Hal", "Baku"))
    parser.add_argument(
        "--include-children",
        action="store_true",
        help="Also label one-step non-terminal children under top root average actions.",
    )
    parser.add_argument(
        "--child-top-k",
        type=int,
        default=2,
        help="Number of top root seconds per role to branch when --include-children is set.",
    )
    parser.add_argument("--out", required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    scenario_names = _split_csv(args.scenarios)
    unknown = [name for name in scenario_names if name not in SCENARIOS]
    if unknown:
        raise SystemExit(f"unknown scenarios: {', '.join(unknown)}")
    seeds = [int(item.strip()) for item in args.seeds.split(",") if item.strip()]
    factories = {name: SCENARIOS[name] for name in scenario_names}

    records = generate_policy_guard_records(
        checkpoint=args.checkpoint,
        scenario_factories=factories,
        seeds=seeds,
        iterations=args.iterations,
        player=args.player,
        include_children=args.include_children,
        child_top_k=args.child_top_k,
        policy_ensemble_size=args.policy_ensemble_size,
        policy_uniform_mix=args.policy_uniform_mix,
        resolve_at_critical=args.resolve_at_critical,
        resolve_horizon=args.resolve_horizon,
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    save_policy_guard_records(records, out)
    print(
        f"Wrote {len(records)} policy_guard targets to {out}; "
        f"breakdown={source_breakdown([record.target for record in records])}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
