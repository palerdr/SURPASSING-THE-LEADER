#!/usr/bin/env python3
"""Generate policy/value targets from the Tier A interval tablebase.

Rows are emitted only for states whose Tier A root interval width is at or
below ``--max-width``. The value label is the interval midpoint; the policy
label is the root strategy from a one-ply selective solve whose frontier is
the Tier A evaluator. This is a training-data bridge from tablebase coverage
to the existing ValueNet policy/value heads.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from environment.cfr.evaluator import TerminalOnlyEvaluator
from environment.cfr.exact import ExactSearchConfig
from environment.cfr.selective import selective_solve
from hal.value_net import extract_features
from scripts.run_tier_a_decision_event import make_game, policy_vectors
from training.tablebase.tier_a import TierAEvaluator, TierALookup
from training.value_targets import ValueTarget, save_targets, source_breakdown


SOURCE_TIER_A = "tier_a"


def _linspace_ints(lo: int, hi: int, count: int) -> list[int]:
    if count <= 0:
        return []
    return sorted({int(round(x)) for x in np.linspace(lo, hi, count)})


def build_state_specs(args) -> list[dict]:
    """Return deterministic candidate specs for Tier A target sampling."""
    cylinders = sorted(
        set(
            _linspace_ints(0, 299, args.cylinder_points)
            + _linspace_ints(220, 249, args.near_boundary_points)
            + [0, 60, 120, 180, 220, 230, 235, 239, 240, 241, 250, 280, 299]
        )
    )
    ttds = sorted(set(_linspace_ints(60, 299, args.ttd_points) + [60, 120, 180, 239, 240, 299]))
    if args.ttd_min is not None:
        ttds = [ttd for ttd in ttds if ttd >= args.ttd_min]
    if args.ttd_max is not None:
        ttds = [ttd for ttd in ttds if ttd <= args.ttd_max]
    specs: list[dict] = []

    if args.death_filter in ("all", "d0"):
        for half in (1, 2):
            for ch in cylinders:
                for cb in cylinders:
                    specs.append({"half": half, "hal_cyl": ch, "baku_cyl": cb})

    if args.include_d1 and args.death_filter in ("all", "d1", "d1_hal", "d1_baku"):
        for ttd in ttds:
            for half in (1, 2):
                for ch in cylinders:
                    for cb in cylinders:
                        if args.death_filter in ("all", "d1", "d1_hal"):
                            specs.append(
                                {
                                    "half": half,
                                    "hal_cyl": ch,
                                    "baku_cyl": cb,
                                    "hal_deaths": 1,
                                    "hal_ttd": ttd,
                                }
                            )
                        if args.death_filter in ("all", "d1", "d1_baku"):
                            specs.append(
                                {
                                    "half": half,
                                    "hal_cyl": ch,
                                    "baku_cyl": cb,
                                    "baku_deaths": 1,
                                    "baku_ttd": ttd,
                                }
                            )
    rng = np.random.default_rng(args.seed)
    order = rng.permutation(len(specs))
    return [specs[int(i)] for i in order]


def generate_targets(args) -> tuple[list[ValueTarget], dict]:
    lookup = TierALookup(args.tier_a_dir, verify=args.verify_manifest)
    evaluator = TierAEvaluator(
        TerminalOnlyEvaluator(),
        lookup=lookup,
        max_width=args.runtime_width,
        use_midpoint_for_wide=True,
    )
    config = ExactSearchConfig()
    targets: list[ValueTarget] = []
    stats = {
        "candidates": 0,
        "hits": 0,
        "accepted": 0,
        "wide": 0,
        "miss_reasons": {},
        "accepted_by_artifact": {},
        "max_width_seen": 0.0,
    }

    for spec in build_state_specs(args):
        if args.limit is not None and stats["accepted"] >= args.limit:
            break
        stats["candidates"] += 1
        game = make_game(**spec)
        lookup_result = lookup.lookup(game)
        if lookup_result.interval is None:
            reason = lookup_result.miss_reason or "unknown"
            stats["miss_reasons"][reason] = stats["miss_reasons"].get(reason, 0) + 1
            continue
        interval = lookup_result.interval
        stats["hits"] += 1
        stats["max_width_seen"] = max(stats["max_width_seen"], interval.width)
        if interval.width > args.max_width:
            stats["wide"] += 1
            continue

        solve = selective_solve(game, args.policy_horizon, config, evaluator=evaluator)
        drop, check, drop_mask, check_mask = policy_vectors(game, solve)
        targets.append(
            ValueTarget(
                features=extract_features(game),
                value=float(interval.midpoint),
                source=args.source,
                horizon=args.policy_horizon,
                dropper_dist=drop,
                checker_dist=check,
                dropper_legal_mask=drop_mask,
                checker_legal_mask=check_mask,
                unresolved_probability=float(interval.width),
            )
        )
        stats["accepted"] += 1
        stats["accepted_by_artifact"][interval.source] = (
            stats["accepted_by_artifact"].get(interval.source, 0) + 1
        )

    return targets, stats


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=str(Path("checkpoints") / "tier_a_targets.npz"))
    parser.add_argument(
        "--source",
        default=SOURCE_TIER_A,
        help="Source tag written into each ValueTarget row.",
    )
    parser.add_argument("--tier-a-dir", default=str(Path("checkpoints") / "tablebase" / "tier_a"))
    parser.add_argument("--max-width", type=float, default=0.05)
    parser.add_argument("--runtime-width", type=float, default=0.10)
    parser.add_argument("--policy-horizon", type=int, default=1)
    parser.add_argument("--cylinder-points", type=int, default=24)
    parser.add_argument("--near-boundary-points", type=int, default=30)
    parser.add_argument("--ttd-points", type=int, default=10)
    parser.add_argument(
        "--death-filter",
        choices=("all", "d0", "d1", "d1_hal", "d1_baku"),
        default="all",
        help="Restrict generated candidate states by death epoch/owner.",
    )
    parser.add_argument("--ttd-min", type=int, default=None)
    parser.add_argument("--ttd-max", type=int, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--include-d1", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--verify-manifest", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    start = time.time()
    targets, stats = generate_targets(args)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    save_targets(targets, out)
    elapsed = time.time() - start

    print(f"Saved {len(targets)} Tier A targets to {out}")
    print(f"Breakdown: {source_breakdown(targets)}")
    print(f"Stats: {stats}")
    print(f"Elapsed: {elapsed:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
