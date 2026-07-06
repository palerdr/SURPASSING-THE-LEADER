#!/usr/bin/env python3
"""Run a repeatable Tier A frontier event and recommend the next long run.

This script compares ordinary depth-limited best-response brackets against
the same probe with Tier A intervals available at the truncation frontier.
It is meant to answer one question: is the tablebase useful enough as a
certified search frontier to justify the next expensive generation/training
run?
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stl.play.agent import DEFAULT_CHECKPOINT
from stl.commands.exploitability import SCENARIOS, build_policy
from stl.learning.strength import best_response_interval
from stl.solver.tablebase import TierALookup, frontier_interval_fn


DEFAULT_SCENARIOS = "postleap_230,postleap_d1_hal_120_230"
DEFAULT_POLICIES = "uniform,agent"
DEFAULT_ACCEPTED_HOLDOUT_MSE = 0.055905345689954


def _split_csv(raw: str) -> list[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


def _support_mass(policy_spec: str, explicit: float | None) -> float:
    if explicit is not None:
        return explicit
    return 0.999 if policy_spec == "net" else 1.0


def _interval(result) -> dict:
    return {
        "lo": float(result.lo),
        "hi": float(result.hi),
        "width": float(result.width),
        "states": int(result.states_solved),
        "engine_expansions": int(result.engine_expansions),
        "policy_queries": int(result.policy_queries),
        "frontier_hits": int(result.frontier_hits),
        "tablebase_frontier_hits": int(result.tablebase_frontier_hits),
        "elapsed_seconds": round(float(result.elapsed_seconds), 4),
    }


def run_case(args, lookup: TierALookup, *, scenario: str, policy_spec: str) -> dict:
    game = SCENARIOS[scenario]()
    policy, label = build_policy(
        policy_spec,
        iterations=args.iterations,
        seed=args.seed,
        checkpoint=args.checkpoint,
    )
    support_mass = _support_mass(policy_spec, args.support_mass)
    baseline = best_response_interval(
        game,
        policy,
        depth=args.depth,
        frozen_name=args.frozen,
        support_mass=support_mass,
        max_states=args.max_states,
    )
    tier_a = best_response_interval(
        game,
        policy,
        depth=args.depth,
        frozen_name=args.frozen,
        support_mass=support_mass,
        max_states=args.max_states,
        frontier_fn=frontier_interval_fn(lookup, max_width=args.tier_a_max_width),
    )
    coverage = 0.0
    if tier_a.frontier_hits:
        coverage = tier_a.tablebase_frontier_hits / tier_a.frontier_hits
    return {
        "scenario": scenario,
        "policy": policy_spec,
        "policy_label": label,
        "support_mass": support_mass,
        "baseline": _interval(baseline),
        "tier_a_frontier": _interval(tier_a),
        "width_reduction": float(baseline.width - tier_a.width),
        "width_reduction_fraction": (
            float((baseline.width - tier_a.width) / baseline.width)
            if baseline.width > 0.0
            else 0.0
        ),
        "tablebase_frontier_coverage": float(coverage),
    }


def summarize(rows: list[dict]) -> dict:
    reductions = [row["width_reduction"] for row in rows]
    fractions = [row["width_reduction_fraction"] for row in rows]
    coverages = [row["tablebase_frontier_coverage"] for row in rows]
    supported = [
        row
        for row in rows
        if row["tier_a_frontier"]["tablebase_frontier_hits"] > 0
    ]
    return {
        "cases": len(rows),
        "supported_cases": len(supported),
        "median_width_reduction": float(statistics.median(reductions)) if reductions else 0.0,
        "min_width_reduction": float(min(reductions)) if reductions else 0.0,
        "max_width_reduction": float(max(reductions)) if reductions else 0.0,
        "median_width_reduction_fraction": float(statistics.median(fractions)) if fractions else 0.0,
        "median_tablebase_frontier_coverage": float(statistics.median(coverages)) if coverages else 0.0,
        "total_frontier_hits": int(sum(row["tier_a_frontier"]["frontier_hits"] for row in rows)),
        "total_tablebase_frontier_hits": int(
            sum(row["tier_a_frontier"]["tablebase_frontier_hits"] for row in rows)
        ),
    }


def _width_tag(width: float) -> str:
    text = f"{width:.4f}".rstrip("0").rstrip(".")
    return "w" + text.replace(".", "")


def build_next_run(args, summary: dict) -> dict:
    width_tag = _width_tag(args.next_target_max_width)
    target_path = Path("checkpoints") / f"tier_a_targets_{args.next_target_count // 1000}k_{width_tag}.npz"
    out_dir = Path("checkpoints") / f"gen_tier_a_aux_{args.next_target_count // 1000}k_{width_tag}"
    out_targets = Path("checkpoints") / f"gen_tier_a_aux_{args.next_target_count // 1000}k_{width_tag}_targets.npz"

    generate_command = (
        f"python scripts/run_tier_a_targets.py --out {target_path} "
        f"--limit {args.next_target_count} --max-width {args.next_target_max_width} "
        f"--runtime-width 0.0 --policy-horizon 1 --include-d1 "
        f"--verify-manifest --seed {args.seed}"
    )
    train_command = (
        f"python scripts/run_gen_iteration.py --in-checkpoint {args.checkpoint} "
        f"--out-dir {out_dir} --out-targets {out_targets} "
        f"--anchor-targets checkpoints/ceiling_corpus.npz "
        f"--extra-targets {target_path} "
        f"--held-out-targets checkpoints/ceiling_holdout_clean.npz "
        f"--iterations {args.next_iterations} --epochs {args.next_epochs} "
        f"--hidden-dim 192 --split-interior --weight-decay 1e-4 "
        f"--tablebase-weight 15 --tier-a-weight {args.next_tier_a_weight} "
        f"--tier-a-policy-weight {args.next_tier_a_policy_weight} "
        f"--tier-a-replicate 1 --prev-gen-holdout-mse {args.prev_gen_holdout_mse} "
        f"--no-enforce-monotonicity"
    )

    enough_signal = (
        summary["supported_cases"] >= max(1, summary["cases"] // 2)
        and summary["median_width_reduction_fraction"] >= args.min_reduction_fraction
        and summary["median_tablebase_frontier_coverage"] >= args.min_coverage
    )
    if enough_signal:
        recommendation = (
            "Run a longer low-weight Tier A auxiliary generation/training pass. "
            "Keep it experimental and compare against the accepted checkpoint; "
            "do not enable Tier A runtime leaf replacement by default."
        )
        status = "ready_for_aux_generation"
    else:
        recommendation = (
            "Do not spend a long training run yet. Expand the tablebase frontier "
            "or run deeper exploitability probes until coverage and width "
            "reduction clear the thresholds."
        )
        status = "needs_more_frontier_signal"

    return {
        "status": status,
        "recommendation": recommendation,
        "target_generation_command": generate_command,
        "training_command": train_command,
        "acceptance_rule": (
            "Accept only if held-out overall MSE beats the prior checkpoint and "
            "per-source calibration remains inside gate thresholds."
        ),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--tier-a-dir", default=str(Path("checkpoints") / "tablebase" / "tier_a"))
    parser.add_argument("--out-dir", default=str(Path("checkpoints") / "tier_a_frontier_event"))
    parser.add_argument("--scenarios", default=DEFAULT_SCENARIOS)
    parser.add_argument("--policies", default=DEFAULT_POLICIES)
    parser.add_argument("--depth", type=int, default=1)
    parser.add_argument("--frozen", default="Hal", choices=("Hal", "Baku"))
    parser.add_argument("--support-mass", type=float, default=None)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-states", type=int, default=250_000)
    parser.add_argument("--tier-a-max-width", type=float, default=None)
    parser.add_argument("--min-reduction-fraction", type=float, default=0.25)
    parser.add_argument("--min-coverage", type=float, default=0.5)
    parser.add_argument("--next-target-count", type=int, default=50_000)
    parser.add_argument("--next-target-max-width", type=float, default=0.01)
    parser.add_argument("--next-iterations", type=int, default=300)
    parser.add_argument("--next-epochs", type=int, default=120)
    parser.add_argument("--next-tier-a-weight", type=float, default=0.10)
    parser.add_argument("--next-tier-a-policy-weight", type=float, default=0.25)
    parser.add_argument("--prev-gen-holdout-mse", type=float, default=DEFAULT_ACCEPTED_HOLDOUT_MSE)
    parser.add_argument("--quick", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.quick:
        args.policies = "uniform"
        args.scenarios = "postleap_230"
        args.iterations = min(args.iterations, 10)
        args.max_states = min(args.max_states, 50_000)

    scenarios = _split_csv(args.scenarios)
    policies = _split_csv(args.policies)
    unknown = [name for name in scenarios if name not in SCENARIOS]
    if unknown:
        raise SystemExit(f"unknown scenarios: {', '.join(unknown)}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    lookup = TierALookup(args.tier_a_dir, verify=True)
    start = time.time()
    rows = []
    for policy_spec in policies:
        for scenario in scenarios:
            print(f"running policy={policy_spec} scenario={scenario}", flush=True)
            rows.append(run_case(args, lookup, scenario=scenario, policy_spec=policy_spec))

    summary = summarize(rows)
    report = {
        "config": vars(args),
        "manifest_entries": len(lookup.verify_manifest()),
        "rows": rows,
        "summary": summary,
        "next_run": build_next_run(args, summary),
        "elapsed_seconds": round(time.time() - start, 2),
    }
    report_path = out_dir / "report.json"
    with report_path.open("w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)

    print(f"Tier A frontier event report: {report_path}")
    print(
        "summary: "
        f"supported={summary['supported_cases']}/{summary['cases']} "
        f"median_width_reduction={summary['median_width_reduction']:.4f} "
        f"median_fraction={summary['median_width_reduction_fraction']:.3f} "
        f"median_coverage={summary['median_tablebase_frontier_coverage']:.3f}"
    )
    print(f"next status: {report['next_run']['status']}")
    print(report["next_run"]["recommendation"])
    print("target generation:", report["next_run"]["target_generation_command"])
    print("training:", report["next_run"]["training_command"])
    print(f"elapsed: {report['elapsed_seconds']}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
