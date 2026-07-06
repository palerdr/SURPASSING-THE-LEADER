#!/usr/bin/env python3
"""Train/evaluate a ValueNet pilot with additional Tier A targets."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stl.learning.bootstrap import calibration_check
from stl.learning.train import TrainConfig, train
from stl.learning.targets import (
    SOURCE_EXACT_HORIZON_2,
    SOURCE_EXACT_HORIZON_3,
    SOURCE_TABLEBASE,
    SOURCE_TABLEBASE_INTERIOR,
    SOURCE_TERMINAL,
    load_targets_as_records,
    save_targets,
    source_breakdown,
    _generate_tablebase_targets,
    _generate_terminal_targets,
    TRAINING_TERMINAL_CONFIGS,
)
from stl.solver.exact import ExactSearchConfig


SOURCE_TIER_A = "tier_a"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-targets", default=str(Path("checkpoints") / "ceiling_corpus.npz"))
    parser.add_argument("--tier-a-targets", required=True)
    parser.add_argument("--held-out-targets", default=str(Path("checkpoints") / "ceiling_holdout_clean.npz"))
    parser.add_argument("--out-dir", default=str(Path("checkpoints") / "tier_a_training_pilot"))
    parser.add_argument("--merged-targets", default=None)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--hidden-dim", type=int, default=192)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--policy-loss-weight", type=float, default=1.0)
    parser.add_argument("--tier-a-weight", type=float, default=5.0)
    parser.add_argument("--tier-a-policy-weight", type=float, default=1.0)
    parser.add_argument("--tablebase-weight", type=float, default=15.0)
    parser.add_argument("--tablebase-replicate", type=int, default=30)
    parser.add_argument("--interior-replicate", type=int, default=30)
    parser.add_argument("--terminal-replicate", type=int, default=1)
    parser.add_argument("--add-anchor-classes", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--terminal-weight", type=float, default=30.0)
    parser.add_argument("--horizon-weight", type=float, default=10.0)
    parser.add_argument("--interior-weight", type=float, default=2.0)
    parser.add_argument("--prev-best-mse", type=float, default=0.055905345689954)
    args = parser.parse_args()

    start = time.time()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    merged_path = Path(args.merged_targets) if args.merged_targets else out_dir / "merged_targets.npz"

    base = load_targets_as_records(args.base_targets)
    tier_a = load_targets_as_records(args.tier_a_targets)
    merged = base + tier_a
    if args.add_anchor_classes:
        cfg = ExactSearchConfig()
        merged.extend(_generate_terminal_targets(cfg, TRAINING_TERMINAL_CONFIGS))
        merged.extend(_generate_tablebase_targets(cfg, split_interior=True))
        tablebase = [t for t in merged if t.source == SOURCE_TABLEBASE]
        interior = [t for t in merged if t.source == SOURCE_TABLEBASE_INTERIOR]
        terminal = [t for t in merged if t.source == SOURCE_TERMINAL]
        if args.tablebase_replicate > 1 and tablebase:
            merged.extend(tablebase * (args.tablebase_replicate - 1))
        if args.interior_replicate > 1 and interior:
            merged.extend(interior * (args.interior_replicate - 1))
        if args.terminal_replicate > 1 and terminal:
            merged.extend(terminal * (args.terminal_replicate - 1))
    save_targets(merged, merged_path)

    cfg = TrainConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        seed=args.seed,
        device=args.device,
        hidden_dim=args.hidden_dim,
        weight_decay=args.weight_decay,
        policy_loss_weight=args.policy_loss_weight,
        policy_source_weights=((SOURCE_TIER_A, args.tier_a_policy_weight),),
        early_stopping_patience=max(10, min(40, args.epochs // 2)),
        source_weights=(
            (SOURCE_TERMINAL, args.terminal_weight),
            (SOURCE_EXACT_HORIZON_2, args.horizon_weight),
            (SOURCE_EXACT_HORIZON_3, args.horizon_weight),
            (SOURCE_TABLEBASE, args.tablebase_weight),
            (SOURCE_TABLEBASE_INTERIOR, args.interior_weight),
            (SOURCE_TIER_A, args.tier_a_weight),
        ),
    )
    result = train(merged_path, out_dir, cfg)
    held_out = load_targets_as_records(args.held_out_targets)
    report = calibration_check(
        checkpoint_path=result.checkpoint_path,
        held_out_targets=held_out,
        device=args.device,
    )

    summary = {
        "base_targets": args.base_targets,
        "tier_a_targets": args.tier_a_targets,
        "merged_targets": str(merged_path),
        "out_dir": str(out_dir),
        "source_breakdown": source_breakdown(merged),
        "train": {
            "best_val_mse": result.best_val_mse,
            "best_epoch": result.best_epoch,
            "best_per_source_mse": result.best_per_source_mse,
            "checkpoint": result.checkpoint_path,
        },
        "held_out": {
            "overall_mse": report.overall_mse,
            "mse_per_source": report.mse_per_source,
            "mean_unresolved_probability_per_source": report.mean_unresolved_probability_per_source,
            "improved_vs_prev_best": report.overall_mse < args.prev_best_mse,
            "prev_best_mse": args.prev_best_mse,
        },
        "elapsed_seconds": round(time.time() - start, 2),
    }
    with (out_dir / "summary.json").open("w") as fh:
        json.dump(summary, fh, indent=2)

    print(f"Merged targets: {merged_path}")
    print(f"Breakdown: {summary['source_breakdown']}")
    print(f"Best train-val MSE: {result.best_val_mse:.5f} at epoch {result.best_epoch}")
    print(f"Held-out overall MSE: {report.overall_mse:.5f}")
    print(f"Held-out per-source MSE: {report.mse_per_source}")
    print(f"Improved vs {args.prev_best_mse:.5f}: {summary['held_out']['improved_vs_prev_best']}")
    print(f"Summary: {out_dir / 'summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
