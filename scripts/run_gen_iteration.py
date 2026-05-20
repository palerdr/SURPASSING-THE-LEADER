"""Run one AlphaZero-style generation step: MCTS-bootstrap → anchor merge →
class-rebalanced training → calibration gate.

Usage:
    python scripts/run_gen_iteration.py \
        --in-checkpoint checkpoints/gen1_v8/best.pt \
        --out-dir checkpoints/gen2/ \
        --out-targets checkpoints/gen2_targets.npz \
        --anchor-targets checkpoints/gen0_targets.npz \
        --held-out-targets checkpoints/holdout_targets.npz \
        --iterations 1000 \
        --epochs 150

Bakes in the hyperparameters that produced gen-1 v8 passing the calibration
gate: terminal=30, h2=10, h3=10 source weights, tablebase corpus replication
100x (because <5 tablebase records can't drive gradient via weight alone).
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from training.bootstrap_loop import (
    BootstrapConfig,
    CalibrationGateError,
    calibration_check,
    enforce_calibration_gate,
)
from training.train_value_net import (
    TrainConfig,
    load_checkpoint,
    make_predict_fn,
    train,
)
from training.value_targets import (
    SOURCE_EXACT_HORIZON_2,
    SOURCE_EXACT_HORIZON_3,
    SOURCE_TABLEBASE,
    SOURCE_TERMINAL,
    generate_mcts_bootstrap_targets,
    load_targets_as_records,
    save_targets,
    source_breakdown,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--in-checkpoint", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--out-targets", required=True)
    parser.add_argument("--anchor-targets", default=None)
    parser.add_argument("--held-out-targets", required=True)
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--tablebase-replicate", type=int, default=100)
    parser.add_argument("--terminal-weight", type=float, default=30.0)
    parser.add_argument("--horizon-weight", type=float, default=10.0)
    parser.add_argument("--policy-loss-weight", type=float, default=1.0)
    parser.add_argument("--prev-gen-holdout-mse", type=float, default=None)
    parser.add_argument("--tablebase-mse-threshold", type=float, default=0.01)
    parser.add_argument("--max-unresolved-per-source", type=float, default=0.35)
    parser.add_argument(
        "--subgame-resolve-at-critical",
        action="store_true",
        help="Use deeper root subgame re-solving for MCTS bootstrap labels at critical states.",
    )
    args = parser.parse_args()

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    print(f"[1/5] Loading evaluator: {args.in_checkpoint}", flush=True)
    model = load_checkpoint(args.in_checkpoint, device=args.device)
    predict_fn = make_predict_fn(model, device=args.device)

    print(
        f"[2/5] MCTS bootstrap sweep at {args.iterations} iter (this is the long step)",
        flush=True,
    )
    t0 = time.time()
    targets = generate_mcts_bootstrap_targets(
        predict_fn,
        iterations_per_state=args.iterations,
        seed=args.seed,
        subgame_resolve_at_critical=args.subgame_resolve_at_critical,
    )
    print(
        f"  Bootstrap done in {time.time() - t0:.1f}s; {len(targets)} records",
        flush=True,
    )

    if args.anchor_targets:
        print(f"[3/5] Merging anchors from: {args.anchor_targets}", flush=True)
        anchors = load_targets_as_records(args.anchor_targets)
        targets = targets + anchors
        print(f"  After anchor merge: {len(targets)} records", flush=True)

    if args.tablebase_replicate > 1:
        tb = [t for t in targets if t.source == SOURCE_TABLEBASE]
        if tb:
            replicas = tb * (args.tablebase_replicate - 1)
            targets = targets + replicas
            print(
                f"  Tablebase replicated {args.tablebase_replicate}x: {len(tb)} -> "
                f"{len(tb) * args.tablebase_replicate} records",
                flush=True,
            )

    print(f"  Final corpus: {len(targets)} records  breakdown: {source_breakdown(targets)}", flush=True)
    save_targets(targets, args.out_targets)
    print(f"  Saved: {args.out_targets}", flush=True)

    print(f"[4/5] Training on rebalanced corpus", flush=True)
    cfg = TrainConfig(
        epochs=args.epochs,
        seed=args.seed,
        device=args.device,
        source_weights=(
            (SOURCE_TERMINAL, args.terminal_weight),
            (SOURCE_EXACT_HORIZON_2, args.horizon_weight),
            (SOURCE_EXACT_HORIZON_3, args.horizon_weight),
        ),
        policy_loss_weight=args.policy_loss_weight,
        early_stopping_patience=40,
    )
    train_result = train(args.out_targets, args.out_dir, cfg)
    print(
        f"  Trained: best val MSE {train_result.best_val_mse:.5f} "
        f"at epoch {train_result.best_epoch}",
        flush=True,
    )
    print(f"  Per-source val MSE: {train_result.best_per_source_mse}", flush=True)

    print(f"[5/5] Calibration check vs held-out: {args.held_out_targets}", flush=True)
    held_out = load_targets_as_records(args.held_out_targets)
    report = calibration_check(
        checkpoint_path=train_result.checkpoint_path,
        held_out_targets=held_out,
        device=args.device,
    )
    print(f"  Overall MSE on held-out: {report.overall_mse:.5f}", flush=True)
    print(f"  Per-source MSE: {report.mse_per_source}", flush=True)
    print(
        f"  Mean unresolved per source: {report.mean_unresolved_probability_per_source}",
        flush=True,
    )

    gate_cfg = BootstrapConfig(
        tablebase_mse_threshold=args.tablebase_mse_threshold,
        max_unresolved_per_source=args.max_unresolved_per_source,
        required_sources=(
            SOURCE_TERMINAL,
            SOURCE_TABLEBASE,
            SOURCE_EXACT_HORIZON_2,
            SOURCE_EXACT_HORIZON_3,
        ),
    )
    try:
        enforce_calibration_gate(report, gate_cfg)
        print()
        print("CALIBRATION GATE PASSED", flush=True)
        if args.prev_gen_holdout_mse is not None:
            delta = (args.prev_gen_holdout_mse - report.overall_mse) / args.prev_gen_holdout_mse * 100
            verdict = "improvement" if report.overall_mse < args.prev_gen_holdout_mse else "regression"
            print(
                f"  vs prev-gen baseline {args.prev_gen_holdout_mse:.5f}: {delta:+.1f}% ({verdict})",
                flush=True,
            )
        return 0
    except CalibrationGateError as e:
        print()
        print(f"CALIBRATION GATE FAILED: {e}", flush=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
