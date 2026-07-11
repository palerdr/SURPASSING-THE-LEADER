"""Run one AlphaZero-style generation step: MCTS-bootstrap -> anchor merge ->
class-rebalanced training -> calibration gate.

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

from stl.learning.bootstrap import (
    BootstrapConfig,
    CalibrationGateError,
    calibration_check,
    enforce_calibration_gate,
)
from stl.learning.train import (
    TrainConfig,
    load_checkpoint,
    make_predict_fn,
    train,
)
from stl.learning.targets import (
    SOURCE_EXACT_HORIZON_2,
    SOURCE_EXACT_HORIZON_3,
    SOURCE_TABLEBASE,
    SOURCE_TABLEBASE_INTERIOR,
    SOURCE_TERMINAL,
    generate_mcts_bootstrap_targets,
    load_targets_as_records,
    save_targets,
    source_breakdown,
)


def monotonicity_verdict(new_mse: float, prev_mse: float) -> bool:
    """True iff the new generation strictly improved on the previous one.

    The AlphaZero acceptance rule (charter section 2 criterion 5): the new gen's
    held-out overall MSE must be STRICTLY below the prior gen's. Equal or
    worse is a regression.
    """
    return new_mse < prev_mse


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--in-checkpoint", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--out-targets", required=True)
    parser.add_argument("--anchor-targets", default=None)
    parser.add_argument(
        "--extra-targets",
        action="append",
        default=None,
        help="Additional saved ValueTarget .npz files to append after anchors "
        "(for example Tier A policy/value targets). Repeatable.",
    )
    parser.add_argument("--held-out-targets", required=True)
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--init-checkpoint",
        default=None,
        help="Optional checkpoint to warm-start the trained generation from. "
        "The --in-checkpoint evaluator is still used to generate MCTS labels.",
    )
    parser.add_argument("--tablebase-replicate", type=int, default=100)
    parser.add_argument(
        "--split-interior",
        action="store_true",
        help="Phase I-2 fix: label the F-2 interior pins as their own "
        "SOURCE_TABLEBASE_INTERIOR training class (instead of lumping them into "
        "SOURCE_TABLEBASE, where 660 boundary +/-1 pins drown the 3 interior pins "
        "and saturate the net toward +/-1). Pairs with the interior balancing block.",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.0,
        help="Adam L2 regularization, to curb the wider net's boundary-tablebase "
        "overfit (hidden=192 held-out tablebase 0.0117 vs 0.0067). 0.0 = off.",
    )
    parser.add_argument("--terminal-weight", type=float, default=30.0)
    parser.add_argument("--horizon-weight", type=float, default=10.0)
    parser.add_argument(
        "--tablebase-weight",
        type=float,
        default=1.0,
        help="Per-source loss weight for boundary tablebase pins. Default 1.0 "
        "relies on replication alone, which the wider hidden=192 net overfits "
        "(held-out tablebase 0.0117 > 0.01 gate). 15.0 made it learn the boundary "
        "value structure that generalizes (held-out 0.0024). See I-2 progress log.",
    )
    parser.add_argument(
        "--interior-weight",
        type=float,
        default=2.0,
        help="Per-source loss weight for the split interior class (with "
        "--split-interior). Counters the boundary-pin imbalance.",
    )
    parser.add_argument("--policy-loss-weight", type=float, default=1.0)
    parser.add_argument("--prev-gen-holdout-mse", type=float, default=None)
    parser.add_argument(
        "--enforce-monotonicity",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Fail (exit 1) when --prev-gen-holdout-mse is given and the new "
        "overall held-out MSE is not strictly below it (charter section 2 criterion "
        "5: AlphaZero accept). --no-enforce-monotonicity restores the old "
        "report-only behavior.",
    )
    parser.add_argument("--tablebase-mse-threshold", type=float, default=0.01)
    parser.add_argument("--tier-a-weight", type=float, default=1.0)
    parser.add_argument("--tier-a-policy-weight", type=float, default=1.0)
    parser.add_argument("--tier-a-replicate", type=int, default=1)
    parser.add_argument("--max-unresolved-per-source", type=float, default=0.35)
    parser.add_argument(
        "--hidden-dim",
        type=int,
        default=64,
        help="ValueNet hidden width (Phase I). Default 64 (13.7K params, "
        "original). 128 gives 35.5K params (2.6x capacity) for fitting more "
        "diverse tablebase pins; 192 gives ~65.4K params (Phase I-2, under the "
        "raised 70K guard).",
    )
    parser.add_argument(
        "--per-source-mse-threshold",
        action="append",
        default=None,
        metavar="SOURCE:VALUE",
        help="Per-source MSE ceiling for the calibration gate. Format "
        "'source_name:value' (e.g. exact_horizon_3:0.15). Repeat for "
        "multiple. Catches the 'overfit tablebase, collapse h2/h3' "
        "failure mode that the strict tablebase-only gate misses.",
    )
    parser.add_argument(
        "--subgame-resolve-at-critical",
        action="store_true",
        help="Use bounded root subgame re-solving for MCTS bootstrap labels at critical states.",
    )
    parser.add_argument(
        "--subgame-resolve-horizon",
        type=int,
        default=1,
        help="Half-round horizon for critical root resolve. Default 1 bounds the resolve to the current turn.",
    )
    parser.add_argument(
        "--subgame-resolve-cfr-iters",
        type=int,
        default=2000,
        help="CFR+ iterations for the bounded critical root resolve.",
    )
    parser.add_argument(
        "--bootstrap-critical-only",
        action="store_true",
        help="Generate MCTS bootstrap labels only for states flagged by is_critical(). "
        "Anchor/tablebase classes are still merged afterward.",
    )
    parser.add_argument(
        "--bootstrap-max-states",
        type=int,
        default=None,
        help="Cap the number of non-anchor MCTS bootstrap states. Useful for bounded "
        "critical-resolve runs that would otherwise take hours without artifacts.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()

    per_source_mse_thresholds: dict[str, float] | None = None
    if args.per_source_mse_threshold:
        per_source_mse_thresholds = {}
        for item in args.per_source_mse_threshold:
            if ":" not in item:
                raise SystemExit(
                    f"--per-source-mse-threshold expects 'source_name:value', got {item!r}"
                )
            name, raw_value = item.rsplit(":", 1)
            try:
                per_source_mse_thresholds[name] = float(raw_value)
            except ValueError as exc:
                raise SystemExit(
                    f"--per-source-mse-threshold {item!r}: value is not a float ({exc})"
                ) from exc

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    print(f"[1/5] Loading evaluator: {args.in_checkpoint}", flush=True)
    model = load_checkpoint(args.in_checkpoint, device=args.device)
    predict_fn = make_predict_fn(model, device=args.device)

    print(
        f"[2/5] MCTS bootstrap sweep at {args.iterations} iter (this is the long step)",
        flush=True,
    )
    if args.bootstrap_critical_only or args.bootstrap_max_states is not None:
        print(
            "  Bootstrap scope: "
            f"critical_only={args.bootstrap_critical_only} "
            f"max_states={args.bootstrap_max_states}",
            flush=True,
        )
    t0 = time.time()
    targets = generate_mcts_bootstrap_targets(
        predict_fn,
        iterations_per_state=args.iterations,
        seed=args.seed,
        subgame_resolve_at_critical=args.subgame_resolve_at_critical,
        subgame_resolve_horizon=args.subgame_resolve_horizon,
        subgame_resolve_cfr_iters=args.subgame_resolve_cfr_iters,
        bootstrap_critical_only=args.bootstrap_critical_only,
        bootstrap_max_states=args.bootstrap_max_states,
        split_interior=args.split_interior,
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
    else:
        print("[3/5] No anchor targets supplied", flush=True)

    if args.extra_targets:
        for path in args.extra_targets:
            extra = load_targets_as_records(path)
            if args.tier_a_replicate > 1:
                tier_a = [t for t in extra if t.source == "tier_a"]
                if tier_a:
                    extra = extra + tier_a * (args.tier_a_replicate - 1)
                    print(
                        f"  Tier A replicated {args.tier_a_replicate}x in {path}: "
                        f"{len(tier_a)} -> {len(tier_a) * args.tier_a_replicate} records",
                        flush=True,
                    )
            targets = targets + extra
            print(f"  Added extra targets from {path}: +{len(extra)} records", flush=True)

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

    # ── Interior-anchor balancing (Phase I-2 fix) ──────────────────────────────
    # With --split-interior the 3 interior pins are their own class. But the
    # boundary-tablebase block just replicated to ~660 records; left at 3 records
    # weight 1.0, the interior pins are drowned ~220:1 and the net saturates them
    # toward ±1 (hidden=192 predicted +0.567 for the −0.373 target). Counter that.
    interior_weight = 1.0
    if args.split_interior:
        interior = [t for t in targets if t.source == SOURCE_TABLEBASE_INTERIOR]
        n_boundary = sum(1 for t in targets if t.source == SOURCE_TABLEBASE)
        if interior:
            # Balance interior against the ~n_boundary boundary pins on two axes:
            #  (1) record count — replicate the 3 unique interior pins up to the
            #      boundary count so neither class dominates by sheer volume (the
            #      original 90:660 = 12% share is what drowned them);
            #  (2) gradient priority — a modest extra loss weight, because the
            #      interior targets (+0.568 / −0.373 / −0.024) sit in the hard,
            #      non-saturated middle of tanh and need more gradient per sample
            #      than the easy ±1 boundary pins. Memorization target ≤ 0.05.
            interior_replicate = max(1, round(n_boundary / len(interior)))
            interior_weight = args.interior_weight

            if interior_replicate > 1:
                targets = targets + interior * (interior_replicate - 1)
            print(
                f"  Interior anchors: {len(interior)} unique x {interior_replicate} "
                f"(loss weight {interior_weight}) vs {n_boundary} boundary pins",
                flush=True,
            )

    print(f"  Final corpus: {len(targets)} records  breakdown: {source_breakdown(targets)}", flush=True)
    save_targets(targets, args.out_targets)
    print(f"  Saved: {args.out_targets}", flush=True)

    print(f"[4/5] Training on rebalanced corpus (hidden_dim={args.hidden_dim})", flush=True)
    cfg = TrainConfig(
        allow_legacy_targets=True,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        seed=args.seed,
        device=args.device,
        hidden_dim=args.hidden_dim,
        init_checkpoint=args.init_checkpoint,
        weight_decay=args.weight_decay,
        source_weights=(
            (SOURCE_TERMINAL, args.terminal_weight),
            (SOURCE_EXACT_HORIZON_2, args.horizon_weight),
            (SOURCE_EXACT_HORIZON_3, args.horizon_weight),
            (SOURCE_TABLEBASE, args.tablebase_weight),
            (SOURCE_TABLEBASE_INTERIOR, interior_weight),
            ("tier_a", args.tier_a_weight),
        ),
        policy_loss_weight=args.policy_loss_weight,
        policy_source_weights=(("tier_a", args.tier_a_policy_weight),),
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
        per_source_mse_thresholds=per_source_mse_thresholds,
    )
    try:
        enforce_calibration_gate(report, gate_cfg)
        print()
        print("CALIBRATION GATE PASSED", flush=True)
        if args.prev_gen_holdout_mse is not None:
            improved = monotonicity_verdict(report.overall_mse, args.prev_gen_holdout_mse)
            delta = (args.prev_gen_holdout_mse - report.overall_mse) / args.prev_gen_holdout_mse * 100
            verdict = "improvement" if improved else "regression"
            print(
                f"  vs prev-gen baseline {args.prev_gen_holdout_mse:.5f}: {delta:+.1f}% ({verdict})",
                flush=True,
            )
            if not improved and args.enforce_monotonicity:
                print(
                    "MONOTONICITY GATE FAILED: held-out MSE "
                    f"{report.overall_mse:.5f} >= prev-gen {args.prev_gen_holdout_mse:.5f} "
                    "(use --no-enforce-monotonicity to report without failing)",
                    flush=True,
                )
                return 1
        return 0
    except CalibrationGateError as e:
        print()
        print(f"CALIBRATION GATE FAILED: {e}", flush=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
