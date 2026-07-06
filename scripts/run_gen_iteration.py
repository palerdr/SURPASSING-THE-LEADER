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
import json
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
    SOURCE_TABLEBASE_INTERIOR,
    SOURCE_TERMINAL,
    generate_mcts_bootstrap_targets,
    load_targets_as_records,
    save_targets,
    source_breakdown,
)


def monotonicity_verdict(new_mse: float, prev_mse: float) -> bool:
    """True iff the new generation strictly improved on the previous one.

    The AlphaZero acceptance rule (charter §2 criterion 5): the new gen's
    held-out overall MSE must be STRICTLY below the prior gen's. Equal or
    worse is a regression.
    """
    return new_mse < prev_mse


def parse_thresholds(items: list[str] | None) -> dict[str, float] | None:
    if not items:
        return None
    parsed: dict[str, float] = {}
    for item in items:
        if ":" not in item:
            raise SystemExit(
                f"--per-source-mse-threshold expects 'source_name:value', got {item!r}"
            )
        name, raw_value = item.rsplit(":", 1)
        try:
            parsed[name] = float(raw_value)
        except ValueError as exc:
            raise SystemExit(
                f"--per-source-mse-threshold {item!r}: value is not a float ({exc})"
            ) from exc
    return parsed


def parse_source_weights(items: list[str] | None) -> tuple[tuple[str, float], ...]:
    parsed = parse_thresholds(items)
    if not parsed:
        return ()
    return tuple(parsed.items())


def load_optional_json(path: str | None):
    if not path:
        return None
    with Path(path).open(encoding="utf-8") as fh:
        return json.load(fh)


def canary_gate_decision(canary_report: dict | None, *, min_wins_delta: int = 0) -> dict | None:
    """Return a pass/fail decision for an optional ladder-style canary report."""
    if canary_report is None:
        return None
    try:
        wins_delta = int(canary_report["aggregate"]["delta"]["wins_delta"])
    except (KeyError, TypeError, ValueError):
        return {
            "status": "failed",
            "reason": "canary report is missing aggregate.delta.wins_delta",
            "wins_delta": None,
            "min_wins_delta": int(min_wins_delta),
        }
    passed = wins_delta >= int(min_wins_delta)
    return {
        "status": "passed" if passed else "failed",
        "reason": None
        if passed
        else f"canary wins_delta {wins_delta} < required {int(min_wins_delta)}",
        "wins_delta": wins_delta,
        "min_wins_delta": int(min_wins_delta),
        "score_rate_delta": canary_report.get("aggregate", {})
        .get("delta", {})
        .get("score_rate_delta"),
    }


def write_gate_report(
    *,
    args,
    train_result,
    report,
    gate_passed: bool,
    gate_error: str | None,
    beats_prev_gen: bool | None,
    bootstrap_report: dict,
    training_corpus_report: dict,
) -> Path:
    """Write the calibration artifact consumed by promotion events."""
    out_dir = Path(args.out_dir)
    canary_result = load_optional_json(args.canary_report)
    canary_gate = canary_gate_decision(
        canary_result,
        min_wins_delta=getattr(args, "canary_min_wins_delta", 0),
    )
    summary = {
        "targets": args.out_targets,
        "train_targets": args.out_targets,
        "extra_targets": args.extra_targets or [],
        "out_dir": str(out_dir),
        "checkpoint": train_result.checkpoint_path,
        "config": {
            **vars(args),
            "source": "run_gen_iteration",
        },
        "train_best_val_mse": train_result.best_val_mse,
        "train_best_epoch": train_result.best_epoch,
        "train_selection_metric": args.selection_metric,
        "train_best_selection_score": train_result.best_selection_score,
        "train_best_per_source_mse": train_result.best_per_source_mse,
        "bootstrap": bootstrap_report,
        "training_corpus": training_corpus_report,
        "canary_result": canary_result,
        "canary_gate": canary_gate,
        "held_out_overall_mse": report.overall_mse,
        "held_out_mse_per_source": report.mse_per_source,
        "held_out_mean_unresolved_probability_per_source": (
            report.mean_unresolved_probability_per_source
        ),
        "gate_passed": gate_passed,
        "gate_error": gate_error,
        "beats_prev_gen": beats_prev_gen,
    }
    report_path = out_dir / "gate_report.json"
    with report_path.open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    return report_path


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
    parser.add_argument(
        "--reference-checkpoint",
        default=None,
        help="Optional checkpoint whose value/policy outputs define distillation penalties.",
    )
    parser.add_argument("--value-distill-weight", type=float, default=0.0)
    parser.add_argument("--policy-distill-weight", type=float, default=0.0)
    parser.add_argument("--tablebase-replicate", type=int, default=100)
    parser.add_argument(
        "--split-interior",
        action="store_true",
        help="Phase I-2 fix: label the F-2 interior pins as their own "
        "SOURCE_TABLEBASE_INTERIOR training class (instead of lumping them into "
        "SOURCE_TABLEBASE, where 660 boundary ±1 pins drown the 3 interior pins "
        "and saturate the net toward ±1). Pairs with the interior balancing block.",
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
    parser.add_argument(
        "--selection-metric",
        choices=("value_mse", "policy_nll", "value_plus_policy"),
        default="value_mse",
        help=(
            "Validation metric used to choose best.pt. Keep value_mse for "
            "calibration-first runs; use policy_nll only for explicit "
            "policy-head repair experiments."
        ),
    )
    parser.add_argument("--prev-gen-holdout-mse", type=float, default=None)
    parser.add_argument(
        "--enforce-monotonicity",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Fail (exit 1) when --prev-gen-holdout-mse is given and the new "
        "overall held-out MSE is not strictly below it (charter §2 criterion "
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
        "original). 128 gives 35.5K params (2.6× capacity) for fitting more "
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
        help="Use deeper root subgame re-solving for MCTS bootstrap labels at critical states.",
    )
    parser.add_argument(
        "--bootstrap-critical-only",
        action="store_true",
        help="Generate MCTS bootstrap labels only for states flagged by is_critical(). "
        "Anchor/tablebase classes are still merged afterward.",
    )
    parser.add_argument(
        "--subgame-resolve-horizon",
        type=int,
        default=3,
        help="Selective-solve horizon used with --subgame-resolve-at-critical. "
        "Default 3 keeps bounded critical-resolve experiments operational; "
        "use 4 only after horizon-3 canaries pass.",
    )
    parser.add_argument(
        "--bootstrap-max-states",
        type=int,
        default=None,
        help="Cap the number of non-anchor MCTS bootstrap states. Useful for bounded "
        "critical-resolve runs that would otherwise take hours without artifacts.",
    )
    parser.add_argument(
        "--source-weight",
        action="append",
        default=None,
        metavar="SOURCE:VALUE",
        help="Additional/override per-source value-loss weight. Repeatable.",
    )
    parser.add_argument(
        "--policy-source-weight",
        action="append",
        default=None,
        metavar="SOURCE:VALUE",
        help="Additional/override per-source policy-loss weight. Repeatable.",
    )
    parser.add_argument(
        "--force-train-source",
        action="append",
        default=None,
        metavar="SOURCE",
        help=(
            "Move validation feature groups containing this source into the "
            "training split. Useful for sparse trace-repair rows that must "
            "actually train."
        ),
    )
    parser.add_argument("--early-stopping-patience", type=int, default=40)
    parser.add_argument(
        "--canary-report",
        default=None,
        help="Optional JSON diagnostic to embed in gate_report.json, e.g. a "
        "pattern_reader smoke report run before full promotion.",
    )
    parser.add_argument(
        "--canary-min-wins-delta",
        type=int,
        default=0,
        help=(
            "Minimum aggregate wins_delta required when --canary-report is supplied. "
            "Default 0 rejects canary regressions."
        ),
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()

    per_source_mse_thresholds = parse_thresholds(args.per_source_mse_threshold)
    extra_source_weights = parse_source_weights(args.source_weight)
    extra_policy_source_weights = parse_source_weights(args.policy_source_weight)

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
        bootstrap_critical_only=args.bootstrap_critical_only,
        bootstrap_max_states=args.bootstrap_max_states,
        split_interior=args.split_interior,
    )
    bootstrap_runtime = time.time() - t0
    bootstrap_report = {
        "runtime_seconds": round(bootstrap_runtime, 3),
        "rows_generated": len(targets),
        "source_breakdown": source_breakdown(targets),
        "critical_only": args.bootstrap_critical_only,
        "max_states": args.bootstrap_max_states,
        "subgame_resolve_at_critical": args.subgame_resolve_at_critical,
        "subgame_resolve_horizon": args.subgame_resolve_horizon,
        "iterations_per_state": args.iterations,
    }
    print(
        f"  Bootstrap done in {bootstrap_runtime:.1f}s; {len(targets)} records",
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
                f"  Interior anchors: {len(interior)} unique × {interior_replicate} "
                f"(loss weight {interior_weight}) vs {n_boundary} boundary pins",
                flush=True,
            )

    print(f"  Final corpus: {len(targets)} records  breakdown: {source_breakdown(targets)}", flush=True)
    training_corpus_report = {
        "rows": len(targets),
        "source_breakdown": source_breakdown(targets),
    }
    save_targets(targets, args.out_targets)
    print(f"  Saved: {args.out_targets}", flush=True)

    print(f"[4/5] Training on rebalanced corpus (hidden_dim={args.hidden_dim})", flush=True)
    cfg = TrainConfig(
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        seed=args.seed,
        device=args.device,
        hidden_dim=args.hidden_dim,
        init_checkpoint=args.init_checkpoint,
        reference_checkpoint=args.reference_checkpoint,
        value_distill_weight=args.value_distill_weight,
        policy_distill_weight=args.policy_distill_weight,
        weight_decay=args.weight_decay,
        source_weights=(
            (SOURCE_TERMINAL, args.terminal_weight),
            (SOURCE_EXACT_HORIZON_2, args.horizon_weight),
            (SOURCE_EXACT_HORIZON_3, args.horizon_weight),
            (SOURCE_TABLEBASE, args.tablebase_weight),
            (SOURCE_TABLEBASE_INTERIOR, interior_weight),
            ("tier_a", args.tier_a_weight),
            *extra_source_weights,
        ),
        policy_loss_weight=args.policy_loss_weight,
        policy_source_weights=(
            ("tier_a", args.tier_a_policy_weight),
            *extra_policy_source_weights,
        ),
        early_stopping_patience=args.early_stopping_patience,
        selection_metric=args.selection_metric,
        force_train_sources=tuple(args.force_train_source or ()),
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
    beats_prev_gen = (
        monotonicity_verdict(report.overall_mse, args.prev_gen_holdout_mse)
        if args.prev_gen_holdout_mse is not None
        else None
    )
    canary_result = load_optional_json(args.canary_report)
    canary_gate = canary_gate_decision(
        canary_result,
        min_wins_delta=args.canary_min_wins_delta,
    )
    try:
        enforce_calibration_gate(report, gate_cfg)
        print()
        print("CALIBRATION GATE PASSED", flush=True)
        overall_gate_passed = True
        overall_gate_error = None
        if canary_gate is not None and canary_gate["status"] != "passed":
            overall_gate_passed = False
            overall_gate_error = str(canary_gate["reason"])
        report_path = write_gate_report(
            args=args,
            train_result=train_result,
            report=report,
            gate_passed=overall_gate_passed,
            gate_error=overall_gate_error,
            beats_prev_gen=beats_prev_gen,
            bootstrap_report=bootstrap_report,
            training_corpus_report=training_corpus_report,
        )
        print(f"  Gate report: {report_path}", flush=True)
        if canary_gate is not None:
            print(
                "  Canary gate: "
                f"{canary_gate['status'].upper()} "
                f"wins_delta={canary_gate['wins_delta']} "
                f"required>={canary_gate['min_wins_delta']}",
                flush=True,
            )
            if canary_gate["status"] != "passed":
                print(f"CANARY GATE FAILED: {canary_gate['reason']}", flush=True)
        if args.prev_gen_holdout_mse is not None:
            delta = (args.prev_gen_holdout_mse - report.overall_mse) / args.prev_gen_holdout_mse * 100
            verdict = "improvement" if beats_prev_gen else "regression"
            print(
                f"  vs prev-gen baseline {args.prev_gen_holdout_mse:.5f}: {delta:+.1f}% ({verdict})",
                flush=True,
            )
            if not beats_prev_gen and args.enforce_monotonicity:
                print(
                    "MONOTONICITY GATE FAILED: held-out MSE "
                    f"{report.overall_mse:.5f} >= prev-gen {args.prev_gen_holdout_mse:.5f} "
                    "(use --no-enforce-monotonicity to report without failing)",
                    flush=True,
                )
                return 1
        if not overall_gate_passed:
            return 1
        return 0
    except CalibrationGateError as e:
        report_path = write_gate_report(
            args=args,
            train_result=train_result,
            report=report,
            gate_passed=False,
            gate_error=str(e),
            beats_prev_gen=beats_prev_gen,
            bootstrap_report=bootstrap_report,
            training_corpus_report=training_corpus_report,
        )
        print()
        print(f"CALIBRATION GATE FAILED: {e}", flush=True)
        print(f"  Gate report: {report_path}", flush=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
