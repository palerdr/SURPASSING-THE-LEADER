#!/usr/bin/env python3
"""Train ValueNet on an existing target corpus and run the calibration gate.

This is the training-only companion to ``scripts/run_gen_iteration.py``. Use it
after an expensive generation command has already written ``--targets`` and the
remaining question is whether different source weights can pass the held-out
gate without regenerating MCTS bootstrap labels.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from training.bootstrap_loop import (
    BootstrapConfig,
    CalibrationGateError,
    calibration_check,
    enforce_calibration_gate,
)
from training.target_merge import merge_duplicate_targets
from training.train_value_net import TrainConfig, train
from training.value_targets import (
    SOURCE_EXACT_HORIZON_2,
    SOURCE_EXACT_HORIZON_3,
    SOURCE_MCTS_BOOTSTRAP,
    SOURCE_POLICY_GUARD,
    SOURCE_TABLEBASE,
    SOURCE_TABLEBASE_INTERIOR,
    SOURCE_TERMINAL,
    load_targets_as_records,
    save_targets,
    source_breakdown,
)


def monotonicity_verdict(new_mse: float, prev_mse: float) -> bool:
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
        source, raw_value = item.rsplit(":", 1)
        try:
            parsed[source] = float(raw_value)
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--targets", required=True)
    parser.add_argument(
        "--extra-targets",
        action="append",
        default=None,
        help="Additional saved ValueTarget .npz files to append before training. Repeatable.",
    )
    parser.add_argument(
        "--dedupe-targets",
        action="store_true",
        help=(
            "Average exact duplicate same-contract target rows after loading "
            "--targets and --extra-targets. Writes target_merge_report.json."
        ),
    )
    parser.add_argument(
        "--dedupe-source",
        action="append",
        default=None,
        help=(
            "Source eligible for --dedupe-targets. Repeatable. Defaults to "
            "mcts_bootstrap and policy_guard."
        ),
    )
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--held-out-targets", required=True)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hidden-dim", type=int, default=192)
    parser.add_argument(
        "--trainable-parts",
        default="all",
        choices=("all", "value_head", "policy_head", "heads", "trunk_value"),
    )
    parser.add_argument(
        "--init-checkpoint",
        default=None,
        help="Optional checkpoint to warm-start from instead of random init.",
    )
    parser.add_argument(
        "--reference-checkpoint",
        default=None,
        help="Optional checkpoint whose value/policy outputs define distillation penalties.",
    )
    parser.add_argument("--value-distill-weight", type=float, default=0.0)
    parser.add_argument("--policy-distill-weight", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--terminal-weight", type=float, default=30.0)
    parser.add_argument("--horizon-weight", type=float, default=10.0)
    parser.add_argument("--tablebase-weight", type=float, default=15.0)
    parser.add_argument("--interior-weight", type=float, default=2.0)
    parser.add_argument("--tier-a-weight", type=float, default=0.1)
    parser.add_argument("--tier-a-policy-weight", type=float, default=0.25)
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
    parser.add_argument("--policy-loss-weight", type=float, default=1.0)
    parser.add_argument(
        "--selection-metric",
        choices=("value_mse", "policy_nll", "value_plus_policy"),
        default="value_mse",
        help="Validation metric used to choose best.pt.",
    )
    parser.add_argument(
        "--force-train-source",
        action="append",
        default=None,
        metavar="SOURCE",
        help=(
            "Move validation feature groups containing this source into the "
            "training split. Repeat for sparse repair sources such as "
            "opponent_trace_guard or mcts_bootstrap."
        ),
    )
    parser.add_argument("--early-stopping-patience", type=int, default=40)
    parser.add_argument("--prev-gen-holdout-mse", type=float, default=None)
    parser.add_argument("--tablebase-mse-threshold", type=float, default=0.01)
    parser.add_argument("--tablebase-interior-mse-threshold", type=float, default=0.05)
    parser.add_argument("--max-unresolved-per-source", type=float, default=0.35)
    parser.add_argument(
        "--per-source-mse-threshold",
        action="append",
        default=None,
        metavar="SOURCE:VALUE",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    per_source_mse_thresholds = parse_thresholds(args.per_source_mse_threshold)
    extra_source_weights = parse_source_weights(args.source_weight)
    extra_policy_source_weights = parse_source_weights(args.policy_source_weight)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    records = load_targets_as_records(args.targets)
    if args.extra_targets:
        for path in args.extra_targets:
            extra = load_targets_as_records(path)
            print(f"Extra targets: {path}  +{len(extra)}  breakdown={source_breakdown(extra)}")
            records.extend(extra)
    merge_summary = None
    if args.dedupe_targets:
        merge_sources = (
            set(args.dedupe_source)
            if args.dedupe_source
            else {SOURCE_MCTS_BOOTSTRAP, SOURCE_POLICY_GUARD}
        )
        records, merge_summary = merge_duplicate_targets(
            records,
            merge_sources=merge_sources,
        )
        merge_report_path = out_dir / "target_merge_report.json"
        with merge_report_path.open("w", encoding="utf-8") as fh:
            json.dump(
                {
                    "merge_sources": sorted(merge_sources),
                    "summary": merge_summary.to_json(),
                },
                fh,
                indent=2,
            )
        print(
            f"Target dedupe: merge_sources={sorted(merge_sources)} "
            f"summary={merge_summary.to_json()}  report={merge_report_path}"
        )
    if args.extra_targets or args.dedupe_targets:
        combined_path = out_dir / "training_corpus.npz"
        save_targets(records, combined_path)
        train_targets = combined_path
    else:
        train_targets = Path(args.targets)
    print(f"Corpus: {train_targets}  records={len(records)}  breakdown={source_breakdown(records)}")
    cfg = TrainConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        seed=args.seed,
        device=args.device,
        hidden_dim=args.hidden_dim,
        init_checkpoint=args.init_checkpoint,
        trainable_parts=args.trainable_parts,
        reference_checkpoint=args.reference_checkpoint,
        value_distill_weight=args.value_distill_weight,
        policy_distill_weight=args.policy_distill_weight,
        weight_decay=args.weight_decay,
        source_weights=(
            (SOURCE_TERMINAL, args.terminal_weight),
            (SOURCE_EXACT_HORIZON_2, args.horizon_weight),
            (SOURCE_EXACT_HORIZON_3, args.horizon_weight),
            (SOURCE_TABLEBASE, args.tablebase_weight),
            (SOURCE_TABLEBASE_INTERIOR, args.interior_weight),
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
    result = train(train_targets, out_dir, cfg)
    print(
        f"Trained: best val MSE {result.best_val_mse:.5f} at epoch {result.best_epoch}; "
        f"checkpoint={result.checkpoint_path}"
    )
    print(f"Per-source val MSE: {result.best_per_source_mse}")

    held_out = load_targets_as_records(args.held_out_targets)
    report = calibration_check(
        checkpoint_path=result.checkpoint_path,
        held_out_targets=held_out,
        device=args.device,
    )
    print(f"Held-out overall MSE: {report.overall_mse:.5f}")
    print(f"Held-out per-source MSE: {report.mse_per_source}")
    print(
        "Held-out mean unresolved per source: "
        f"{report.mean_unresolved_probability_per_source}"
    )

    gate_cfg = BootstrapConfig(
        tablebase_mse_threshold=args.tablebase_mse_threshold,
        tablebase_interior_mse_threshold=args.tablebase_interior_mse_threshold,
        max_unresolved_per_source=args.max_unresolved_per_source,
        required_sources=(
            SOURCE_TERMINAL,
            SOURCE_TABLEBASE,
            SOURCE_EXACT_HORIZON_2,
            SOURCE_EXACT_HORIZON_3,
        ),
        per_source_mse_thresholds=per_source_mse_thresholds,
    )
    gate_passed = False
    gate_error = None
    try:
        enforce_calibration_gate(report, gate_cfg)
        gate_passed = True
    except CalibrationGateError as exc:
        gate_error = str(exc)

    improved = None
    if args.prev_gen_holdout_mse is not None:
        improved = monotonicity_verdict(report.overall_mse, args.prev_gen_holdout_mse)

    summary = {
        "targets": args.targets,
        "train_targets": str(train_targets),
        "extra_targets": args.extra_targets or [],
        "target_merge": merge_summary.to_json() if merge_summary is not None else None,
        "out_dir": str(out_dir),
        "checkpoint": result.checkpoint_path,
        "config": vars(args),
        "train_best_val_mse": result.best_val_mse,
        "train_best_epoch": result.best_epoch,
        "train_selection_metric": args.selection_metric,
        "train_best_selection_score": result.best_selection_score,
        "train_best_per_source_mse": result.best_per_source_mse,
        "held_out_overall_mse": report.overall_mse,
        "held_out_mse_per_source": report.mse_per_source,
        "held_out_mean_unresolved_probability_per_source": (
            report.mean_unresolved_probability_per_source
        ),
        "gate_passed": gate_passed,
        "gate_error": gate_error,
        "beats_prev_gen": improved,
    }
    report_path = out_dir / "gate_report.json"
    with report_path.open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)

    print(f"Gate report: {report_path}")
    if gate_passed:
        print("CALIBRATION GATE PASSED")
        if improved is not None:
            verdict = "improvement" if improved else "regression"
            delta = (args.prev_gen_holdout_mse - report.overall_mse) / args.prev_gen_holdout_mse * 100
            print(f"vs prev-gen {args.prev_gen_holdout_mse:.5f}: {delta:+.1f}% ({verdict})")
        return 0
    print(f"CALIBRATION GATE FAILED: {gate_error}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
