"""One-use V5 static evaluation and horizon-aligned bounded MCTS audit."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import tempfile

from stl.commands.gen0_eval import evaluate_checkpoint
from stl.learning.bellman import (
    BellmanGateThresholds,
    evaluate_bellman_gate,
    load_bellman_bundle,
    spot_recheck_bellman_bundle,
)
from stl.learning.holdout import (
    V4_HOLDOUT_GATES,
    claim_holdout_use,
    complete_holdout_use,
    load_bellman_holdout_seal,
    sha256_file,
)
from stl.learning.replay import load_replay_manifest
from stl.learning.train import (
    load_checkpoint,
    load_checkpoint_bundle,
    make_predict_fn,
)
from stl.solver.mcts_conformance import (
    FROZEN_BUDGETS,
    FROZEN_SEEDS,
    MCTSConformanceGateThresholds,
    evaluate_conformance_gate,
    run_bounded_mcts_conformance,
)
from stl.solver.search import TablebaseEvaluator, ValueNetEvaluator


REPORT_SCHEMA = "stl.gen0-bounded-promotion-audit.v1"


def _atomic_json(path: str | Path, payload: dict[str, object]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def run_bounded_promotion_audit(args: argparse.Namespace) -> dict[str, object]:
    seal = load_bellman_holdout_seal(
        args.seal,
        holdout_path=args.bellman_holdout,
        certificate_path=args.bellman_certificates,
        bellman_path=args.bellman_bundle,
        calibration_holdout_path=args.calibration_holdout,
        calibration_certificate_path=args.calibration_certificates,
        calibration_taxonomy_path=args.calibration_taxonomy,
    )
    checkpoint_sha256 = sha256_file(args.checkpoint)
    evaluation_config = {
        "schema": REPORT_SCHEMA,
        "budgets": list(args.budget),
        "seeds": list(args.seed),
        "maximum_candidate_worsening": args.maximum_candidate_worsening,
        "bellman_gates": asdict(BellmanGateThresholds(maximum_saddle_gap=0.05)),
        "calibration_gates": V4_HOLDOUT_GATES,
        "mcts_root_hashes": list(seal["mcts_root_hashes"]),
    }
    claim_holdout_use(
        args.ledger,
        seal_digest=str(seal["seal_digest"]),
        checkpoint_sha256=checkpoint_sha256,
        evaluation_config=evaluation_config,
    )

    checkpoint_bundle = load_checkpoint_bundle(args.checkpoint, device=args.device)
    corpus_digests = set(
        str(value)
        for value in checkpoint_bundle["provenance"].get("corpus_digests", ())
    )
    bellman_train_digest = str(
        load_replay_manifest(args.bellman_train)["data_sha256"]
    )
    bellman_holdout_digest = str(
        load_replay_manifest(args.bellman_holdout)["data_sha256"]
    )
    if bellman_train_digest not in corpus_digests:
        raise ValueError("checkpoint provenance omits Bellman training shard")
    if bellman_holdout_digest in corpus_digests:
        raise ValueError("sealed Bellman holdout leaked into checkpoint provenance")

    model = load_checkpoint(args.checkpoint, device=args.device)
    predict = make_predict_fn(model, device=args.device)
    bundle = load_bellman_bundle(args.bellman_bundle)
    bellman_static = evaluate_bellman_gate(
        bundle,
        predict,
        thresholds=BellmanGateThresholds(maximum_saddle_gap=0.05),
    )
    bellman_recheck = spot_recheck_bellman_bundle(bundle)
    calibration = evaluate_checkpoint(
        checkpoint_path=args.checkpoint,
        train_path=args.calibration_train,
        ruler_path=args.calibration_holdout,
        certificate_path=args.calibration_certificates,
        taxonomy_path=args.calibration_taxonomy,
        device=args.device,
        **V4_HOLDOUT_GATES,
    )
    static_passed = bool(
        bellman_static["passed"]
        and bellman_recheck["passed"]
        and calibration["gate"]["passed"]
    )

    baseline_report = None
    candidate_report = None
    candidate_gate = None
    paired_worsening = None
    mcts_passed = False
    if static_passed:
        roots = list(seal["mcts_root_hashes"])
        baseline = run_bounded_mcts_conformance(
            bundle,
            root_hashes=roots,
            budgets=args.budget,
            seeds=args.seed,
        )
        candidate = run_bounded_mcts_conformance(
            bundle,
            root_hashes=roots,
            budgets=args.budget,
            seeds=args.seed,
            evaluator_factory=lambda: TablebaseEvaluator(
                fallback=ValueNetEvaluator(predict)
            ),
        )
        thresholds = MCTSConformanceGateThresholds(
            evaluation_budget=max(args.budget),
            comparison_budget=sorted(args.budget)[-2],
        )
        gate = evaluate_conformance_gate(candidate, thresholds)
        baseline_by_key = {
            (row.scenario_name, row.budget, row.seed): row
            for row in baseline.records
        }
        paired = []
        for row in candidate.records:
            if row.budget != max(args.budget):
                continue
            reference = baseline_by_key[
                (row.scenario_name, row.budget, row.seed)
            ]
            paired.append(
                {
                    "root": row.scenario_name,
                    "seed": row.seed,
                    "value_error_delta": row.absolute_value_error
                    - reference.absolute_value_error,
                    "saddle_gap_delta": row.full_width_saddle_gap
                    - reference.full_width_saddle_gap,
                }
            )
        maximum_value_worsening = max(
            (float(row["value_error_delta"]) for row in paired), default=0.0
        )
        maximum_gap_worsening = max(
            (float(row["saddle_gap_delta"]) for row in paired), default=0.0
        )
        mcts_passed = bool(
            gate.passed
            and maximum_value_worsening <= args.maximum_candidate_worsening
            and maximum_gap_worsening <= args.maximum_candidate_worsening
        )
        baseline_report = [asdict(row) for row in baseline.records]
        candidate_report = [asdict(row) for row in candidate.records]
        candidate_gate = asdict(gate)
        paired_worsening = {
            "maximum_value_error_worsening": maximum_value_worsening,
            "maximum_saddle_gap_worsening": maximum_gap_worsening,
            "rows": paired,
        }

    passed = bool(static_passed and mcts_passed)
    report = {
        "schema": REPORT_SCHEMA,
        "checkpoint": {
            "path": str(args.checkpoint),
            "sha256": checkpoint_sha256,
        },
        "seal_digest": seal["seal_digest"],
        "static_passed": static_passed,
        "bellman_static": bellman_static,
        "bellman_spot_recheck": bellman_recheck,
        "calibration": calibration,
        "mcts_executed": static_passed,
        "mcts_passed": mcts_passed,
        "mcts_gate": candidate_gate,
        "paired_worsening": paired_worsening,
        "baseline_records": baseline_report,
        "candidate_records": candidate_report,
        "passed": passed,
    }
    _atomic_json(args.out, report)
    complete_holdout_use(args.ledger, report_path=args.out, passed=passed)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--bellman-train", required=True)
    parser.add_argument("--bellman-holdout", required=True)
    parser.add_argument("--bellman-certificates", required=True)
    parser.add_argument("--bellman-bundle", required=True)
    parser.add_argument("--calibration-train", required=True)
    parser.add_argument("--calibration-holdout", required=True)
    parser.add_argument("--calibration-certificates", required=True)
    parser.add_argument("--calibration-taxonomy", required=True)
    parser.add_argument("--seal", required=True)
    parser.add_argument("--ledger", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--budget", type=int, action="append", default=None)
    parser.add_argument("--seed", type=int, action="append", default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--maximum-candidate-worsening", type=float, default=0.02)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    args.budget = list(FROZEN_BUDGETS if args.budget is None else args.budget)
    args.seed = list(FROZEN_SEEDS if args.seed is None else args.seed)
    report = run_bounded_promotion_audit(args)
    print(
        f"[bounded-audit] {'PASS' if report['passed'] else 'FAIL'}; "
        f"static={report['static_passed']} mcts={report['mcts_passed']}",
        flush=True,
    )
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
