"""Audit the selected Gen-0 evaluator inside frozen matrix-game MCTS."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import tempfile

from stl.learning.train import load_checkpoint, make_predict_fn
from stl.solver.mcts_conformance import (
    FROZEN_BUDGETS,
    FROZEN_SEEDS,
    MCTSConformanceGateThresholds,
    evaluate_conformance_gate,
    frozen_horizon_one_scenarios,
    run_mcts_conformance,
)
from stl.solver.search import TablebaseEvaluator, ValueNetEvaluator


REPORT_SCHEMA = "stl.gen0-mcts-evaluator-audit.v1"


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_candidate_mcts_audit(
    *,
    checkpoint_path: str | Path,
    budgets=FROZEN_BUDGETS,
    seeds=FROZEN_SEEDS,
    device: str = "cpu",
    maximum_candidate_worsening: float = 0.02,
) -> dict[str, object]:
    budgets = tuple(sorted(int(value) for value in budgets))
    seeds = tuple(sorted(int(value) for value in seeds))
    if len(budgets) < 2:
        raise ValueError("MCTS evaluator audit requires at least two budgets")
    scenarios = frozen_horizon_one_scenarios()
    baseline = run_mcts_conformance(scenarios, budgets=budgets, seeds=seeds)

    model = load_checkpoint(checkpoint_path, device=device)
    predict = make_predict_fn(model, device=device)
    candidate = run_mcts_conformance(
        scenarios,
        budgets=budgets,
        seeds=seeds,
        evaluator_factory=lambda: TablebaseEvaluator(
            fallback=ValueNetEvaluator(model_fn=predict)
        ),
    )
    thresholds = MCTSConformanceGateThresholds(
        evaluation_budget=max(budgets),
        comparison_budget=sorted(budgets)[-2],
    )
    candidate_gate = evaluate_conformance_gate(candidate, thresholds)
    baseline_by_key = {
        (row.scenario_name, row.budget, row.seed): row for row in baseline.records
    }
    worsenings = []
    for row in candidate.records:
        if row.budget != max(budgets):
            continue
        reference = baseline_by_key[(row.scenario_name, row.budget, row.seed)]
        worsenings.append(
            {
                "scenario": row.scenario_name,
                "seed": row.seed,
                "absolute_value_error_delta": (
                    row.absolute_value_error - reference.absolute_value_error
                ),
                "saddle_gap_delta": (
                    row.full_width_saddle_gap - reference.full_width_saddle_gap
                ),
            }
        )
    maximum_value_worsening = max(
        (float(row["absolute_value_error_delta"]) for row in worsenings),
        default=0.0,
    )
    maximum_gap_worsening = max(
        (float(row["saddle_gap_delta"]) for row in worsenings),
        default=0.0,
    )
    failures = list(candidate_gate.failures)
    if maximum_value_worsening > maximum_candidate_worsening:
        failures.append("candidate evaluator worsens root value error")
    if maximum_gap_worsening > maximum_candidate_worsening:
        failures.append("candidate evaluator worsens full-width saddle gap")
    return {
        "schema": REPORT_SCHEMA,
        "checkpoint": {
            "path": str(checkpoint_path),
            "sha256": _sha256_file(checkpoint_path),
        },
        "budgets": list(budgets),
        "seeds": list(seeds),
        "scenarios": [scenario.name for scenario in scenarios],
        "candidate_gate": asdict(candidate_gate),
        "maximum_candidate_worsening": maximum_candidate_worsening,
        "maximum_value_error_worsening": maximum_value_worsening,
        "maximum_saddle_gap_worsening": maximum_gap_worsening,
        "paired_worsenings": worsenings,
        "baseline_records": [asdict(row) for row in baseline.records],
        "candidate_records": [asdict(row) for row in candidate.records],
        "passed": not failures,
        "failures": failures,
    }


def _atomic_write_json(payload: dict[str, object], path: str | Path) -> None:
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--out", default="outputs/regen2rl/gen0_mcts_evaluator_audit_v3.json"
    )
    parser.add_argument("--budget", action="append", type=int, default=None)
    parser.add_argument("--seed", action="append", type=int, default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--maximum-candidate-worsening", type=float, default=0.02)
    parser.add_argument(
        "--fail-on-gate", action=argparse.BooleanOptionalAction, default=True
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    report = run_candidate_mcts_audit(
        checkpoint_path=args.checkpoint,
        budgets=FROZEN_BUDGETS if args.budget is None else args.budget,
        seeds=FROZEN_SEEDS if args.seed is None else args.seed,
        device=args.device,
        maximum_candidate_worsening=args.maximum_candidate_worsening,
    )
    _atomic_write_json(report, args.out)
    print(
        f"[gen0-mcts] gate={'PASS' if report['passed'] else 'FAIL'}; "
        f"failures={report['failures']}",
        flush=True,
    )
    print(f"[gen0-mcts] report: {args.out}", flush=True)
    return 0 if report["passed"] or not args.fail_on_gate else 2


if __name__ == "__main__":
    raise SystemExit(main())
