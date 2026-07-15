"""Select one horizon-aware candidate using calibration and Bellman development."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import tempfile

from stl.commands.gen0_select import select_development_candidate
from stl.learning.bellman import evaluate_bellman_gate, load_bellman_bundle
from stl.learning.train import load_checkpoint, make_predict_fn


def select_bellman_candidate(args: argparse.Namespace) -> dict[str, object]:
    base = select_development_candidate(
        checkpoint_paths=args.checkpoint,
        train_path=args.train,
        development_path=args.development,
        certificate_path=args.certificates,
        taxonomy_path=args.taxonomy,
        parent_checkpoint_path=args.parent_checkpoint,
        device=args.device,
        tablebase_mse_max=args.tablebase_mse_max,
        tablebase_interior_mse_max=args.tablebase_interior_mse_max,
        exact_saddle_gap_max=args.exact_saddle_gap_max,
        boundary_max_abs_error=args.boundary_max_abs_error,
        interior_max_abs_error=args.interior_max_abs_error,
    )
    bundle = load_bellman_bundle(args.bellman_bundle)
    passing = []
    for candidate in base["candidates"]:
        model = load_checkpoint(candidate["checkpoint"], device=args.device)
        bellman = evaluate_bellman_gate(bundle, make_predict_fn(model, args.device))
        candidate["bellman_gate"] = bellman
        if candidate["gate"]["passed"] and bellman["passed"]:
            passing.append(candidate)
    passing.sort(
        key=lambda candidate: (
            float(candidate["bellman_gate"]["metrics"]["maximum_bellman_residual"]),
            float(candidate["bellman_gate"]["metrics"]["backed_root_mse"]),
            float(candidate["bellman_gate"]["metrics"]["maximum_saddle_gap"]),
            str(candidate["checkpoint_sha256"]),
        )
    )
    selected = passing[0] if passing else None
    base.update(
        {
            "schema": "stl.gen0-bellman-development-selection.v1",
            "passing_candidate_count": len(passing),
            "selected_checkpoint": (
                None if selected is None else selected["checkpoint"]
            ),
            "selected_checkpoint_sha256": (
                None if selected is None else selected["checkpoint_sha256"]
            ),
            "passed": selected is not None,
        }
    )
    return base


def _atomic_json(path: str | Path, payload: dict[str, object]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    os.close(descriptor)
    temporary = Path(name)
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
    parser.add_argument("--checkpoint", action="append", required=True)
    parser.add_argument("--train", required=True)
    parser.add_argument("--development", required=True)
    parser.add_argument("--certificates", required=True)
    parser.add_argument("--taxonomy", required=True)
    parser.add_argument("--bellman-bundle", required=True)
    parser.add_argument("--parent-checkpoint", default=None)
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--tablebase-mse-max", type=float, default=0.005)
    parser.add_argument("--tablebase-interior-mse-max", type=float, default=0.025)
    parser.add_argument("--exact-saddle-gap-max", type=float, default=0.04)
    parser.add_argument("--boundary-max-abs-error", type=float, default=0.10)
    parser.add_argument("--interior-max-abs-error", type=float, default=0.05**0.5)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    report = select_bellman_candidate(args)
    _atomic_json(args.out, report)
    print(
        f"[bellman-select] {'PASS' if report['passed'] else 'FAIL'}; "
        f"selected={report['selected_checkpoint']}",
        flush=True,
    )
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
