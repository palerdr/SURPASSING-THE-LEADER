"""Select one Gen-0 ladder candidate using only the development shard."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile

import numpy as np
import torch

from stl.commands.gen0_eval import (
    _assert_isolated,
    _gate,
    _policy_report,
    _source_value_metrics,
    _taxonomy_value_metrics,
)
from stl.learning.holdout import load_taxonomy
from stl.learning.certificates import certificates_by_state, load_certificate_shard
from stl.learning.replay import (
    ShardRole,
    exact_state_hash,
    load_replay_manifest,
    load_replay_shard,
    reconstruct_game,
)
from stl.learning.train import (
    load_checkpoint,
    load_checkpoint_bundle,
    make_predict_fn,
)


REPORT_SCHEMA = "stl.gen0-development-selection.v1"


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def select_development_candidate(
    *,
    checkpoint_paths: list[str | Path],
    train_path: str | Path,
    development_path: str | Path,
    certificate_path: str | Path,
    device: str = "cpu",
    tablebase_mse_max: float = 0.01,
    tablebase_interior_mse_max: float = 0.05,
    exact_saddle_gap_max: float = 0.05,
    unique_policy_tv_median_max: float = 0.15,
    exact_cutoff_max: float = 0.5,
    taxonomy_path: str | Path | None = None,
    parent_checkpoint_path: str | Path | None = None,
    boundary_max_abs_error: float | None = None,
    interior_max_abs_error: float | None = None,
    preservation_relative: float = 0.10,
    preservation_mse_floor: float = 0.001,
    preservation_max_abs_delta: float = 0.02,
) -> dict[str, object]:
    if not checkpoint_paths:
        raise ValueError("at least one checkpoint is required")
    train_manifest = load_replay_manifest(train_path)
    development_manifest = load_replay_manifest(development_path)
    train = load_replay_shard(
        train_path, for_training=True, expected_role=ShardRole.REPLAY
    )
    development = load_replay_shard(
        development_path, expected_role=ShardRole.DEVELOPMENT
    )
    overlaps = _assert_isolated(train, development)
    certificates = certificates_by_state(
        load_certificate_shard(certificate_path, expected_role=ShardRole.DEVELOPMENT)
    )
    active_hashes = {
        exact_state_hash(record.exact_state)
        for record in development
        if float(record.dropper_dist.sum(dtype=np.float64)) > 0.0
        or float(record.checker_dist.sum(dtype=np.float64)) > 0.0
    }
    if set(certificates) != active_hashes:
        raise ValueError(
            "development certificates do not exactly cover active policy rows"
        )

    train_digest = str(train_manifest["data_sha256"])
    development_digest = str(development_manifest["data_sha256"])
    taxonomy = (
        None
        if taxonomy_path is None
        else load_taxonomy(taxonomy_path, replay_path=development_path)
    )
    parent_metrics = None
    parent_bundle = None
    if parent_checkpoint_path is not None:
        parent_bundle = load_checkpoint_bundle(parent_checkpoint_path, device=device)
        parent_model = load_checkpoint(parent_checkpoint_path, device=device)
        parent_predict = make_predict_fn(parent_model, device=device)
        parent_predictions = np.asarray(
            [
                parent_predict(
                    reconstruct_game(record.exact_state),
                    horizon=record.value_horizon_half_rounds,
                )[0]
                for record in development
            ],
            dtype=np.float64,
        )
        parent_metrics, _mse, _rmse, _brier = _source_value_metrics(
            development, parent_predictions
        )
    candidates: list[dict[str, object]] = []
    for checkpoint_path in sorted(Path(path) for path in checkpoint_paths):
        bundle = load_checkpoint_bundle(checkpoint_path, device=device)
        corpus_digests = set(bundle["provenance"].get("corpus_digests", ()))
        if (
            train_digest not in corpus_digests
            or development_digest not in corpus_digests
        ):
            raise ValueError(
                f"checkpoint does not name both train and development shards: {checkpoint_path}"
            )
        model = load_checkpoint(checkpoint_path, device=device)
        predict = make_predict_fn(model, device=device)
        predictions = np.asarray(
            [
                predict(
                    reconstruct_game(record.exact_state),
                    horizon=record.value_horizon_half_rounds,
                )[0]
                for record in development
            ],
            dtype=np.float64,
        )
        source_metrics, overall_mse, overall_rmse, brier = _source_value_metrics(
            development, predictions
        )
        taxonomy_metrics = (
            None
            if taxonomy is None
            else _taxonomy_value_metrics(development, predictions, taxonomy)
        )
        policy = _policy_report(
            development,
            predict,
            certificates=certificates,
            uniqueness_tolerance=1e-9,
        )
        gate = _gate(
            value_sources=source_metrics,
            policy=policy,
            reload_max_abs_difference=0.0,
            tablebase_mse_max=tablebase_mse_max,
            tablebase_interior_mse_max=tablebase_interior_mse_max,
            exact_saddle_gap_max=exact_saddle_gap_max,
            unique_policy_tv_median_max=unique_policy_tv_median_max,
            exact_cutoff_max=exact_cutoff_max,
            taxonomy_metrics=taxonomy_metrics,
            boundary_max_abs_error=boundary_max_abs_error,
            interior_max_abs_error=interior_max_abs_error,
        )
        frozen_module_identity = None
        trainable_parts = bundle["provenance"].get("resolved_config", {}).get(
            "trainable_parts"
        )
        if trainable_parts == "value_head" and parent_bundle is not None:
            prefixes = ("trunk.", "policy_head.")
            frozen_module_identity = all(
                torch.equal(tensor, parent_bundle["model_state_dict"][name])
                for name, tensor in bundle["model_state_dict"].items()
                if name.startswith(prefixes)
            )
            gate["checks"]["value_head_frozen_module_identity"] = {
                "observed": frozen_module_identity,
                "required": True,
                "passed": frozen_module_identity,
            }
            gate["failures"] = [
                name for name, check in gate["checks"].items() if not check["passed"]
            ]
            gate["passed"] = not gate["failures"]
        if parent_metrics is not None:
            for source in ("exact_horizon_2", "exact_horizon_3", "terminal"):
                parent = parent_metrics[source]
                observed = source_metrics[source]
                mse_limit = float(parent["mse"]) + max(
                    preservation_mse_floor,
                    preservation_relative * float(parent["mse"]),
                )
                max_abs_limit = (
                    float(parent["max_abs_error"]) + preservation_max_abs_delta
                )
                gate["checks"][f"preserve_mse:{source}"] = {
                    "observed": float(observed["mse"]),
                    "maximum": mse_limit,
                    "passed": float(observed["mse"]) <= mse_limit,
                }
                gate["checks"][f"preserve_max_abs:{source}"] = {
                    "observed": float(observed["max_abs_error"]),
                    "maximum": max_abs_limit,
                    "passed": float(observed["max_abs_error"]) <= max_abs_limit,
                }
            gate["failures"] = [
                name for name, check in gate["checks"].items() if not check["passed"]
            ]
            gate["passed"] = not gate["failures"]
        candidates.append(
            {
                "checkpoint": str(checkpoint_path),
                "checkpoint_sha256": _sha256_file(checkpoint_path),
                "hidden_dim": bundle["model_schema"]["hidden_dim"],
                "seed": bundle["provenance"].get("seed"),
                "best_epoch": bundle["training"].get("epoch"),
                "overall_mse": overall_mse,
                "overall_rmse": overall_rmse,
                "brier_score": brier,
                "value_sources": source_metrics,
                "taxonomy_metrics": taxonomy_metrics,
                "policy": policy,
                "gate": gate,
                "frozen_module_identity": frozen_module_identity,
            }
        )

    passing = [candidate for candidate in candidates if candidate["gate"]["passed"]]
    def worst_limit_ratio(candidate: dict[str, object]) -> float:
        ratios: list[float] = []
        for check in candidate["gate"]["checks"].values():
            observed = check.get("observed")
            limit = check.get("maximum")
            if observed is None or limit is None:
                continue
            if float(limit) == 0.0:
                ratios.append(0.0 if float(observed) == 0.0 else math.inf)
            else:
                ratios.append(float(observed) / float(limit))
        return max(ratios, default=0.0)

    passing.sort(
        key=lambda candidate: (
            worst_limit_ratio(candidate),
            float(candidate["overall_mse"]),
            float(candidate["policy"]["saddle_gap"]["maximum"]),
            str(candidate["checkpoint_sha256"]),
        )
    )
    selected = None if not passing else passing[0]
    return {
        "schema": REPORT_SCHEMA,
        "training_shard": {
            "path": str(train_path),
            "sha256": train_digest,
            "records": len(train),
            "source_counts": dict(Counter(record.source for record in train)),
        },
        "development_shard": {
            "path": str(development_path),
            "sha256": development_digest,
            "certificate_path": str(certificate_path),
            "certificate_sha256": _sha256_file(certificate_path),
            "records": len(development),
            "source_counts": dict(Counter(record.source for record in development)),
            "overlaps_with_training": overlaps,
            "taxonomy_path": None if taxonomy_path is None else str(taxonomy_path),
        },
        "parent_checkpoint": (
            None
            if parent_checkpoint_path is None
            else {
                "path": str(parent_checkpoint_path),
                "sha256": _sha256_file(parent_checkpoint_path),
                "value_sources": parent_metrics,
            }
        ),
        "candidate_count": len(candidates),
        "passing_candidate_count": len(passing),
        "selected_checkpoint": (None if selected is None else selected["checkpoint"]),
        "selected_checkpoint_sha256": (
            None if selected is None else selected["checkpoint_sha256"]
        ),
        "candidates": candidates,
        "passed": selected is not None,
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
    parser.add_argument("--checkpoint", action="append", required=True)
    parser.add_argument("--train", required=True)
    parser.add_argument("--development", required=True)
    parser.add_argument("--certificates", required=True)
    parser.add_argument("--taxonomy", default=None)
    parser.add_argument("--parent-checkpoint", default=None)
    parser.add_argument(
        "--out", default="stl/outputs/regen2rl/gen0_development_selection_v3.json"
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--tablebase-mse-max", type=float, default=0.01)
    parser.add_argument("--tablebase-interior-mse-max", type=float, default=0.05)
    parser.add_argument("--exact-saddle-gap-max", type=float, default=0.05)
    parser.add_argument("--unique-policy-tv-median-max", type=float, default=0.15)
    parser.add_argument("--exact-cutoff-max", type=float, default=0.5)
    parser.add_argument("--boundary-max-abs-error", type=float, default=None)
    parser.add_argument("--interior-max-abs-error", type=float, default=None)
    parser.add_argument("--preservation-relative", type=float, default=0.10)
    parser.add_argument("--preservation-mse-floor", type=float, default=0.001)
    parser.add_argument("--preservation-max-abs-delta", type=float, default=0.02)
    parser.add_argument(
        "--fail-on-gate", action=argparse.BooleanOptionalAction, default=True
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    report = select_development_candidate(
        checkpoint_paths=args.checkpoint,
        train_path=args.train,
        development_path=args.development,
        certificate_path=args.certificates,
        device=args.device,
        tablebase_mse_max=args.tablebase_mse_max,
        tablebase_interior_mse_max=args.tablebase_interior_mse_max,
        exact_saddle_gap_max=args.exact_saddle_gap_max,
        unique_policy_tv_median_max=args.unique_policy_tv_median_max,
        exact_cutoff_max=args.exact_cutoff_max,
        taxonomy_path=args.taxonomy,
        parent_checkpoint_path=args.parent_checkpoint,
        boundary_max_abs_error=args.boundary_max_abs_error,
        interior_max_abs_error=args.interior_max_abs_error,
        preservation_relative=args.preservation_relative,
        preservation_mse_floor=args.preservation_mse_floor,
        preservation_max_abs_delta=args.preservation_max_abs_delta,
    )
    _atomic_write_json(report, args.out)
    print(
        f"[gen0-select] candidates={report['candidate_count']}, "
        f"passing={report['passing_candidate_count']}, "
        f"selected={report['selected_checkpoint']}",
        flush=True,
    )
    print(f"[gen0-select] report: {args.out}", flush=True)
    return 0 if report["passed"] or not args.fail_on_gate else 2


if __name__ == "__main__":
    raise SystemExit(main())
