"""Fail-closed comparison of two resolved DTH training configurations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf


MISSING = object()
EXPECTED_DIFFERENCES = {
    "output_dir",
    "initial_checkpoint",
    "model.feature_lift",
}


def _resolved(config_name: str, config_dir: Path) -> dict[str, Any]:
    with initialize_config_dir(version_base="1.3", config_dir=str(config_dir)):
        config = compose(config_name=config_name)
    values = OmegaConf.to_container(config, resolve=True)
    if not isinstance(values, dict):
        raise TypeError("resolved training config must be a mapping")
    return values


def _diff(
    baseline: Any,
    candidate: Any,
    *,
    path: str = "",
) -> list[dict[str, Any]]:
    if isinstance(baseline, dict) and isinstance(candidate, dict):
        keys = sorted(set(baseline) | set(candidate))
        differences = []
        for key in keys:
            child_path = f"{path}.{key}" if path else str(key)
            differences.extend(
                _diff(
                    baseline.get(key, MISSING),
                    candidate.get(key, MISSING),
                    path=child_path,
                )
            )
        return differences
    if baseline == candidate:
        return []
    return [
        {
            "path": path,
            "baseline": "<missing>" if baseline is MISSING else baseline,
            "candidate": "<missing>" if candidate is MISSING else candidate,
        }
    ]


def compare_configs(
    baseline_name: str,
    candidate_name: str,
    *,
    config_dir: str | Path,
    output: str | Path,
) -> dict[str, Any]:
    resolved_dir = Path(config_dir).resolve()
    baseline = _resolved(baseline_name, resolved_dir)
    candidate = _resolved(candidate_name, resolved_dir)
    differences = _diff(baseline, candidate)
    difference_paths = {item["path"] for item in differences}
    report = {
        "baseline_config": baseline_name,
        "candidate_config": candidate_name,
        "differences": differences,
        "expected_difference_paths": sorted(EXPECTED_DIFFERENCES),
        "unexpected_difference_paths": sorted(
            difference_paths - EXPECTED_DIFFERENCES
        ),
        "passed": difference_paths == EXPECTED_DIFFERENCES,
    }
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    if not report["passed"]:
        raise ValueError(f"resolved config comparison failed: {report}")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", default="train_matrix_policy_generalization_balanced_v19")
    parser.add_argument("--candidate", default="train_matrix_policy_boundary_lift_v21")
    parser.add_argument("--config-dir", default="dth/config")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    compare_configs(
        args.baseline,
        args.candidate,
        config_dir=args.config_dir,
        output=args.output,
    )
    print(f"Resolved config comparison passed; proof written to {args.output}")


if __name__ == "__main__":
    main()
