"""Migrate an identity DTH checkpoint to the boundary_v1 input lift."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from torch import Tensor

from dth.network import DTHNetworkConfig, DTHPolicyValueNet, FEATURE_SCHEMA


MIGRATION_METHOD = "boundary_lift_v1_zero_column_append"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _config(payload: Mapping[str, Any]) -> DTHNetworkConfig:
    values = dict(payload.get("model_config", {}))
    values.setdefault("feature_lift", "identity")
    return DTHNetworkConfig(**values)


def migrate_state_dict(
    state_dict: Mapping[str, Tensor],
    source_config: DTHNetworkConfig,
) -> tuple[dict[str, Tensor], DTHNetworkConfig]:
    """Append four zero input columns while preserving every other tensor."""

    if source_config.feature_lift != "identity":
        raise ValueError("source checkpoint must use the identity feature lift")
    if source_config.action_count != 60:
        raise ValueError("source checkpoint must use the canonical 60-action heads")

    target_config = DTHNetworkConfig(
        hidden_width=source_config.hidden_width,
        hidden_layers=source_config.hidden_layers,
        action_count=source_config.action_count,
        horizon_scale=source_config.horizon_scale,
        feature_lift="boundary_v1",
    )
    source_model = DTHPolicyValueNet(source_config)
    source_model.load_state_dict(state_dict, strict=True)
    source_state = source_model.state_dict()
    target_model = DTHPolicyValueNet(target_config)
    target_state = target_model.state_dict()
    if source_state.keys() != target_state.keys():
        raise ValueError("source and target checkpoint parameters differ")

    first_weight = source_state["trunk.0.weight"]
    expected_source_shape = (source_config.hidden_width, len(FEATURE_SCHEMA))
    if tuple(first_weight.shape) != expected_source_shape:
        raise ValueError(
            "source first layer does not match the five-feature external contract"
        )
    migrated_first_weight = torch.zeros_like(target_state["trunk.0.weight"])
    migrated_first_weight[:, : len(FEATURE_SCHEMA)].copy_(first_weight)

    migrated: dict[str, Tensor] = {}
    for name, target_tensor in target_state.items():
        if name == "trunk.0.weight":
            migrated[name] = migrated_first_weight
        else:
            source_tensor = source_state[name]
            if source_tensor.shape != target_tensor.shape:
                raise ValueError(f"parameter shape changed for {name}")
            migrated[name] = source_tensor.clone()
    target_model.load_state_dict(migrated, strict=True)
    return migrated, target_config


def _rows_from_json(value: Any) -> set[tuple[tuple[int, int, int, int], int]]:
    rows: set[tuple[tuple[int, int, int, int], int]] = set()
    if isinstance(value, dict):
        if "state" in value and ("horizon" in value or "horizons" in value):
            horizon_value = value.get("horizon", value.get("horizons"))
            if isinstance(horizon_value, (int, float)):
                state = tuple(int(item) for item in value["state"])
                if len(state) == 4:
                    rows.add((state, int(horizon_value)))
        for child in value.values():
            rows.update(_rows_from_json(child))
    elif isinstance(value, list):
        for child in value:
            rows.update(_rows_from_json(child))
    return rows


def _rows_from_artifact(path: Path) -> set[tuple[tuple[int, int, int, int], int]]:
    if path.suffix.lower() == ".json":
        return _rows_from_json(json.loads(path.read_text(encoding="utf-8")))
    if path.suffix.lower() != ".npz":
        raise ValueError(f"unsupported audit artifact type: {path}")
    with np.load(path, allow_pickle=False) as artifact:
        if "states" not in artifact.files:
            raise ValueError(f"audit artifact has no states array: {path}")
        horizon_name = "horizons" if "horizons" in artifact.files else "horizon"
        if horizon_name not in artifact.files:
            raise ValueError(f"audit artifact has no horizon array: {path}")
        states = np.asarray(artifact["states"])
        horizons = np.asarray(artifact[horizon_name]).reshape(-1)
    if states.ndim != 2 or states.shape[1] != 4:
        raise ValueError(f"audit states must have shape (N, 4): {path}")
    if len(states) != len(horizons):
        raise ValueError(f"audit states and horizons have different lengths: {path}")
    return {
        (tuple(int(item) for item in state), int(horizon))
        for state, horizon in zip(states, horizons, strict=True)
    }


@torch.no_grad()
def audit_prediction_equivalence(
    source_payload: Mapping[str, Any],
    migrated_payload: Mapping[str, Any],
    artifact_paths: list[Path],
    *,
    batch_size: int = 4096,
    tolerance: float = 1e-5,
) -> dict[str, Any]:
    """Compare value and both policy logits over every row in the audit corpus."""

    rows: set[tuple[tuple[int, int, int, int], int]] = set()
    artifact_summaries = []
    for path in artifact_paths:
        if not path.is_file():
            raise FileNotFoundError(f"required audit artifact is missing: {path}")
        artifact_rows = _rows_from_artifact(path)
        if not artifact_rows:
            raise ValueError(f"audit artifact has no state/horizon rows: {path}")
        rows.update(artifact_rows)
        artifact_summaries.append({"path": str(path), "rows": len(artifact_rows)})
    if not rows:
        raise ValueError("prediction audit corpus is empty")

    source_config = _config(source_payload)
    migrated_config = _config(migrated_payload)
    source_model = DTHPolicyValueNet(source_config).eval()
    migrated_model = DTHPolicyValueNet(migrated_config).eval()
    source_model.load_state_dict(source_payload["state_dict"], strict=True)
    migrated_model.load_state_dict(migrated_payload["state_dict"], strict=True)

    ordered_rows = sorted(rows)
    max_errors = {"value": 0.0, "drop_logits": 0.0, "check_logits": 0.0}
    for start in range(0, len(ordered_rows), batch_size):
        batch = ordered_rows[start : start + batch_size]
        states = torch.tensor([row[0] for row in batch], dtype=torch.float32)
        horizons = torch.tensor([row[1] for row in batch], dtype=torch.float32)
        source_outputs = source_model(source_model.encode(states, horizons))
        migrated_outputs = migrated_model(migrated_model.encode(states, horizons))
        for name, source_output, migrated_output in zip(
            ("value", "drop_logits", "check_logits"),
            source_outputs,
            migrated_outputs,
            strict=True,
        ):
            max_errors[name] = max(
                max_errors[name],
                float(torch.max(torch.abs(source_output - migrated_output)).item()),
            )

    overall = max(max_errors.values())
    return {
        "artifacts": artifact_summaries,
        "unique_rows": len(ordered_rows),
        "max_abs_error": max_errors,
        "overall_max_abs_error": overall,
        "tolerance": tolerance,
        "passed": overall <= tolerance,
    }


def migrate_boundary_checkpoint(
    source: str | Path,
    destination: str | Path,
    *,
    audit_artifacts: list[str | Path],
) -> dict[str, Any]:
    """Migrate, fully audit, and atomically materialize a boundary checkpoint."""

    source_path = Path(source)
    destination_path = Path(destination)
    source_payload = torch.load(source_path, map_location="cpu", weights_only=False)
    if "model_config" not in source_payload or "state_dict" not in source_payload:
        raise ValueError("checkpoint must contain model_config and state_dict")
    source_config = _config(source_payload)
    migrated_state, target_config = migrate_state_dict(
        source_payload["state_dict"], source_config
    )
    migrated_payload = dict(source_payload)
    migrated_payload["state_dict"] = migrated_state
    migrated_payload["model_config"] = target_config.to_dict()
    migrated_payload["boundary_lift_migration"] = {
        "method": MIGRATION_METHOD,
        "source_checkpoint": str(source_path),
        "source_checkpoint_sha256": _sha256(source_path),
        "feature_lift": "boundary_v1",
    }
    audit = audit_prediction_equivalence(
        source_payload,
        migrated_payload,
        [Path(path) for path in audit_artifacts],
    )
    if not audit["passed"]:
        raise ValueError(f"migrated checkpoint prediction audit failed: {audit}")
    migrated_payload["boundary_lift_migration"]["prediction_audit"] = audit
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(migrated_payload, destination_path)
    return migrated_payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--destination", required=True)
    parser.add_argument("--audit-artifact", action="append", required=True)
    args = parser.parse_args()
    migrate_boundary_checkpoint(
        args.source,
        args.destination,
        audit_artifacts=args.audit_artifact,
    )
    print(f"Wrote audited boundary checkpoint to {args.destination}", flush=True)


if __name__ == "__main__":
    main()
