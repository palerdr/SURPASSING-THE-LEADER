"""Exact tablebase target construction for ToySTL networks."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from stl.toy.artifacts import load_npz_artifact, write_npz_artifact
from stl.toy.rules import ToyRuleset
from stl.toy.tablebase import build_tablebase


TARGET_SCHEMA = "toy.exact_targets.v2"


def build_exact_targets(
    rules: ToyRuleset,
    *,
    tablebase: dict[str, np.ndarray | dict] | None = None,
) -> dict[str, np.ndarray | dict]:
    tablebase = build_tablebase(rules) if tablebase is None else tablebase
    arrays = tablebase["arrays"]
    active = np.asarray(arrays["policy_active"], dtype=bool)
    state_values = np.asarray(arrays["states"])
    state_id = np.asarray(arrays["state_id"])
    physical_ids = np.asarray(
        [f"{tuple(row)}" for row in state_values],
        dtype="U128",
    )
    target_arrays = {
        "states": state_values[active],
        "horizon": np.asarray(arrays["horizon"])[active],
        "state_id": state_id[active],
        "physical_state_id": physical_ids[active],
        "value": np.asarray(arrays["value"])[active],
        "drop_policy": np.asarray(arrays["drop_policy"])[active],
        "check_policy": np.asarray(arrays["check_policy"])[active],
    }
    metadata = dict(tablebase["metadata"])
    metadata.update(
        {
            "target_rows": int(active.sum()),
            "target_kind": "exact_value_and_lp_policy",
            "horizon_zero_policy_rows_omitted": True,
        }
    )
    return {"arrays": target_arrays, "metadata": metadata}


def write_exact_targets(
    targets: dict[str, np.ndarray | dict],
    output_dir: str | Path,
) -> tuple[Path, Path, dict]:
    output_dir = Path(output_dir)
    return_paths = write_npz_artifact(
        {name: value for name, value in targets["arrays"].items()},
        output_dir / "targets.npz",
        output_dir / "targets.json",
        metadata=targets["metadata"],
        schema_version=TARGET_SCHEMA,
    )
    return output_dir / "targets.npz", output_dir / "targets.json", return_paths


def load_exact_targets(npz_path: str | Path, manifest_path: str | Path) -> tuple[dict[str, np.ndarray], dict]:
    return load_npz_artifact(npz_path, manifest_path, expected_schema_version=TARGET_SCHEMA)
