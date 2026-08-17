"""Exhaustive reachable-state tablebase generation for the abstract game."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from abstract.artifacts import digest_files, digest_json, load_npz_artifact, write_npz_artifact
from abstract.exact import AbstractExactResult, solve_all_reachable
from abstract.rules import (
    FROZEN_REVIVAL_MODEL,
    REVIVAL_BASELINE,
    REVIVAL_TTD_DECAY_PER_DEATH_DOSE,
    AbstractRuleset,
    TIMING_CONVENTION_ID,
)
from abstract.state import AbstractState


TABLEBASE_SCHEMA = "abstract.tablebase.v3"

_ARRAY_DTYPES = {
    "states": "int16",
    "value": "float64",
    "drop_policy": "float32",
    "check_policy": "float32",
    "saddle_gap": "float64",
    "dropper_win_probability": "float64",
    "checker_win_probability": "float64",
}

_METADATA_KEYS = {
    "ruleset_id",
    "state_schema",
    "state_field_names",
    "action_values",
    "action_seconds",
    "action_mapping",
    "timing_convention_id",
    "bucket_seconds",
    "load_cap_units",
    "load_cap_seconds",
    "failed_check_penalty_units",
    "revival_model",
    "state_ids",
    "root_state",
    "reachable_state_count",
    "physical_state_upper_bound",
    "solver",
    "solver_tolerances",
    "build_config_digest",
    "code_config_digest",
}


def _source_digest_inputs() -> list[Path]:
    source_root = Path(__file__).resolve().parent
    repository_root = source_root.parent.parent
    return [
        source_root / "artifacts.py",
        source_root / "state.py",
        source_root / "rules.py",
        source_root / "matrix.py",
        source_root / "exact.py",
        source_root / "tablebase.py",
        repository_root / "uv.lock",
    ]


def _build_config_payload(
    rules: AbstractRuleset,
    root: AbstractState,
) -> dict[str, object]:
    return {
        "schema": TABLEBASE_SCHEMA,
        "ruleset_id": rules.ruleset_id,
        "action_values": rules.action_values,
        "bucket_seconds": rules.bucket_seconds,
        "load_cap_units": rules.load_cap_units,
        "failed_check_penalty_units": rules.failed_check_penalty_units,
        "revival_model": rules.revival_model_metadata,
        "root": rules.state_fields(root),
        "solver_tolerances": {
            "primal_feasibility": 1e-9,
            "dual_feasibility": 1e-9,
            "ipm_optimality": 1e-10,
            "policy_saddle_gap": 2e-7,
        },
    }


def _rules_for_metadata(metadata: dict) -> AbstractRuleset:
    revival = metadata.get("revival_model")
    if revival != {
        "kind": FROZEN_REVIVAL_MODEL,
        "baseline": REVIVAL_BASELINE,
        "st_shape": "linear_pre_failure_load",
        "ttd_decay_per_death_dose": REVIVAL_TTD_DECAY_PER_DEATH_DOSE,
    }:
        raise ValueError("abstract tablebase does not use the frozen revival model")
    ruleset_id = metadata.get("ruleset_id")
    actions = metadata.get("action_values")
    integer_fields = {
        name: metadata.get(name)
        for name in (
            "bucket_seconds",
            "load_cap_units",
            "failed_check_penalty_units",
        )
    }
    if not isinstance(ruleset_id, str) or not ruleset_id:
        raise ValueError("abstract tablebase ruleset ID is invalid")
    if (
        not isinstance(actions, list)
        or not actions
        or any(isinstance(action, bool) or not isinstance(action, int) for action in actions)
    ):
        raise ValueError("abstract tablebase action values are invalid")
    if any(
        isinstance(value, bool) or not isinstance(value, int)
        for value in integer_fields.values()
    ):
        raise ValueError("abstract tablebase ruleset dimensions are invalid")
    return AbstractRuleset(
        ruleset_id=ruleset_id,
        action_values=tuple(actions),
        bucket_seconds=integer_fields["bucket_seconds"],
        load_cap_units=integer_fields["load_cap_units"],
        failed_check_penalty_units=integer_fields["failed_check_penalty_units"],
    )


def state_id(state: AbstractState, rules: AbstractRuleset) -> str:
    payload = f"{rules.ruleset_id}|{rules.state_fields(state)}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _policy_row(result: AbstractExactResult, *, drop: bool, action_size: int) -> np.ndarray:
    actions = result.drop_actions if drop else result.check_actions
    policy = result.dropper_strategy if drop else result.checker_strategy
    row = np.zeros(action_size, dtype=np.float32)
    row[np.asarray(actions, dtype=np.int64) - 1] = np.asarray(policy, dtype=np.float32)
    return row


def build_tablebase(
    rules: AbstractRuleset,
    *,
    root: AbstractState | None = None,
) -> dict[str, np.ndarray | dict]:
    """Build a complete terminal-value tablebase for the root's closure."""

    root = rules.initial_state() if root is None else root
    rules.validate_state(root)
    # Fingerprints describe the implementation that is about to run.  Capture
    # them before invoking the solver so in-process source mutation cannot
    # make the resulting rows claim a different implementation.
    build_config_digest = digest_json(_build_config_payload(rules, root))
    code_config_digest = digest_files(
        _source_digest_inputs(),
        config={"build_config_digest": build_config_digest},
    )
    rows = solve_all_reachable(rules, root=root)
    states = np.asarray([rules.state_fields(state) for state, _result in rows], dtype=np.int16)
    results = [result for _state, result in rows]
    arrays: dict[str, np.ndarray] = {
        "states": states,
        "value": np.asarray([result.value_for_dropper for result in results], dtype=np.float64),
        "drop_policy": np.stack([_policy_row(result, drop=True, action_size=rules.action_size) for result in results]),
        "check_policy": np.stack([_policy_row(result, drop=False, action_size=rules.action_size) for result in results]),
        "saddle_gap": np.asarray([result.saddle_gap for result in results], dtype=np.float64),
        "dropper_win_probability": np.asarray(
            [result.dropper_win_probability for result in results], dtype=np.float64
        ),
        "checker_win_probability": np.asarray(
            [result.checker_win_probability for result in results], dtype=np.float64
        ),
    }
    return {
        "arrays": arrays,
        "metadata": {
            "ruleset_id": rules.ruleset_id,
            "state_schema": rules.schema_version,
            "state_field_names": list(rules.state_field_names),
            "action_values": list(rules.action_values),
            "action_seconds": [rules.action_seconds(action) for action in rules.action_values],
            "action_mapping": {str(action): rules.action_seconds(action) for action in rules.action_values},
            "timing_convention_id": TIMING_CONVENTION_ID,
            "bucket_seconds": rules.bucket_seconds,
            "load_cap_units": rules.load_cap_units,
            "load_cap_seconds": rules.load_cap_seconds,
            "failed_check_penalty_units": rules.failed_check_penalty_units,
            "revival_model": rules.revival_model_metadata,
            "state_ids": "derived_on_lookup_or_export_sha256",
            "root_state": list(rules.state_fields(root)),
            "reachable_state_count": len(rows),
            "physical_state_upper_bound": rules.physical_state_upper_bound,
            "solver": "exhaustive_reachable_acyclic_dynamic_programming_with_lp",
            "solver_tolerances": {
                "primal_feasibility": 1e-9,
                "dual_feasibility": 1e-9,
                "ipm_optimality": 1e-10,
                "policy_saddle_gap": 2e-7,
            },
            "build_config_digest": build_config_digest,
            "code_config_digest": code_config_digest,
        },
    }


def write_tablebase(
    tablebase: dict[str, np.ndarray | dict],
    output_dir: str | Path,
) -> tuple[Path, Path, dict]:
    output_dir = Path(output_dir)
    npz_path = output_dir / "tablebase.npz"
    manifest_path = output_dir / "tablebase.json"
    manifest = write_npz_artifact(
        {name: value for name, value in tablebase["arrays"].items()},
        npz_path,
        manifest_path,
        metadata=dict(tablebase["metadata"]),
        schema_version=TABLEBASE_SCHEMA,
    )
    return npz_path, manifest_path, manifest


def load_tablebase(npz_path: str | Path, manifest_path: str | Path) -> dict[str, np.ndarray | dict]:
    arrays, manifest = load_npz_artifact(
        npz_path,
        manifest_path,
        expected_schema_version=TABLEBASE_SCHEMA,
    )
    metadata = manifest.get("metadata")
    if not isinstance(metadata, dict):
        raise ValueError("abstract tablebase metadata is malformed")
    if set(metadata) != _METADATA_KEYS:
        raise ValueError("abstract tablebase metadata key set is incompatible")
    rules = _rules_for_metadata(metadata)
    root_fields = metadata.get("root_state")
    if (
        not isinstance(root_fields, list)
        or len(root_fields) != 4
        or any(isinstance(value, bool) or not isinstance(value, int) for value in root_fields)
    ):
        raise ValueError("abstract tablebase root state is invalid")
    root = rules.validate_state(AbstractState(*root_fields))
    expected_metadata = {
        "state_schema": rules.schema_version,
        "state_field_names": list(rules.state_field_names),
        "action_values": list(rules.action_values),
        "action_seconds": [rules.action_seconds(action) for action in rules.action_values],
        "action_mapping": {
            str(action): rules.action_seconds(action) for action in rules.action_values
        },
        "timing_convention_id": TIMING_CONVENTION_ID,
        "bucket_seconds": rules.bucket_seconds,
        "load_cap_units": rules.load_cap_units,
        "load_cap_seconds": rules.load_cap_seconds,
        "failed_check_penalty_units": rules.failed_check_penalty_units,
        "revival_model": rules.revival_model_metadata,
        "state_ids": "derived_on_lookup_or_export_sha256",
        "root_state": root_fields,
        "physical_state_upper_bound": rules.physical_state_upper_bound,
        "solver": "exhaustive_reachable_acyclic_dynamic_programming_with_lp",
        "solver_tolerances": _build_config_payload(rules, root)["solver_tolerances"],
    }
    for field, expected in expected_metadata.items():
        if metadata.get(field) != expected:
            raise ValueError(f"abstract tablebase metadata is incompatible at {field}")

    reachable = metadata.get("reachable_state_count")
    if (
        isinstance(reachable, bool)
        or not isinstance(reachable, int)
        or not 0 < reachable <= rules.physical_state_upper_bound
    ):
        raise ValueError("abstract tablebase reachable-state count is invalid")
    expected_shapes = {
        "states": (reachable, 4),
        "value": (reachable,),
        "drop_policy": (reachable, rules.action_size),
        "check_policy": (reachable, rules.action_size),
        "saddle_gap": (reachable,),
        "dropper_win_probability": (reachable,),
        "checker_win_probability": (reachable,),
    }
    if set(arrays) != set(expected_shapes):
        raise ValueError("abstract tablebase array set is incompatible")
    for name, shape in expected_shapes.items():
        if arrays[name].shape != shape or str(arrays[name].dtype) != _ARRAY_DTYPES[name]:
            raise ValueError(f"abstract tablebase array contract is invalid for {name}")

    expected_build_digest = digest_json(_build_config_payload(rules, root))
    if metadata.get("build_config_digest") != expected_build_digest:
        raise ValueError("abstract tablebase build configuration is stale")
    expected_code_digest = digest_files(
        _source_digest_inputs(),
        config={"build_config_digest": expected_build_digest},
    )
    if metadata.get("code_config_digest") != expected_code_digest:
        raise ValueError("abstract tablebase code/configuration digest is stale")
    return {"arrays": arrays, "metadata": manifest["metadata"]}
