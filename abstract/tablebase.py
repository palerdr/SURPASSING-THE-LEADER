"""Exhaustive reachable-state tablebase generation for the abstract game."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from abstract.artifacts import digest_files, load_npz_artifact, write_npz_artifact
from abstract.exact import AbstractExactResult, solve_all_reachable
from abstract.rules import AbstractRuleset, TIMING_CONVENTION_ID
from abstract.state import AbstractState


TABLEBASE_SCHEMA = "abstract.tablebase.v2"


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
    source_root = Path(__file__).resolve().parent
    code_config_digest = digest_files(
        [source_root / name for name in ("state.py", "rules.py", "matrix.py", "exact.py", "tablebase.py")],
        config={"ruleset_id": rules.ruleset_id, "root": rules.state_fields(root)},
    )
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
    return {"arrays": arrays, "metadata": manifest["metadata"]}
