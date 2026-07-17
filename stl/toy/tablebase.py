"""Exhaustive ToySTL-v0 tablebase generation and lookup."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from stl.toy.artifacts import digest_files, load_npz_artifact, write_npz_artifact
from stl.toy.exact import ToyExactResult, solve_all_states
from stl.toy.rules import ToyRuleset
from stl.toy.state import ToyState


TABLEBASE_SCHEMA = "toy.tablebase.v2"


def state_horizon_id(state: ToyState, horizon: int, rules: ToyRuleset) -> str:
    payload = f"{rules.ruleset_id}|{horizon}|{rules.state_fields(state)}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _policy_row(result: ToyExactResult, *, drop: bool, action_size: int) -> np.ndarray:
    actions = result.drop_actions if drop else result.check_actions
    policy = result.dropper_strategy if drop else result.checker_strategy
    row = np.zeros(action_size, dtype=np.float32)
    if actions:
        row[np.asarray(actions, dtype=np.int64) - 1] = np.asarray(policy, dtype=np.float32)
    return row


def build_tablebase(rules: ToyRuleset, *, max_horizon: int | None = None) -> dict[str, np.ndarray | dict]:
    if rules.ruleset_id != "bucket12_fixed50":
        raise ValueError("build_tablebase is exhaustive only for ToySTL-v0")
    rows = solve_all_states(rules, max_horizon=max_horizon)
    action_size = rules.action_size
    states = np.asarray([rules.state_fields(state) for state, _h, _r in rows], dtype=np.int16)
    horizons = np.asarray([h for _state, h, _r in rows], dtype=np.int16)
    results = [result for _state, _h, result in rows]
    state_ids = np.asarray(
        [state_horizon_id(state, h, rules) for state, h, _r in rows],
        dtype="U64",
    )
    arrays: dict[str, np.ndarray] = {
        "states": states,
        "horizon": horizons,
        "state_id": state_ids,
        "value": np.asarray([result.value_for_hal for result in results], dtype=np.float64),
        "drop_policy": np.stack(
            [_policy_row(result, drop=True, action_size=action_size) for result in results]
        ),
        "check_policy": np.stack(
            [_policy_row(result, drop=False, action_size=action_size) for result in results]
        ),
        "saddle_gap": np.asarray([result.saddle_gap for result in results], dtype=np.float64),
        "truncated_probability": np.asarray(
            [result.truncated_probability for result in results], dtype=np.float64
        ),
        "hal_win_probability": np.asarray(
            [result.hal_win_probability for result in results], dtype=np.float64
        ),
        "baku_win_probability": np.asarray(
            [result.baku_win_probability for result in results], dtype=np.float64
        ),
        "policy_active": np.asarray([bool(h > 0) for h in horizons], dtype=np.bool_),
    }
    arrays["metadata"] = np.asarray([], dtype=np.uint8)
    source_root = Path(__file__).resolve().parent
    code_config_digest = digest_files(
        [source_root / name for name in ("state.py", "rules.py", "matrix.py", "exact.py", "tablebase.py")],
        config={"ruleset_id": rules.ruleset_id, "max_horizon": rules.max_half_rounds if max_horizon is None else max_horizon},
    )
    return {
        "arrays": arrays,
        "metadata": {
            "ruleset_id": rules.ruleset_id,
            "state_schema": rules.schema_version,
            "state_field_names": list(rules.state_field_names),
            "feature_names": list(rules.feature_names),
            "action_values": list(rules.action_values),
            "action_seconds": [rules.action_seconds(action) for action in rules.action_values],
            "action_mapping": {
                str(action): rules.action_seconds(action) for action in rules.action_values
            },
            "load_cap_units": rules.load_cap_units,
            "load_cap_seconds": rules.load_cap_seconds,
            "revival_mode": rules.revival_mode,
            "revival_probability": rules.fixed_revival_probability,
            "max_horizon": rules.max_half_rounds if max_horizon is None else max_horizon,
            "state_count": 60 * 60 * 2,
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
    metadata = dict(tablebase["metadata"])
    npz_path = output_dir / "tablebase.npz"
    manifest_path = output_dir / "tablebase.json"
    arrays = {name: value for name, value in tablebase["arrays"].items() if name != "metadata"}
    manifest = write_npz_artifact(
        arrays,
        npz_path,
        manifest_path,
        metadata=metadata,
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
