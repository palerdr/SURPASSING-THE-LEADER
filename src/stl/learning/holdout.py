"""Hash-bound tablebase taxonomy and single-candidate holdout controls."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Mapping

from stl.learning.contracts import canonical_config_json, config_digest
from stl.learning.replay import load_replay_manifest


TAXONOMY_SCHEMA = "stl.tablebase-taxonomy.v1"
HOLDOUT_SEAL_SCHEMA = "stl.gen0-holdout-seal.v1"
HOLDOUT_LEDGER_SCHEMA = "stl.gen0-holdout-use-ledger.v1"
BELLMAN_HOLDOUT_SEAL_SCHEMA = "stl.gen0-bellman-holdout-seal.v1"

V4_HOLDOUT_GATES: dict[str, float] = {
    "tablebase_mse_max": 0.01,
    "tablebase_interior_mse_max": 0.05,
    "boundary_max_abs_error": 0.10,
    "interior_max_abs_error": 0.05**0.5,
    "exact_horizon_mse_max": 0.01,
    "exact_max_abs_error": 0.25,
    "terminal_mse_max": 0.001,
    "terminal_max_abs_error": 0.05,
    "exact_saddle_gap_max": 0.05,
    "unique_policy_tv_median_max": 0.15,
    "exact_cutoff_max": 0.5,
}


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def taxonomy_path_for(replay_path: str | Path) -> Path:
    path = Path(replay_path)
    return path.with_name(f"{path.stem}.taxonomy.json")


def _atomic_json(path: str | Path, payload: Mapping[str, object]) -> None:
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


def write_taxonomy(
    replay_path: str | Path,
    entries: Mapping[str, Mapping[str, object]],
) -> Path:
    manifest = load_replay_manifest(replay_path)
    path = taxonomy_path_for(replay_path)
    payload: dict[str, object] = {
        "schema": TAXONOMY_SCHEMA,
        "replay_data_sha256": str(manifest["data_sha256"]),
        "record_count": int(manifest["record_count"]),
        "entries": {key: dict(value) for key, value in sorted(entries.items())},
    }
    payload["entries_sha256"] = hashlib.sha256(
        canonical_config_json(payload["entries"]).encode("utf-8")
    ).hexdigest()
    _atomic_json(path, payload)
    return path


def load_taxonomy(
    path: str | Path,
    *,
    replay_path: str | Path | None = None,
) -> dict[str, dict[str, object]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("schema") != TAXONOMY_SCHEMA:
        raise ValueError("unsupported tablebase taxonomy schema")
    entries = payload.get("entries")
    if not isinstance(entries, dict):
        raise ValueError("tablebase taxonomy entries must be an object")
    expected = hashlib.sha256(
        canonical_config_json(entries).encode("utf-8")
    ).hexdigest()
    if payload.get("entries_sha256") != expected:
        raise ValueError("tablebase taxonomy entry digest mismatch")
    if replay_path is not None:
        manifest = load_replay_manifest(replay_path)
        if payload.get("replay_data_sha256") != manifest.get("data_sha256"):
            raise ValueError("tablebase taxonomy is bound to another replay shard")
    return {str(key): dict(value) for key, value in entries.items()}


def write_holdout_seal(
    path: str | Path,
    *,
    holdout_path: str | Path,
    certificate_path: str | Path,
    taxonomy_path: str | Path,
    generation_plan_digest: str,
    blocked_artifacts: Mapping[str, str],
) -> dict[str, object]:
    holdout_manifest = load_replay_manifest(holdout_path)
    taxonomy = load_taxonomy(taxonomy_path, replay_path=holdout_path)
    payload: dict[str, object] = {
        "schema": HOLDOUT_SEAL_SCHEMA,
        "holdout_path": str(holdout_path),
        "holdout_data_sha256": str(holdout_manifest["data_sha256"]),
        "holdout_record_count": int(holdout_manifest["record_count"]),
        "certificate_path": str(certificate_path),
        "certificate_sha256": sha256_file(certificate_path),
        "taxonomy_path": str(taxonomy_path),
        "taxonomy_sha256": sha256_file(taxonomy_path),
        "taxonomy_entry_count": len(taxonomy),
        "generation_plan_digest": generation_plan_digest,
        "blocked_artifacts": dict(sorted(blocked_artifacts.items())),
        "gates": dict(V4_HOLDOUT_GATES),
    }
    payload["seal_digest"] = config_digest(payload)
    _atomic_json(path, payload)
    return payload


def write_bellman_holdout_seal(
    path: str | Path,
    *,
    holdout_path: str | Path,
    certificate_path: str | Path,
    bellman_path: str | Path,
    calibration_holdout_path: str | Path,
    calibration_certificate_path: str | Path,
    calibration_taxonomy_path: str | Path,
    generation_plan_digest: str,
    blocked_artifacts: Mapping[str, str],
    bellman_gates: Mapping[str, float],
    mcts_root_hashes: list[str],
) -> dict[str, object]:
    """Seal the bounded-frontier graph and fresh calibration ruler together."""

    holdout_manifest = load_replay_manifest(holdout_path)
    calibration_manifest = load_replay_manifest(calibration_holdout_path)
    load_taxonomy(
        calibration_taxonomy_path, replay_path=calibration_holdout_path
    )
    payload: dict[str, object] = {
        "schema": BELLMAN_HOLDOUT_SEAL_SCHEMA,
        "holdout_path": str(holdout_path),
        "holdout_data_sha256": str(holdout_manifest["data_sha256"]),
        "holdout_record_count": int(holdout_manifest["record_count"]),
        "certificate_path": str(certificate_path),
        "certificate_sha256": sha256_file(certificate_path),
        "bellman_path": str(bellman_path),
        "bellman_sha256": sha256_file(bellman_path),
        "calibration_holdout_path": str(calibration_holdout_path),
        "calibration_holdout_sha256": str(calibration_manifest["data_sha256"]),
        "calibration_certificate_path": str(calibration_certificate_path),
        "calibration_certificate_sha256": sha256_file(
            calibration_certificate_path
        ),
        "calibration_taxonomy_path": str(calibration_taxonomy_path),
        "calibration_taxonomy_sha256": sha256_file(calibration_taxonomy_path),
        "generation_plan_digest": generation_plan_digest,
        "blocked_artifacts": dict(sorted(blocked_artifacts.items())),
        "bellman_gates": dict(sorted(bellman_gates.items())),
        "calibration_gates": dict(V4_HOLDOUT_GATES),
        "mcts_root_hashes": list(mcts_root_hashes),
    }
    payload["seal_digest"] = config_digest(payload)
    _atomic_json(path, payload)
    return payload


def load_bellman_holdout_seal(
    path: str | Path,
    *,
    holdout_path: str | Path,
    certificate_path: str | Path,
    bellman_path: str | Path,
    calibration_holdout_path: str | Path,
    calibration_certificate_path: str | Path,
    calibration_taxonomy_path: str | Path,
) -> dict[str, object]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("schema") != BELLMAN_HOLDOUT_SEAL_SCHEMA:
        raise ValueError("unsupported Bellman holdout seal schema")
    digest_payload = dict(payload)
    stored_digest = digest_payload.pop("seal_digest", None)
    if stored_digest != config_digest(digest_payload):
        raise ValueError("Bellman holdout seal digest mismatch")
    holdout_manifest = load_replay_manifest(holdout_path)
    calibration_manifest = load_replay_manifest(calibration_holdout_path)
    checks = {
        "holdout_data_sha256": str(holdout_manifest["data_sha256"]),
        "certificate_sha256": sha256_file(certificate_path),
        "bellman_sha256": sha256_file(bellman_path),
        "calibration_holdout_sha256": str(calibration_manifest["data_sha256"]),
        "calibration_certificate_sha256": sha256_file(
            calibration_certificate_path
        ),
        "calibration_taxonomy_sha256": sha256_file(calibration_taxonomy_path),
    }
    for name, observed in checks.items():
        if payload.get(name) != observed:
            raise ValueError(f"Bellman holdout seal {name} mismatch")
    load_taxonomy(
        calibration_taxonomy_path, replay_path=calibration_holdout_path
    )
    return payload


def load_holdout_seal(
    path: str | Path,
    *,
    holdout_path: str | Path,
    certificate_path: str | Path,
    taxonomy_path: str | Path,
) -> dict[str, object]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("schema") != HOLDOUT_SEAL_SCHEMA:
        raise ValueError("unsupported holdout seal schema")
    digest_payload = dict(payload)
    stored_digest = digest_payload.pop("seal_digest", None)
    if stored_digest != config_digest(digest_payload):
        raise ValueError("holdout seal digest mismatch")
    manifest = load_replay_manifest(holdout_path)
    checks = {
        "holdout_data_sha256": str(manifest["data_sha256"]),
        "certificate_sha256": sha256_file(certificate_path),
        "taxonomy_sha256": sha256_file(taxonomy_path),
    }
    for name, observed in checks.items():
        if payload.get(name) != observed:
            raise ValueError(f"holdout seal {name} mismatch")
    load_taxonomy(taxonomy_path, replay_path=holdout_path)
    return payload


def claim_holdout_use(
    ledger_path: str | Path,
    *,
    seal_digest: str,
    checkpoint_sha256: str,
    evaluation_config: Mapping[str, object],
) -> dict[str, object]:
    path = Path(ledger_path)
    config_sha256 = config_digest(dict(evaluation_config))
    identity = {
        "schema": HOLDOUT_LEDGER_SCHEMA,
        "seal_digest": seal_digest,
        "checkpoint_sha256": checkpoint_sha256,
        "evaluation_config_sha256": config_sha256,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL)
    except FileExistsError:
        existing = json.loads(path.read_text(encoding="utf-8"))
        for key, value in identity.items():
            if existing.get(key) != value:
                raise ValueError(
                    "sealed holdout was already bound to another candidate or config"
                )
        return existing
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        json.dump({**identity, "status": "started"}, stream, indent=2, sort_keys=True)
        stream.write("\n")
    return {**identity, "status": "started"}


def complete_holdout_use(
    ledger_path: str | Path,
    *,
    report_path: str | Path,
    passed: bool,
) -> None:
    path = Path(ledger_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload.update(
        {
            "status": "completed",
            "report_path": str(report_path),
            "report_sha256": sha256_file(report_path),
            "passed": bool(passed),
        }
    )
    _atomic_json(path, payload)


__all__ = [
    "HOLDOUT_LEDGER_SCHEMA",
    "HOLDOUT_SEAL_SCHEMA",
    "BELLMAN_HOLDOUT_SEAL_SCHEMA",
    "TAXONOMY_SCHEMA",
    "V4_HOLDOUT_GATES",
    "claim_holdout_use",
    "complete_holdout_use",
    "load_holdout_seal",
    "load_bellman_holdout_seal",
    "load_taxonomy",
    "sha256_file",
    "taxonomy_path_for",
    "write_holdout_seal",
    "write_bellman_holdout_seal",
    "write_taxonomy",
]
