"""Auditable exact root-matrix certificates for V3 policy targets."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Mapping, Sequence

import numpy as np

from stl.engine.actions import ACTION_SIZE
from stl.learning.replay import ShardRole
from stl.solver.exact import ExactSolveResult


CERTIFICATE_SCHEMA = "stl.exact-policy-certificate.v1"
CERTIFICATE_MANIFEST_SCHEMA = "stl.exact-policy-certificate-manifest.v1"
MAX_DROP_ACTIONS = ACTION_SIZE - 1
MAX_CHECK_ACTIONS = ACTION_SIZE - 2


@dataclass(frozen=True)
class ExactPolicyCertificate:
    """One exact root matrix and its equilibrium, keyed to a replay state."""

    state_hash: str
    search_config_digest: str
    horizon: int
    drop_actions: tuple[int, ...]
    check_actions: tuple[int, ...]
    payoff_for_hal: np.ndarray
    dropper_strategy: np.ndarray
    checker_strategy: np.ndarray
    value_for_hal: float
    unresolved_probability: float

    def __post_init__(self) -> None:
        payoff = np.asarray(self.payoff_for_hal, dtype=np.float64).copy()
        dropper = np.asarray(self.dropper_strategy, dtype=np.float64).copy()
        checker = np.asarray(self.checker_strategy, dtype=np.float64).copy()
        payoff.setflags(write=False)
        dropper.setflags(write=False)
        checker.setflags(write=False)
        object.__setattr__(self, "payoff_for_hal", payoff)
        object.__setattr__(self, "dropper_strategy", dropper)
        object.__setattr__(self, "checker_strategy", checker)
        _validate_certificate(self)


def certificate_from_result(
    *,
    state_hash: str,
    search_config_digest: str,
    result: ExactSolveResult,
) -> ExactPolicyCertificate:
    if result.payoff_for_hal is None:
        raise ValueError("exact policy certificate requires a root payoff matrix")
    return ExactPolicyCertificate(
        state_hash=state_hash,
        search_config_digest=search_config_digest,
        horizon=int(result.half_round_horizon),
        drop_actions=tuple(int(action) for action in result.drop_actions),
        check_actions=tuple(int(action) for action in result.check_actions),
        payoff_for_hal=result.payoff_for_hal,
        dropper_strategy=result.dropper_strategy,
        checker_strategy=result.checker_strategy,
        value_for_hal=float(result.value_for_hal),
        unresolved_probability=float(result.unresolved_probability),
    )


def _validate_digest(value: str, name: str) -> None:
    if len(value) != 64:
        raise ValueError(f"{name} must be a SHA-256 hex digest")
    try:
        int(value, 16)
    except ValueError as exc:
        raise ValueError(f"{name} must be a SHA-256 hex digest") from exc


def _validate_distribution(values: np.ndarray, size: int, name: str) -> None:
    if values.shape != (size,):
        raise ValueError(f"{name} shape {values.shape} does not match {(size,)}")
    if not np.all(np.isfinite(values)) or np.any(values < 0.0):
        raise ValueError(f"{name} must be finite and nonnegative")
    if not math.isclose(float(values.sum()), 1.0, abs_tol=1e-8):
        raise ValueError(f"{name} must sum to one")


def _validate_certificate(certificate: ExactPolicyCertificate) -> None:
    _validate_digest(certificate.state_hash, "state_hash")
    _validate_digest(certificate.search_config_digest, "search_config_digest")
    if certificate.horizon <= 0:
        raise ValueError("certificate horizon must be positive")
    if not certificate.drop_actions or not certificate.check_actions:
        raise ValueError("certificate action sets must be non-empty")
    if len(set(certificate.drop_actions)) != len(certificate.drop_actions):
        raise ValueError("drop actions must be unique")
    if len(set(certificate.check_actions)) != len(certificate.check_actions):
        raise ValueError("check actions must be unique")
    if not all(1 <= action < ACTION_SIZE for action in certificate.drop_actions):
        raise ValueError("drop actions are outside the literal-second schema")
    if not all(1 <= action < ACTION_SIZE - 1 for action in certificate.check_actions):
        raise ValueError("check actions are outside the literal-second schema")
    expected = (len(certificate.drop_actions), len(certificate.check_actions))
    if certificate.payoff_for_hal.shape != expected:
        raise ValueError(
            f"payoff shape {certificate.payoff_for_hal.shape} does not match {expected}"
        )
    if not np.all(np.isfinite(certificate.payoff_for_hal)):
        raise ValueError("payoff matrix must be finite")
    _validate_distribution(
        certificate.dropper_strategy, len(certificate.drop_actions), "dropper strategy"
    )
    _validate_distribution(
        certificate.checker_strategy, len(certificate.check_actions), "checker strategy"
    )
    if not math.isfinite(certificate.value_for_hal) or not (
        -1.000001 <= certificate.value_for_hal <= 1.000001
    ):
        raise ValueError("certificate value must be finite and in [-1, 1]")
    if not math.isfinite(certificate.unresolved_probability) or not (
        0.0 <= certificate.unresolved_probability <= 1.0
    ):
        raise ValueError("certificate unresolved probability must be in [0, 1]")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json(value: object) -> bytes:
    return (
        json.dumps(value, allow_nan=False, separators=(",", ":"), sort_keys=True) + "\n"
    ).encode("utf-8")


def certificate_manifest_path(path: str | Path) -> Path:
    source = Path(path)
    return source.with_name(source.name + ".manifest.json")


def save_certificate_shard(
    certificates: Sequence[ExactPolicyCertificate],
    path: str | Path,
    *,
    shard_role: ShardRole,
    generation_provenance: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Atomically publish a pickle-free certificate shard and commit manifest."""

    destination = Path(path)
    if destination.suffix.lower() != ".npz":
        raise ValueError("certificate shard path must end in .npz")
    manifest_path = certificate_manifest_path(destination)
    if destination.exists() or manifest_path.exists():
        raise ValueError(f"refusing to overwrite certificate shard {destination}")
    role = ShardRole(shard_role)
    rows = tuple(certificates)
    seen: set[str] = set()
    for row in rows:
        _validate_certificate(row)
        if row.state_hash in seen:
            raise ValueError(f"duplicate certificate state hash {row.state_hash}")
        seen.add(row.state_hash)

    count = len(rows)
    payoffs = np.full(
        (count, MAX_DROP_ACTIONS, MAX_CHECK_ACTIONS), np.nan, dtype=np.float64
    )
    drop_actions = np.full((count, MAX_DROP_ACTIONS), -1, dtype=np.int16)
    check_actions = np.full((count, MAX_CHECK_ACTIONS), -1, dtype=np.int16)
    dropper_strategies = np.zeros((count, MAX_DROP_ACTIONS), dtype=np.float64)
    checker_strategies = np.zeros((count, MAX_CHECK_ACTIONS), dtype=np.float64)
    for index, row in enumerate(rows):
        d_count = len(row.drop_actions)
        c_count = len(row.check_actions)
        payoffs[index, :d_count, :c_count] = row.payoff_for_hal
        drop_actions[index, :d_count] = row.drop_actions
        check_actions[index, :c_count] = row.check_actions
        dropper_strategies[index, :d_count] = row.dropper_strategy
        checker_strategies[index, :c_count] = row.checker_strategy
    arrays = {
        "state_hashes": np.asarray([row.state_hash for row in rows], dtype="<U64"),
        "search_config_digests": np.asarray(
            [row.search_config_digest for row in rows], dtype="<U64"
        ),
        "horizons": np.asarray([row.horizon for row in rows], dtype=np.int16),
        "drop_counts": np.asarray(
            [len(row.drop_actions) for row in rows], dtype=np.int16
        ),
        "check_counts": np.asarray(
            [len(row.check_actions) for row in rows], dtype=np.int16
        ),
        "drop_actions": drop_actions,
        "check_actions": check_actions,
        "payoffs_for_hal": payoffs,
        "dropper_strategies": dropper_strategies,
        "checker_strategies": checker_strategies,
        "values_for_hal": np.asarray(
            [row.value_for_hal for row in rows], dtype=np.float64
        ),
        "unresolved_probabilities": np.asarray(
            [row.unresolved_probability for row in rows], dtype=np.float64
        ),
    }

    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".npz", dir=destination.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    manifest_tmp = manifest_path.with_name(f".{manifest_path.name}.{os.getpid()}.tmp")
    try:
        np.savez_compressed(temporary, **arrays)
        manifest = {
            "schema": CERTIFICATE_MANIFEST_SCHEMA,
            "certificate_schema": CERTIFICATE_SCHEMA,
            "record_count": count,
            "shard_role": role.value,
            "data_file": destination.name,
            "data_sha256": _sha256_file(temporary),
            "state_hashes_sha256": hashlib.sha256(
                "\n".join(sorted(seen)).encode("ascii")
            ).hexdigest(),
            "generation_provenance": dict(generation_provenance or {}),
        }
        manifest_tmp.write_bytes(_canonical_json(manifest))
        os.replace(temporary, destination)
        os.replace(manifest_tmp, manifest_path)
        return manifest
    finally:
        temporary.unlink(missing_ok=True)
        manifest_tmp.unlink(missing_ok=True)


def load_certificate_shard(
    path: str | Path,
    *,
    expected_role: ShardRole | None = None,
) -> list[ExactPolicyCertificate]:
    source = Path(path)
    manifest_path = certificate_manifest_path(source)
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid certificate manifest: {exc}") from exc
    if manifest.get("schema") != CERTIFICATE_MANIFEST_SCHEMA:
        raise ValueError("unsupported certificate manifest schema")
    role = ShardRole(str(manifest.get("shard_role")))
    if expected_role is not None and role is not ShardRole(expected_role):
        raise ValueError(
            f"certificate shard role {role.value!r} does not match "
            f"required role {ShardRole(expected_role).value!r}"
        )
    if manifest.get("data_file") != source.name:
        raise ValueError("certificate manifest data file mismatch")
    if manifest.get("data_sha256") != _sha256_file(source):
        raise ValueError("certificate payload SHA-256 mismatch")
    try:
        with np.load(source, allow_pickle=False) as payload:
            arrays = {
                name: np.array(payload[name], copy=True) for name in payload.files
            }
    except (OSError, KeyError, ValueError) as exc:
        raise ValueError(f"invalid certificate payload: {exc}") from exc

    required = {
        "state_hashes",
        "search_config_digests",
        "horizons",
        "drop_counts",
        "check_counts",
        "drop_actions",
        "check_actions",
        "payoffs_for_hal",
        "dropper_strategies",
        "checker_strategies",
        "values_for_hal",
        "unresolved_probabilities",
    }
    if set(arrays) != required:
        raise ValueError("certificate payload arrays do not match schema")
    count = int(manifest.get("record_count", -1))
    if arrays["state_hashes"].shape != (count,):
        raise ValueError("certificate record count mismatch")
    rows: list[ExactPolicyCertificate] = []
    for index in range(count):
        d_count = int(arrays["drop_counts"][index])
        c_count = int(arrays["check_counts"][index])
        rows.append(
            ExactPolicyCertificate(
                state_hash=str(arrays["state_hashes"][index]),
                search_config_digest=str(arrays["search_config_digests"][index]),
                horizon=int(arrays["horizons"][index]),
                drop_actions=tuple(
                    int(value) for value in arrays["drop_actions"][index, :d_count]
                ),
                check_actions=tuple(
                    int(value) for value in arrays["check_actions"][index, :c_count]
                ),
                payoff_for_hal=arrays["payoffs_for_hal"][index, :d_count, :c_count],
                dropper_strategy=arrays["dropper_strategies"][index, :d_count],
                checker_strategy=arrays["checker_strategies"][index, :c_count],
                value_for_hal=float(arrays["values_for_hal"][index]),
                unresolved_probability=float(arrays["unresolved_probabilities"][index]),
            )
        )
    hashes = {row.state_hash for row in rows}
    if len(hashes) != len(rows):
        raise ValueError("duplicate state hash in certificate payload")
    observed_hash_digest = hashlib.sha256(
        "\n".join(sorted(hashes)).encode("ascii")
    ).hexdigest()
    if manifest.get("state_hashes_sha256") != observed_hash_digest:
        raise ValueError("certificate state-hash digest mismatch")
    return rows


def certificates_by_state(
    certificates: Sequence[ExactPolicyCertificate],
) -> dict[str, ExactPolicyCertificate]:
    result = {certificate.state_hash: certificate for certificate in certificates}
    if len(result) != len(certificates):
        raise ValueError("duplicate exact-policy certificate state hash")
    return result


__all__ = [
    "CERTIFICATE_MANIFEST_SCHEMA",
    "CERTIFICATE_SCHEMA",
    "ExactPolicyCertificate",
    "certificate_from_result",
    "certificate_manifest_path",
    "certificates_by_state",
    "load_certificate_shard",
    "save_certificate_shard",
]
