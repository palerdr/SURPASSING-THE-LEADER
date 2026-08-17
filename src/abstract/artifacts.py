"""Versioned NPZ/JSON artifact helpers for the exact abstract pipeline."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Mapping

import numpy as np


def canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def digest_json(value: object) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def digest_files(paths: list[str | Path], *, config: object | None = None) -> str:
    """Hash ordered source/config inputs for artifact provenance."""

    resolved = [Path(path).resolve() for path in paths]
    if not resolved:
        raise ValueError("at least one source path is required")
    common_root = Path(os.path.commonpath([str(path) for path in resolved]))
    if common_root in resolved:
        common_root = common_root.parent
    digest = hashlib.sha256(b"abstract-source-config-bundle-v1\0")
    for path in resolved:
        label = path.relative_to(common_root).as_posix().encode("utf-8")
        payload = path.read_bytes()
        digest.update(len(label).to_bytes(8, "big"))
        digest.update(label)
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    if config is not None:
        label = b"config"
        payload = canonical_json(config).encode("utf-8")
        digest.update(len(label).to_bytes(8, "big"))
        digest.update(label)
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    return digest.hexdigest()


def write_npz_artifact(
    arrays: Mapping[str, np.ndarray],
    npz_path: str | Path,
    manifest_path: str | Path,
    *,
    metadata: Mapping[str, object],
    schema_version: str,
) -> dict:
    """Write arrays atomically and emit a deterministic manifest."""

    npz_path = Path(npz_path)
    manifest_path = Path(manifest_path)
    npz_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.NamedTemporaryFile(dir=npz_path.parent, suffix=".npz", delete=False) as temp:
        temp_path = Path(temp.name)
    try:
        np.savez_compressed(temp_path, **{name: np.asarray(value) for name, value in arrays.items()})
        os.replace(temp_path, npz_path)
    finally:
        temp_path.unlink(missing_ok=True)

    manifest = {
        "schema_version": schema_version,
        "metadata": dict(metadata),
        "arrays": {
            name: {
                "shape": list(np.asarray(value).shape),
                "dtype": str(np.asarray(value).dtype),
            }
            for name, value in sorted(arrays.items())
        },
        "npz_sha256": sha256_file(npz_path),
    }
    manifest_path.write_text(canonical_json(manifest) + "\n", encoding="utf-8")
    return manifest


def load_npz_artifact(
    npz_path: str | Path,
    manifest_path: str | Path,
    *,
    expected_schema_version: str | None = None,
) -> tuple[dict[str, np.ndarray], dict]:
    npz_path = Path(npz_path)
    manifest_path = Path(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict) or set(manifest) != {
        "schema_version",
        "metadata",
        "arrays",
        "npz_sha256",
    }:
        raise ValueError("artifact manifest key set is malformed")
    if expected_schema_version is not None and manifest.get("schema_version") != expected_schema_version:
        raise ValueError(
            f"artifact schema mismatch: expected {expected_schema_version!r}, "
            f"got {manifest.get('schema_version')!r}"
        )
    actual_digest = sha256_file(npz_path)
    if actual_digest != manifest.get("npz_sha256"):
        raise ValueError("NPZ digest does not match artifact manifest")

    with np.load(npz_path, allow_pickle=False) as loaded:
        arrays = {name: np.asarray(loaded[name]) for name in loaded.files}
    manifest_arrays = manifest.get("arrays")
    if not isinstance(manifest_arrays, dict):
        raise ValueError("artifact manifest must declare its arrays")
    if set(arrays) != set(manifest_arrays):
        raise ValueError("NPZ and manifest array sets do not match exactly")
    for name, spec in manifest_arrays.items():
        if not isinstance(spec, dict) or set(spec) != {"shape", "dtype"}:
            raise ValueError(f"manifest array contract is malformed for {name!r}")
        if name not in arrays:
            raise ValueError(f"manifest array {name!r} is missing from NPZ")
        if list(arrays[name].shape) != spec["shape"] or str(arrays[name].dtype) != spec["dtype"]:
            raise ValueError(f"manifest shape/dtype mismatch for array {name!r}")
    return arrays, manifest
