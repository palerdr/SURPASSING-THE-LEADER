"""Versioned NPZ/JSON artifact helpers for the toy pipeline."""

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

    digest = hashlib.sha256()
    for path in paths:
        path = Path(path)
        digest.update(str(path).encode("utf-8"))
        digest.update(path.read_bytes())
    if config is not None:
        digest.update(canonical_json(config).encode("utf-8"))
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
    for name, spec in manifest.get("arrays", {}).items():
        if name not in arrays:
            raise ValueError(f"manifest array {name!r} is missing from NPZ")
        if list(arrays[name].shape) != spec["shape"] or str(arrays[name].dtype) != spec["dtype"]:
            raise ValueError(f"manifest shape/dtype mismatch for array {name!r}")
    return arrays, manifest
