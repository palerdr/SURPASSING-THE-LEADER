"""Verified read access to a completed STL leap artifact.

The paper and a future leap-aware Hal read the leap tables through this
module. It sits outside ``solver/``, so the builder hash, which reads
``solver/leap_*.py`` by glob, does not cover it. It opens an artifact
read-only.

``open_leap`` checks the manifest before it returns a table:

- the schema is ``stl-leap-values-v1``;
- the builder marked the artifact complete;
- the SHA-256 of the DTH tail values (``V.npy``) equals ``dth_sha256``.

With ``verify="files"`` it also re-hashes each file that the manifest lists.
That mode reads the whole artifact, about 82 GB for ``leap-full-native``.

A key names a table as the builder does: ``("H1", minute)``,
``("H2", minute)`` or ``("REV", clock)``. ``checker`` and ``dropper`` are
profile indices that ``LeapTable.profile_index`` returns. The mapping from a
live ``WorldState`` to keys waits for the leap-aware Hal.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from stl.solver.leap_audit import PackedReader, certify_matrix, explicit_matrix, stage_inputs
from stl.solver.leap_build import is_window, key_name
from stl.solver.leap_profiles import WIN, profiles


SCHEMA = "stl-leap-values-v1"
TOLERANCE = 1e-6
VERIFY_MODES = ("manifest", "files")
ARTIFACT_ENV = "STL_LEAP_ARTIFACT"
DTH_ENV = "STL_LEAP_DTH"

_PACKAGE = Path(__file__).resolve().parent
DEFAULT_ARTIFACT = _PACKAGE / "outputs" / "leap-full-native"
DEFAULT_DTH = _PACKAGE.parent / "dth_compact" / "artifacts" / "V.npy"


@dataclass(frozen=True)
class LeapIdentity:
    """The facts that bind a figure or a policy to one artifact."""

    artifact: Path
    dth_values: Path
    schema: str
    builder_sha256: str
    dth_sha256: str
    verified: str


def _sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _default(name: str, fallback: Path) -> Path:
    """Read a default path from the environment, or use the package-relative one."""

    value = os.environ.get(name)
    if not value:
        return fallback
    path = Path(value).expanduser()
    if not path.is_absolute():
        raise ValueError(f"{name} must hold an absolute path, not {value!r}")
    return path


class LeapTable:
    """Stored values and stage certificates of one verified leap artifact."""

    def __init__(self, identity: LeapIdentity, dth: np.ndarray):
        self._identity = identity
        self._reader = PackedReader(identity.artifact / "tables")
        self._dth = dth

    @property
    def identity(self) -> LeapIdentity:
        return self._identity

    def profile_index(self, st: int, ttd: int) -> int:
        """Return the quotient profile of a player with squandered time st and TTD ttd."""

        p = profiles()
        found = np.flatnonzero((p.st == st) & (p.ttd == ttd))
        if len(found) != 1:
            raise ValueError(f"no profile has st={st} and ttd={ttd}")
        return int(found[0])

    def value(self, key: tuple[str, int], checker: int, dropper: int) -> float:
        """Return the stored value of one state, from the Dropper's side.

        The implicit WIN dropper profile reads as -1, as the builder stores it.
        """

        key = (str(key[0]), int(key[1]))
        if not 0 <= dropper <= WIN:
            raise ValueError(f"dropper profile {dropper} is out of range")
        column = int(profiles().idx0[dropper]) if key[0] == "REV" else dropper
        if column < 0:
            raise ValueError(f"{key_name(key)} stores no column for dropper profile {dropper}")
        return float(self._reader.get(key, checker, column))

    def stage(self, key: tuple[str, int], checker: int, dropper: int) -> dict:
        """Re-certify one stored value with an independent LP on its stage matrix.

        The stage matrix comes from the stored child values and the DTH tail.
        The method raises ValueError when the LP finds no certificate with a
        saddle gap of at most 1e-6, or when the certified value differs from
        the stored value by more than 1e-6. The result holds the stored value,
        the Bellman residual and the certificate of
        ``leap_audit.certify_matrix``.
        """

        key = (str(key[0]), int(key[1]))
        stored = self.value(key, checker, dropper)
        success, failure = stage_inputs(self._reader, key, checker, dropper, self._dth)
        where = f"{key_name(key)} checker {checker} dropper {dropper}"
        # certify_matrix raises RuntimeError when no LP attempt passes its 1e-6 gap gate.
        try:
            certificate = certify_matrix(explicit_matrix(success, failure, is_window(key)))
        except RuntimeError as error:
            raise ValueError(f"{where}: {error}") from error
        residual = abs(stored - certificate["value"])
        if residual > TOLERANCE:
            raise ValueError(f"{where}: Bellman residual {residual:.3g} exceeds {TOLERANCE:g}")
        if certificate["gap"] > TOLERANCE:
            raise ValueError(f"{where}: saddle gap {certificate['gap']:.3g} exceeds {TOLERANCE:g}")
        return {"stored_value": stored, "bellman_residual": residual, **certificate}


def open_leap(
    artifact: str | os.PathLike[str] | None = None,
    *,
    dth_values: str | os.PathLike[str] | None = None,
    verify: str = "manifest",
) -> LeapTable:
    """Open a completed leap artifact after its manifest checks pass.

    ``artifact`` defaults to ``STL_LEAP_ARTIFACT``, else to
    ``src/stl/outputs/leap-full-native``. ``dth_values`` defaults to
    ``STL_LEAP_DTH``, else to ``src/dth_compact/artifacts/V.npy``. Both
    defaults resolve from this package's directory, and an environment path
    must be absolute.
    """

    if verify not in VERIFY_MODES:
        raise ValueError(f"verify must be one of {VERIFY_MODES}, not {verify!r}")
    root = (Path(artifact) if artifact is not None else _default(ARTIFACT_ENV, DEFAULT_ARTIFACT)).resolve()
    dth_path = (Path(dth_values) if dth_values is not None else _default(DTH_ENV, DEFAULT_DTH)).resolve()
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"no leap manifest at {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != SCHEMA:
        raise ValueError(f"{manifest_path}: schema {manifest.get('schema')!r} is not {SCHEMA!r}")
    if manifest.get("complete") is not True:
        raise ValueError(f"{manifest_path}: the builder did not mark the artifact complete")
    if _sha256(dth_path) != manifest["dth_sha256"]:
        raise ValueError(f"{dth_path}: the DTH tail differs from the audited solve")
    if verify == "files":
        for name, expected in manifest["files"].items():
            if _sha256(root / name) != expected:
                raise ValueError(f"{root / name}: the file differs from its manifest hash")
    identity = LeapIdentity(
        artifact=root,
        dth_values=dth_path,
        schema=SCHEMA,
        builder_sha256=str(manifest["builder_sha256"]),
        dth_sha256=str(manifest["dth_sha256"]),
        verified=verify,
    )
    return LeapTable(identity, np.load(dth_path, mmap_mode="r"))
