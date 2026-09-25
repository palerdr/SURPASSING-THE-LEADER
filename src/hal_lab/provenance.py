"""Verify the SHA-256 bindings of the frozen Hal evidence records.

A frozen record is a tracked file that holds the protocol or the results of a
finished Hal experiment, or that a sealed record binds. ``RECORDS`` lists each
record by its path at the ``pre-restructure`` tag. A record binds files by
SHA-256, and this module checks each binding against the source that still
holds the bound bytes:

- The verifier reads a path that git tracked at the tag from the tag, the
  bytes that ``git show pre-restructure:<path>`` prints. A later edit or move
  of the file leaves the result unchanged.
- The verifier reads any other path, such as an ignored ``outputs/...``
  report, from disk.

``evidence/PATH_MAP.toml`` resolves the keys that are not repository paths:
an absolute path under a recorded checkout root, and a bare file name. It
also maps a historical path to its current path. A change that moves a record
or an ignored output edits the map and leaves the record's bytes alone.

Each binding gets one status:

- ``verified``: the bound bytes hash to the recorded value.
- ``mismatch``: the bound bytes hash to another value.
- ``missing``: no file exists at the bound path.
- ``unverifiable``: the record names no file for the hash, or the hash is an
  artifact digest. The entry states the reason.

The report holds the SHA-256 of each record too. ``--report`` prints the
report as JSON, and ``--check`` compares a fresh report with
``evidence/verification_baseline.json``. The module reads the repository and
writes no file.

    uv run python -m hal_lab.provenance --report
    uv run python -m hal_lab.provenance --check
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
import tomllib
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath

import yaml


ROOT = Path(__file__).resolve().parents[2]
EVIDENCE_DIR = Path(__file__).resolve().parent / "evidence"
PATH_MAP = EVIDENCE_DIR / "PATH_MAP.toml"
BASELINE = EVIDENCE_DIR / "verification_baseline.json"
TAG = "pre-restructure"
SCHEMA = "hal-lab-evidence-verification-v1"

# The frozen records, by path at TAG. PATH_MAP.toml gives each current path.
RECORDS = (
    # The six sealed records. Tests read them at these paths, so they stay in
    # src/arena/config.
    "src/arena/config/aggro_hal_adaptive_exploitation_goal_v1.json",
    "src/arena/config/aggro_hal_adaptive_memory_v1.yaml",
    "src/arena/config/aggro_hal_tactical_baseline_v1.json",
    "src/arena/config/aggro_hal_v1.yaml",
    "src/arena/config/pm_hal_confirmation_v3.json",
    "src/arena/config/pm_hal_evaluation_v3.json",
    # The protocols and results of the finished one-shot experiments.
    "src/arena/config/external_hal_prior_v1_results.json",
    "src/arena/config/neural_pilots_v1_results.json",
    "src/arena/config/perfect_hal_bayes_v2.json",
    "src/arena/config/perfect_hal_bayes_v2_results.json",
    "src/arena/config/perfect_hal_bayes_v2_selection.json",
    "src/arena/config/perfect_hal_ensemble_v1.json",
    "src/arena/config/pm_hal_controller_v2.json",
    "src/arena/config/pm_hal_evaluation_v1.json",
    "src/arena/config/pm_hal_evaluation_v2.json",
    "src/arena/config/selector_study_v1_results.json",
    "src/arena/config/translated_hal_v1_results.json",
    "src/arena/config/translated_hal_v1_selection.json",
)

STATUSES = ("verified", "mismatch", "missing", "unverifiable")
_SHA256 = re.compile(r"[0-9a-f]{64}")
# A mapping key that names a file: a path with a slash, or a bare file name
# with an extension. A table such as {"src/arena/match.py": "<sha256>"} uses
# such keys.
_FILE_KEY = re.compile(r"\S*/\S*|[^/\s]+\.[A-Za-z0-9]+")
# A hash field whose file a sibling field names, and the way the record hashed
# that file.
_PAIRED_FIELDS = {
    "sha256": ("path", "bytes"),
    "manifest_sha256": ("manifest_path", "bytes"),
    "report_sha256": ("report_path", "bytes"),
    "validation_sha256": ("validation_report", "bytes"),
    "canonical_yaml_sha256": ("path", "canonical-json"),
    "goal_manifest_canonical_json_sha256": ("goal_manifest", "canonical-json"),
}
# Hash fields that record an artifact digest, which no single file carries.
_DIGEST_SUFFIXES = ("digest", "schema_hash")


class TagUnavailable(RuntimeError):
    """The checkout has no pre-restructure tag, or git cannot read the tag."""


@dataclass(frozen=True)
class Binding:
    """One SHA-256 value in a record, with the name of the file it binds."""

    pointer: str
    expected: str
    # The record's name for the bound file, as the record writes it. None
    # when the record names no file for this hash.
    key: str | None
    method: str | None
    reason: str | None


@dataclass(frozen=True)
class PathMap:
    """The contents of PATH_MAP.toml."""

    absolute_roots: tuple[str, ...] = ()
    basenames: Mapping[str, str] = field(default_factory=dict)
    moves: Mapping[str, str] = field(default_factory=dict)

    def historical(self, key: str) -> tuple[str | None, str | None]:
        """Return the repository path that a record key named at the tag.

        The second item gives the reason when the key resolves to no path.
        """

        if key.startswith("/"):
            for root in self.absolute_roots:
                if key.startswith(root):
                    return key[len(root) :], None
            return None, "absolute path outside every root in PATH_MAP.toml"
        if "/" not in key:
            if key in self.basenames:
                return self.basenames[key], None
            return None, "bare file name with no entry in PATH_MAP.toml"
        return key, None

    def current(self, historical: str) -> str:
        """Return the current path of a file by its path at the tag."""

        return self.moves.get(historical, historical)


@dataclass(frozen=True)
class Difference:
    """One place where a fresh report differs from the baseline."""

    record: str
    # The JSON pointer of the binding, or None for the record's own bytes.
    pointer: str | None
    # "git" or "disk" for a binding; None for a record or the report header.
    source: str | None
    message: str

    def __str__(self) -> str:
        where = self.record + (self.pointer or "")
        return f"{where}: {self.message}" if where else self.message


def _repository_path(path: str) -> str:
    if not path or path.startswith("/") or "\\" in path or ".." in PurePosixPath(path).parts:
        raise ValueError(f"PATH_MAP.toml: {path!r} is not a repository path")
    return path


def load_path_map(path: Path = PATH_MAP) -> PathMap:
    document = tomllib.loads(path.read_text(encoding="utf-8"))
    if document.get("schema_version") != 1:
        raise ValueError(f"{path}: schema_version must be 1")
    roots = tuple(str(root) for root in document.get("absolute_roots", ()))
    for root in roots:
        if not (root.startswith("/") and root.endswith("/")):
            raise ValueError(f"{path}: absolute root {root!r} must start and end with '/'")
    basenames = {
        str(name): _repository_path(str(target)) for name, target in document.get("basename", {}).items()
    }
    for name in basenames:
        if "/" in name:
            raise ValueError(f"{path}: basename key {name!r} holds a '/'")
    moves = {
        _repository_path(str(old)): _repository_path(str(new))
        for old, new in document.get("path", {}).items()
    }
    return PathMap(roots, basenames, moves)


class TagTree:
    """The files that git tracked at the tag."""

    def __init__(self, root: Path = ROOT, tag: str = TAG) -> None:
        self.root = root
        self.tag = tag
        try:
            commit = self._git("rev-parse", "--verify", "--quiet", f"refs/tags/{tag}^{{commit}}")
        except RuntimeError as error:
            raise TagUnavailable(f"{root} has no readable tag {tag!r}: {error}") from error
        self.commit = commit.decode("ascii").strip()
        listing = self._git("ls-tree", "-r", "-z", "--name-only", self.commit)
        self.paths = frozenset(name.decode("utf-8") for name in listing.split(b"\0") if name)

    def _git(self, *args: str) -> bytes:
        try:
            result = subprocess.run(["git", "-C", str(self.root), *args], capture_output=True, check=False)
        except OSError as error:
            raise RuntimeError(f"git cannot run: {error}") from error
        if result.returncode != 0:
            detail = result.stderr.decode("utf-8", errors="replace").strip()
            raise RuntimeError(f"git {' '.join(args)} exited {result.returncode}: {detail}")
        return result.stdout

    def read(self, path: str) -> bytes:
        return self._git("cat-file", "blob", f"{self.commit}:{path}")


def canonical_json_sha256(value: object) -> str:
    """Hash a JSON value with the canonical form of the Aggro trainer.

    The rule copies ``_canonical_json_sha256`` in train_aggro_hal.py, which
    wrote the canonical hashes that the Aggro records hold.
    """

    canonical = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _load(data: bytes, path: str) -> object:
    return yaml.safe_load(data) if path.endswith((".yaml", ".yml")) else json.loads(data)


def digest(data: bytes, method: str, path: str) -> str:
    """Hash a file's bytes the way a record hashed them."""

    if method == "bytes":
        return hashlib.sha256(data).hexdigest()
    if method == "canonical-json":
        return canonical_json_sha256(_load(data, path))
    raise ValueError(f"unknown hash method {method!r}")


def _pointer(parts: tuple[str, ...]) -> str:
    return "".join("/" + part.replace("~", "~0").replace("/", "~1") for part in parts)


def _binding(parent: Mapping[str, object], name: str, expected: str, pointer: str) -> Binding:
    if _FILE_KEY.fullmatch(name):
        return Binding(pointer, expected, name, "bytes", None)
    if name in _PAIRED_FIELDS:
        sibling, method = _PAIRED_FIELDS[name]
        target = parent.get(sibling)
        if isinstance(target, str):
            return Binding(pointer, expected, target, method, None)
    if name.endswith(_DIGEST_SUFFIXES):
        return Binding(pointer, expected, None, None, "artifact digest, not the hash of one file")
    if any(str(sibling) == "url" or str(sibling).endswith("_url") for sibling in parent):
        return Binding(pointer, expected, None, None, "the hash binds a download URL")
    return Binding(pointer, expected, None, None, "no field beside the hash names its file")


def bindings_of(document: object) -> list[Binding]:
    """List every SHA-256 value in a record, in document order."""

    found: list[Binding] = []

    def visit(value: object, parts: tuple[str, ...]) -> None:
        if isinstance(value, Mapping):
            for raw_name, item in value.items():
                name = str(raw_name)
                if isinstance(item, str) and _SHA256.fullmatch(item):
                    found.append(_binding(value, name, item, _pointer(parts + (name,))))
                else:
                    visit(item, parts + (name,))
        elif isinstance(value, list):
            for index, item in enumerate(value):
                if isinstance(item, str) and _SHA256.fullmatch(item):
                    reason = "a list holds the hash without a file"
                    found.append(Binding(_pointer(parts + (str(index),)), item, None, None, reason))
                else:
                    visit(item, parts + (str(index),))

    visit(document, ())
    return found


class _Verifier:
    def __init__(self, root: Path, tag: TagTree, path_map: PathMap) -> None:
        self.root = root
        self.tag = tag
        self.path_map = path_map
        self._bytes: dict[tuple[str, str], bytes | None] = {}

    def _read(self, source: str, path: str) -> bytes | None:
        key = (source, path)
        if key not in self._bytes:
            if source == "git":
                self._bytes[key] = self.tag.read(path)
            else:
                file = self.root / self.path_map.current(path)
                self._bytes[key] = file.read_bytes() if file.is_file() else None
        return self._bytes[key]

    def verify(self, binding: Binding) -> dict[str, object]:
        entry: dict[str, object] = {
            "pointer": binding.pointer,
            "expected": binding.expected,
            "path": None,
            "source": None,
            "method": binding.method,
            "status": "unverifiable",
            "reason": binding.reason,
        }
        if binding.key is None or binding.method is None:
            return entry
        path, reason = self.path_map.historical(binding.key)
        if path is None:
            entry["reason"] = reason
            return entry
        entry["path"] = path
        entry["source"] = "git" if path in self.tag.paths else "disk"
        if entry["source"] == "disk":
            file = self.root / self.path_map.current(path)
            if not file.exists():
                entry["status"] = "missing"
                return entry
            if not file.is_file():
                entry["reason"] = "the path names a directory"
                return entry
        data = self._read(str(entry["source"]), path)
        assert data is not None
        try:
            actual = digest(data, binding.method, path)
        except (ValueError, yaml.YAMLError):
            entry["reason"] = "the file does not parse for its canonical hash"
            return entry
        entry["status"] = "verified" if actual == binding.expected else "mismatch"
        return entry


def report(
    root: Path = ROOT,
    *,
    path_map: PathMap | None = None,
    tag: TagTree | None = None,
) -> dict[str, object]:
    """Verify every binding of every record and return the report."""

    path_map = load_path_map() if path_map is None else path_map
    tag = TagTree(root) if tag is None else tag
    verifier = _Verifier(root, tag, path_map)
    summary = dict.fromkeys(STATUSES, 0)
    records: dict[str, object] = {}
    for name in RECORDS:
        file = root / path_map.current(name)
        if not file.is_file():
            records[name] = {"sha256": None, "bindings": []}
            continue
        data = file.read_bytes()
        entries = [verifier.verify(binding) for binding in bindings_of(_load(data, name))]
        for entry in entries:
            summary[str(entry["status"])] += 1
        records[name] = {"sha256": hashlib.sha256(data).hexdigest(), "bindings": entries}
    return {
        "schema": SCHEMA,
        "tag": tag.tag,
        "tag_commit": tag.commit,
        "summary": summary,
        "records": records,
    }


def compare(baseline: Mapping[str, object], current: Mapping[str, object]) -> list[Difference]:
    """List the differences of a fresh report from the baseline.

    The summary counts derive from the bindings, so the comparison skips them.
    """

    differences = [
        Difference("", None, None, f"{name}: {baseline.get(name)!r} -> {current.get(name)!r}")
        for name in ("schema", "tag", "tag_commit")
        if baseline.get(name) != current.get(name)
    ]
    old_records = baseline.get("records", {})
    new_records = current.get("records", {})
    assert isinstance(old_records, Mapping) and isinstance(new_records, Mapping)
    for name in dict.fromkeys([*old_records, *new_records]):
        old, new = old_records.get(name), new_records.get(name)
        if old is None or new is None:
            absent = "the baseline" if old is None else "the report"
            differences.append(Difference(name, None, None, f"record absent from {absent}"))
            continue
        if old.get("sha256") != new.get("sha256"):
            message = f"sha256: {old.get('sha256')} -> {new.get('sha256')}"
            differences.append(Difference(name, None, None, message))
        old_bindings = {entry["pointer"]: entry for entry in old.get("bindings", [])}
        new_bindings = {entry["pointer"]: entry for entry in new.get("bindings", [])}
        for pointer in dict.fromkeys([*old_bindings, *new_bindings]):
            before, after = old_bindings.get(pointer), new_bindings.get(pointer)
            if before == after:
                continue
            source = (after or before).get("source")
            if before is None or after is None:
                message = f"binding absent from {'the baseline' if before is None else 'the report'}"
            else:
                message = "; ".join(
                    f"{key}: {before.get(key)!r} -> {after.get(key)!r}"
                    for key in dict.fromkeys([*before, *after])
                    if before.get(key) != after.get(key)
                )
            differences.append(Difference(name, pointer, source, message))
    return differences


def dumps(value: object) -> str:
    return json.dumps(value, indent=2) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m hal_lab.provenance",
        description=(
            f"Verify the SHA-256 bindings of the frozen Hal evidence records against git history at {TAG!r}."
        ),
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--report", action="store_true", help="print the verification report as JSON")
    mode.add_argument("--check", action="store_true", help="compare a fresh report with the baseline")
    parser.add_argument(
        "--baseline",
        type=Path,
        default=BASELINE,
        help=f"the baseline that --check reads (default: {BASELINE.relative_to(ROOT).as_posix()})",
    )
    args = parser.parse_args(argv)
    try:
        current = report()
    except TagUnavailable as error:
        print(f"provenance: {error}", file=sys.stderr)
        return 2
    if args.report:
        sys.stdout.write(dumps(current))
        return 0
    differences = compare(json.loads(args.baseline.read_text(encoding="utf-8")), current)
    for difference in differences:
        print(difference)
    if differences:
        print(f"provenance: {len(differences)} differences from {args.baseline}", file=sys.stderr)
        return 1
    counts = ", ".join(f"{count} {status}" for status, count in current["summary"].items())
    print(f"provenance: the report equals the baseline ({counts})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
