"""Versioned, reconstructable replay records for the Python RL pipeline.

This module deliberately does not replace the legacy ``ValueTarget`` corpus
format yet.  It provides the strict V2 boundary that later self-play and
training phases can adopt without teaching the solver or engine about replay.

Replay shards use a pickle-free NPZ payload and a canonical JSON manifest.  A
writer publishes the payload first and the manifest last; the manifest is the
commit marker, and loaders verify its payload digest before reading arrays.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, fields
from enum import StrEnum
import hashlib
import json
import math
import os
from pathlib import Path
import uuid

import numpy as np

from stl.engine.actions import ACTION_SIZE, legal_mask
from stl.engine.game import Game, Player, Referee
from stl.learning.model import (
    FEATURE_DIM,
    FEATURE_SCHEMA_VERSION,
    LEGACY_FEATURE_DIM,
    LEGACY_FEATURE_SCHEMA_VERSION,
    extract_features,
    extract_features_v1,
)
from stl.solver.exact import ExactPublicState, exact_public_state


RECORD_SCHEMA_V2 = "stl.training-record.v2"
REPLAY_MANIFEST_SCHEMA_V2 = "stl.replay-shard.v2"
ENGINE_SCHEMA_V1 = "stl.exact-public-state.v1"
ACTION_SCHEMA_V1 = "stl.literal-seconds.padding0.normal1-60.baku-leap-dropper61.v1"
FEATURE_SCHEMA_V1 = LEGACY_FEATURE_SCHEMA_VERSION
FEATURE_SCHEMA_V2 = FEATURE_SCHEMA_VERSION
DEFAULT_FEATURE_SCHEMA = FEATURE_SCHEMA_V2

_SHA256_HEX_LENGTH = 64
_MANIFEST_SUFFIX = ".manifest.json"


class ReplayValidationError(ValueError):
    """Raised when a replay record, payload, or manifest is not trustworthy."""


class TargetKind(StrEnum):
    """Semantic meaning of a value target, independent of its source label."""

    EXACT_VALUE = "exact_value"
    TABLEBASE_VALUE = "tablebase_value"
    SEARCH_BOOTSTRAP_VALUE = "search_bootstrap_value"
    TERMINAL_OUTCOME = "terminal_outcome"


class ShardRole(StrEnum):
    """Whether a shard may participate in replay/training."""

    REPLAY = "replay"
    EXTERNAL_RULER = "external_ruler"


def _readonly_float32(values: np.ndarray | Sequence[float]) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32).copy()
    array.setflags(write=False)
    return array


def _readonly_bool(values: np.ndarray | Sequence[bool]) -> np.ndarray:
    array = np.asarray(values, dtype=np.bool_).copy()
    array.setflags(write=False)
    return array


def _normalize_rng_seeds(
    seeds: Mapping[str, int] | Sequence[tuple[str, int]],
) -> tuple[tuple[str, int], ...]:
    items = seeds.items() if isinstance(seeds, Mapping) else seeds
    normalized = tuple(sorted((str(name), int(seed)) for name, seed in items))
    names = [name for name, _seed in normalized]
    if len(names) != len(set(names)):
        raise ReplayValidationError("rng_seeds contains duplicate names")
    if any(not name for name in names):
        raise ReplayValidationError("rng seed names must be non-empty")
    return normalized


@dataclass(frozen=True)
class TrainingRecordV2:
    """One reconstructable policy/value training row.

    ``source`` is an operational provenance tag.  ``target_kind`` separately
    declares the mathematical meaning of ``value`` so callers never infer it
    from a source string or overload a search-horizon field with iterations.
    """

    features: np.ndarray
    exact_state: ExactPublicState
    value: float
    target_kind: TargetKind
    source: str
    dropper_dist: np.ndarray
    checker_dist: np.ndarray
    dropper_legal_mask: np.ndarray
    checker_legal_mask: np.ndarray
    episode_id: str = ""
    half_round_index: int = 0
    truncated: bool = False
    parent_checkpoint_digest: str = ""
    search_config_digest: str = ""
    engine_schema: str = ENGINE_SCHEMA_V1
    action_schema: str = ACTION_SCHEMA_V1
    feature_schema: str = DEFAULT_FEATURE_SCHEMA
    rng_seeds: tuple[tuple[str, int], ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "features", _readonly_float32(self.features))
        object.__setattr__(
            self, "exact_state", _normalize_exact_state(self.exact_state)
        )
        object.__setattr__(self, "value", float(self.value))
        object.__setattr__(self, "dropper_dist", _readonly_float32(self.dropper_dist))
        object.__setattr__(self, "checker_dist", _readonly_float32(self.checker_dist))
        object.__setattr__(
            self, "dropper_legal_mask", _readonly_bool(self.dropper_legal_mask)
        )
        object.__setattr__(
            self, "checker_legal_mask", _readonly_bool(self.checker_legal_mask)
        )
        object.__setattr__(self, "target_kind", TargetKind(self.target_kind))
        object.__setattr__(self, "episode_id", str(self.episode_id))
        object.__setattr__(self, "source", str(self.source))
        object.__setattr__(self, "half_round_index", int(self.half_round_index))
        object.__setattr__(self, "truncated", bool(self.truncated))
        object.__setattr__(
            self, "parent_checkpoint_digest", str(self.parent_checkpoint_digest)
        )
        object.__setattr__(self, "search_config_digest", str(self.search_config_digest))
        object.__setattr__(self, "engine_schema", str(self.engine_schema))
        object.__setattr__(self, "action_schema", str(self.action_schema))
        object.__setattr__(self, "feature_schema", str(self.feature_schema))
        object.__setattr__(self, "rng_seeds", _normalize_rng_seeds(self.rng_seeds))


@dataclass(frozen=True)
class FeatureCollision:
    """Distinct exact states collapsed to one feature vector."""

    feature_hash: str
    record_indices: tuple[int, ...]
    exact_state_hashes: tuple[str, ...]
    values: tuple[float, ...]

    @property
    def value_span(self) -> float:
        return max(self.values) - min(self.values)


_STATE_FIELD_NAMES = tuple(field.name for field in fields(ExactPublicState))
_STATE_FLOAT_FIELDS = {
    "p1_physicality",
    "p1_cylinder",
    "p1_ttd",
    "p2_physicality",
    "p2_cylinder",
    "p2_ttd",
    "game_clock",
}
_STATE_INT_FIELDS = {
    "p1_deaths",
    "p2_deaths",
    "referee_cprs",
    "current_half",
    "round_num",
}
_STATE_BOOL_FIELDS = {"p1_alive", "p2_alive", "game_over"}
_STATE_STR_FIELDS = {"p1_name", "p2_name", "first_dropper_name"}
_STATE_OPTIONAL_STR_FIELDS = {"winner_name", "loser_name"}


def _canonical_json_bytes(value: object) -> bytes:
    try:
        text = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise ReplayValidationError(
            f"value is not canonical-JSON serializable: {exc}"
        ) from exc
    return (text + "\n").encode("utf-8")


def _normalize_exact_state(
    state: ExactPublicState | Mapping[str, object],
) -> ExactPublicState:
    raw = asdict(state) if isinstance(state, ExactPublicState) else dict(state)
    actual = set(raw)
    expected = set(_STATE_FIELD_NAMES)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise ReplayValidationError(
            f"ExactPublicState fields mismatch: missing={missing}, extra={extra}"
        )

    for name in _STATE_FLOAT_FIELDS:
        value = raw[name]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ReplayValidationError(f"ExactPublicState.{name} must be numeric")
        if not math.isfinite(float(value)):
            raise ReplayValidationError(f"ExactPublicState.{name} must be finite")
        raw[name] = float(value)
    for name in _STATE_INT_FIELDS:
        value = raw[name]
        if isinstance(value, bool) or not isinstance(value, int):
            raise ReplayValidationError(f"ExactPublicState.{name} must be an integer")
        raw[name] = int(value)
    for name in _STATE_BOOL_FIELDS:
        if not isinstance(raw[name], bool):
            raise ReplayValidationError(f"ExactPublicState.{name} must be boolean")
    for name in _STATE_STR_FIELDS:
        if not isinstance(raw[name], str):
            raise ReplayValidationError(f"ExactPublicState.{name} must be a string")
    for name in _STATE_OPTIONAL_STR_FIELDS:
        if raw[name] is not None and not isinstance(raw[name], str):
            raise ReplayValidationError(
                f"ExactPublicState.{name} must be a string or null"
            )
    return ExactPublicState(**raw)


def canonical_exact_state_json(
    state: ExactPublicState | Mapping[str, object],
) -> str:
    """Serialize an exact public state with stable key ordering and floats."""

    normalized = _normalize_exact_state(state)
    return _canonical_json_bytes(asdict(normalized)).decode("utf-8").rstrip("\n")


def exact_state_from_json(payload: str) -> ExactPublicState:
    try:
        raw = json.loads(payload)
    except (TypeError, json.JSONDecodeError) as exc:
        raise ReplayValidationError(f"invalid ExactPublicState JSON: {exc}") from exc
    if not isinstance(raw, dict):
        raise ReplayValidationError("ExactPublicState JSON must contain an object")
    state = _normalize_exact_state(raw)
    if canonical_exact_state_json(state) != payload:
        raise ReplayValidationError("ExactPublicState JSON is not canonical")
    return state


def exact_state_hash(state: ExactPublicState | Mapping[str, object]) -> str:
    payload = canonical_exact_state_json(state).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def reconstruct_game(state: ExactPublicState | Mapping[str, object]) -> Game:
    """Rebuild a mutable engine game and prove its public state is identical."""

    normalized = _normalize_exact_state(state)
    p1 = Player(name=normalized.p1_name, physicality=normalized.p1_physicality)
    p1.cylinder = normalized.p1_cylinder
    p1.ttd = normalized.p1_ttd
    p1.deaths = normalized.p1_deaths
    p1.alive = normalized.p1_alive

    p2 = Player(name=normalized.p2_name, physicality=normalized.p2_physicality)
    p2.cylinder = normalized.p2_cylinder
    p2.ttd = normalized.p2_ttd
    p2.deaths = normalized.p2_deaths
    p2.alive = normalized.p2_alive

    if p1.name == p2.name:
        raise ReplayValidationError("ExactPublicState player names must be distinct")
    by_name = {p1.name: p1, p2.name: p2}

    def player_named(name: str | None, field_name: str) -> Player | None:
        if name is None:
            return None
        if name not in by_name:
            raise ReplayValidationError(
                f"ExactPublicState.{field_name}={name!r} is not a stored player"
            )
        return by_name[name]

    first_dropper = player_named(normalized.first_dropper_name, "first_dropper_name")
    referee = Referee()
    referee.cprs_performed = normalized.referee_cprs
    game = Game(
        player1=p1,
        player2=p2,
        referee=referee,
        first_dropper=first_dropper,
    )
    game.seed(0)
    game.game_clock = normalized.game_clock
    game.current_half = normalized.current_half
    game.round_num = normalized.round_num
    game.game_over = normalized.game_over
    game.winner = player_named(normalized.winner_name, "winner_name")
    game.loser = player_named(normalized.loser_name, "loser_name")
    rebuilt = exact_public_state(game)
    if rebuilt != normalized:
        raise ReplayValidationError(
            "reconstructed Game does not equal its stored ExactPublicState"
        )
    return game


def _validate_digest(value: str, field_name: str) -> None:
    if not value:
        return
    if len(value) != _SHA256_HEX_LENGTH or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise ReplayValidationError(
            f"{field_name} must be empty or a lowercase SHA-256 digest"
        )


def _expected_role_masks(game: Game) -> tuple[np.ndarray, np.ndarray]:
    if game.game_over:
        empty = np.zeros(ACTION_SIZE, dtype=np.bool_)
        return empty, empty.copy()
    dropper, checker = game.get_roles_for_half(game.current_half)
    duration = game.get_turn_duration()
    return (
        legal_mask(dropper.name, "dropper", duration),
        legal_mask(checker.name, "checker", duration),
    )


def _validate_policy(
    distribution: np.ndarray,
    mask: np.ndarray,
    *,
    name: str,
) -> None:
    if distribution.shape != (ACTION_SIZE,):
        raise ReplayValidationError(
            f"{name} must have action size {ACTION_SIZE}, got {distribution.shape}"
        )
    if mask.shape != (ACTION_SIZE,):
        raise ReplayValidationError(
            f"{name} legal mask must have action size {ACTION_SIZE}, got {mask.shape}"
        )
    if not np.isfinite(distribution).all() or np.any(distribution < 0.0):
        raise ReplayValidationError(f"{name} must be finite and non-negative")
    if mask[0] or distribution[0] != 0.0:
        raise ReplayValidationError(f"{name} action index 0 must be illegal padding")
    if np.any(distribution[~mask] != 0.0):
        raise ReplayValidationError(f"{name} assigns probability to an illegal action")
    total = float(distribution.sum(dtype=np.float64))
    if total != 0.0 and not math.isclose(total, 1.0, abs_tol=1e-5):
        raise ReplayValidationError(f"{name} probability mass must sum to 0 or 1")


def validate_record(
    record: TrainingRecordV2,
    *,
    expected_feature_schema: str | None = DEFAULT_FEATURE_SCHEMA,
    expected_feature_dim: int | None = FEATURE_DIM,
) -> Game:
    """Validate one record and return its exactly reconstructed game."""

    if record.engine_schema != ENGINE_SCHEMA_V1:
        raise ReplayValidationError(
            f"unsupported engine schema {record.engine_schema!r}"
        )
    if record.action_schema != ACTION_SCHEMA_V1:
        raise ReplayValidationError(
            f"unsupported action schema {record.action_schema!r}"
        )
    if (
        expected_feature_schema is not None
        and record.feature_schema != expected_feature_schema
    ):
        raise ReplayValidationError(
            f"feature schema mismatch: {record.feature_schema!r} != "
            f"{expected_feature_schema!r}"
        )
    if expected_feature_dim is not None and record.features.shape != (
        expected_feature_dim,
    ):
        raise ReplayValidationError(
            f"feature dimension mismatch: {record.features.shape} != "
            f"{(expected_feature_dim,)}"
        )
    if not np.isfinite(record.features).all():
        raise ReplayValidationError("features must be finite")
    if not math.isfinite(record.value) or not -1.000001 <= record.value <= 1.000001:
        raise ReplayValidationError("value must be finite and in [-1, 1]")
    if not record.source:
        raise ReplayValidationError("source must be non-empty")
    if record.half_round_index < 0:
        raise ReplayValidationError("half_round_index must be non-negative")
    _validate_digest(record.parent_checkpoint_digest, "parent_checkpoint_digest")
    _validate_digest(record.search_config_digest, "search_config_digest")

    game = reconstruct_game(record.exact_state)
    _validate_policy(
        record.dropper_dist, record.dropper_legal_mask, name="dropper_dist"
    )
    _validate_policy(
        record.checker_dist, record.checker_legal_mask, name="checker_dist"
    )
    expected_drop_mask, expected_check_mask = _expected_role_masks(game)
    if not np.array_equal(record.dropper_legal_mask, expected_drop_mask):
        raise ReplayValidationError("dropper legal mask does not match the exact state")
    if not np.array_equal(record.checker_legal_mask, expected_check_mask):
        raise ReplayValidationError("checker legal mask does not match the exact state")

    if record.feature_schema == FEATURE_SCHEMA_V2:
        rebuilt_features = extract_features(game)
        if not np.array_equal(record.features, rebuilt_features):
            raise ReplayValidationError(
                "stored V2 features do not match features reconstructed from exact state"
            )
    elif record.feature_schema == FEATURE_SCHEMA_V1:
        if record.features.shape != (LEGACY_FEATURE_DIM,):
            raise ReplayValidationError("stored V1 feature dimension is invalid")
        if not np.array_equal(record.features, extract_features_v1(game)):
            raise ReplayValidationError(
                "stored V1 features do not match the explicit legacy adapter"
            )
    return game


def manifest_path_for(shard_path: str | Path) -> Path:
    path = Path(shard_path)
    return path.with_name(path.name + _MANIFEST_SUFFIX)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _array_digest(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(contiguous.dtype).encode("ascii"))
    digest.update(_canonical_json_bytes(list(contiguous.shape)))
    digest.update(contiguous.tobytes(order="C"))
    return digest.hexdigest()


_ARRAY_NAMES = (
    "features",
    "values",
    "exact_states_json",
    "exact_state_hashes",
    "target_kinds",
    "sources",
    "dropper_dists",
    "checker_dists",
    "dropper_legal_masks",
    "checker_legal_masks",
    "episode_ids",
    "half_round_indices",
    "truncated",
    "parent_checkpoint_digests",
    "search_config_digests",
    "rng_seeds_json",
)


def _records_to_arrays(
    records: Sequence[TrainingRecordV2], *, feature_dim: int
) -> dict[str, np.ndarray]:
    count = len(records)
    if count:
        features = np.stack([record.features for record in records]).astype(np.float32)
        dropper_dists = np.stack([record.dropper_dist for record in records]).astype(
            np.float32
        )
        checker_dists = np.stack([record.checker_dist for record in records]).astype(
            np.float32
        )
        dropper_masks = np.stack(
            [record.dropper_legal_mask for record in records]
        ).astype(np.bool_)
        checker_masks = np.stack(
            [record.checker_legal_mask for record in records]
        ).astype(np.bool_)
    else:
        features = np.zeros((0, feature_dim), dtype=np.float32)
        dropper_dists = np.zeros((0, ACTION_SIZE), dtype=np.float32)
        checker_dists = np.zeros((0, ACTION_SIZE), dtype=np.float32)
        dropper_masks = np.zeros((0, ACTION_SIZE), dtype=np.bool_)
        checker_masks = np.zeros((0, ACTION_SIZE), dtype=np.bool_)

    state_json = [canonical_exact_state_json(record.exact_state) for record in records]
    return {
        "features": features,
        "values": np.asarray([record.value for record in records], dtype=np.float32),
        "exact_states_json": np.asarray(state_json, dtype=np.str_),
        "exact_state_hashes": np.asarray(
            [exact_state_hash(record.exact_state) for record in records], dtype=np.str_
        ),
        "target_kinds": np.asarray(
            [record.target_kind.value for record in records], dtype=np.str_
        ),
        "sources": np.asarray([record.source for record in records], dtype=np.str_),
        "dropper_dists": dropper_dists,
        "checker_dists": checker_dists,
        "dropper_legal_masks": dropper_masks,
        "checker_legal_masks": checker_masks,
        "episode_ids": np.asarray(
            [record.episode_id for record in records], dtype=np.str_
        ),
        "half_round_indices": np.asarray(
            [record.half_round_index for record in records], dtype=np.int64
        ),
        "truncated": np.asarray(
            [record.truncated for record in records], dtype=np.bool_
        ),
        "parent_checkpoint_digests": np.asarray(
            [record.parent_checkpoint_digest for record in records], dtype=np.str_
        ),
        "search_config_digests": np.asarray(
            [record.search_config_digest for record in records], dtype=np.str_
        ),
        "rng_seeds_json": np.asarray(
            [
                _canonical_json_bytes(dict(record.rng_seeds))
                .decode("utf-8")
                .rstrip("\n")
                for record in records
            ],
            dtype=np.str_,
        ),
    }


def _validate_array_layout(
    arrays: Mapping[str, np.ndarray],
    *,
    count: int,
    feature_dim: int,
    action_size: int,
) -> None:
    names = set(arrays)
    required = set(_ARRAY_NAMES)
    if names != required:
        raise ReplayValidationError(
            f"replay arrays mismatch: missing={sorted(required - names)}, "
            f"extra={sorted(names - required)}"
        )
    for name, array in arrays.items():
        if array.dtype.hasobject:
            raise ReplayValidationError(f"array {name!r} has forbidden object dtype")
        if array.ndim == 0 or array.shape[0] != count:
            raise ReplayValidationError(
                f"array {name!r} record count {array.shape} does not match {count}"
            )

    expected_shapes = {
        "features": (count, feature_dim),
        "values": (count,),
        "dropper_dists": (count, action_size),
        "checker_dists": (count, action_size),
        "dropper_legal_masks": (count, action_size),
        "checker_legal_masks": (count, action_size),
    }
    for name, shape in expected_shapes.items():
        if arrays[name].shape != shape:
            raise ReplayValidationError(
                f"array {name!r} shape {arrays[name].shape} does not match {shape}"
            )
    if arrays["features"].dtype != np.float32:
        raise ReplayValidationError("features must use float32")
    if arrays["values"].dtype != np.float32:
        raise ReplayValidationError("values must use float32")
    for name in ("dropper_dists", "checker_dists"):
        if arrays[name].dtype != np.float32:
            raise ReplayValidationError(f"{name} must use float32")
    for name in ("dropper_legal_masks", "checker_legal_masks", "truncated"):
        if arrays[name].dtype != np.bool_:
            raise ReplayValidationError(f"{name} must use bool")
    if arrays["half_round_indices"].dtype != np.int64:
        raise ReplayValidationError("half_round_indices must use int64")


def _array_manifest(arrays: Mapping[str, np.ndarray]) -> dict[str, dict[str, object]]:
    return {
        name: {
            "dtype": str(array.dtype),
            "shape": list(array.shape),
            "sha256": _array_digest(array),
        }
        for name, array in sorted(arrays.items())
    }


def save_replay_shard(
    records: Sequence[TrainingRecordV2],
    shard_path: str | Path,
    *,
    shard_role: ShardRole = ShardRole.REPLAY,
    expected_feature_schema: str = DEFAULT_FEATURE_SCHEMA,
    expected_feature_dim: int = FEATURE_DIM,
) -> dict[str, object]:
    """Validate and atomically publish a V2 replay shard and its manifest."""

    path = Path(shard_path)
    if path.suffix.lower() != ".npz":
        raise ReplayValidationError("replay shard path must end in .npz")
    role = ShardRole(shard_role)
    records = tuple(records)
    for record in records:
        validate_record(
            record,
            expected_feature_schema=expected_feature_schema,
            expected_feature_dim=expected_feature_dim,
        )
    arrays = _records_to_arrays(records, feature_dim=expected_feature_dim)
    _validate_array_layout(
        arrays,
        count=len(records),
        feature_dim=expected_feature_dim,
        action_size=ACTION_SIZE,
    )

    path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path = manifest_path_for(path)
    nonce = f"{os.getpid()}.{uuid.uuid4().hex}"
    payload_tmp = path.with_name(f".{path.name}.{nonce}.tmp.npz")
    manifest_tmp = manifest_path.with_name(f".{manifest_path.name}.{nonce}.tmp")
    try:
        np.savez_compressed(payload_tmp, **arrays)
        with np.load(payload_tmp, allow_pickle=False) as stored:
            stored_arrays = {
                name: np.array(stored[name], copy=True) for name in stored.files
            }
        _validate_array_layout(
            stored_arrays,
            count=len(records),
            feature_dim=expected_feature_dim,
            action_size=ACTION_SIZE,
        )
        manifest: dict[str, object] = {
            "schema": REPLAY_MANIFEST_SCHEMA_V2,
            "record_schema": RECORD_SCHEMA_V2,
            "engine_schema": ENGINE_SCHEMA_V1,
            "action_schema": ACTION_SCHEMA_V1,
            "feature_schema": expected_feature_schema,
            "feature_dim": expected_feature_dim,
            "action_size": ACTION_SIZE,
            "record_count": len(records),
            "shard_role": role.value,
            "data_file": path.name,
            "data_sha256": _sha256_file(payload_tmp),
            "arrays": _array_manifest(stored_arrays),
            "parent_checkpoint_digests": sorted(
                {record.parent_checkpoint_digest for record in records}
            ),
            "search_config_digests": sorted(
                {record.search_config_digest for record in records}
            ),
        }
        manifest_tmp.write_bytes(_canonical_json_bytes(manifest))
        os.replace(payload_tmp, path)
        # The manifest is the commit marker and is intentionally published last.
        os.replace(manifest_tmp, manifest_path)
        return manifest
    finally:
        payload_tmp.unlink(missing_ok=True)
        manifest_tmp.unlink(missing_ok=True)


def load_replay_manifest(shard_path: str | Path) -> dict[str, object]:
    path = Path(shard_path)
    manifest_path = manifest_path_for(path)
    if not manifest_path.exists():
        raise ReplayValidationError(f"replay manifest not found: {manifest_path}")
    try:
        raw_manifest = manifest_path.read_bytes()
        manifest = json.loads(raw_manifest)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ReplayValidationError(f"invalid replay manifest: {exc}") from exc
    if not isinstance(manifest, dict):
        raise ReplayValidationError("replay manifest must contain a JSON object")
    if raw_manifest != _canonical_json_bytes(manifest):
        raise ReplayValidationError("replay manifest is not canonical JSON")
    if manifest.get("schema") != REPLAY_MANIFEST_SCHEMA_V2:
        raise ReplayValidationError(
            f"replay manifest schema mismatch: {manifest.get('schema')!r}"
        )
    if manifest.get("record_schema") != RECORD_SCHEMA_V2:
        raise ReplayValidationError("training record schema mismatch")
    if manifest.get("engine_schema") != ENGINE_SCHEMA_V1:
        raise ReplayValidationError("engine schema mismatch")
    if manifest.get("action_schema") != ACTION_SCHEMA_V1:
        raise ReplayValidationError("action schema mismatch")
    if manifest.get("data_file") != path.name:
        raise ReplayValidationError("manifest data_file does not match shard path")
    try:
        ShardRole(str(manifest.get("shard_role")))
    except ValueError as exc:
        raise ReplayValidationError("unknown replay shard role") from exc
    return manifest


def _manifest_int(manifest: Mapping[str, object], name: str) -> int:
    value = manifest.get(name)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ReplayValidationError(f"manifest {name} must be an integer")
    return value


def _verify_manifest_arrays(
    arrays: Mapping[str, np.ndarray], manifest: Mapping[str, object]
) -> None:
    specs = manifest.get("arrays")
    if not isinstance(specs, dict) or set(specs) != set(_ARRAY_NAMES):
        raise ReplayValidationError("manifest array inventory is incomplete")
    for name, array in arrays.items():
        spec = specs.get(name)
        if not isinstance(spec, dict):
            raise ReplayValidationError(f"manifest array spec missing for {name}")
        if spec.get("dtype") != str(array.dtype):
            raise ReplayValidationError(f"manifest dtype mismatch for {name}")
        if spec.get("shape") != list(array.shape):
            raise ReplayValidationError(f"manifest shape mismatch for {name}")
        if spec.get("sha256") != _array_digest(array):
            raise ReplayValidationError(f"manifest array digest mismatch for {name}")


def _records_from_arrays(
    arrays: Mapping[str, np.ndarray], manifest: Mapping[str, object]
) -> list[TrainingRecordV2]:
    records: list[TrainingRecordV2] = []
    count = _manifest_int(manifest, "record_count")
    for index in range(count):
        state_payload = str(arrays["exact_states_json"][index])
        state = exact_state_from_json(state_payload)
        stored_hash = str(arrays["exact_state_hashes"][index])
        if stored_hash != exact_state_hash(state):
            raise ReplayValidationError(f"exact state hash mismatch at record {index}")
        seed_payload = str(arrays["rng_seeds_json"][index])
        try:
            seed_map = json.loads(seed_payload)
        except json.JSONDecodeError as exc:
            raise ReplayValidationError(
                f"invalid RNG seed JSON at record {index}"
            ) from exc
        if not isinstance(seed_map, dict):
            raise ReplayValidationError(
                f"RNG seed JSON at record {index} must contain an object"
            )
        if _canonical_json_bytes(seed_map).decode("utf-8").rstrip("\n") != seed_payload:
            raise ReplayValidationError(
                f"RNG seed JSON at record {index} is not canonical"
            )
        try:
            target_kind = TargetKind(str(arrays["target_kinds"][index]))
        except ValueError as exc:
            raise ReplayValidationError(
                f"unknown target kind at record {index}"
            ) from exc
        records.append(
            TrainingRecordV2(
                features=arrays["features"][index],
                exact_state=state,
                value=float(arrays["values"][index]),
                target_kind=target_kind,
                source=str(arrays["sources"][index]),
                dropper_dist=arrays["dropper_dists"][index],
                checker_dist=arrays["checker_dists"][index],
                dropper_legal_mask=arrays["dropper_legal_masks"][index],
                checker_legal_mask=arrays["checker_legal_masks"][index],
                episode_id=str(arrays["episode_ids"][index]),
                half_round_index=int(arrays["half_round_indices"][index]),
                truncated=bool(arrays["truncated"][index]),
                parent_checkpoint_digest=str(
                    arrays["parent_checkpoint_digests"][index]
                ),
                search_config_digest=str(arrays["search_config_digests"][index]),
                engine_schema=str(manifest["engine_schema"]),
                action_schema=str(manifest["action_schema"]),
                feature_schema=str(manifest["feature_schema"]),
                rng_seeds=seed_map,
            )
        )
    return records


def load_replay_shard(
    shard_path: str | Path,
    *,
    for_training: bool = False,
    expected_feature_schema: str = DEFAULT_FEATURE_SCHEMA,
    expected_feature_dim: int = FEATURE_DIM,
    expected_parent_checkpoint_digest: str | None = None,
) -> list[TrainingRecordV2]:
    """Load a shard only after manifest, hashes, shapes, and states validate."""

    path = Path(shard_path)
    manifest = load_replay_manifest(path)
    role = ShardRole(str(manifest["shard_role"]))
    if for_training and role is ShardRole.EXTERNAL_RULER:
        raise ReplayValidationError(
            "external-ruler shards are forbidden in training or replay"
        )
    feature_schema = manifest.get("feature_schema")
    if feature_schema != expected_feature_schema:
        raise ReplayValidationError(
            f"feature schema mismatch: {feature_schema!r} != "
            f"{expected_feature_schema!r}"
        )
    feature_dim = _manifest_int(manifest, "feature_dim")
    if feature_dim != expected_feature_dim:
        raise ReplayValidationError(
            f"feature dimension mismatch: {feature_dim} != {expected_feature_dim}"
        )
    action_size = _manifest_int(manifest, "action_size")
    if action_size != ACTION_SIZE:
        raise ReplayValidationError(
            f"action size mismatch: {action_size} != {ACTION_SIZE}"
        )
    count = _manifest_int(manifest, "record_count")
    if count < 0:
        raise ReplayValidationError("record_count must be non-negative")
    if not path.exists():
        raise ReplayValidationError(f"replay payload not found: {path}")
    if manifest.get("data_sha256") != _sha256_file(path):
        raise ReplayValidationError("replay payload SHA-256 mismatch")

    try:
        with np.load(path, allow_pickle=False) as payload:
            arrays = {
                name: np.array(payload[name], copy=True) for name in payload.files
            }
    except (OSError, KeyError, ValueError) as exc:
        raise ReplayValidationError(f"unsafe or invalid replay payload: {exc}") from exc
    _validate_array_layout(
        arrays,
        count=count,
        feature_dim=feature_dim,
        action_size=action_size,
    )
    _verify_manifest_arrays(arrays, manifest)
    parent_digests = sorted(
        {str(value) for value in arrays["parent_checkpoint_digests"]}
    )
    search_digests = sorted({str(value) for value in arrays["search_config_digests"]})
    if manifest.get("parent_checkpoint_digests") != parent_digests:
        raise ReplayValidationError("manifest parent checkpoint digests mismatch")
    if manifest.get("search_config_digests") != search_digests:
        raise ReplayValidationError("manifest search config digests mismatch")
    records = _records_from_arrays(arrays, manifest)
    for record in records:
        validate_record(
            record,
            expected_feature_schema=expected_feature_schema,
            expected_feature_dim=expected_feature_dim,
        )
    if expected_parent_checkpoint_digest is not None:
        _validate_digest(
            expected_parent_checkpoint_digest, "expected_parent_checkpoint_digest"
        )
        parents = {record.parent_checkpoint_digest for record in records}
        if parents != {expected_parent_checkpoint_digest}:
            raise ReplayValidationError(
                "checkpoint parent digest does not match replay provenance"
            )
    return records


class _DisjointSet:
    def __init__(self, size: int) -> None:
        self.parent = list(range(size))
        self.rank = [0] * size

    def find(self, item: int) -> int:
        while self.parent[item] != item:
            self.parent[item] = self.parent[self.parent[item]]
            item = self.parent[item]
        return item

    def union(self, left: int, right: int) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root == right_root:
            return
        if self.rank[left_root] < self.rank[right_root]:
            left_root, right_root = right_root, left_root
        self.parent[right_root] = left_root
        if self.rank[left_root] == self.rank[right_root]:
            self.rank[left_root] += 1


def grouped_split_indices(
    records: Sequence[TrainingRecordV2],
    *,
    validation_fraction: float = 0.1,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Split connected groups sharing an exact state *or* an episode.

    Union-find is required here: state A and state B may share episode X while
    state B also appears in episode Y.  Treating either key independently or
    using a composite key would allow that connected component to leak.
    Empty episode IDs denote non-episodic anchors and do not all form one group.
    """

    if not 0.0 < validation_fraction < 1.0:
        raise ReplayValidationError("validation_fraction must be between 0 and 1")
    count = len(records)
    if count == 0:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)
    groups = _DisjointSet(count)
    state_owner: dict[str, int] = {}
    episode_owner: dict[str, int] = {}
    for index, record in enumerate(records):
        state_key = exact_state_hash(record.exact_state)
        if state_key in state_owner:
            groups.union(index, state_owner[state_key])
        else:
            state_owner[state_key] = index
        if record.episode_id:
            if record.episode_id in episode_owner:
                groups.union(index, episode_owner[record.episode_id])
            else:
                episode_owner[record.episode_id] = index

    components: dict[int, list[int]] = {}
    for index in range(count):
        components.setdefault(groups.find(index), []).append(index)
    ordered = sorted(components.values(), key=lambda component: component[0])
    if len(ordered) < 2:
        raise ReplayValidationError(
            "at least two disconnected state/episode groups are required to split"
        )
    rng = np.random.default_rng(seed)
    permutation = rng.permutation(len(ordered))
    validation_group_count = min(
        len(ordered) - 1,
        max(1, int(round(len(ordered) * validation_fraction))),
    )
    validation_positions = set(
        int(item) for item in permutation[:validation_group_count]
    )
    train: list[int] = []
    validation: list[int] = []
    for position, component in enumerate(ordered):
        destination = validation if position in validation_positions else train
        destination.extend(component)
    return (
        np.asarray(sorted(train), dtype=np.int64),
        np.asarray(sorted(validation), dtype=np.int64),
    )


def _feature_hash(features: np.ndarray) -> str:
    array = np.asarray(features, dtype=np.float32)
    digest = hashlib.sha256()
    digest.update(_canonical_json_bytes(list(array.shape)))
    digest.update(np.ascontiguousarray(array).tobytes(order="C"))
    return digest.hexdigest()


def audit_feature_collisions(
    records: Sequence[TrainingRecordV2],
    *,
    divergent_only: bool = True,
    value_tolerance: float = 1e-6,
) -> list[FeatureCollision]:
    """Report distinct exact states sharing features, optionally by value split."""

    if value_tolerance < 0.0:
        raise ReplayValidationError("value_tolerance must be non-negative")
    grouped: dict[str, list[int]] = {}
    for index, record in enumerate(records):
        grouped.setdefault(_feature_hash(record.features), []).append(index)

    collisions: list[FeatureCollision] = []
    for feature_key, indices in sorted(grouped.items()):
        state_hashes = tuple(
            sorted({exact_state_hash(records[index].exact_state) for index in indices})
        )
        if len(state_hashes) < 2:
            continue
        values = tuple(float(records[index].value) for index in indices)
        collision = FeatureCollision(
            feature_hash=feature_key,
            record_indices=tuple(indices),
            exact_state_hashes=state_hashes,
            values=values,
        )
        if divergent_only and collision.value_span <= value_tolerance:
            continue
        collisions.append(collision)
    return collisions


__all__ = [
    "ACTION_SCHEMA_V1",
    "DEFAULT_FEATURE_SCHEMA",
    "ENGINE_SCHEMA_V1",
    "FEATURE_SCHEMA_V1",
    "FEATURE_SCHEMA_V2",
    "RECORD_SCHEMA_V2",
    "REPLAY_MANIFEST_SCHEMA_V2",
    "FeatureCollision",
    "ReplayValidationError",
    "ShardRole",
    "TargetKind",
    "TrainingRecordV2",
    "audit_feature_collisions",
    "canonical_exact_state_json",
    "exact_state_from_json",
    "exact_state_hash",
    "grouped_split_indices",
    "load_replay_manifest",
    "load_replay_shard",
    "manifest_path_for",
    "reconstruct_game",
    "save_replay_shard",
    "validate_record",
]
