"""Current fail-closed exact DTH value, interval, census, and queue store."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sqlite3
import struct
import time
from typing import Iterator, Literal, Sequence
import zlib

import numpy as np

from dth.solver import (
    CertifiedSolution,
    NTState,
    SADDLE_GAP_TOLERANCE,
    SOLVER_VERSION,
    STATE_ENCODING_VERSION,
    TARGET_SCHEMA,
    ValueInterval,
    canonical_damage_rank,
    canonical_state_id,
    solver_schema_hash,
    state_from_canonical_id,
    validate_live_state,
)


TABLEBASE_SCHEMA_VERSION = "dth-certified-interval-tablebase-v2"
COMPLETE_NAMESPACE = "complete-game"
FINITE_NAMESPACE = "finite-horizon"
Scope = Literal["complete-game-exact", "finite-horizon-exact"]
_EXPECTED_TABLES = {
    "metadata",
    "values",
    "policy_cache",
    "roots",
    "states",
    "rank_layers",
}
_EXPECTED_INDEXES = {"ix_frontier"}


class TablebaseError(RuntimeError):
    pass


class TablebaseSchemaError(TablebaseError):
    pass


class TablebaseCorruptionError(TablebaseError):
    pass


class DuplicateSolutionError(TablebaseError):
    pass


@dataclass
class TablebaseMetrics:
    value_lookup_hits: int = 0
    value_lookup_misses: int = 0
    value_lookup_seconds: float = 0.0
    policy_lookup_hits: int = 0
    policy_lookup_misses: int = 0
    policy_deserialization_seconds: float = 0.0
    policy_validation_seconds: float = 0.0
    certificate_validation_seconds: float = 0.0
    durable_commit_seconds: float = 0.0
    verify_seconds: float = 0.0


@dataclass(frozen=True)
class StoredValue:
    state: NTState
    lower_bound: float
    upper_bound: float
    value: float | None
    saddle_gap: float | None
    damage_rank: int
    child_dependencies: int
    exact: bool
    scope: Scope
    horizon: int | None

    @property
    def interval(self) -> ValueInterval:
        return ValueInterval(self.lower_bound, self.upper_bound)


def _namespace(scope: Scope) -> str:
    if scope == "complete-game-exact":
        return COMPLETE_NAMESPACE
    if scope == "finite-horizon-exact":
        return FINITE_NAMESPACE
    raise ValueError(f"unsupported exact scope {scope!r}")


def _storage_horizon(scope: Scope, horizon: int | None) -> int:
    if scope == "complete-game-exact":
        if horizon is not None:
            raise ValueError("complete-game rows cannot carry a finite horizon")
        return -1
    if horizon is None or horizon <= 0:
        raise ValueError("finite-horizon rows require horizon >= 1")
    return int(horizon)


def _policy_bytes(policy: Sequence[float], *, role: str) -> bytes:
    values = np.asarray(policy, dtype="<f8")
    if values.shape != (60,) or not np.all(np.isfinite(values)):
        raise TablebaseCorruptionError(f"{role} policy must be finite length 60")
    if np.any(values < -1e-12) or abs(float(values.sum()) - 1.0) > 1e-10:
        raise TablebaseCorruptionError(f"{role} policy is not a probability vector")
    return zlib.compress(values.tobytes(order="C"), level=9)


def _policy_from_bytes(raw: bytes, *, role: str) -> tuple[float, ...]:
    try:
        decoded = zlib.decompress(raw)
    except zlib.error as exc:
        raise TablebaseCorruptionError(f"{role} policy compression is corrupt") from exc
    if len(decoded) != 60 * 8:
        raise TablebaseCorruptionError(f"{role} policy payload has the wrong length")
    values = np.frombuffer(decoded, dtype="<f8")
    _policy_bytes(values, role=role)
    return tuple(float(value) for value in values)


def _value_digest(
    *,
    namespace: str,
    horizon: int,
    state_id: int,
    lower: float,
    upper: float,
    value: float | None,
    saddle_gap: float | None,
    rank: int,
    dependencies: int,
    exact: bool,
) -> bytes:
    payload = {
        "namespace": namespace,
        "horizon": horizon,
        "state_id": state_id,
        "lower": float(lower).hex(),
        "upper": float(upper).hex(),
        "value": None if value is None else float(value).hex(),
        "saddle_gap": None if saddle_gap is None else float(saddle_gap).hex(),
        "rank": rank,
        "dependencies": dependencies,
        "exact": exact,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).digest()


def _policy_digest(
    value_digest: bytes,
    drop_policy: bytes,
    check_policy: bytes,
) -> bytes:
    return hashlib.sha256(value_digest + drop_policy + check_policy).digest()


class CertifiedTablebase:
    """One schema and one transaction domain for the current exact workflow."""

    def __init__(self, path: str | Path, *, verify_on_open: bool = True) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        existed = self.path.exists()
        self.connection = sqlite3.connect(self.path, timeout=60.0)
        self.connection.row_factory = sqlite3.Row
        self.connection.execute("PRAGMA foreign_keys = ON")
        self.connection.execute("PRAGMA journal_mode = WAL")
        self.connection.execute("PRAGMA synchronous = FULL")
        self.metrics = TablebaseMetrics()
        if existed:
            self._assert_existing_artifact_shape()
            self._validate_metadata()
        else:
            self._initialize_schema()
        if verify_on_open:
            self.verify(full=True)

    def __enter__(self) -> "CertifiedTablebase":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def close(self) -> None:
        self.connection.close()

    @contextmanager
    def transaction(self) -> Iterator[None]:
        self.connection.execute("BEGIN IMMEDIATE")
        try:
            yield
        except BaseException:
            self.connection.rollback()
            raise
        else:
            self.connection.commit()

    @property
    def metadata(self) -> dict[str, str]:
        return {
            str(row["key"]): str(row["value"])
            for row in self.connection.execute(
                "SELECT key, value FROM metadata ORDER BY key"
            )
        }

    def _table_names(self) -> set[str]:
        return {
            str(row[0])
            for row in self.connection.execute(
                "SELECT name FROM sqlite_master "
                "WHERE type='table' AND name NOT LIKE 'sqlite_%'"
            )
        }

    def _index_names(self) -> set[str]:
        return {
            str(row[0])
            for row in self.connection.execute(
                "SELECT name FROM sqlite_master "
                "WHERE type='index' AND name NOT LIKE 'sqlite_%'"
            )
        }

    def _assert_existing_artifact_shape(self) -> None:
        tables = self._table_names()
        if tables != _EXPECTED_TABLES:
            raise TablebaseSchemaError(
                "existing SQLite artifact is not the current exact DTH schema "
                f"(tables={sorted(tables)!r})"
            )
        indexes = self._index_names()
        if indexes != _EXPECTED_INDEXES:
            raise TablebaseSchemaError(
                "existing SQLite artifact is missing the current exact DTH indexes "
                f"(indexes={sorted(indexes)!r})"
            )

    def _initialize_schema(self) -> None:
        self.connection.executescript(
            """
            CREATE TABLE metadata (
                key TEXT PRIMARY KEY NOT NULL,
                value TEXT NOT NULL
            ) WITHOUT ROWID;

            CREATE TABLE "values" (
                namespace TEXT NOT NULL,
                horizon INTEGER NOT NULL,
                state_id INTEGER NOT NULL,
                lower_bound REAL NOT NULL,
                upper_bound REAL NOT NULL,
                value REAL,
                saddle_gap REAL,
                damage_rank INTEGER NOT NULL,
                child_dependencies INTEGER NOT NULL,
                is_exact INTEGER NOT NULL CHECK(is_exact IN (0, 1)),
                certificate_sha256 BLOB NOT NULL
                    CHECK(length(certificate_sha256) = 32),
                PRIMARY KEY(namespace, horizon, state_id),
                CHECK(lower_bound >= -1.0 AND upper_bound <= 1.0),
                CHECK(lower_bound <= upper_bound),
                CHECK(child_dependencies BETWEEN 0 AND 61),
                CHECK(
                    (is_exact = 0 AND value IS NULL AND saddle_gap IS NULL)
                    OR
                    (is_exact = 1 AND value IS NOT NULL
                     AND saddle_gap >= 0.0 AND saddle_gap <= 0.000001)
                )
            ) WITHOUT ROWID;

            CREATE TABLE policy_cache (
                namespace TEXT NOT NULL,
                horizon INTEGER NOT NULL,
                state_id INTEGER NOT NULL,
                drop_policy BLOB NOT NULL,
                check_policy BLOB NOT NULL,
                certificate_sha256 BLOB NOT NULL
                    CHECK(length(certificate_sha256) = 32),
                created_utc TEXT NOT NULL,
                PRIMARY KEY(namespace, horizon, state_id),
                FOREIGN KEY(namespace, horizon, state_id)
                    REFERENCES "values"(namespace, horizon, state_id)
                    ON DELETE CASCADE
            ) WITHOUT ROWID;

            CREATE TABLE roots (
                root_ordinal INTEGER PRIMARY KEY NOT NULL,
                raw_state_id INTEGER NOT NULL,
                state_id INTEGER NOT NULL
            ) WITHOUT ROWID;

            CREATE TABLE states (
                state_id INTEGER PRIMARY KEY NOT NULL,
                damage_rank INTEGER NOT NULL,
                census_status INTEGER NOT NULL CHECK(census_status IN (0, 1)),
                unique_child_edges INTEGER,
                new_unique_states INTEGER,
                solve_status INTEGER NOT NULL CHECK(solve_status IN (0, 1, 2))
            ) WITHOUT ROWID;

            CREATE INDEX ix_frontier
                ON states(census_status, damage_rank, state_id);

            CREATE TABLE rank_layers (
                damage_rank INTEGER PRIMARY KEY NOT NULL,
                state_count INTEGER NOT NULL,
                frontier_count INTEGER NOT NULL,
                expanded_count INTEGER NOT NULL,
                edge_count INTEGER NOT NULL,
                new_state_discoveries INTEGER NOT NULL
            ) WITHOUT ROWID;
            """
        )
        expected = self._expected_metadata()
        with self.transaction():
            for key, value in expected.items():
                self.connection.execute(
                    "INSERT INTO metadata(key, value) VALUES (?, ?)",
                    (
                        key,
                        datetime.now(timezone.utc).isoformat()
                        if key == "created_utc"
                        else value,
                    ),
                )

    @staticmethod
    def _expected_metadata() -> dict[str, str]:
        return {
            "created_utc": "",
            "policy_codec": "zlib-f64le-v1",
            "rules_schema_hash": solver_schema_hash(),
            "solver_version": SOLVER_VERSION,
            "state_encoding": STATE_ENCODING_VERSION,
            "tablebase_schema_version": TABLEBASE_SCHEMA_VERSION,
            "target_schema": TARGET_SCHEMA,
        }

    def _validate_metadata(self) -> None:
        expected = self._expected_metadata()
        existing = self.metadata
        if set(existing) != set(expected):
            raise TablebaseSchemaError("exact tablebase metadata keys mismatch")
        for key, value in expected.items():
            if key == "created_utc":
                if not existing[key]:
                    raise TablebaseSchemaError("tablebase creation timestamp is missing")
            elif existing[key] != value:
                raise TablebaseSchemaError(
                    f"tablebase {key} mismatch: {existing[key]!r} != {value!r}"
                )

    def database_size_bytes(self) -> int:
        return self.path.stat().st_size if self.path.exists() else 0

    def checkpointed_size_bytes(self) -> int:
        self.connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        return self.database_size_bytes()

    def bind_roots(self, roots: Sequence[Sequence[int]]) -> tuple[NTState, ...]:
        normalized = tuple(dict.fromkeys(validate_live_state(root) for root in roots))
        if not normalized:
            raise ValueError("exact workflow needs at least one root")
        existing = self.connection.execute(
            "SELECT root_ordinal, raw_state_id, state_id "
            "FROM roots ORDER BY root_ordinal"
        ).fetchall()
        expected = [
            (index, _raw_state_id(root), canonical_state_id(root))
            for index, root in enumerate(normalized)
        ]
        if existing:
            actual = [tuple(int(value) for value in row) for row in existing]
            if actual != expected:
                raise TablebaseSchemaError(
                    "artifact root manifest differs from the requested roots"
                )
            return normalized
        with self.transaction():
            for ordinal, raw_id, state_id in expected:
                self.connection.execute(
                    "INSERT INTO roots(root_ordinal, raw_state_id, state_id) "
                    "VALUES (?, ?, ?)",
                    (ordinal, raw_id, state_id),
                )
        return normalized

    def _validate_value_row(
        self,
        row: sqlite3.Row,
        *,
        requested_state: NTState,
        scope: Scope,
        horizon: int | None,
    ) -> StoredValue:
        namespace = _namespace(scope)
        stored_horizon = _storage_horizon(scope, horizon)
        state_id = canonical_state_id(requested_state)
        if (
            str(row["namespace"]) != namespace
            or int(row["horizon"]) != stored_horizon
            or int(row["state_id"]) != state_id
        ):
            raise TablebaseCorruptionError("value lookup returned the wrong key")
        lower = float(row["lower_bound"])
        upper = float(row["upper_bound"])
        exact = bool(int(row["is_exact"]))
        value = None if row["value"] is None else float(row["value"])
        gap = None if row["saddle_gap"] is None else float(row["saddle_gap"])
        rank = int(row["damage_rank"])
        dependencies = int(row["child_dependencies"])
        try:
            ValueInterval(lower, upper)
        except ValueError as exc:
            raise TablebaseCorruptionError("stored interval is invalid") from exc
        if rank != canonical_damage_rank(requested_state):
            raise TablebaseCorruptionError("stored damage rank mismatches state key")
        if not 0 <= dependencies <= 61:
            raise TablebaseCorruptionError("stored dependency count is invalid")
        if exact:
            if (
                value is None
                or gap is None
                or not math.isfinite(value)
                or not lower - 1e-10 <= value <= upper + 1e-10
                or not 0.0 <= gap <= SADDLE_GAP_TOLERANCE
            ):
                raise TablebaseCorruptionError("stored exact scalar certificate is invalid")
        elif value is not None or gap is not None:
            raise TablebaseCorruptionError("non-exact interval carries exact scalars")
        expected = _value_digest(
            namespace=namespace,
            horizon=stored_horizon,
            state_id=state_id,
            lower=lower,
            upper=upper,
            value=value,
            saddle_gap=gap,
            rank=rank,
            dependencies=dependencies,
            exact=exact,
        )
        if bytes(row["certificate_sha256"]) != expected:
            raise TablebaseCorruptionError("value interval certificate digest mismatch")
        return StoredValue(
            state=requested_state,
            lower_bound=lower,
            upper_bound=upper,
            value=value,
            saddle_gap=gap,
            damage_rank=rank,
            child_dependencies=dependencies,
            exact=exact,
            scope=scope,
            horizon=horizon,
        )

    def get_value(
        self,
        raw: NTState,
        *,
        scope: Scope = "complete-game-exact",
        horizon: int | None = None,
    ) -> StoredValue | None:
        state = validate_live_state(raw)
        namespace = _namespace(scope)
        stored_horizon = _storage_horizon(scope, horizon)
        started = time.perf_counter()
        try:
            row = self.connection.execute(
                'SELECT * FROM "values" WHERE namespace=? AND horizon=? AND state_id=?',
                (namespace, stored_horizon, canonical_state_id(state)),
            ).fetchone()
            if row is None:
                self.metrics.value_lookup_misses += 1
                return None
            self.metrics.value_lookup_hits += 1
            return self._validate_value_row(
                row, requested_state=state, scope=scope, horizon=horizon
            )
        finally:
            self.metrics.value_lookup_seconds += time.perf_counter() - started

    def get_complete_value(self, raw: NTState) -> StoredValue | None:
        return self.get_value(raw)

    def get_interval(self, raw: NTState) -> ValueInterval | None:
        stored = self.get_complete_value(raw)
        return None if stored is None else stored.interval

    def _write_value(
        self,
        *,
        state: NTState,
        scope: Scope,
        horizon: int | None,
        interval: ValueInterval,
        value: float | None,
        saddle_gap: float | None,
        child_dependencies: int,
        exact: bool,
        replace_interval: bool,
    ) -> None:
        namespace = _namespace(scope)
        stored_horizon = _storage_horizon(scope, horizon)
        state_id = canonical_state_id(state)
        rank = canonical_damage_rank(state)
        digest = _value_digest(
            namespace=namespace,
            horizon=stored_horizon,
            state_id=state_id,
            lower=interval.lower_bound,
            upper=interval.upper_bound,
            value=value,
            saddle_gap=saddle_gap,
            rank=rank,
            dependencies=child_dependencies,
            exact=exact,
        )
        verb = "INSERT OR REPLACE" if replace_interval else "INSERT"
        self.connection.execute(
            f"""
            {verb} INTO "values"(
                namespace, horizon, state_id, lower_bound, upper_bound,
                value, saddle_gap, damage_rank, child_dependencies,
                is_exact, certificate_sha256
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                namespace,
                stored_horizon,
                state_id,
                interval.lower_bound,
                interval.upper_bound,
                value,
                saddle_gap,
                rank,
                child_dependencies,
                int(exact),
                digest,
            ),
        )

    def put_interval(
        self,
        raw: NTState,
        interval: ValueInterval,
        *,
        child_dependencies: int,
    ) -> None:
        """Persist a conservative incomplete enclosure, never an approximate value."""

        state = validate_live_state(raw)
        prior = self.get_complete_value(state)
        if prior is not None and prior.exact:
            return
        if prior is not None and (
            interval.lower_bound < prior.lower_bound - 1e-12
            or interval.upper_bound > prior.upper_bound + 1e-12
        ):
            raise TablebaseCorruptionError("interval refinement widened a prior bound")
        with self.transaction():
            self._write_value(
                state=state,
                scope="complete-game-exact",
                horizon=None,
                interval=interval,
                value=None,
                saddle_gap=None,
                child_dependencies=child_dependencies,
                exact=False,
                replace_interval=True,
            )

    @staticmethod
    def _validate_solution(solution: CertifiedSolution) -> None:
        state = validate_live_state(solution.state)
        scalars = (
            solution.value,
            solution.lower_bound,
            solution.upper_bound,
            solution.saddle_gap,
        )
        if not all(math.isfinite(value) for value in scalars):
            raise TablebaseCorruptionError("solution contains a non-finite scalar")
        if not (
            -1.0 <= solution.lower_bound <= solution.value <= solution.upper_bound <= 1.0
        ):
            raise TablebaseCorruptionError("solution scalar interval is invalid")
        if not 0.0 <= solution.saddle_gap <= SADDLE_GAP_TOLERANCE:
            raise TablebaseCorruptionError("solution saddle gap exceeds 1e-6")
        if solution.damage_rank != sum(state):
            raise TablebaseCorruptionError("solution damage rank mismatches public state")
        if not 0 <= solution.child_dependencies <= 61:
            raise TablebaseCorruptionError("solution dependency count is invalid")

    def commit_complete(self, solution: CertifiedSolution) -> None:
        """Atomically commit an exact value and mark its claimed queue row done."""

        self._validate_solution(solution)
        state = validate_live_state(solution.state)
        state_id = canonical_state_id(state)
        interval = ValueInterval(solution.lower_bound, solution.upper_bound)
        started = time.perf_counter()
        try:
            with self.transaction():
                existing = self.connection.execute(
                    'SELECT is_exact FROM "values" '
                    'WHERE namespace=? AND horizon=-1 AND state_id=?',
                    (COMPLETE_NAMESPACE, state_id),
                ).fetchone()
                if existing is not None and int(existing["is_exact"]) == 1:
                    raise DuplicateSolutionError(
                        f"duplicate complete-game value for {state!r}"
                    )
                queue = self.connection.execute(
                    "SELECT solve_status FROM states WHERE state_id=?",
                    (state_id,),
                ).fetchone()
                if queue is None or int(queue["solve_status"]) != 1:
                    raise TablebaseCorruptionError(
                        "exact commit requires an atomically claimed census row"
                    )
                self._write_value(
                    state=state,
                    scope="complete-game-exact",
                    horizon=None,
                    interval=interval,
                    value=float(solution.value),
                    saddle_gap=float(solution.saddle_gap),
                    child_dependencies=solution.child_dependencies,
                    exact=True,
                    replace_interval=existing is not None,
                )
                updated = self.connection.execute(
                    "UPDATE states SET solve_status=2 "
                    "WHERE state_id=? AND solve_status=1",
                    (state_id,),
                ).rowcount
                if updated != 1:
                    raise TablebaseCorruptionError("queue commit lost its claim")
        finally:
            self.metrics.durable_commit_seconds += time.perf_counter() - started

    def cache_policy(self, solution: CertifiedSolution) -> None:
        """Optionally cache policies for a queried root; internal rows stay scalar."""

        self._validate_solution(solution)
        namespace = _namespace(solution.scope)  # type: ignore[arg-type]
        horizon = _storage_horizon(solution.scope, solution.horizon)  # type: ignore[arg-type]
        state_id = canonical_state_id(solution.state)
        row = self.connection.execute(
            'SELECT certificate_sha256, is_exact FROM "values" '
            "WHERE namespace=? AND horizon=? AND state_id=?",
            (namespace, horizon, state_id),
        ).fetchone()
        if row is None or int(row["is_exact"]) != 1:
            raise TablebaseCorruptionError("policy cache requires an exact value row")
        drop = _policy_bytes(solution.drop_policy, role="Dropper")
        check = _policy_bytes(solution.check_policy, role="Checker")
        digest = _policy_digest(bytes(row["certificate_sha256"]), drop, check)
        with self.transaction():
            self.connection.execute(
                """
                INSERT OR REPLACE INTO policy_cache(
                    namespace, horizon, state_id, drop_policy, check_policy,
                    certificate_sha256, created_utc
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    namespace,
                    horizon,
                    state_id,
                    drop,
                    check,
                    digest,
                    datetime.now(timezone.utc).isoformat(),
                ),
            )

    def get_cached_policy(
        self,
        raw: NTState,
        *,
        scope: Scope = "complete-game-exact",
        horizon: int | None = None,
    ) -> CertifiedSolution | None:
        state = validate_live_state(raw)
        stored = self.get_value(state, scope=scope, horizon=horizon)
        if stored is None or not stored.exact or stored.value is None or stored.saddle_gap is None:
            return None
        namespace = _namespace(scope)
        stored_horizon = _storage_horizon(scope, horizon)
        row = self.connection.execute(
            """
            SELECT p.drop_policy, p.check_policy, p.certificate_sha256,
                   v.certificate_sha256 AS value_digest
            FROM policy_cache AS p
            JOIN "values" AS v USING(namespace, horizon, state_id)
            WHERE p.namespace=? AND p.horizon=? AND p.state_id=?
            """,
            (namespace, stored_horizon, canonical_state_id(state)),
        ).fetchone()
        if row is None:
            self.metrics.policy_lookup_misses += 1
            return None
        self.metrics.policy_lookup_hits += 1
        started = time.perf_counter()
        drop_raw = bytes(row["drop_policy"])
        check_raw = bytes(row["check_policy"])
        drop = _policy_from_bytes(drop_raw, role="Dropper")
        check = _policy_from_bytes(check_raw, role="Checker")
        self.metrics.policy_deserialization_seconds += time.perf_counter() - started
        validation_started = time.perf_counter()
        if bytes(row["certificate_sha256"]) != _policy_digest(
            bytes(row["value_digest"]), drop_raw, check_raw
        ):
            raise TablebaseCorruptionError("cached root policy digest mismatch")
        self.metrics.policy_validation_seconds += (
            time.perf_counter() - validation_started
        )
        return CertifiedSolution(
            state=state,
            value=stored.value,
            drop_policy=drop,
            check_policy=check,
            lower_bound=stored.lower_bound,
            upper_bound=stored.upper_bound,
            saddle_gap=stored.saddle_gap,
            damage_rank=sum(state),
            scope=scope,
            horizon=horizon,
            child_dependencies=stored.child_dependencies,
        )

    def put_finite(self, solution: CertifiedSolution) -> None:
        self._validate_solution(solution)
        if solution.scope != "finite-horizon-exact" or solution.horizon is None:
            raise TablebaseCorruptionError("put_finite requires a finite certificate")
        state = validate_live_state(solution.state)
        with self.transaction():
            self._write_value(
                state=state,
                scope="finite-horizon-exact",
                horizon=solution.horizon,
                interval=ValueInterval(solution.lower_bound, solution.upper_bound),
                value=solution.value,
                saddle_gap=solution.saddle_gap,
                child_dependencies=solution.child_dependencies,
                exact=True,
                replace_interval=False,
            )
        self.cache_policy(solution)

    def get_finite(self, raw: NTState, horizon: int) -> CertifiedSolution | None:
        return self.get_cached_policy(
            raw, scope="finite-horizon-exact", horizon=horizon
        )

    def iter_values(
        self,
        *,
        exact_only: bool = False,
    ) -> Iterator[StoredValue]:
        where = "WHERE is_exact=1" if exact_only else ""
        rows = self.connection.execute(
            f'SELECT * FROM "values" {where} '
            "ORDER BY namespace, horizon, state_id"
        ).fetchall()
        for row in rows:
            scope: Scope = (
                "complete-game-exact"
                if str(row["namespace"]) == COMPLETE_NAMESPACE
                else "finite-horizon-exact"
            )
            horizon = None if int(row["horizon"]) == -1 else int(row["horizon"])
            state = state_from_canonical_id(int(row["state_id"]))
            yield self._validate_value_row(
                row, requested_state=state, scope=scope, horizon=horizon
            )

    def verify(self, *, full: bool = True) -> dict[str, int | float]:
        started = time.perf_counter()
        try:
            if self.connection.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
                raise TablebaseCorruptionError("SQLite integrity check failed")
            self._assert_existing_artifact_shape()
            exact = 0
            intervals = 0
            maximum_gap = 0.0
            if full:
                for stored in self.iter_values():
                    intervals += 1
                    if stored.exact:
                        exact += 1
                        maximum_gap = max(maximum_gap, stored.saddle_gap or 0.0)
                for row in self.connection.execute(
                    "SELECT namespace, horizon, state_id FROM policy_cache"
                ):
                    scope: Scope = (
                        "complete-game-exact"
                        if str(row["namespace"]) == COMPLETE_NAMESPACE
                        else "finite-horizon-exact"
                    )
                    state = state_from_canonical_id(int(row["state_id"]))
                    horizon = None if int(row["horizon"]) == -1 else int(row["horizon"])
                    if self.get_cached_policy(state, scope=scope, horizon=horizon) is None:
                        raise TablebaseCorruptionError("policy cache row disappeared")
            else:
                intervals = int(
                    self.connection.execute('SELECT COUNT(*) FROM "values"').fetchone()[0]
                )
                exact = int(
                    self.connection.execute(
                        'SELECT COUNT(*) FROM "values" WHERE is_exact=1'
                    ).fetchone()[0]
                )
                row = self.connection.execute(
                    'SELECT COALESCE(MAX(saddle_gap), 0.0) FROM "values"'
                ).fetchone()
                maximum_gap = float(row[0])
            policies = int(
                self.connection.execute("SELECT COUNT(*) FROM policy_cache").fetchone()[0]
            )
            return {
                "intervals": intervals,
                "exact_values": exact,
                "cached_root_policies": policies,
                "maximum_saddle_gap": maximum_gap,
            }
        finally:
            self.metrics.verify_seconds += time.perf_counter() - started

    def deterministic_snapshot(self) -> bytes:
        metadata = {
            key: value for key, value in self.metadata.items() if key != "created_utc"
        }
        values = [
            tuple(
                bytes(value).hex() if isinstance(value, bytes) else value
                for value in row
            )
            for row in self.connection.execute(
                'SELECT * FROM "values" ORDER BY namespace, horizon, state_id'
            )
        ]
        policies = [
            tuple(
                None
                if index == 6
                else bytes(value).hex()
                if isinstance(value, bytes)
                else value
                for index, value in enumerate(row)
            )
            for row in self.connection.execute(
                "SELECT * FROM policy_cache ORDER BY namespace, horizon, state_id"
            )
        ]
        states = [
            tuple(value for value in row)
            for row in self.connection.execute("SELECT * FROM states ORDER BY state_id")
        ]
        layers = [
            tuple(value for value in row)
            for row in self.connection.execute(
                "SELECT * FROM rank_layers ORDER BY damage_rank"
            )
        ]
        roots = [
            tuple(value for value in row)
            for row in self.connection.execute(
                "SELECT * FROM roots ORDER BY root_ordinal"
            )
        ]
        return json.dumps(
            {
                "metadata": metadata,
                "values": values,
                "policies": policies,
                "states": states,
                "layers": layers,
                "roots": roots,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode()


def _raw_state_id(state: NTState) -> int:
    """Local explicit alias: root manifests retain the full public state."""

    from dth.solver import encode_raw_state_id

    return encode_raw_state_id(state)
