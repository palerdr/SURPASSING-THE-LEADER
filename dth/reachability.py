"""Factorized external-memory closure census and persistent rank scheduler."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
import ctypes
import json
import os
import sqlite3
import time
from typing import Iterable, Sequence

import numpy as np

from dth.solver import (
    CertifiedSolution,
    NTState,
    SADDLE_GAP_TOLERANCE,
    ValueInterval,
    bellman_value_interval,
    canonical_damage_rank,
    canonical_state_id,
    certify_complete_game_matrix,
    complete_game_dependencies,
    continuation_class_values,
    damage_rank,
    decode_raw_state_id,
    encode_raw_state_id,
    failure_dead_quotient,
    reconstruct_transition_class_matrix,
    solver_schema_hash,
    state_from_canonical_id,
    validate_live_state,
)
from dth.tablebase import (
    COMPLETE_NAMESPACE,
    CertifiedTablebase,
    StoredValue,
    TablebaseCorruptionError,
)


CENSUS_REPORT_VERSION = "dth-reachability-report-current"
LIVE_STATE_SPACE_UPPER_BOUND = 300 * 301 * 300 * 301


class CensusError(RuntimeError):
    pass


class CensusSchemaError(CensusError):
    pass


class CensusCorruptionError(CensusError):
    pass


@dataclass(frozen=True)
class CensusRun:
    stop_reason: str
    expansions: int
    new_states: int
    new_edges: int
    elapsed_seconds: float
    peak_memory_bytes: int


@dataclass
class RankSolveMetrics:
    states_committed: int = 0
    dependency_construction_seconds: float = 0.0
    sqlite_value_lookup_seconds: float = 0.0
    matrix_reconstruction_seconds: float = 0.0
    structured_solving_seconds: float = 0.0
    highs_fallback_seconds: float = 0.0
    durable_commit_seconds: float = 0.0
    queue_seconds: float = 0.0
    elapsed_seconds: float = 0.0
    starting_rank: int | None = None
    ending_rank: int | None = None


def encode_state_id(raw: Sequence[int]) -> int:
    """Lossless public-state packing retained for reports and root manifests."""

    return encode_raw_state_id(raw)


def decode_state_id(raw: int) -> NTState:
    return decode_raw_state_id(raw)


def failure_dead_reachability_bitsets(
    roots: Sequence[Sequence[int]],
) -> dict[str, object]:
    """Count the exact quotient closure with checker-turn row bitsets.

    Bit ``b-1`` in row ``a`` records quotient state ``(a,b)``, where each
    coordinate is remaining ST capacity.  A live successful lag ``l`` maps
    ``(a,b)`` to ``(b,a-l)`` for ``1 <= l < a``.  Failed checks and lags
    ``l >= a`` are terminal.  This is the complete quotient transition rule.
    """

    normalized = tuple(validate_live_state(root) for root in roots)
    quotient_roots: list[tuple[int, int]] = []
    for root in normalized:
        quotient = failure_dead_quotient(root)
        if quotient is None:
            raise ValueError("bitset closure requires failure-dead roots")
        quotient_roots.append(quotient)
    rows = [0] * 61
    frontier = sorted(set(quotient_roots))
    for checker_remaining, dropper_remaining in frontier:
        rows[checker_remaining] |= 1 << (dropper_remaining - 1)
    cursor = 0
    while cursor < len(frontier):
        checker_remaining, dropper_remaining = frontier[cursor]
        cursor += 1
        for lag in range(1, checker_remaining):
            child = (dropper_remaining, checker_remaining - lag)
            bit = 1 << (child[1] - 1)
            if rows[child[0]] & bit:
                continue
            rows[child[0]] |= bit
            frontier.append(child)
    rank_counts: dict[str, int] = {}
    for checker_remaining, dropper_remaining in frontier:
        rank = 1800 - checker_remaining - dropper_remaining
        rank_counts[str(rank)] = rank_counts.get(str(rank), 0) + 1
    return {
        "quotient": "failure-dead-remaining-capacity-v1",
        "exact_equivalence_classes": len(frontier),
        "checker_turn_bitsets_hex": {
            str(index): hex(bits)
            for index, bits in enumerate(rows)
            if bits
        },
        "classes_by_damage_rank": dict(
            sorted(rank_counts.items(), key=lambda item: int(item[0]))
        ),
    }


def _peak_resident_memory_bytes() -> int:
    if os.name == "nt":
        from ctypes import wintypes

        class ProcessMemoryCounters(ctypes.Structure):
            _fields_ = [
                ("cb", ctypes.c_ulong),
                ("PageFaultCount", ctypes.c_ulong),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
            ]

        counters = ProcessMemoryCounters()
        counters.cb = ctypes.sizeof(counters)
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        psapi = ctypes.WinDLL("psapi", use_last_error=True)
        kernel32.GetCurrentProcess.restype = wintypes.HANDLE
        psapi.GetProcessMemoryInfo.argtypes = (
            wintypes.HANDLE,
            ctypes.POINTER(ProcessMemoryCounters),
            wintypes.DWORD,
        )
        psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
        ok = psapi.GetProcessMemoryInfo(
            kernel32.GetCurrentProcess(), ctypes.byref(counters), counters.cb
        )
        return int(counters.PeakWorkingSetSize) if ok else 0
    import resource

    peak = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return peak if os.uname().sysname == "Darwin" else peak * 1024


def _canonical_dependencies(state_id: int) -> tuple[int, ...]:
    state = state_from_canonical_id(state_id)
    parent_rank = canonical_damage_rank(state)
    children = {
        canonical_state_id(child) for child in complete_game_dependencies(state)
    }
    for child_id in children:
        child = state_from_canonical_id(child_id)
        if canonical_damage_rank(child) <= parent_rank:
            raise CensusCorruptionError(
                "quotient dependency does not strictly increase damage rank"
            )
    return tuple(
        sorted(
            children,
            key=lambda child_id: (
                canonical_damage_rank(state_from_canonical_id(child_id)),
                child_id,
            ),
        )
    )


class ReachabilityCensus:
    """Deterministic census over the same database used for exact commits."""

    def __init__(
        self,
        tablebase: CertifiedTablebase,
        roots: Sequence[Sequence[int]],
    ) -> None:
        self.tablebase = tablebase
        self.connection = tablebase.connection
        self.roots = tablebase.bind_roots(roots)
        self._initialize_roots()
        self.verify(full=False)

    @property
    def complete(self) -> bool:
        return (
            int(
                self.connection.execute(
                    "SELECT COUNT(*) FROM states WHERE census_status=0"
                ).fetchone()[0]
            )
            == 0
        )

    def _initialize_roots(self) -> None:
        if self.connection.execute("SELECT 1 FROM states LIMIT 1").fetchone() is not None:
            return
        inserted_by_rank: dict[int, int] = {}
        with self.tablebase.transaction():
            for root in self.roots:
                state_id = canonical_state_id(root)
                rank = canonical_damage_rank(root)
                inserted = self.connection.execute(
                    """
                    INSERT OR IGNORE INTO states(
                        state_id, damage_rank, census_status,
                        unique_child_edges, new_unique_states, solve_status
                    ) VALUES (?, ?, 0, NULL, NULL, 0)
                    """,
                    (state_id, rank),
                ).rowcount
                if inserted:
                    inserted_by_rank[rank] = inserted_by_rank.get(rank, 0) + 1
            for rank, count in inserted_by_rank.items():
                self.connection.execute(
                    """
                    INSERT INTO rank_layers(
                        damage_rank, state_count, frontier_count,
                        expanded_count, edge_count, new_state_discoveries
                    ) VALUES (?, ?, ?, 0, 0, 0)
                    """,
                    (rank, count, count),
                )

    def verify(self, *, full: bool) -> dict[str, int]:
        states = int(self.connection.execute("SELECT COUNT(*) FROM states").fetchone()[0])
        frontier = int(
            self.connection.execute(
                "SELECT COUNT(*) FROM states WHERE census_status=0"
            ).fetchone()[0]
        )
        expanded = states - frontier
        totals = self.connection.execute(
            """
            SELECT COALESCE(SUM(state_count),0),
                   COALESCE(SUM(frontier_count),0),
                   COALESCE(SUM(expanded_count),0)
            FROM rank_layers
            """
        ).fetchone()
        if tuple(int(value) for value in totals) != (states, frontier, expanded):
            raise CensusCorruptionError("rank-layer totals disagree with states")
        if full:
            for row in self.connection.execute(
                "SELECT state_id, damage_rank, solve_status FROM states"
            ):
                state = state_from_canonical_id(int(row["state_id"]))
                if canonical_damage_rank(state) != int(row["damage_rank"]):
                    raise CensusCorruptionError("state key has a mismatched damage rank")
                exact = self.connection.execute(
                    'SELECT is_exact FROM "values" '
                    "WHERE namespace=? AND horizon=-1 AND state_id=?",
                    (COMPLETE_NAMESPACE, int(row["state_id"])),
                ).fetchone()
                if (exact is not None and int(exact["is_exact"]) == 1) != (
                    int(row["solve_status"]) == 2
                ):
                    raise CensusCorruptionError(
                        "queue status and atomic exact value commit disagree"
                    )
            aggregate = [
                tuple(int(value) for value in row)
                for row in self.connection.execute(
                    """
                    SELECT damage_rank, COUNT(*),
                           SUM(CASE WHEN census_status=0 THEN 1 ELSE 0 END),
                           SUM(CASE WHEN census_status=1 THEN 1 ELSE 0 END),
                           COALESCE(SUM(unique_child_edges),0),
                           COALESCE(SUM(new_unique_states),0)
                    FROM states GROUP BY damage_rank ORDER BY damage_rank
                    """
                )
            ]
            stored = [
                tuple(int(value) for value in row)
                for row in self.connection.execute(
                    "SELECT * FROM rank_layers ORDER BY damage_rank"
                )
            ]
            if aggregate != stored:
                raise CensusCorruptionError("rank layers disagree with state facts")
        return {"states": states, "frontier": frontier, "expanded": expanded}

    def run(
        self,
        *,
        max_expansions: int | None,
        max_states: int | None,
        max_seconds: float | None,
    ) -> CensusRun:
        if max_expansions is not None and max_expansions < 0:
            raise ValueError("max_expansions must be nonnegative")
        if max_states is not None and max_states < len(self.roots):
            raise ValueError("max_states cannot be smaller than roots")
        if max_seconds is not None and max_seconds < 0:
            raise ValueError("max_seconds must be nonnegative")
        started = time.monotonic()
        start_count = int(self.connection.execute("SELECT COUNT(*) FROM states").fetchone()[0])
        count = start_count
        expansions = 0
        edges = 0
        stop_reason = "complete"
        try:
            while True:
                if max_expansions is not None and expansions >= max_expansions:
                    stop_reason = "max-expansions"
                    break
                if max_seconds is not None and time.monotonic() - started >= max_seconds:
                    stop_reason = "max-seconds"
                    break
                row = self.connection.execute(
                    """
                    SELECT state_id, damage_rank FROM states
                    WHERE census_status=0 ORDER BY damage_rank, state_id LIMIT 1
                    """
                ).fetchone()
                if row is None:
                    break
                state_id = int(row["state_id"])
                children = _canonical_dependencies(state_id)
                existing = 0
                if children:
                    placeholders = ",".join("?" for _ in children)
                    existing = int(
                        self.connection.execute(
                            f"SELECT COUNT(*) FROM states WHERE state_id IN ({placeholders})",
                            children,
                        ).fetchone()[0]
                    )
                pending = len(children) - existing
                if max_states is not None and count + pending > max_states:
                    stop_reason = "max-states"
                    break
                inserted_by_rank: dict[int, int] = {}
                inserted = 0
                with self.tablebase.transaction():
                    current = self.connection.execute(
                        "SELECT census_status FROM states WHERE state_id=?",
                        (state_id,),
                    ).fetchone()
                    if current is None or int(current["census_status"]) != 0:
                        raise CensusCorruptionError("frontier state changed unexpectedly")
                    for child_id in children:
                        rank = canonical_damage_rank(state_from_canonical_id(child_id))
                        did_insert = self.connection.execute(
                            """
                            INSERT OR IGNORE INTO states(
                                state_id, damage_rank, census_status,
                                unique_child_edges, new_unique_states, solve_status
                            ) VALUES (?, ?, 0, NULL, NULL, 0)
                            """,
                            (child_id, rank),
                        ).rowcount
                        if did_insert:
                            inserted += 1
                            inserted_by_rank[rank] = inserted_by_rank.get(rank, 0) + 1
                    for rank, rank_count in inserted_by_rank.items():
                        self.connection.execute(
                            """
                            INSERT INTO rank_layers(
                                damage_rank, state_count, frontier_count,
                                expanded_count, edge_count, new_state_discoveries
                            ) VALUES (?, ?, ?, 0, 0, 0)
                            ON CONFLICT(damage_rank) DO UPDATE SET
                                state_count=state_count+excluded.state_count,
                                frontier_count=frontier_count+excluded.frontier_count
                            """,
                            (rank, rank_count, rank_count),
                        )
                    self.connection.execute(
                        """
                        UPDATE states SET census_status=1,
                            unique_child_edges=?, new_unique_states=?
                        WHERE state_id=?
                        """,
                        (len(children), inserted, state_id),
                    )
                    self.connection.execute(
                        """
                        UPDATE rank_layers SET
                            frontier_count=frontier_count-1,
                            expanded_count=expanded_count+1,
                            edge_count=edge_count+?,
                            new_state_discoveries=new_state_discoveries+?
                        WHERE damage_rank=?
                        """,
                        (len(children), inserted, int(row["damage_rank"])),
                    )
                expansions += 1
                edges += len(children)
                count += inserted
        except KeyboardInterrupt:
            stop_reason = "keyboard-interrupt"
        return CensusRun(
            stop_reason=stop_reason,
            expansions=expansions,
            new_states=count - start_count,
            new_edges=edges,
            elapsed_seconds=time.monotonic() - started,
            peak_memory_bytes=_peak_resident_memory_bytes(),
        )

    def report(self, run: CensusRun) -> dict[str, object]:
        verified = self.verify(full=True)
        layers = self.connection.execute(
            "SELECT * FROM rank_layers ORDER BY damage_rank"
        ).fetchall()
        states = verified["states"]
        edges = sum(int(row["edge_count"]) for row in layers)
        attempts = len({canonical_state_id(root) for root in self.roots}) + edges
        complete = verified["frontier"] == 0
        rank_layers = {
            str(int(row["damage_rank"])): {
                "states": int(row["state_count"]),
                "frontier": int(row["frontier_count"]),
                "expanded": int(row["expanded_count"]),
                "unique_edges": int(row["edge_count"]),
                "new_states_discovered": int(row["new_state_discoveries"]),
            }
            for row in layers
        }
        branching = {
            str(int(row[0])): int(row[1])
            for row in self.connection.execute(
                """
                SELECT unique_child_edges, COUNT(*) FROM states
                WHERE census_status=1 GROUP BY unique_child_edges
                ORDER BY unique_child_edges
                """
            )
        }
        quotient_bitsets = None
        if all(failure_dead_quotient(root) is not None for root in self.roots):
            quotient_bitsets = failure_dead_reachability_bitsets(self.roots)
            if complete and quotient_bitsets["exact_equivalence_classes"] != states:
                raise CensusCorruptionError("bitset count disagrees with persisted census")
        size = self.tablebase.checkpointed_size_bytes()
        return {
            "schema_version": CENSUS_REPORT_VERSION,
            "rules_schema_hash": solver_schema_hash(),
            "roots": [list(root) for root in self.roots],
            "state_identity": "failure-dead quotient classes plus raw live states",
            "completion_status": "complete" if complete else "bounded-incomplete",
            "stop_reason": run.stop_reason,
            "unique_reachable_states": states if complete else None,
            "reachable_states_lower_bound": states,
            "universal_live_state_upper_bound": LIVE_STATE_SPACE_UPPER_BOUND,
            "states_and_edges_by_damage_rank": rank_layers,
            "unique_child_branching_distribution": branching,
            "deduplication": {
                "discovery_attempts": attempts,
                "duplicate_attempts": attempts - states,
                "duplicate_elimination_ratio": (
                    (attempts - states) / attempts if attempts else 0.0
                ),
            },
            "frontier_growth_per_rank": rank_layers,
            "elapsed_seconds": run.elapsed_seconds,
            "peak_memory_bytes": run.peak_memory_bytes,
            "persistent_store_size_bytes": size,
            "persistent_bytes_per_state": size / states if states else None,
            "continuation": asdict(run),
            "verification": verified,
            "failure_dead_bitset_proof": quotient_bitsets,
            "projected_closure": (
                {"kind": "exact-closure", "reachable_states": states}
                if complete
                else None
            ),
            "projection_note": (
                "The exhausted frontier makes this quotient-aware closure exact."
                if complete
                else "No projection is made; the persisted count is a lower bound."
            ),
        }

    def claim_next_rank(self, *, limit: int) -> tuple[int, ...]:
        if limit <= 0:
            raise ValueError("queue claim limit must be positive")
        if not self.complete:
            raise CensusError("census must complete before exact solving")
        with self.tablebase.transaction():
            self.connection.execute(
                "UPDATE states SET solve_status=0 WHERE solve_status=1"
            )
            rank_row = self.connection.execute(
                "SELECT MAX(damage_rank) FROM states WHERE solve_status=0"
            ).fetchone()
            if rank_row[0] is None:
                return ()
            rank = int(rank_row[0])
            rows = self.connection.execute(
                """
                SELECT state_id FROM states
                WHERE solve_status=0 AND damage_rank=?
                ORDER BY state_id LIMIT ?
                """,
                (rank, limit),
            ).fetchall()
            state_ids = tuple(int(row["state_id"]) for row in rows)
            for state_id in state_ids:
                self.connection.execute(
                    "UPDATE states SET solve_status=1 WHERE state_id=?",
                    (state_id,),
                )
        return state_ids

    def release_claims(self, state_ids: Iterable[int]) -> None:
        with self.tablebase.transaction():
            for state_id in state_ids:
                self.connection.execute(
                    "UPDATE states SET solve_status=0 "
                    "WHERE state_id=? AND solve_status=1",
                    (state_id,),
                )

    def queue_counts(self) -> dict[str, int]:
        counts = {0: 0, 1: 0, 2: 0}
        for row in self.connection.execute(
            "SELECT solve_status, COUNT(*) FROM states GROUP BY solve_status"
        ):
            counts[int(row[0])] = int(row[1])
        return {
            "pending": counts[0],
            "in_progress": counts[1],
            "committed": counts[2],
        }

    def deterministic_snapshot(self) -> bytes:
        return self.tablebase.deterministic_snapshot()


def _matrix_from_exact_children(
    state: NTState,
    tablebase: CertifiedTablebase,
    metrics: RankSolveMetrics | None = None,
) -> tuple[np.ndarray, int]:
    dependency_started = time.perf_counter()
    children = complete_game_dependencies(state)
    if metrics is not None:
        metrics.dependency_construction_seconds += (
            time.perf_counter() - dependency_started
        )
    parent_rank = canonical_damage_rank(state)
    values: dict[int, float] = {}
    lookup_started = time.perf_counter()
    for child in children:
        child_id = canonical_state_id(child)
        if canonical_damage_rank(child) <= parent_rank:
            raise TablebaseCorruptionError("dependency rank is not strictly greater")
        if child_id in values:
            continue
        stored = tablebase.get_complete_value(child)
        if stored is None or not stored.exact or stored.value is None:
            raise TablebaseCorruptionError(
                f"state {state!r} has an uncommitted exact child {child!r}"
            )
        values[child_id] = stored.value
    if metrics is not None:
        metrics.sqlite_value_lookup_seconds += time.perf_counter() - lookup_started
    matrix_started = time.perf_counter()
    successful, failed = continuation_class_values(
        state, lambda child: values[canonical_state_id(child)]
    )
    matrix = reconstruct_transition_class_matrix(successful, failed)
    if metrics is not None:
        metrics.matrix_reconstruction_seconds += time.perf_counter() - matrix_started
    return matrix, len({canonical_state_id(child) for child in children})


def reconstruct_policy_from_certified_children(
    raw: NTState,
    tablebase: CertifiedTablebase,
    *,
    cache: bool = False,
) -> CertifiedSolution:
    state = validate_live_state(raw)
    stored = tablebase.get_complete_value(state)
    if stored is None or not stored.exact:
        raise TablebaseCorruptionError("queried root has no exact committed value")
    matrix, child_count = _matrix_from_exact_children(state, tablebase)
    solution = certify_complete_game_matrix(
        state, matrix, child_dependencies=child_count
    )
    if stored.value is None or abs(stored.value - solution.value) > 1e-10:
        raise TablebaseCorruptionError("reconstructed policy disagrees with root value")
    if cache and tablebase.get_cached_policy(state) is None:
        tablebase.cache_policy(solution)
    return solution


def bellman_recertify(
    raw: NTState,
    tablebase: CertifiedTablebase,
) -> CertifiedSolution:
    state = validate_live_state(raw)
    solution = reconstruct_policy_from_certified_children(state, tablebase)
    cached = tablebase.get_cached_policy(state)
    if cached is not None:
        matrix, _ = _matrix_from_exact_children(state, tablebase)
        induced = max(
            0.0,
            float(
                np.max(matrix @ np.asarray(cached.check_policy))
                - np.min(matrix.T @ np.asarray(cached.drop_policy))
            ),
        )
        if induced > SADDLE_GAP_TOLERANCE:
            raise TablebaseCorruptionError(
                f"cached root policy fails Bellman audit: {induced}"
            )
    return solution


def propagate_bellman_interval(
    raw: NTState,
    tablebase: CertifiedTablebase,
    *,
    persist: bool = True,
) -> ValueInterval:
    """Compute a certified global bound; absent children remain exactly [-1,1]."""

    state = validate_live_state(raw)
    interval = bellman_value_interval(state, tablebase.get_interval)
    if persist:
        tablebase.put_interval(
            state,
            interval,
            child_dependencies=len(
                {canonical_state_id(child) for child in complete_game_dependencies(state)}
            ),
        )
    return interval


class RankLayerSolver:
    """Descending-rank exact solver with atomic value/queue commits."""

    def __init__(self, census: ReachabilityCensus) -> None:
        self.census = census
        self.tablebase = census.tablebase
        self.metrics = RankSolveMetrics()

    def run(
        self,
        *,
        max_new_solutions: int | None,
        max_seconds: float | None,
        batch_size: int,
        workers: int = 1,
    ) -> dict[str, object]:
        if max_new_solutions is not None and max_new_solutions < 0:
            raise ValueError("max_new_solutions must be nonnegative")
        if max_seconds is not None and max_seconds < 0:
            raise ValueError("max_seconds must be nonnegative")
        if batch_size <= 0 or workers <= 0:
            raise ValueError("batch_size and workers must be positive")
        started = time.monotonic()
        stop_reason = "complete"
        while True:
            if (
                max_new_solutions is not None
                and self.metrics.states_committed >= max_new_solutions
            ):
                stop_reason = "max-new-solutions"
                break
            if max_seconds is not None and time.monotonic() - started >= max_seconds:
                stop_reason = "max-seconds"
                break
            remaining = (
                batch_size
                if max_new_solutions is None
                else min(
                    batch_size,
                    max_new_solutions - self.metrics.states_committed,
                )
            )
            queue_started = time.perf_counter()
            claimed = self.census.claim_next_rank(limit=remaining)
            self.metrics.queue_seconds += time.perf_counter() - queue_started
            if not claimed:
                break
            rank = canonical_damage_rank(state_from_canonical_id(claimed[0]))
            if any(
                canonical_damage_rank(state_from_canonical_id(state_id)) != rank
                for state_id in claimed
            ):
                self.census.release_claims(claimed)
                raise CensusCorruptionError("one batch mixed damage ranks")
            if self.metrics.starting_rank is None:
                self.metrics.starting_rank = rank
            try:
                prepared = [
                    (
                        state_id,
                        state_from_canonical_id(state_id),
                        *_matrix_from_exact_children(
                            state_from_canonical_id(state_id),
                            self.tablebase,
                            self.metrics,
                        ),
                    )
                    for state_id in claimed
                ]

                def certify(
                    item: tuple[int, NTState, np.ndarray, int],
                ) -> tuple[int, CertifiedSolution, float, str]:
                    state_id, state, matrix, child_count = item
                    solve_started = time.perf_counter()
                    backend: list[str] = []
                    solution = certify_complete_game_matrix(
                        state,
                        matrix,
                        child_dependencies=child_count,
                        backend_out=backend,
                    )
                    return (
                        state_id,
                        solution,
                        time.perf_counter() - solve_started,
                        backend[0],
                    )

                if workers == 1 or len(prepared) == 1:
                    solved = [certify(item) for item in prepared]
                else:
                    with ThreadPoolExecutor(max_workers=workers) as executor:
                        solved = list(executor.map(certify, prepared))
                for state_id, solution, solve_seconds, backend in sorted(
                    solved, key=lambda item: item[0]
                ):
                    if backend == "structured-full-support":
                        self.metrics.structured_solving_seconds += solve_seconds
                    else:
                        self.metrics.highs_fallback_seconds += solve_seconds
                    commit_started = time.perf_counter()
                    self.tablebase.commit_complete(solution)
                    self.metrics.durable_commit_seconds += (
                        time.perf_counter() - commit_started
                    )
                    self.metrics.states_committed += 1
                    self.metrics.ending_rank = solution.damage_rank
            except BaseException:
                self.census.release_claims(claimed)
                raise
        self.metrics.elapsed_seconds += time.monotonic() - started
        queue = self.census.queue_counts()
        size = self.tablebase.checkpointed_size_bytes()
        return {
            "stop_reason": stop_reason,
            "completion_status": (
                "complete"
                if queue["pending"] == 0 and queue["in_progress"] == 0
                else "bounded-incomplete"
            ),
            "queue": queue,
            "metrics": asdict(self.metrics),
            "tablebase_metrics": asdict(self.tablebase.metrics),
            "persistent_store_size_bytes": size,
            "persistent_bytes_per_state": (
                size / queue["committed"] if queue["committed"] else None
            ),
        }
