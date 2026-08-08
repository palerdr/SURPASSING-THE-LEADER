"""Reproducible hard-tail audit for a completed DTH quotient tablebase.

The build stores the broad solver route per class, but not the LP sub-route or
the equilibrium policies.  This audit therefore reconstructs that evidence
from the finished value table:

1. every LP-routed class is re-solved through the complete fallback ladder;
2. a deterministic class-stratified pool of support-routed classes is screened
   with the full-size equalizer; and
3. the lowest-mass full-support certificates in that pool are cross-checked by
   the independent two-LP oracle.

The output is generated data under ``src/dth/artifacts`` and remains ignored.
"""

from __future__ import annotations

from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import time
from typing import Any, Callable, Iterable, Iterator

import numpy as np

from dth.complete_tablebase import (
    SOLVER_KIND_LP,
    SOLVER_KIND_SUPPORT,
    CompleteTablebase,
    _FULL_SUPPORT,
    _solve_matrix_ipm,
    _solve_matrix_tightened,
    attempt_support_solution,
    build_profile_table,
    class_transition_values,
)
from dth.packed import QuotientProfileTable
from dth.solver import (
    SADDLE_GAP_TOLERANCE,
    reconstruct_transition_class_matrix,
    solve_matrix,
)
from dth.support_solver import certify, solve_matrix_single_lp


AUDIT_SCHEMA_VERSION = "dth.complete-hard-tail-audit.v1"

_WORKER_TABLE: QuotientProfileTable | None = None
_WORKER_VALUES: np.ndarray | None = None


def _kind_ids(kind: np.ndarray, target: int, *, chunk: int = 16_000_000) -> np.ndarray:
    """Return target-route class ids without allocating a full-size bool mask."""

    parts: list[np.ndarray] = []
    for start in range(0, len(kind), chunk):
        local = np.flatnonzero(np.asarray(kind[start : start + chunk]) == target)
        if len(local):
            parts.append(local.astype(np.int64, copy=False) + start)
    return np.concatenate(parts) if parts else np.empty(0, dtype=np.int64)


def _stratified_kind_ids(
    kind: np.ndarray, target: int, *, requested: int
) -> np.ndarray:
    """Sample evenly across class-id space, retaining anchors on ``target``."""

    if requested <= 0 or len(kind) == 0:
        return np.empty(0, dtype=np.int64)
    anchors = np.linspace(0, len(kind) - 1, num=requested, dtype=np.int64)
    selected = anchors[np.asarray(kind[anchors]) == target]
    return np.unique(selected)


def _solve_lp_ladder(
    matrix: np.ndarray,
) -> tuple[float, np.ndarray, np.ndarray, str]:
    """Re-run the production LP fallback while retaining its full policies."""

    try:
        value, drop, check = solve_matrix_single_lp(matrix)
        route = "single-lp-dual"
    except (ValueError, RuntimeError, AttributeError, np.linalg.LinAlgError):
        try:
            value, drop, check = solve_matrix(matrix)
            route = "highs"
        except RuntimeError:
            try:
                value, drop, check = _solve_matrix_ipm(matrix)
                route = "highs-ipm"
            except RuntimeError:
                value, drop, check = _solve_matrix_tightened(matrix)
                route = "highs-tightened"
    return float(value), drop, check, route


def _audit_lp_class(
    table: QuotientProfileTable, values: np.ndarray, class_id: int
) -> dict[str, Any]:
    success, failed = class_transition_values(table, class_id, values)
    matrix = reconstruct_transition_class_matrix(success, failed)
    value, drop, check, route = _solve_lp_ladder(matrix)
    certified, gap = certify(matrix, drop, check)
    return {
        "class_id": int(class_id),
        "route": route,
        "stored_deviation": abs(certified - float(values[class_id])),
        "saddle_gap": gap,
        "value": value,
    }


def _audit_equalizer_class(
    table: QuotientProfileTable, values: np.ndarray, class_id: int
) -> dict[str, Any]:
    success, failed = class_transition_values(table, class_id, values)
    solution = attempt_support_solution(
        success, failed, _FULL_SUPPORT, _FULL_SUPPORT
    )
    if solution is None:
        return {"class_id": int(class_id), "accepted": False}
    value, drop, check = solution
    matrix = reconstruct_transition_class_matrix(success, failed)
    certified, gap = certify(matrix, drop, check)
    return {
        "class_id": int(class_id),
        "accepted": True,
        "stored_deviation": abs(certified - float(values[class_id])),
        "saddle_gap": gap,
        "minimum_mass": float(min(drop.min(), check.min())),
        "drop_support": int(np.count_nonzero(drop > 0.0)),
        "check_support": int(np.count_nonzero(check > 0.0)),
        "equalizer_value": float(value),
    }


def _audit_oracle_class(
    table: QuotientProfileTable,
    values: np.ndarray,
    class_id: int,
    equalizer_value: float,
) -> dict[str, Any]:
    success, failed = class_transition_values(table, class_id, values)
    matrix = reconstruct_transition_class_matrix(success, failed)
    value, drop, check = solve_matrix(matrix)
    certified, gap = certify(matrix, drop, check)
    return {
        "class_id": int(class_id),
        "stored_deviation": abs(certified - float(values[class_id])),
        "equalizer_deviation": abs(certified - float(equalizer_value)),
        "saddle_gap": gap,
        "oracle_value": float(value),
    }


def _init_worker(artifact_dir: str) -> None:
    global _WORKER_TABLE, _WORKER_VALUES
    tablebase = CompleteTablebase(Path(artifact_dir), verify_hashes=False)
    _WORKER_TABLE = build_profile_table()
    _WORKER_VALUES = tablebase._arrays["value"]


def _require_worker_state() -> tuple[QuotientProfileTable, np.ndarray]:
    if _WORKER_TABLE is None or _WORKER_VALUES is None:
        raise RuntimeError("complete-tablebase audit worker was not initialized")
    return _WORKER_TABLE, _WORKER_VALUES


def _lp_worker(class_id: int) -> dict[str, Any]:
    table, values = _require_worker_state()
    return _audit_lp_class(table, values, int(class_id))


def _equalizer_worker(class_id: int) -> dict[str, Any]:
    table, values = _require_worker_state()
    return _audit_equalizer_class(table, values, int(class_id))


def _oracle_worker(payload: tuple[int, float]) -> dict[str, Any]:
    table, values = _require_worker_state()
    class_id, equalizer_value = payload
    return _audit_oracle_class(table, values, int(class_id), float(equalizer_value))


def _mapped(
    function: Callable[[Any], dict[str, Any]],
    items: Iterable[Any],
    *,
    artifact_dir: Path,
    workers: int,
    chunksize: int,
) -> Iterator[dict[str, Any]]:
    if workers <= 1:
        _init_worker(str(artifact_dir))
        yield from map(function, items)
        return
    with ProcessPoolExecutor(
        max_workers=workers,
        initializer=_init_worker,
        initargs=(str(artifact_dir),),
    ) as pool:
        yield from pool.map(function, items, chunksize=chunksize)


def _max_field(rows: list[dict[str, Any]], field: str) -> tuple[float, int | None]:
    if not rows:
        return 0.0, None
    worst = max(rows, key=lambda row: float(row[field]))
    return float(worst[field]), int(worst["class_id"])


def run_complete_audit(config) -> dict[str, Any]:
    """Run the canonical hard-tail audit and write a compact JSON report."""

    artifact_dir = Path(config.artifact_dir)
    output_path = Path(config.output_path)
    workers = int(config.workers)
    chunksize = int(config.chunksize)
    progress_every = int(config.progress_every)
    started = time.perf_counter()

    # Verify every artifact digest once in the parent before workers reopen the
    # arrays read-only without repeating the multi-gigabyte hash scan.
    tablebase = CompleteTablebase(artifact_dir)
    if not tablebase._canonical:
        raise RuntimeError("the hard-tail audit requires the canonical tablebase")
    values = tablebase._arrays["value"]
    kind = tablebase._arrays["solver_kind"]

    lp_ids = _kind_ids(kind, SOLVER_KIND_LP)
    expected_lp = int(tablebase.metadata["lp_states"])
    if len(lp_ids) != expected_lp:
        raise RuntimeError(
            f"LP route count mismatch: array has {len(lp_ids)}, manifest has {expected_lp}"
        )

    lp_routes: Counter[str] = Counter()
    lp_route_class_ids: dict[str, list[int]] = {}
    lp_worst_deviation = (0.0, None)
    lp_worst_gap = (0.0, None)
    for index, row in enumerate(
        _mapped(
            _lp_worker,
            (int(class_id) for class_id in lp_ids),
            artifact_dir=artifact_dir,
            workers=workers,
            chunksize=chunksize,
        ),
        start=1,
    ):
        lp_routes[str(row["route"])] += 1
        if row["route"] != "single-lp-dual":
            lp_route_class_ids.setdefault(str(row["route"]), []).append(
                int(row["class_id"])
            )
        if float(row["stored_deviation"]) > lp_worst_deviation[0]:
            lp_worst_deviation = (
                float(row["stored_deviation"]),
                int(row["class_id"]),
            )
        if float(row["saddle_gap"]) > lp_worst_gap[0]:
            lp_worst_gap = (float(row["saddle_gap"]), int(row["class_id"]))
        if progress_every and index % progress_every == 0:
            print(f"[complete-audit] LP {index:,}/{len(lp_ids):,}", flush=True)

    if lp_worst_deviation[0] > SADDLE_GAP_TOLERANCE:
        raise RuntimeError(
            f"LP tail exceeds stored-value tolerance: {lp_worst_deviation}"
        )
    expected_non_default = sum(
        int(tablebase.metadata.get(key, 0))
        for key in ("lp_highs", "lp_ipm", "lp_tightened")
    )
    observed_non_default = sum(len(ids) for ids in lp_route_class_ids.values())
    if observed_non_default != expected_non_default:
        raise RuntimeError(
            "non-default LP count changed: "
            f"audit found {observed_non_default}, production manifest records "
            f"{expected_non_default}"
        )

    candidate_ids = _stratified_kind_ids(
        kind,
        SOLVER_KIND_SUPPORT,
        requested=int(config.equalizer_candidates),
    )
    equalizer_rows = list(
        _mapped(
            _equalizer_worker,
            (int(class_id) for class_id in candidate_ids),
            artifact_dir=artifact_dir,
            workers=workers,
            chunksize=chunksize,
        )
    )
    accepted = [row for row in equalizer_rows if bool(row["accepted"])]
    full_support = [
        row
        for row in accepted
        if int(row["drop_support"]) == 60 and int(row["check_support"]) == 60
    ]
    if not full_support:
        raise RuntimeError("stratified equalizer screen found no full-support classes")
    equalizer_worst_deviation = _max_field(accepted, "stored_deviation")
    equalizer_worst_gap = _max_field(accepted, "saddle_gap")
    if equalizer_worst_deviation[0] > SADDLE_GAP_TOLERANCE:
        raise RuntimeError(
            "equalizer screen exceeds stored-value tolerance: "
            f"{equalizer_worst_deviation}"
        )

    target_count = min(int(config.equalizer_crosscheck), len(full_support))
    near_degenerate = sorted(
        full_support,
        key=lambda row: (float(row["minimum_mass"]), int(row["class_id"])),
    )[:target_count]
    oracle_rows = list(
        _mapped(
            _oracle_worker,
            (
                (int(row["class_id"]), float(row["equalizer_value"]))
                for row in near_degenerate
            ),
            artifact_dir=artifact_dir,
            workers=workers,
            chunksize=chunksize,
        )
    )
    oracle_worst_stored = _max_field(oracle_rows, "stored_deviation")
    oracle_worst_equalizer = _max_field(oracle_rows, "equalizer_deviation")
    oracle_worst_gap = _max_field(oracle_rows, "saddle_gap")
    if max(oracle_worst_stored[0], oracle_worst_equalizer[0]) > SADDLE_GAP_TOLERANCE:
        raise RuntimeError(
            "near-degenerate LP cross-check exceeds tolerance: "
            f"stored={oracle_worst_stored}, equalizer={oracle_worst_equalizer}"
        )

    manifest = json.loads((artifact_dir / "tablebase.json").read_text(encoding="utf-8"))
    report = {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "artifact": {
            "schema_version": manifest["schema_version"],
            "class_count": int(tablebase.metadata["class_count"]),
            "value_sha256": manifest["arrays"]["value"]["sha256"],
            "solver_kind_sha256": manifest["arrays"]["solver_kind"]["sha256"],
        },
        "config": {
            "workers": workers,
            "chunksize": chunksize,
            "equalizer_candidates_requested": int(config.equalizer_candidates),
            "equalizer_crosscheck_requested": int(config.equalizer_crosscheck),
        },
        "lp_tail": {
            "classes": int(len(lp_ids)),
            "route_counts": dict(sorted(lp_routes.items())),
            "production_non_default_classes": expected_non_default,
            "current_non_default_class_ids_by_route": {
                route: sorted(class_ids)
                for route, class_ids in sorted(lp_route_class_ids.items())
            },
            "worst_stored_deviation": lp_worst_deviation[0],
            "worst_stored_deviation_class": lp_worst_deviation[1],
            "worst_saddle_gap": lp_worst_gap[0],
            "worst_saddle_gap_class": lp_worst_gap[1],
        },
        "equalizer_screen": {
            "stratified_candidates": int(len(candidate_ids)),
            "accepted_full_size_equalizers": int(len(accepted)),
            "strict_full_support": int(len(full_support)),
            "minimum_mass": float(
                min(float(row["minimum_mass"]) for row in full_support)
            ),
            "worst_stored_deviation": equalizer_worst_deviation[0],
            "worst_stored_deviation_class": equalizer_worst_deviation[1],
            "worst_saddle_gap": equalizer_worst_gap[0],
            "worst_saddle_gap_class": equalizer_worst_gap[1],
        },
        "near_degenerate_lp_crosscheck": {
            "classes": int(len(oracle_rows)),
            "selection": "lowest minimum policy mass among strict-full-support stratified candidates",
            "selected_class_ids": [int(row["class_id"]) for row in near_degenerate],
            "selected_minimum_mass": float(
                min(float(row["minimum_mass"]) for row in near_degenerate)
            ),
            "worst_stored_deviation": oracle_worst_stored[0],
            "worst_stored_deviation_class": oracle_worst_stored[1],
            "worst_equalizer_vs_lp_deviation": oracle_worst_equalizer[0],
            "worst_equalizer_vs_lp_deviation_class": oracle_worst_equalizer[1],
            "worst_saddle_gap": oracle_worst_gap[0],
            "worst_saddle_gap_class": oracle_worst_gap[1],
        },
        "elapsed_seconds": time.perf_counter() - started,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(output_path)
    return report


def main() -> None:
    import hydra
    from omegaconf import DictConfig

    @hydra.main(version_base="1.3", config_path="config", config_name="complete_audit_v1")
    def _entry(config: DictConfig) -> None:
        report = run_complete_audit(config)
        print(
            "complete hard-tail audit finished in "
            f"{report['elapsed_seconds']:.1f}s; report at {config.output_path}"
        )

    _entry()


if __name__ == "__main__":
    main()
