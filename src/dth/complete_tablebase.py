"""Complete-game tablebase over the packed TTD-dead quotient.

This module owns the dense complete-game solver artifact: one float64 value
per quotient class of ``dth.packed``, computed by dynamic programming in
strictly decreasing class potential, plus a one-byte solver-routing record per
class. This module's Python path is the behavioral authority for the optional
Rust kernel per ``docs/DTH_COMPLETE_PARITY.md``.

Solve ladder per class, in pinned order (``LADDER_ID``):

1. pure saddle point, from the 61 transition-class values in O(60) via the
   Toeplitz structure of the DTH matrix (``matrix[d, c] = success[c - d]``
   when ``c >= d`` else ``failed``);
2. warm-started support solve: guess the equilibrium support of a spatial
   neighbour solved in the previous potential layer, solve the square
   equalizer system, and certify against the full matrix at the frozen
   ``SADDLE_GAP_TOLERANCE``;
3. full-support solve: the identical equalizer mechanism at k = 60, which is
   the structured full-support path and dominates the endgame regions;
4. LP residue: the certified single-LP-dual rung of ``dth.support_solver``,
   then the two-LP HiGHS oracle, an HiGHS IPM retry, and finally the oracle
   under tightened HiGHS dual-simplex tolerances — every retry changes or
   tightens the solver, never the gate.

Every rung returns a certificate midpoint measured against the full matrix at
the same 1e-6 gate; nothing is accepted on a weaker test.  Warm guesses are
precomputed per state from the previous layer's support table (no chaining
inside any kernel), which makes artifact bytes independent of worker count
and work partition by construction.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import math
import os
import tempfile
import time
from collections import deque
from dataclasses import dataclass, field
from numbers import Integral
from pathlib import Path
from typing import Any

import numpy as np

from dth.packed import (
    PACKED_CLASS_ENCODING,
    PROFILE_COUNT,
    QuotientProfileTable,
    build_profile_table,
    decode_class,
    encode_class,
)
from dth.solver import (
    SADDLE_GAP_TOLERANCE,
    reconstruct_transition_class_matrix,
    solve_matrix,
    solver_schema_hash,
)
from dth.support_solver import (
    certify,
    solve_certified_matrix_fast,
    solve_matrix_single_lp,
)

__all__ = [
    "COMPLETE_TABLEBASE_SCHEMA",
    "COMPLETE_BUILD_SCHEMA",
    "WARM_START_POLICY",
    "SOLVER_KIND_PURE",
    "SOLVER_KIND_SUPPORT",
    "SOLVER_KIND_LP",
    "CompleteTablebaseBuilder",
    "CompleteTablebase",
    "build_dead_band_reference",
    "buckets_from_potential",
    "toeplitz_saddle",
    "support_of_policy",
    "attempt_support_solution",
    "recertify_class",
]

COMPLETE_TABLEBASE_SCHEMA = "dth.complete-tablebase.v2"
COMPLETE_BUILD_SCHEMA = "dth.complete-tablebase-build.v2"
WARM_START_POLICY = "prev-layer-neighbor-v1"
SOLVER_KIND_PURE = 0
SOLVER_KIND_SUPPORT = 1
SOLVER_KIND_LP = 2
_POLICY_MASS_EPS = 1e-9
_PIVOT_EPS = 1e-12
_SUPPORTS_FILE = "warm-supports.npz"
_FULL_SUPPORT = tuple(range(60))
# The pinned rung order; part of every build's config digest.
LADDER_ID = "pure/warm-support/full-support/lp-v1"
_RUST_SOURCE_BUNDLE_ALGORITHM = "sha256-framed-source-bundle-v1"
_RUST_SOURCE_BUNDLE_DOMAIN = b"stl-rust-source-bundle-v1\0"

_COMPLETE_METADATA_KEYS = {
    "class_encoding",
    "canonical_table",
    "table_digest",
    "profile_count",
    "class_count",
    "max_class_potential",
    "solver_schema_hash",
    "saddle_gap_tolerance",
    "build_config_digest",
    "warm_start",
    "warm_start_policy",
    "max_support",
    "policy_mass_eps",
    "ladder",
    "solver_kinds",
    "pure_states",
    "support_states",
    "warm_hits",
    "full_support_hits",
    "lp_states",
    "lp_single_dual",
    "lp_highs",
    "lp_ipm",
    "lp_tightened",
    "warm_attempts",
    "execution_backends",
    "recertified_samples",
    "recertified_worst_gap",
    "code_config_digest",
}


# --------------------------------------------------------------------------
# Artifact helpers (fresh implementations; peer projects must not be imported)
# --------------------------------------------------------------------------


def _canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _digest_json(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _update_frame(digest: Any, label: str, payload: bytes) -> None:
    label_bytes = label.encode("utf-8")
    digest.update(len(label_bytes).to_bytes(8, "big"))
    digest.update(label_bytes)
    digest.update(len(payload).to_bytes(8, "big"))
    digest.update(payload)


def _digest_files(paths: list[Path], *, config: object) -> str:
    resolved = [path.resolve() for path in paths]
    if not resolved:
        raise ValueError("at least one source path is required")
    common_root = Path(os.path.commonpath([str(path) for path in resolved]))
    if common_root in resolved:
        common_root = common_root.parent
    digest = hashlib.sha256(b"dth-source-config-bundle-v1\0")
    for path in resolved:
        _update_frame(
            digest,
            path.relative_to(common_root).as_posix(),
            path.read_bytes(),
        )
    _update_frame(digest, "config", _canonical_json(config).encode("utf-8"))
    return digest.hexdigest()


def _rust_source_bundle_digest() -> str:
    """Digest the exact compile-time source bundle embedded by build.rs."""

    source_workspace = Path(__file__).resolve().parent.parent
    repository_root = source_workspace.parent
    crate_root = source_workspace / "crates" / "dth_complete"
    entries = (
        ("Cargo.toml", crate_root / "Cargo.toml"),
        ("build.rs", crate_root / "build.rs"),
        ("src/lib.rs", crate_root / "src" / "lib.rs"),
        ("Cargo.lock", repository_root / "Cargo.lock"),
    )
    digest = hashlib.sha256(_RUST_SOURCE_BUNDLE_DOMAIN)
    for label, path in entries:
        _update_frame(digest, label, path.read_bytes())
    return digest.hexdigest()


def _source_digest_inputs(*, include_rust: bool) -> list[Path]:
    """Return every implementation input that can affect persisted values."""

    source_root = Path(__file__).resolve().parent
    repository_root = source_root.parent.parent
    inputs = [
        source_root / "packed.py",
        source_root / "solver.py",
        source_root / "support_solver.py",
        source_root / "complete_tablebase.py",
        repository_root / "uv.lock",
    ]
    if include_rust:
        source_workspace = source_root.parent
        inputs.extend(
            (
                source_workspace / "crates" / "dth_complete" / "Cargo.toml",
                source_workspace / "crates" / "dth_complete" / "build.rs",
                source_workspace / "crates" / "dth_complete" / "src" / "lib.rs",
                repository_root / "Cargo.lock",
            )
        )
    return inputs


def _implementation_digest(*, include_rust: bool) -> str:
    return _digest_files(
        _source_digest_inputs(include_rust=include_rust),
        config={
            "artifact_schema": COMPLETE_TABLEBASE_SCHEMA,
            "build_schema": COMPLETE_BUILD_SCHEMA,
            "execution_backend": "rust" if include_rust else "python",
        },
    )


def _build_config_payload(
    *,
    canonical_table: bool,
    table_digest: str,
    warm_start: bool,
    max_support: int,
    include_rust: bool,
) -> dict[str, object]:
    """Construct the source-bound resume contract used by builders and readers."""

    return {
        "schema": COMPLETE_BUILD_SCHEMA,
        "class_encoding": PACKED_CLASS_ENCODING,
        "canonical_table": canonical_table,
        "table_digest": table_digest,
        "solver_schema_hash": solver_schema_hash(),
        "saddle_gap_tolerance": SADDLE_GAP_TOLERANCE,
        "ladder": LADDER_ID,
        "warm_start": warm_start,
        "warm_start_policy": WARM_START_POLICY if warm_start else None,
        "max_support": max_support,
        "policy_mass_eps": _POLICY_MASS_EPS,
        "implementation_digest": _implementation_digest(include_rust=include_rust),
    }


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, suffix=".json", delete=False
    ) as handle:
        temporary = Path(handle.name)
        handle.write(_canonical_json(payload) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    with tempfile.NamedTemporaryFile(
        dir=path.parent, suffix=".npz", delete=False
    ) as handle:
        temporary = Path(handle.name)
        np.savez(handle, **arrays)
        handle.flush()
        os.fsync(handle.fileno())
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _open_npy(path: Path, *, mode: str, dtype: str, shape: tuple[int, ...]) -> np.memmap:
    if mode == "w+":
        return np.lib.format.open_memmap(
            path, mode=mode, dtype=np.dtype(dtype), shape=shape
        )
    loaded = np.load(path, mmap_mode=mode, allow_pickle=False)
    if loaded.shape != shape or loaded.dtype != np.dtype(dtype):
        raise ValueError(
            f"complete-tablebase array {path.name} has shape/dtype {loaded.shape}/{loaded.dtype}, "
            f"expected {shape}/{np.dtype(dtype)}"
        )
    return loaded


def buckets_from_potential(potential: np.ndarray) -> tuple[np.ndarray, ...]:
    """Group profile ids by per-profile potential, the layer iteration basis."""

    highest = int(potential.max())
    return tuple(
        np.flatnonzero(potential == value).astype(np.uint32)
        for value in range(highest + 1)
    )


def _table_digest(table: QuotientProfileTable) -> str:
    digest = hashlib.sha256(b"dth-quotient-profile-table-v2\0")
    _update_frame(digest, "class_encoding", PACKED_CLASS_ENCODING.encode("utf-8"))
    for name in (
        "alive_id_by_st_ttd",
        "st_by_profile",
        "ttd_by_profile",
        "potential_by_profile",
        "revival_by_profile",
        "success_child_by_profile",
        "failure_child_by_profile",
    ):
        array = np.ascontiguousarray(getattr(table, name))
        _update_frame(digest, f"{name}.dtype", str(array.dtype).encode("ascii"))
        _update_frame(
            digest,
            f"{name}.shape",
            _canonical_json(list(array.shape)).encode("ascii"),
        )
        _update_frame(digest, f"{name}.data", array.tobytes())
    _update_frame(
        digest,
        "bucket_profiles.count",
        len(table.bucket_profiles).to_bytes(8, "big"),
    )
    for index, raw_bucket in enumerate(table.bucket_profiles):
        bucket = np.ascontiguousarray(raw_bucket)
        prefix = f"bucket_profiles[{index}]"
        _update_frame(digest, f"{prefix}.dtype", str(bucket.dtype).encode("ascii"))
        _update_frame(
            digest,
            f"{prefix}.shape",
            _canonical_json(list(bucket.shape)).encode("ascii"),
        )
        _update_frame(digest, f"{prefix}.data", bucket.tobytes())
    return digest.hexdigest()


# --------------------------------------------------------------------------
# Pinned solve primitives shared with (and mirrored by) the Rust kernel
# --------------------------------------------------------------------------


def toeplitz_saddle(
    success: np.ndarray, failed: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """O(60)-per-state pure-saddle reductions over transition-class values.

    ``success`` is ``(n, 60)`` and ``failed`` is ``(n,)``.  Row ``d`` of the
    implied matrix holds the prefix ``success[0 .. 59-d]`` on and above the
    diagonal and ``failed`` below it, so both reductions are prefix scans:

    - ``row_min[d] = min(prefix-min of success at 59-d, failed if d > 0)``
    - ``col_max[c] = max(prefix-max of success at c, failed if c < 59)``

    Returns ``(gap, maximin, minimax)`` with ``gap = minimax - maximin``.
    Min/max are exact and order-independent, so this is bit-stable across
    backends given bit-identical class values.
    """

    prefix_min = np.minimum.accumulate(success, axis=1)
    row_min = prefix_min[:, ::-1].copy()
    row_min[:, 1:] = np.minimum(row_min[:, 1:], failed[:, None])
    maximin = row_min.max(axis=1)

    prefix_max = np.maximum.accumulate(success, axis=1)
    col_max = prefix_max.copy()
    col_max[:, :-1] = np.maximum(col_max[:, :-1], failed[:, None])
    minimax = col_max.min(axis=1)
    return minimax - maximin, maximin, minimax


def support_of_policy(policy: np.ndarray, *, max_support: int) -> tuple[int, ...]:
    """Pinned support extraction: threshold, top-mass trim, ascending order."""

    indices = np.flatnonzero(policy > _POLICY_MASS_EPS)
    if len(indices) > max_support:
        by_mass = sorted(indices, key=lambda index: (-policy[index], index))
        indices = np.array(sorted(by_mass[:max_support]), dtype=np.int64)
    return tuple(int(index) for index in indices)


def _matrix_cell(success: np.ndarray, failed: float, drop: int, check: int) -> float:
    return float(success[check - drop]) if check >= drop else failed


def _solve_linear_pinned(a: np.ndarray, b: np.ndarray) -> np.ndarray | None:
    """Gaussian elimination, partial pivoting, lowest-index ties, pinned order.

    Mutates its inputs.  Row updates are elementwise ``a[r] -= factor * a[p]``
    (one multiply and one subtract per element, no FMA), so a Rust kernel
    looping the same expressions per element reproduces every intermediate
    bit; the back-substitution sum is sequential ascending for the same
    reason.  Fails closed on any pivot magnitude below 1e-12.
    """

    n = len(b)
    for column in range(n):
        pivot = column + int(np.argmax(np.abs(a[column:, column])))
        if abs(float(a[pivot, column])) < _PIVOT_EPS:
            return None
        if pivot != column:
            a[[column, pivot]] = a[[pivot, column]]
            b[[column, pivot]] = b[[pivot, column]]
        for row in range(column + 1, n):
            factor = float(a[row, column]) / float(a[column, column])
            if factor != 0.0:
                a[row, column] = 0.0
                a[row, column + 1 :] -= factor * a[column, column + 1 :]
                b[row] -= factor * b[column]
    solution = np.zeros(n, dtype=np.float64)
    for row in range(n - 1, -1, -1):
        accumulated = float(b[row])
        for column in range(row + 1, n):
            accumulated -= float(a[row, column]) * float(solution[column])
        solution[row] = accumulated / float(a[row, row])
    return solution


def attempt_support_solution(
    success: np.ndarray,
    failed: float,
    rows: tuple[int, ...],
    cols: tuple[int, ...],
    *,
    tolerance: float = SADDLE_GAP_TOLERANCE,
) -> tuple[float, np.ndarray, np.ndarray] | None:
    """Solve a guessed square support and certify against the full matrix.

    The support is trimmed to ``k = min(len(rows), len(cols))`` leading
    (ascending) indices.  Both equalizer systems are solved by the pinned
    elimination above; negative mass beyond -1e-12 or a certificate gap above
    ``tolerance`` fails closed.  Returns ``(value, drop_policy, check_policy)``
    on the 60-action simplex, or ``None``.
    """

    k = min(len(rows), len(cols))
    if k == 0:
        return None
    rows = rows[:k]
    cols = cols[:k]

    submatrix = np.empty((k, k), dtype=np.float64)
    for i in range(k):
        for j in range(k):
            submatrix[i, j] = _matrix_cell(success, failed, rows[i] + 1, cols[j] + 1)

    def solve_side(equalized: np.ndarray) -> np.ndarray | None:
        a = np.zeros((k + 1, k + 1), dtype=np.float64)
        a[:k, :k] = equalized
        a[:k, k] = -1.0
        a[k, :k] = 1.0
        b = np.zeros(k + 1, dtype=np.float64)
        b[k] = 1.0
        return _solve_linear_pinned(a, b)

    check_solution = solve_side(submatrix)
    drop_solution = solve_side(submatrix.T.copy())
    if check_solution is None or drop_solution is None:
        return None
    check_mass = check_solution[:k]
    drop_mass = drop_solution[:k]
    if float(check_mass.min()) < -_PIVOT_EPS or float(drop_mass.min()) < -_PIVOT_EPS:
        return None
    check_mass = np.maximum(check_mass, 0.0)
    drop_mass = np.maximum(drop_mass, 0.0)
    check_total = 0.0
    for mass in check_mass:
        check_total += float(mass)
    drop_total = 0.0
    for mass in drop_mass:
        drop_total += float(mass)
    if check_total <= 0.0 or drop_total <= 0.0:
        return None

    drop_policy = np.zeros(60, dtype=np.float64)
    check_policy = np.zeros(60, dtype=np.float64)
    for i in range(k):
        drop_policy[rows[i]] = drop_mass[i] / drop_total
        check_policy[cols[i]] = check_mass[i] / check_total

    # O(60k) certificate against the full matrix, ascending-index summation.
    upper = -np.inf
    for drop in range(60):
        payoff = 0.0
        for j in range(k):
            payoff += check_policy[cols[j]] * _matrix_cell(
                success, failed, drop + 1, cols[j] + 1
            )
        if payoff > upper:
            upper = payoff
    lower = np.inf
    for check in range(60):
        payoff = 0.0
        for i in range(k):
            payoff += drop_policy[rows[i]] * _matrix_cell(
                success, failed, rows[i] + 1, check + 1
            )
        if payoff < lower:
            lower = payoff
    if max(0.0, upper - lower) > tolerance:
        return None
    return (lower + upper) / 2.0, drop_policy, check_policy


def _solve_matrix_tightened(matrix: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    """The two-LP oracle under tightened HiGHS feasibility tolerances.

    scipy's HiGHS defaults occasionally return a primal/dual pair whose
    certificate gap exceeds the frozen 1e-6 gate on ill-conditioned DTH
    matrices.  This retry tightens the *solver*, never the gate: the same
    certificate is measured against the full matrix and a failure still
    raises, so nothing is accepted on a weaker test.
    """

    from scipy.optimize import linprog

    options = {
        "primal_feasibility_tolerance": 1e-10,
        "dual_feasibility_tolerance": 1e-10,
    }
    rows, cols = matrix.shape
    drop_result = linprog(
        c=np.concatenate([np.zeros(rows), [-1.0]]),
        A_ub=np.hstack([-matrix.T, np.ones((cols, 1))]),
        b_ub=np.zeros(cols),
        A_eq=np.hstack([np.ones((1, rows)), np.zeros((1, 1))]),
        b_eq=np.array([1.0]),
        bounds=[(0.0, None)] * rows + [(None, None)],
        method="highs-ds",
        options=options,
    )
    if not drop_result.success:
        raise RuntimeError(f"tightened Dropper LP failed: {drop_result.message}")
    check_result = linprog(
        c=np.concatenate([np.zeros(cols), [1.0]]),
        A_ub=np.hstack([matrix, -np.ones((rows, 1))]),
        b_ub=np.zeros(rows),
        A_eq=np.hstack([np.ones((1, cols)), np.zeros((1, 1))]),
        b_eq=np.array([1.0]),
        bounds=[(0.0, None)] * cols + [(None, None)],
        method="highs-ds",
        options=options,
    )
    if not check_result.success:
        raise RuntimeError(f"tightened Checker LP failed: {check_result.message}")
    drop = np.clip(drop_result.x[:-1], 0.0, None)
    check = np.clip(check_result.x[:-1], 0.0, None)
    drop /= drop.sum()
    check /= check.sum()
    value, _ = certify(matrix, drop, check)
    return value, drop, check


def _solve_matrix_ipm(matrix: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    """Solve the two LPs with HiGHS' interior-point method.

    A small set of ill-conditioned transition matrices can make HiGHS' dual
    simplex path report status 15 even though the same feasible game solves
    cleanly with IPM. This is a solver-path retry only: the returned pair is
    still certified against the full matrix at the frozen saddle-gap gate.
    """

    from scipy.optimize import linprog

    rows, cols = matrix.shape
    drop_result = linprog(
        c=np.concatenate([np.zeros(rows), [-1.0]]),
        A_ub=np.hstack([-matrix.T, np.ones((cols, 1))]),
        b_ub=np.zeros(cols),
        A_eq=np.hstack([np.ones((1, rows)), np.zeros((1, 1))]),
        b_eq=np.array([1.0]),
        bounds=[(0.0, None)] * rows + [(None, None)],
        method="highs-ipm",
    )
    if not drop_result.success:
        raise RuntimeError(f"IPM Dropper LP failed: {drop_result.message}")
    check_result = linprog(
        c=np.concatenate([np.zeros(cols), [1.0]]),
        A_ub=np.hstack([matrix, -np.ones((rows, 1))]),
        b_ub=np.zeros(rows),
        A_eq=np.hstack([np.ones((1, cols)), np.zeros((1, 1))]),
        b_eq=np.array([1.0]),
        bounds=[(0.0, None)] * cols + [(None, None)],
        method="highs-ipm",
    )
    if not check_result.success:
        raise RuntimeError(f"IPM Checker LP failed: {check_result.message}")
    drop = np.clip(drop_result.x[:-1], 0.0, None)
    check = np.clip(check_result.x[:-1], 0.0, None)
    drop /= drop.sum()
    check /= check.sum()
    value, _ = certify(matrix, drop, check)
    return value, drop, check


def _solve_residue(
    success: np.ndarray, failed: float, *, max_support: int
) -> tuple[float, tuple[int, ...], tuple[int, ...], str]:
    """Certified LP path shared verbatim by both backends.

    Four attempts, all certified against the full matrix at the frozen
    tolerance: single LP with dual extraction, the two-LP oracle, an IPM
    retry, then the oracle under tightened HiGHS tolerances.  A matrix that
    fails all four aborts the build rather than store an uncertified value.
    """

    matrix = reconstruct_transition_class_matrix(success, failed)
    try:
        value, drop, check = solve_matrix_single_lp(matrix)
        backend = "single-lp-dual"
    except (ValueError, RuntimeError, AttributeError, np.linalg.LinAlgError):
        try:
            value, drop, check = solve_matrix(matrix)
            backend = "highs"
        except RuntimeError:
            try:
                value, drop, check = _solve_matrix_ipm(matrix)
                backend = "highs-ipm"
            except RuntimeError:
                value, drop, check = _solve_matrix_tightened(matrix)
                backend = "highs-tightened"
    return (
        float(value),
        support_of_policy(drop, max_support=max_support),
        support_of_policy(check, max_support=max_support),
        backend,
    )


def _residue_worker(
    payload: tuple[int, bytes, float, int],
) -> tuple[int, float, tuple[int, ...], tuple[int, ...], str]:
    index, success_bytes, failed, max_support = payload
    success = np.frombuffer(success_bytes, dtype=np.float64)
    value, drop_support, check_support, backend = _solve_residue(
        success, failed, max_support=max_support
    )
    return index, value, drop_support, check_support, backend


def class_transition_values(
    table: QuotientProfileTable,
    class_id: int,
    value: np.ndarray,
) -> tuple[np.ndarray, float]:
    """Rebuild one class's 61 transition-class values from stored children.

    Sixty successful-check continuations indexed by squandered-time lag, then
    the failed-check expectation.  Child values are negated because the roles
    swap across every live edge, and an absent child is the terminal win for
    the mover.
    """

    profile_count = int(len(table.st_by_profile))
    checker, dropper = divmod(int(class_id), profile_count)
    success_profiles = table.success_child_by_profile[checker].astype(np.int64)
    success = np.where(
        success_profiles < 0,
        1.0,
        -value[np.maximum(dropper * profile_count + success_profiles, 0)],
    )
    failure_profile = int(table.failure_child_by_profile[checker])
    if failure_profile < 0:
        failed = 1.0
    else:
        revival = float(table.revival_by_profile[checker])
        failed = revival * (
            -float(value[dropper * profile_count + failure_profile])
        ) + (1.0 - revival)
    return success, float(failed)


def class_certificate(
    table: QuotientProfileTable,
    class_id: int,
    value: np.ndarray,
    *,
    max_support: int = 12,
) -> tuple[float, np.ndarray, np.ndarray, float]:
    """Re-derive one class's full certificate: value, both policies, and gap.

    The artifact stores values only, so a player that wants to *act* on a
    class rebuilds its matrix from stored children and solves it here.  The
    returned pair is certified against that full matrix at the frozen
    tolerance exactly as the sweep certified it, so acting on this costs one
    matrix solve and concedes nothing.
    """

    success, failed = class_transition_values(table, class_id, value)
    matrix = reconstruct_transition_class_matrix(success, failed)
    solved, drop, check, _ = solve_certified_matrix_fast(matrix)
    gap = float(np.max(matrix @ check) - np.min(matrix.T @ drop))
    if gap > SADDLE_GAP_TOLERANCE:
        raise RuntimeError(f"class {class_id} certificate gap too large: {gap}")
    del max_support
    return float(solved), drop, check, max(0.0, gap)


def recertify_class(
    table: QuotientProfileTable,
    class_id: int,
    value: np.ndarray,
    *,
    max_support: int = 12,
) -> float:
    """Rebuild one class's 61 class values from stored children and re-solve.

    The Bellman-recertify-on-demand primitive: per-class certificates are not
    persisted because, given the child values, the whole certificate can be
    re-derived at this cost whenever it is questioned.
    """

    success, failed = class_transition_values(table, class_id, value)
    gap, maximin, minimax = toeplitz_saddle(success[None, :], np.array([failed]))
    if gap[0] <= SADDLE_GAP_TOLERANCE:
        return float((maximin[0] + minimax[0]) / 2.0)
    fresh_value, _, _, _ = _solve_residue(success, failed, max_support=max_support)
    return fresh_value


# --------------------------------------------------------------------------
# Builder
# --------------------------------------------------------------------------


@dataclass
class CompleteTablebaseBuilder:
    """Checkpointed descending-potential sweep over the packed class space.

    ``table=None`` builds the canonical quotient; tests inject small synthetic
    ``QuotientProfileTable`` instances to exercise the full pipeline without a
    2.16 GiB artifact.  Only canonical-table artifacts carry state semantics.
    """

    output_dir: Path
    backend: str = "auto"
    warm_start: bool = True
    max_support: int = 12
    lp_workers: int = 1
    work_item_profiles: int = 4096
    progress_every: int = 0
    table: QuotientProfileTable | None = None
    _metrics: dict[str, float] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        self.output_dir = Path(self.output_dir)
        if self.backend not in {"auto", "python", "rust"}:
            raise ValueError("backend must be 'auto', 'python', or 'rust'")
        if not 1 <= self.max_support <= 60:
            raise ValueError("max_support must be in 1..60")
        if self.lp_workers < 1 or self.work_item_profiles < 1:
            raise ValueError("worker and work-item sizes must be positive")
        if self.progress_every < 0:
            raise ValueError("progress_every must be nonnegative")
        self._rate_window: deque[tuple[int, float]] = deque(maxlen=12)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._table = self.table if self.table is not None else build_profile_table()
        self._canonical = self.table is None
        self._profile_count = int(len(self._table.st_by_profile))
        self._class_count = self._profile_count * self._profile_count
        self._buckets = self._table.bucket_profiles
        self._max_class_potential = 2 * (len(self._buckets) - 1)
        self._progress_path = self.output_dir / "build-progress.json"
        self._rust_kernel = self._load_rust_kernel()
        self._active_backend = "rust" if self._rust_kernel is not None else "python"
        self._build_config = _build_config_payload(
            canonical_table=self._canonical,
            table_digest=_table_digest(self._table),
            warm_start=bool(self.warm_start),
            max_support=int(self.max_support),
            include_rust=self._active_backend == "rust",
        )
        self._config_digest = _digest_json(self._build_config)
        if self._progress_path.exists():
            self._progress = json.loads(self._progress_path.read_text(encoding="utf-8"))
            if self._progress.get("schema_version") != COMPLETE_BUILD_SCHEMA:
                raise ValueError("unsupported complete-tablebase checkpoint schema")
            if self._progress.get("config_digest") != self._config_digest:
                raise ValueError(
                    "checkpoint configuration does not match the requested build"
                )
        else:
            self._progress = None

    # ------------------------------------------------------------- plumbing
    def _load_rust_kernel(self) -> Any | None:
        if self.backend == "python":
            return None
        try:
            module = importlib.import_module("dth_complete_rs")
        except ImportError:
            if self.backend == "rust":
                raise RuntimeError(
                    "Rust backend requested but dth_complete_rs is not installed; "
                    "see src/crates/docs/BUILD.md"
                )
            return None
        expected = "dth-complete-parity-v1"
        if getattr(module, "PARITY_CONTRACT_VERSION", None) != expected:
            raise RuntimeError(
                "dth_complete_rs does not match the Python parity contract"
            )
        if (
            getattr(module, "SOURCE_BUNDLE_DIGEST_ALGORITHM", None)
            != _RUST_SOURCE_BUNDLE_ALGORITHM
            or getattr(module, "SOURCE_BUNDLE_DIGEST", None)
            != _rust_source_bundle_digest()
        ):
            raise RuntimeError(
                "dth_complete_rs was not compiled from the current Rust source bundle"
            )
        return module

    @property
    def phase(self) -> str:
        return "init" if self._progress is None else str(self._progress["phase"])

    def _save_progress(self) -> None:
        _atomic_json(self._progress_path, self._progress)

    def _verify_completed_artifact(self) -> None:
        """Verify that a completed checkpoint still names an intact artifact."""

        expected = self._progress.get("manifest_sha256")
        manifest_path = self.output_dir / "tablebase.json"
        if not _is_sha256(expected):
            raise RuntimeError("completed DTH checkpoint has no valid manifest digest")
        if not manifest_path.is_file() or _sha256_file(manifest_path) != expected:
            raise RuntimeError("completed DTH checkpoint manifest is missing or corrupt")
        try:
            CompleteTablebase(self.output_dir, verify_hashes=True)
        except (FileNotFoundError, OSError, TypeError, ValueError) as exc:
            raise RuntimeError("completed DTH tablebase failed verification") from exc

    def _array_specs(self) -> dict[str, tuple[str, tuple[int, ...]]]:
        return {
            "value": ("float64", (self._class_count,)),
            "solver_kind": ("uint8", (self._class_count,)),
        }

    def _open_arrays(self, mode: str) -> dict[str, np.memmap]:
        return {
            name: _open_npy(self.output_dir / f"{name}.npy", mode=mode, dtype=dtype, shape=shape)
            for name, (dtype, shape) in self._array_specs().items()
        }

    def initialize(self) -> None:
        """Allocate NaN-filled hot arrays and the initial progress manifest."""

        if self._progress is not None:
            return
        arrays = self._open_arrays("w+")
        chunk = 16_000_000
        for start in range(0, self._class_count, chunk):
            arrays["value"][start : start + chunk] = np.nan
        for array in arrays.values():
            array.flush()
        self._progress = {
            "schema_version": COMPLETE_BUILD_SCHEMA,
            "config_digest": self._config_digest,
            "phase": "sweep",
            "completed_potential": self._max_class_potential + 1,
            "pure_states": 0,
            "support_states": 0,
            "warm_hits": 0,
            "full_support_hits": 0,
            "lp_states": 0,
            "lp_single_dual": 0,
            "lp_highs": 0,
            "lp_ipm": 0,
            "lp_tightened": 0,
            "warm_attempts": 0,
            "execution_backends": [],
            "warm_supports_sha256": None,
        }
        self._save_progress()

    # ------------------------------------------------------- warm-start table
    def _load_prev_supports(self) -> dict[int, tuple[tuple[int, ...], tuple[int, ...]]]:
        path = self.output_dir / _SUPPORTS_FILE
        if not self.warm_start or self.phase == "complete":
            return {}
        completed = int(self._progress["completed_potential"])
        expected_digest = self._progress.get("warm_supports_sha256")
        if completed == self._max_class_potential + 1:
            if expected_digest is not None:
                raise RuntimeError("initial DTH checkpoint has an unexpected support digest")
            return {}
        if not path.is_file():
            raise RuntimeError("warm-support checkpoint is missing")
        if not _is_sha256(expected_digest) or _sha256_file(path) != expected_digest:
            raise RuntimeError("warm-support checkpoint digest is missing or corrupt")
        try:
            with np.load(path, allow_pickle=False) as payload:
                if set(payload.files) != {"potential", "classes", "rows", "cols"}:
                    raise RuntimeError("warm-support checkpoint array set is invalid")
                potential = np.asarray(payload["potential"])
                classes = np.asarray(payload["classes"])
                rows = np.asarray(payload["rows"])
                cols = np.asarray(payload["cols"])
        except (OSError, TypeError, ValueError) as exc:
            raise RuntimeError("warm-support checkpoint cannot be decoded") from exc
        if potential.shape != () or potential.dtype != np.dtype("int64"):
            raise RuntimeError("warm-support potential tag is malformed")
        if int(potential) != completed:
            raise RuntimeError(
                "warm-support checkpoint does not match completed_potential"
            )
        if classes.dtype != np.dtype("uint64") or classes.ndim != 1:
            raise RuntimeError("warm-support class IDs are malformed")
        expected_shape = (len(classes), self.max_support)
        if (
            rows.dtype != np.dtype("int32")
            or cols.dtype != np.dtype("int32")
            or rows.shape != expected_shape
            or cols.shape != expected_shape
        ):
            raise RuntimeError("warm-support rows or columns are malformed")
        if len(classes):
            if int(classes[-1]) >= self._class_count or np.any(
                classes[1:] <= classes[:-1]
            ):
                raise RuntimeError("warm-support class IDs are not sorted and unique")
        for support_matrix in (rows, cols):
            for support in support_matrix:
                sentinels = np.flatnonzero(support == -1)
                stop = int(sentinels[0]) if len(sentinels) else len(support)
                values = support[:stop]
                if (
                    np.any(support[stop:] != -1)
                    or np.any(values < 0)
                    or np.any(values > 59)
                    or (len(values) > 1 and np.any(np.diff(values) <= 0))
                ):
                    raise RuntimeError("warm-support padding or ordering is invalid")
        supports = {}
        for index in range(len(classes)):
            supports[int(classes[index])] = (
                tuple(int(v) for v in rows[index] if v >= 0),
                tuple(int(v) for v in cols[index] if v >= 0),
            )
        return supports

    def _store_supports(
        self,
        potential: int,
        supports: dict[int, tuple[tuple[int, ...], tuple[int, ...]]],
    ) -> str | None:
        if not self.warm_start:
            return None
        classes = np.array(sorted(supports), dtype=np.uint64)
        rows = np.full((len(classes), self.max_support), -1, dtype=np.int32)
        cols = np.full((len(classes), self.max_support), -1, dtype=np.int32)
        for index, class_id in enumerate(classes):
            drop_support, check_support = supports[int(class_id)]
            rows[index, : len(drop_support)] = drop_support
            cols[index, : len(check_support)] = check_support
        path = self.output_dir / _SUPPORTS_FILE
        _atomic_npz(
            path,
            {"potential": np.int64(potential), "classes": classes, "rows": rows, "cols": cols},
        )
        return _sha256_file(path)

    def _warm_guesses(self, checker: int, dropper: int) -> tuple[int, ...]:
        """Neighbour classes one potential step up, checker shift then dropper."""

        guesses = []
        checker_shift = int(self._table.success_child_by_profile[checker, 0])
        if checker_shift >= 0:
            guesses.append(checker_shift * self._profile_count + dropper)
        dropper_shift = int(self._table.success_child_by_profile[dropper, 0])
        if dropper_shift >= 0:
            guesses.append(checker * self._profile_count + dropper_shift)
        return tuple(guesses)

    # ------------------------------------------------------------- the sweep
    def sweep(self, *, stop_after_layers: int | None = None) -> bool:
        """Advance whole descending layers; return whether the build finished."""

        self.initialize()
        if self.phase == "complete":
            self._verify_completed_artifact()
            return True
        arrays = self._open_arrays("r+")
        prev_supports = self._load_prev_supports()
        layers_done = 0
        pool = None
        try:
            if self.lp_workers > 1:
                from concurrent.futures import ProcessPoolExecutor

                pool = ProcessPoolExecutor(max_workers=self.lp_workers)
            self._announce_start()
            while int(self._progress["completed_potential"]) > 0:
                potential = int(self._progress["completed_potential"]) - 1
                started = time.perf_counter()
                counters, next_supports = self._solve_layer(
                    potential, arrays, prev_supports, pool
                )
                # Timed apart from the solve because a layer that is slow to
                # commit and one that is slow to solve want different fixes.
                solved_at = time.perf_counter()
                for array in arrays.values():
                    array.flush()
                support_digest = self._store_supports(potential, next_supports)
                for key, delta in counters.items():
                    self._progress[key] = int(self._progress.get(key, 0)) + delta
                self._progress["completed_potential"] = potential
                self._progress["warm_supports_sha256"] = support_digest
                if self._active_backend not in self._progress["execution_backends"]:
                    self._progress["execution_backends"].append(self._active_backend)
                self._save_progress()
                finished_at = time.perf_counter()
                self._metrics[f"layer_{potential}_seconds"] = finished_at - started
                self._metrics[f"layer_{potential}_solve_seconds"] = solved_at - started
                self._metrics[f"layer_{potential}_commit_seconds"] = (
                    finished_at - solved_at
                )
                prev_supports = next_supports
                layers_done += 1
                self._report_layer(
                    potential,
                    counters,
                    solve_seconds=solved_at - started,
                    commit_seconds=finished_at - solved_at,
                    layers_done=layers_done,
                )
                if stop_after_layers is not None and layers_done >= stop_after_layers:
                    return False
        finally:
            if pool is not None:
                pool.shutdown()
        self._finalize(arrays)
        return True

    def _solved_class_count(self) -> int:
        return sum(
            int(self._progress.get(key, 0))
            for key in ("pure_states", "support_states", "lp_states")
        )

    def _announce_start(self) -> None:
        """One header line, so a multi-hour session says what it is doing."""

        if not self.progress_every:
            return
        solved = self._solved_class_count()
        print(
            f"[complete] resuming at Phi={int(self._progress['completed_potential'])} "
            f"of {self._max_class_potential}; {solved:,} of {self._class_count:,} "
            f"classes solved ({100.0 * solved / self._class_count:.2f}%); "
            f"backend={self._active_backend} lp_workers={self.lp_workers} "
            f"warm_start={self.warm_start}",
            flush=True,
        )

    def _report_layer(
        self,
        potential: int,
        counters: dict[str, int],
        *,
        solve_seconds: float,
        commit_seconds: float,
        layers_done: int,
    ) -> None:
        """Per-layer progress: rate, route mix, and a rolling-rate estimate.

        The estimate uses the trailing window rather than the session mean
        because layer sizes vary by fifty times across the sweep, so a mean
        over early layers would misdescribe the layers still to come.
        """

        if not self.progress_every or layers_done % self.progress_every:
            return
        layer_classes = (
            int(counters.get("pure_states", 0))
            + int(counters.get("support_states", 0))
            + int(counters.get("lp_states", 0))
        )
        elapsed = solve_seconds + commit_seconds
        self._rate_window.append((layer_classes, elapsed))
        window_classes = sum(item[0] for item in self._rate_window)
        window_seconds = sum(item[1] for item in self._rate_window)
        rate = window_classes / window_seconds if window_seconds > 0.0 else 0.0
        solved = self._solved_class_count()
        remaining = self._class_count - solved
        eta_hours = remaining / rate / 3600.0 if rate > 0.0 else float("inf")
        lp = int(counters.get("lp_states", 0))
        print(
            f"[complete] Phi={potential:4d} {layer_classes:9,} classes "
            f"in {elapsed:7.2f}s ({solve_seconds:6.2f}+{commit_seconds:5.2f}) "
            f"| {layer_classes / elapsed if elapsed > 0 else 0.0:9,.0f}/s "
            f"| pure {int(counters.get('pure_states', 0)):6,} "
            f"full {int(counters.get('full_support_hits', 0)):8,} "
            f"warm {int(counters.get('warm_hits', 0)):6,} "
            f"lp {lp:6,} ({100.0 * lp / layer_classes if layer_classes else 0.0:5.2f}%) "
            f"| {100.0 * solved / self._class_count:5.2f}% done "
            f"| eta {eta_hours:5.2f}h",
            flush=True,
        )

    def _solve_layer(
        self,
        potential: int,
        arrays: dict[str, np.memmap],
        prev_supports: dict[int, tuple[tuple[int, ...], tuple[int, ...]]],
        pool,
    ) -> tuple[dict[str, int], dict[int, tuple[tuple[int, ...], tuple[int, ...]]]]:
        if self._rust_kernel is not None:
            return self._solve_layer_rust(potential, arrays, prev_supports, pool)
        return self._solve_layer_python(potential, arrays, prev_supports, pool)

    def _layer_rectangles(self, potential: int) -> list[tuple[np.ndarray, np.ndarray]]:
        highest = len(self._buckets) - 1
        rectangles = []
        for checker_potential in range(
            max(0, potential - highest), min(highest, potential) + 1
        ):
            checker_bucket = self._buckets[checker_potential]
            dropper_bucket = self._buckets[potential - checker_potential]
            if len(checker_bucket) and len(dropper_bucket):
                rectangles.append((checker_bucket, dropper_bucket))
        return rectangles

    def _solve_layer_python(
        self,
        potential: int,
        arrays: dict[str, np.memmap],
        prev_supports: dict[int, tuple[tuple[int, ...], tuple[int, ...]]],
        pool,
    ) -> tuple[dict[str, int], dict[int, tuple[tuple[int, ...], tuple[int, ...]]]]:
        value = arrays["value"]
        kind = arrays["solver_kind"]
        table = self._table
        count = self._profile_count
        counters = {
            "pure_states": 0,
            "support_states": 0,
            "warm_hits": 0,
            "full_support_hits": 0,
            "lp_states": 0,
            "lp_single_dual": 0,
            "lp_highs": 0,
            "lp_ipm": 0,
            "lp_tightened": 0,
            "warm_attempts": 0,
        }
        residues: list[tuple[int, np.ndarray, float]] = []
        next_supports: dict[int, tuple[tuple[int, ...], tuple[int, ...]]] = {}

        for checker_bucket, dropper_bucket in self._layer_rectangles(potential):
            checkers = checker_bucket.astype(np.int64)
            success = table.success_child_by_profile[checkers].astype(np.int64)
            failure = table.failure_child_by_profile[checkers].astype(np.int64)
            revival = table.revival_by_profile[checkers]
            width = len(checkers)
            block = max(1, 32_768 // width)
            for start in range(0, len(dropper_bucket), block):
                droppers = dropper_bucket[start : start + block].astype(np.int64)
                child = droppers[:, None, None] * count + success[None, :, :]
                success_values = np.where(
                    success[None, :, :] < 0,
                    1.0,
                    -value[np.maximum(child, 0)],
                )
                failure_child = droppers[:, None] * count + failure[None, :]
                failure_values = np.where(
                    failure[None, :] < 0,
                    1.0,
                    revival[None, :] * (-value[np.maximum(failure_child, 0)])
                    + (1.0 - revival[None, :]),
                )
                flat_success = success_values.reshape(-1, 60)
                flat_failure = failure_values.reshape(-1)
                if not np.all(np.isfinite(flat_success)) or not np.all(
                    np.isfinite(flat_failure)
                ):
                    raise RuntimeError(
                        f"layer {potential} read an unsolved child value"
                    )
                gap, maximin, minimax = toeplitz_saddle(flat_success, flat_failure)
                classes = (
                    checkers[None, :] * count + droppers[:, None]
                ).reshape(-1)
                pure = gap <= SADDLE_GAP_TOLERANCE
                pure_indices = np.flatnonzero(pure)
                value[classes[pure_indices]] = (
                    maximin[pure_indices] + minimax[pure_indices]
                ) / 2.0
                kind[classes[pure_indices]] = SOLVER_KIND_PURE
                counters["pure_states"] += int(len(pure_indices))

                for flat in np.flatnonzero(~pure):
                    class_id = int(classes[flat])
                    checker, dropper = divmod(class_id, count)
                    solution = None
                    if self.warm_start:
                        for guess_class in self._warm_guesses(checker, dropper):
                            guess = prev_supports.get(guess_class)
                            if guess is None:
                                continue
                            counters["warm_attempts"] += 1
                            solution = attempt_support_solution(
                                flat_success[flat], float(flat_failure[flat]),
                                guess[0], guess[1],
                            )
                            if solution is not None:
                                counters["warm_hits"] += 1
                                break
                    if solution is None:
                        # The k = 60 case of the same pinned mechanism is the
                        # full-support structured solve; endgame regions are
                        # dominated by near-full supports (measured 2026-07-30).
                        solution = attempt_support_solution(
                            flat_success[flat], float(flat_failure[flat]),
                            _FULL_SUPPORT, _FULL_SUPPORT,
                        )
                        if solution is not None:
                            counters["full_support_hits"] += 1
                    if solution is not None:
                        solved_value, drop_policy, check_policy = solution
                        value[class_id] = solved_value
                        kind[class_id] = SOLVER_KIND_SUPPORT
                        counters["support_states"] += 1
                        next_supports[class_id] = (
                            support_of_policy(drop_policy, max_support=self.max_support),
                            support_of_policy(check_policy, max_support=self.max_support),
                        )
                    else:
                        residues.append(
                            (class_id, flat_success[flat].copy(), float(flat_failure[flat]))
                        )

        self._solve_residues(residues, pool, counters, next_supports, value, kind)
        return counters, next_supports

    def _solve_residues(
        self,
        residues: list[tuple[int, np.ndarray, float]],
        pool,
        counters: dict[str, int],
        next_supports: dict[int, tuple[tuple[int, ...], tuple[int, ...]]],
        value: np.memmap,
        kind: np.memmap,
    ) -> None:
        """The LP tail, shared verbatim by the Python and Rust backends."""

        residues.sort(key=lambda item: item[0])
        if pool is not None and len(residues) > 1:
            payloads = [
                (class_id, success.tobytes(), failed, self.max_support)
                for class_id, success, failed in residues
            ]
            solved = pool.map(_residue_worker, payloads, chunksize=64)
        else:
            solved = (
                (class_id, *_solve_residue(success, failed, max_support=self.max_support))
                for class_id, success, failed in residues
            )
        backend_keys = {
            "single-lp-dual": "lp_single_dual",
            "highs": "lp_highs",
            "highs-ipm": "lp_ipm",
            "highs-tightened": "lp_tightened",
        }
        for class_id, solved_value, drop_support, check_support, backend in solved:
            value[class_id] = solved_value
            kind[class_id] = SOLVER_KIND_LP
            counters["lp_states"] += 1
            counters[backend_keys[backend]] += 1
            next_supports[class_id] = (drop_support, check_support)

    def _solve_layer_rust(
        self,
        potential: int,
        arrays: dict[str, np.memmap],
        prev_supports: dict[int, tuple[tuple[int, ...], tuple[int, ...]]],
        pool,
    ) -> tuple[dict[str, int], dict[int, tuple[tuple[int, ...], tuple[int, ...]]]]:
        table = self._table
        pool_segments: list[np.ndarray] = []
        work: list[int] = []
        offset = 0
        for checker_bucket, dropper_bucket in self._layer_rectangles(potential):
            checker_offset = offset
            pool_segments.append(checker_bucket)
            offset += len(checker_bucket)
            dropper_base = offset
            pool_segments.append(dropper_bucket)
            offset += len(dropper_bucket)
            for start in range(0, len(dropper_bucket), self.work_item_profiles):
                length = min(self.work_item_profiles, len(dropper_bucket) - start)
                work.extend(
                    (checker_offset, len(checker_bucket), dropper_base + start, length)
                )
        profile_pool = (
            np.concatenate(pool_segments).astype(np.uint32)
            if pool_segments
            else np.zeros(0, dtype=np.uint32)
        )
        guess_classes = np.array(sorted(prev_supports), dtype=np.uint64)
        guess_rows = np.full((len(guess_classes), self.max_support), -1, dtype=np.int32)
        guess_cols = np.full((len(guess_classes), self.max_support), -1, dtype=np.int32)
        for index, class_id in enumerate(guess_classes):
            drop_support, check_support = prev_supports[int(class_id)]
            guess_rows[index, : len(drop_support)] = drop_support
            guess_cols[index, : len(check_support)] = check_support

        (
            residue_classes,
            residue_success,
            residue_failed,
            hit_classes,
            hit_rows,
            hit_cols,
            pure_count,
            warm_hits,
            full_hits,
            warm_attempts,
        ) = self._rust_kernel.sweep_layer_rs(
            np.asarray(work, dtype=np.uint64),
            profile_pool,
            np.ascontiguousarray(table.success_child_by_profile, dtype=np.int32).reshape(-1),
            np.ascontiguousarray(table.failure_child_by_profile, dtype=np.int32),
            np.ascontiguousarray(table.revival_by_profile, dtype=np.float64),
            self._profile_count,
            guess_classes,
            guess_rows.reshape(-1),
            guess_cols.reshape(-1),
            arrays["value"],
            arrays["solver_kind"],
            SADDLE_GAP_TOLERANCE,
            self.max_support,
            bool(self.warm_start),
        )
        counters = {
            "pure_states": int(pure_count),
            "support_states": int(warm_hits) + int(full_hits),
            "warm_hits": int(warm_hits),
            "full_support_hits": int(full_hits),
            "lp_states": 0,
            "lp_single_dual": 0,
            "lp_highs": 0,
            "lp_ipm": 0,
            "lp_tightened": 0,
            "warm_attempts": int(warm_attempts),
        }
        next_supports: dict[int, tuple[tuple[int, ...], tuple[int, ...]]] = {}
        hit_rows = np.asarray(hit_rows).reshape(-1, self.max_support)
        hit_cols = np.asarray(hit_cols).reshape(-1, self.max_support)
        for index, class_id in enumerate(np.asarray(hit_classes)):
            next_supports[int(class_id)] = (
                tuple(int(v) for v in hit_rows[index] if v >= 0),
                tuple(int(v) for v in hit_cols[index] if v >= 0),
            )
        residue_success = np.asarray(residue_success).reshape(-1, 60)
        residues = [
            (
                int(np.asarray(residue_classes)[index]),
                residue_success[index].copy(),
                float(np.asarray(residue_failed)[index]),
            )
            for index in range(len(np.asarray(residue_classes)))
        ]
        self._solve_residues(
            residues, pool, counters, next_supports, arrays["value"], arrays["solver_kind"]
        )
        return counters, next_supports

    # ------------------------------------------------------------- finalize
    def _finalize(self, arrays: dict[str, np.memmap]) -> None:
        value = arrays["value"]
        kind = arrays["solver_kind"]
        chunk = 16_000_000
        for start in range(0, self._class_count, chunk):
            values = np.asarray(value[start : start + chunk])
            kinds = np.asarray(kind[start : start + chunk])
            if not np.all(np.isfinite(values)):
                raise RuntimeError("cannot finalize with unsolved classes")
            if np.any(np.abs(values) > 1.0 + 1e-9):
                raise RuntimeError("complete-tablebase value lies outside [-1, 1]")
            if np.any(kinds > SOLVER_KIND_LP):
                raise RuntimeError("complete-tablebase solver_kind contains an unknown route")

        recertified, worst_gap = self._sampled_recertification(value, kind)

        include_rust = "rust" in self._progress["execution_backends"]
        digest_inputs = _source_digest_inputs(include_rust=include_rust)
        code_config_digest = _digest_files(
            digest_inputs, config={"build_config_digest": self._config_digest}
        )
        array_manifest = {}
        for name in self._array_specs():
            path = self.output_dir / f"{name}.npy"
            array_manifest[name] = {
                "file": path.name,
                "shape": list(self._array_specs()[name][1]),
                "dtype": self._array_specs()[name][0],
                "sha256": _sha256_file(path),
            }
        manifest = {
            "schema_version": COMPLETE_TABLEBASE_SCHEMA,
            "metadata": {
                "class_encoding": PACKED_CLASS_ENCODING,
                "canonical_table": self._canonical,
                "table_digest": _table_digest(self._table),
                "profile_count": self._profile_count,
                "class_count": self._class_count,
                "max_class_potential": self._max_class_potential,
                "solver_schema_hash": solver_schema_hash(),
                "saddle_gap_tolerance": SADDLE_GAP_TOLERANCE,
                "build_config_digest": self._config_digest,
                "warm_start": bool(self.warm_start),
                "warm_start_policy": WARM_START_POLICY if self.warm_start else None,
                "max_support": int(self.max_support),
                "policy_mass_eps": _POLICY_MASS_EPS,
                "ladder": LADDER_ID,
                "solver_kinds": {"pure": 0, "support": 1, "lp": 2},
                "pure_states": int(self._progress["pure_states"]),
                "support_states": int(self._progress["support_states"]),
                "warm_hits": int(self._progress["warm_hits"]),
                "full_support_hits": int(self._progress["full_support_hits"]),
                "lp_states": int(self._progress["lp_states"]),
                "lp_single_dual": int(self._progress["lp_single_dual"]),
                "lp_highs": int(self._progress["lp_highs"]),
                "lp_ipm": int(self._progress.get("lp_ipm", 0)),
                "lp_tightened": int(self._progress.get("lp_tightened", 0)),
                "warm_attempts": int(self._progress["warm_attempts"]),
                "execution_backends": list(self._progress["execution_backends"]),
                "recertified_samples": recertified,
                "recertified_worst_gap": worst_gap,
                "code_config_digest": code_config_digest,
            },
            "arrays": array_manifest,
        }
        _atomic_json(self.output_dir / "tablebase.json", manifest)
        supports_path = self.output_dir / _SUPPORTS_FILE
        supports_path.unlink(missing_ok=True)
        self._progress["warm_supports_sha256"] = None
        self._progress["phase"] = "complete"
        self._progress["manifest_sha256"] = _sha256_file(self.output_dir / "tablebase.json")
        self._save_progress()
        self._verify_completed_artifact()

    def _sampled_recertification(
        self, value: np.memmap, kind: np.memmap, *, per_layer: int = 4
    ) -> tuple[int, float]:
        """Deterministically re-derive and re-check sampled stored values.

        Takes the first ``per_layer`` classes of every layer in iteration
        order, plus every LP-routed class among them; a stored value more than
        the frozen tolerance away from a fresh solve refuses to finalize.
        """

        worst = 0.0
        samples = 0
        for potential in range(self._max_class_potential, -1, -1):
            taken = 0
            for checker_bucket, dropper_bucket in self._layer_rectangles(potential):
                for dropper in dropper_bucket:
                    for checker in checker_bucket:
                        if taken >= per_layer:
                            break
                        class_id = int(checker) * self._profile_count + int(dropper)
                        fresh = recertify_class(
                            self._table, class_id, value, max_support=self.max_support
                        )
                        gap = abs(fresh - float(value[class_id]))
                        if gap > SADDLE_GAP_TOLERANCE:
                            raise RuntimeError(
                                f"recertification failed at class {class_id}: "
                                f"stored {float(value[class_id])!r} vs fresh {fresh!r}"
                            )
                        worst = max(worst, gap)
                        samples += 1
                        taken += 1
                    if taken >= per_layer:
                        break
                if taken >= per_layer:
                    break
        return samples, worst

    def run(self) -> Path:
        """Build to completion and return the manifest path."""

        self.sweep()
        return self.output_dir / "tablebase.json"


# --------------------------------------------------------------------------
# Read-only facade
# --------------------------------------------------------------------------


@dataclass
class CompleteTablebase:
    """Digest-verified read-only view over the completed DTH artifact."""

    artifact_dir: Path
    verify_hashes: bool = True

    def __post_init__(self) -> None:
        self.artifact_dir = Path(self.artifact_dir)
        manifest_path = self.artifact_dir / "tablebase.json"
        if not manifest_path.exists():
            raise FileNotFoundError(f"no complete tablebase manifest at {manifest_path}")
        self._manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not isinstance(self._manifest, dict) or set(self._manifest) != {
            "schema_version",
            "metadata",
            "arrays",
        }:
            raise ValueError("malformed complete-tablebase manifest key set")
        if self._manifest.get("schema_version") != COMPLETE_TABLEBASE_SCHEMA:
            raise ValueError(
                f"unsupported complete-tablebase schema "
                f"{self._manifest.get('schema_version')!r}"
            )
        try:
            metadata = self._manifest["metadata"]
            array_manifest = self._manifest["arrays"]
            if not isinstance(metadata, dict) or not isinstance(array_manifest, dict):
                raise TypeError
            if set(metadata) != _COMPLETE_METADATA_KEYS:
                raise ValueError("complete-tablebase metadata key set is incompatible")
            canonical_table = metadata["canonical_table"]
            if not isinstance(canonical_table, bool):
                raise ValueError("canonical_table must be a JSON boolean")
            self._canonical = canonical_table
            profile_count = metadata["profile_count"]
            class_count = metadata["class_count"]
            if any(
                isinstance(value, bool) or not isinstance(value, int)
                for value in (profile_count, class_count)
            ):
                raise ValueError("profile and class counts must be integers")
            self._profile_count = profile_count
            self._class_count = class_count
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("malformed complete-tablebase manifest") from exc

        if self._profile_count <= 0 or self._class_count != self._profile_count**2:
            raise ValueError("complete-tablebase class dimensions are inconsistent")
        if metadata.get("class_encoding") != PACKED_CLASS_ENCODING:
            raise ValueError("complete-tablebase class encoding is incompatible")
        if metadata.get("solver_schema_hash") != solver_schema_hash():
            raise ValueError("complete-tablebase rules hash does not match current DTH")
        if metadata.get("saddle_gap_tolerance") != SADDLE_GAP_TOLERANCE:
            raise ValueError("complete-tablebase saddle-gap tolerance is incompatible")
        if metadata.get("ladder") != LADDER_ID:
            raise ValueError("complete-tablebase solve ladder is incompatible")
        if metadata.get("solver_kinds") != {"pure": 0, "support": 1, "lp": 2}:
            raise ValueError("complete-tablebase solver-kind encoding is incompatible")
        if metadata.get("policy_mass_eps") != _POLICY_MASS_EPS:
            raise ValueError("complete-tablebase policy-mass threshold is incompatible")

        warm_start = metadata.get("warm_start")
        if not isinstance(warm_start, bool):
            raise ValueError("complete-tablebase warm_start must be a JSON boolean")
        expected_warm_policy = WARM_START_POLICY if warm_start else None
        if metadata.get("warm_start_policy") != expected_warm_policy:
            raise ValueError("complete-tablebase warm-start policy is incompatible")
        max_support = metadata.get("max_support")
        if isinstance(max_support, bool) or not isinstance(max_support, int) or not 1 <= max_support <= 60:
            raise ValueError("complete-tablebase max_support is invalid")

        backends = metadata.get("execution_backends")
        if backends not in (["python"], ["rust"]):
            raise ValueError("complete-tablebase execution provenance is invalid")
        include_rust = "rust" in backends
        table_digest = metadata.get("table_digest")
        if not _is_sha256(table_digest):
            raise ValueError("complete-tablebase table digest is missing")
        for field_name in (
            "solver_schema_hash",
            "build_config_digest",
            "code_config_digest",
        ):
            if not _is_sha256(metadata.get(field_name)):
                raise ValueError(
                    f"complete-tablebase {field_name} digest is malformed"
                )
        maximum_potential = metadata.get("max_class_potential")
        if (
            isinstance(maximum_potential, bool)
            or not isinstance(maximum_potential, int)
            or maximum_potential < 0
        ):
            raise ValueError("complete-tablebase maximum potential is invalid")
        if self._canonical:
            if self._profile_count != PROFILE_COUNT:
                raise ValueError("canonical manifest disagrees with the profile count")
            if metadata.get("max_class_potential") != 1200:
                raise ValueError("canonical manifest has the wrong potential schedule")
            current_table_digest = _table_digest(build_profile_table())
            if table_digest != current_table_digest:
                raise ValueError("canonical profile table does not match current DTH rules")

        expected_build_config = _build_config_payload(
            canonical_table=self._canonical,
            table_digest=table_digest,
            warm_start=warm_start,
            max_support=max_support,
            include_rust=include_rust,
        )
        build_config_digest = metadata.get("build_config_digest")
        if build_config_digest != _digest_json(expected_build_config):
            raise ValueError(
                "complete-tablebase build configuration or implementation source is stale"
            )
        expected_code_digest = _digest_files(
            _source_digest_inputs(include_rust=include_rust),
            config={"build_config_digest": build_config_digest},
        )
        if metadata.get("code_config_digest") != expected_code_digest:
            raise ValueError("complete-tablebase code/configuration digest is stale")

        route_counts = tuple(
            metadata.get(name) for name in ("pure_states", "support_states", "lp_states")
        )
        if any(
            isinstance(count, bool) or not isinstance(count, int) or count < 0
            for count in route_counts
        ):
            raise ValueError("complete-tablebase routing counts are invalid")
        route_total = sum(route_counts)
        if route_total != self._class_count:
            raise ValueError("complete-tablebase routing counts are inconsistent")
        detail_names = (
            "warm_hits",
            "full_support_hits",
            "lp_single_dual",
            "lp_highs",
            "lp_ipm",
            "lp_tightened",
            "warm_attempts",
        )
        detail_counts = {name: metadata.get(name) for name in detail_names}
        if any(
            isinstance(count, bool) or not isinstance(count, int) or count < 0
            for count in detail_counts.values()
        ):
            raise ValueError("complete-tablebase detailed routing counts are invalid")
        if (
            detail_counts["warm_hits"] + detail_counts["full_support_hits"]
            != metadata["support_states"]
            or sum(
                detail_counts[name]
                for name in ("lp_single_dual", "lp_highs", "lp_ipm", "lp_tightened")
            )
            != metadata["lp_states"]
            or detail_counts["warm_hits"] > detail_counts["warm_attempts"]
        ):
            raise ValueError("complete-tablebase detailed routing counts are inconsistent")
        recertified_samples = metadata.get("recertified_samples")
        if (
            isinstance(recertified_samples, bool)
            or not isinstance(recertified_samples, int)
            or recertified_samples <= 0
        ):
            raise ValueError("complete-tablebase recertified sample count is invalid")
        recertified_gap = metadata.get("recertified_worst_gap")
        if (
            isinstance(recertified_gap, bool)
            or not isinstance(recertified_gap, (int, float))
            or not math.isfinite(float(recertified_gap))
            or not 0.0 <= float(recertified_gap) <= SADDLE_GAP_TOLERANCE
        ):
            raise ValueError("complete-tablebase recertification metadata is invalid")

        expected_arrays = {
            "value": {"file": "value.npy", "shape": [self._class_count], "dtype": "float64"},
            "solver_kind": {
                "file": "solver_kind.npy",
                "shape": [self._class_count],
                "dtype": "uint8",
            },
        }
        if set(array_manifest) != set(expected_arrays):
            raise ValueError("complete-tablebase array set is incompatible")
        self._arrays = {}
        for name, expected in expected_arrays.items():
            spec = array_manifest[name]
            if (
                not isinstance(spec, dict)
                or set(spec) != {*expected, "sha256"}
                or any(spec.get(field) != value for field, value in expected.items())
            ):
                raise ValueError(f"complete-tablebase array contract is invalid for {name}")
            digest = spec.get("sha256")
            if not _is_sha256(digest):
                raise ValueError(f"complete-tablebase array digest is invalid for {name}")
            path = self.artifact_dir / spec["file"]
            if self.verify_hashes and _sha256_file(path) != digest:
                raise ValueError(f"complete-tablebase array {name} fails its manifest digest")
            self._arrays[name] = _open_npy(
                path, mode="r", dtype=spec["dtype"], shape=tuple(spec["shape"])
            )

    @property
    def metadata(self) -> dict[str, Any]:
        return dict(self._manifest["metadata"])

    def value_of_class(self, index: int) -> float:
        if isinstance(index, (bool, np.bool_)) or not isinstance(index, Integral):
            raise LookupError(f"class index {index!r} must be a literal integer")
        normalized = int(index)
        if not 0 <= normalized < self._class_count:
            raise LookupError(f"class index {index} is outside this artifact")
        return float(self._arrays["value"][normalized])

    def lookup(self, state, *, recertify: bool = False) -> dict[str, Any]:
        """Resolve one live state; raises ``LookupError`` off the domain."""

        if not self._canonical:
            raise RuntimeError(
                "state lookup requires an artifact built from the canonical table"
            )
        try:
            index = encode_class(tuple(state))
        except ValueError as error:
            raise LookupError(str(error)) from error
        result = {
            "state": decode_class(index),
            "class_index": index,
            "value": float(self._arrays["value"][index]),
            "solver_kind": int(self._arrays["solver_kind"][index]),
        }
        if recertify:
            fresh = recertify_class(
                build_profile_table(),
                index,
                self._arrays["value"],
                max_support=int(self._manifest["metadata"]["max_support"]),
            )
            gap = abs(fresh - result["value"])
            if gap > SADDLE_GAP_TOLERANCE:
                raise RuntimeError(
                    f"stored value fails recertification at class {index}: {gap}"
                )
            result["recertification_gap"] = gap
        return result

    def certificate(self, state) -> dict[str, Any]:
        """Resolve one live state to a playable certificate.

        Returns the stored value alongside a freshly derived equilibrium pair
        and its saddle gap, so a player can act on a class the artifact stores
        only the value of.  Raises ``LookupError`` off the domain, exactly as
        :meth:`lookup` does, and ``RuntimeError`` if the rebuilt certificate
        misses the frozen tolerance.
        """

        if not self._canonical:
            raise RuntimeError(
                "state lookup requires an artifact built from the canonical table"
            )
        try:
            index = encode_class(tuple(state))
        except ValueError as error:
            raise LookupError(str(error)) from error
        value, drop, check, gap = class_certificate(
            build_profile_table(),
            index,
            self._arrays["value"],
            max_support=int(self._manifest["metadata"]["max_support"]),
        )
        stored = float(self._arrays["value"][index])
        if abs(stored - value) > SADDLE_GAP_TOLERANCE:
            raise RuntimeError(
                f"stored value disagrees with its certificate at class {index}: "
                f"{abs(stored - value)}"
            )
        return {
            "state": decode_class(index),
            "class_index": index,
            "value": stored,
            "certificate_value": value,
            "drop_policy": drop,
            "check_policy": check,
            "saddle_gap": gap,
            "solver_kind": int(self._arrays["solver_kind"][index]),
        }


# --------------------------------------------------------------------------
# Independent dead-band reference
# --------------------------------------------------------------------------


def build_dead_band_reference(*, min_total: int = 0) -> np.ndarray:
    """Solve the dead-profile x dead-profile sub-DAG independently.

    When both profiles are dead every failed check is a certain win for the
    Dropper, so the sub-DAG closes over the 300 x 300 remaining-ST grid; this
    is the per-player generalization of ``solver.failure_dead_quotient``.  The
    solver here deliberately shares no sweep machinery: it is an independent
    audit oracle for the corresponding region of the complete artifact.
    Returns values indexed ``checker_st * 300 + dropper_st`` with NaN below
    ``min_total = checker_st + dropper_st``.
    """

    from dth.support_solver import solve_certified_matrix_fast

    values = np.full(300 * 300, np.nan, dtype=np.float64)
    for total in range(598, min_total - 1, -1):
        for checker_st in range(max(0, total - 299), min(299, total) + 1):
            dropper_st = total - checker_st
            success = np.empty(60, dtype=np.float64)
            for lag in range(1, 61):
                grown = checker_st + lag
                success[lag - 1] = (
                    1.0 if grown >= 300 else -values[dropper_st * 300 + grown]
                )
            matrix = reconstruct_transition_class_matrix(success, 1.0)
            solved_value, _, _, _ = solve_certified_matrix_fast(matrix)
            values[checker_st * 300 + dropper_st] = solved_value
    return values


# --------------------------------------------------------------------------
# Hydra entry point
# --------------------------------------------------------------------------


def run_complete(config) -> dict[str, Any]:
    """Run one checkpointed sweep session and write its JSON report."""

    from omegaconf import OmegaConf

    started = time.perf_counter()
    builder = CompleteTablebaseBuilder(
        output_dir=Path(config.output_dir),
        backend=str(config.backend),
        warm_start=bool(config.warm_start),
        max_support=int(config.max_support),
        lp_workers=int(config.lp_workers),
        work_item_profiles=int(config.work_item_profiles),
        progress_every=int(config.get("progress_every", 0)),
    )
    stop_after = config.sweep.stop_after_layers
    finished = builder.sweep(
        stop_after_layers=None if stop_after is None else int(stop_after)
    )
    report = {
        "config": OmegaConf.to_container(config, resolve=True),
        "finished": bool(finished),
        "phase": builder.phase,
        "elapsed_seconds": time.perf_counter() - started,
        "progress": builder._progress,
        "layer_seconds": dict(sorted(builder._metrics.items())),
    }
    _atomic_json(Path(config.report_path), report)
    return report


def main() -> None:
    import hydra
    from omegaconf import DictConfig

    @hydra.main(version_base="1.3", config_path="config", config_name="complete_full_v1")
    def _entry(config: DictConfig) -> None:
        report = run_complete(config)
        print(
            f"complete sweep {'finished' if report['finished'] else 'checkpointed'} "
            f"in {report['elapsed_seconds']:.1f}s; report at {config.report_path}"
        )

    _entry()


if __name__ == "__main__":
    main()
