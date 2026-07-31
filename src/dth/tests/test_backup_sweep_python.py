"""The Python authority sweep: primitives, end-to-end builds, resume, anchors.

The builder is exercised on small synthetic profile tables (same structural
invariants as the canonical quotient, ~kilobyte artifacts) and cross-checked
against an independent per-class resolution through the certified ladder.
The dead-band reference ties the new machinery to ``exact_band_v1.sqlite``.
"""

import json
import sqlite3
from pathlib import Path

import numpy as np
import pytest

from dth.backup_tablebase import (
    BACKUP_TABLEBASE_SCHEMA,
    BackupTablebase,
    BackupTablebaseBuilder,
    attempt_support_solution,
    buckets_from_potential,
    build_dead_band_reference,
    support_of_policy,
    toeplitz_saddle,
)
from dth.packed import QuotientProfileTable
from dth.solver import (
    SADDLE_GAP_TOLERANCE,
    reconstruct_transition_class_matrix,
    solve_matrix,
)
from dth.support_solver import solve_certified_matrix_fast

BAND_PATH = Path(__file__).resolve().parents[1] / "artifacts" / "exact_band_v1.sqlite"


def make_synthetic_table(count: int = 40) -> QuotientProfileTable:
    """A miniature game with the canonical structural shape.

    Profile ``i`` has potential ``i``; success under lag ``L`` moves to
    ``i + L`` (terminal W past the end); profiles with ``i + 7 < count`` are
    revivable to profile ``i + 7``, the rest are dead.  Values are irregular
    enough to exercise all three ladder rungs.
    """

    st_by = np.arange(count, dtype=np.int16)
    alive = st_by + 7 < count
    ttd_by = np.where(alive, 0, -1).astype(np.int16)
    potential = np.arange(count, dtype=np.int16)
    revival = np.where(alive, 0.85 - 0.6 * st_by / count, 0.0).astype(np.float64)
    success = np.full((count, 60), -1, dtype=np.int32)
    for profile in range(count):
        for lag in range(1, 61):
            if profile + lag < count:
                success[profile, lag - 1] = profile + lag
    failure = np.where(alive, st_by.astype(np.int32) + 7, -1).astype(np.int32)
    return QuotientProfileTable(
        alive_id_by_st_ttd=np.full((300, 301), -1, dtype=np.int32),
        st_by_profile=st_by,
        ttd_by_profile=ttd_by,
        potential_by_profile=potential,
        revival_by_profile=revival,
        success_child_by_profile=success,
        failure_child_by_profile=failure,
        bucket_profiles=buckets_from_potential(potential),
    )


def independent_class_values(table: QuotientProfileTable) -> np.ndarray:
    """Solve every class through the certified ladder, no sweep machinery."""

    count = len(table.st_by_profile)
    values = np.full(count * count, np.nan, dtype=np.float64)
    order = np.argsort(
        [
            -(int(table.potential_by_profile[c]) + int(table.potential_by_profile[d]))
            for c in range(count)
            for d in range(count)
        ],
        kind="stable",
    )
    for class_id in order:
        checker, dropper = divmod(int(class_id), count)
        success = np.empty(60, dtype=np.float64)
        for lag in range(1, 61):
            child = int(table.success_child_by_profile[checker, lag - 1])
            success[lag - 1] = 1.0 if child < 0 else -values[dropper * count + child]
        failure_child = int(table.failure_child_by_profile[checker])
        if failure_child < 0:
            failed = 1.0
        else:
            revival = float(table.revival_by_profile[checker])
            failed = revival * (-values[dropper * count + failure_child]) + (
                1.0 - revival
            )
        matrix = reconstruct_transition_class_matrix(success, failed)
        values[class_id], _, _, _ = solve_certified_matrix_fast(matrix)
    return values


def test_toeplitz_saddle_matches_full_matrix_reductions() -> None:
    # The O(60) prefix-scan reductions must agree exactly with the literal
    # row-min / col-max of the reconstructed matrix — including which side of
    # the diagonal carries the failed value.
    rng = np.random.default_rng(20260730)
    success = rng.uniform(-1.0, 1.0, size=(50, 60))
    failed = rng.uniform(-1.0, 1.0, size=50)
    gap, maximin, minimax = toeplitz_saddle(success, failed)
    for row in range(50):
        matrix = reconstruct_transition_class_matrix(success[row], float(failed[row]))
        assert float(maximin[row]) == float(matrix.min(axis=1).max())
        assert float(minimax[row]) == float(matrix.max(axis=0).min())
        assert float(gap[row]) == float(
            matrix.max(axis=0).min() - matrix.min(axis=1).max()
        )


def test_support_attempt_reproduces_the_oracle_and_fails_closed() -> None:
    rng = np.random.default_rng(7)
    reproduced = 0
    for _ in range(30):
        success = rng.uniform(-1.0, 1.0, size=60)
        failed = float(rng.uniform(-1.0, 1.0))
        matrix = reconstruct_transition_class_matrix(success, failed)
        value, drop, check = solve_matrix(matrix)
        rows = support_of_policy(drop, max_support=12)
        cols = support_of_policy(check, max_support=12)
        solution = attempt_support_solution(success, failed, rows, cols)
        if solution is not None:
            solved_value, drop_policy, check_policy = solution
            assert abs(solved_value - value) <= SADDLE_GAP_TOLERANCE
            assert drop_policy.min() >= 0.0 and check_policy.min() >= 0.0
            assert abs(drop_policy.sum() - 1.0) < 1e-12
            reproduced += 1
        # A deliberately wrong support must fail closed, never mis-certify.
        wrong = attempt_support_solution(
            success, failed, (59 - rows[0],), (59 - cols[0],)
        )
        if wrong is not None:
            wrong_value = wrong[0]
            assert abs(wrong_value - value) <= SADDLE_GAP_TOLERANCE
    # Random matrices often have non-square supports, which the square-trim
    # correctly refuses; the mechanism only has to work when it certifies.
    # The real-game hit rate is a measured quantity, not a test invariant.
    assert reproduced >= 5


@pytest.mark.slow  # 34s: sweeps a synthetic table and re-solves every class independently
def test_python_sweep_matches_independent_resolution(tmp_path) -> None:
    table = make_synthetic_table()
    builder = BackupTablebaseBuilder(
        output_dir=tmp_path / "sweep", backend="python", table=table
    )
    assert builder.sweep() is True
    tablebase = BackupTablebase(tmp_path / "sweep")
    reference = independent_class_values(table)
    count = len(table.st_by_profile)
    for class_id in range(count * count):
        stored = tablebase.value_of_class(class_id)
        # Two independently certified midpoints of the same game differ by at
        # most the sum of their half-gaps.
        assert abs(stored - reference[class_id]) <= SADDLE_GAP_TOLERANCE
    metadata = tablebase.metadata
    routed = (
        metadata["pure_states"] + metadata["support_states"] + metadata["lp_states"]
    )
    assert routed == count * count
    assert metadata["canonical_table"] is False


@pytest.mark.slow  # 48s: builds, interrupts, resumes, and diffs the artifact bytes
def test_interrupted_build_resumes_byte_for_byte(tmp_path) -> None:
    table = make_synthetic_table()
    straight = tmp_path / "straight"
    BackupTablebaseBuilder(output_dir=straight, backend="python", table=table).sweep()

    interrupted = tmp_path / "interrupted"
    first = BackupTablebaseBuilder(
        output_dir=interrupted, backend="python", table=table
    )
    assert first.sweep(stop_after_layers=7) is False
    second = BackupTablebaseBuilder(
        output_dir=interrupted, backend="python", table=table
    )
    assert second.sweep(stop_after_layers=11) is False
    third = BackupTablebaseBuilder(
        output_dir=interrupted, backend="python", table=table
    )
    assert third.sweep() is True

    for name in ("value.npy", "solver_kind.npy", "tablebase.json"):
        assert (straight / name).read_bytes() == (interrupted / name).read_bytes()


def test_checkpoint_rejects_a_different_configuration(tmp_path) -> None:
    table = make_synthetic_table()
    builder = BackupTablebaseBuilder(
        output_dir=tmp_path / "build", backend="python", table=table
    )
    builder.sweep(stop_after_layers=2)
    with pytest.raises(ValueError, match="configuration"):
        BackupTablebaseBuilder(
            output_dir=tmp_path / "build",
            backend="python",
            table=table,
            warm_start=False,
        )


def test_corrupted_artifact_fails_closed(tmp_path) -> None:
    table = make_synthetic_table(count=20)
    target = tmp_path / "artifact"
    BackupTablebaseBuilder(output_dir=target, backend="python", table=table).sweep()
    BackupTablebase(target)  # intact artifact opens

    value_path = target / "value.npy"
    payload = bytearray(value_path.read_bytes())
    payload[-9] ^= 0xFF
    value_path.write_bytes(bytes(payload))
    with pytest.raises(ValueError, match="digest"):
        BackupTablebase(target)

    manifest_path = target / "tablebase.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["schema_version"] = "dth.backup-tablebase.v0"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="schema"):
        BackupTablebase(target)


@pytest.mark.skipif(not BAND_PATH.exists(), reason="exact_band_v1.sqlite not present")
@pytest.mark.slow  # 31s: solves the dead-dead band against the shipped exact band
def test_dead_band_reference_matches_exact_band_v1() -> None:
    # The shipped band solves the both-STs>=240 quotient: classes keyed by
    # remaining capacities in a disjoint negative id range.  The dead-band
    # reference solves the per-player generalization independently; on the
    # shared 3,541 classes the two must agree within certificate width.
    reference = build_dead_band_reference(min_total=480)
    connection = sqlite3.connect(f"file:{BAND_PATH}?mode=ro", uri=True)
    rows = connection.execute(
        "SELECT state_id, value FROM 'values' "
        "WHERE namespace='complete-game' AND is_exact=1 AND state_id < 0"
    ).fetchall()
    connection.close()
    assert len(rows) == 3_541
    worst = 0.0
    for state_id, stored in rows:
        offset = -1 - int(state_id)
        checker_remaining, dropper_remaining = divmod(offset, 60)
        checker_st = 299 - checker_remaining
        dropper_st = 299 - dropper_remaining
        mine = float(reference[checker_st * 300 + dropper_st])
        assert np.isfinite(mine)
        worst = max(worst, abs(mine - float(stored)))
    assert worst <= 5e-7
    root = float(reference[240 * 300 + 240])
    assert abs(root - 0.3372132166291093) <= 1e-9
