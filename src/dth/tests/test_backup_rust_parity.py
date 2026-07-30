"""Cross-backend parity for the dth_backup_rs kernel.

Python (`dth.backup_tablebase`) is the behavioral authority; the Rust kernel
must reproduce class values and solver routing bit for bit on the same
inputs.  Contract: ``src/dth/docs/DTH_BACKUP_PARITY.md``.
"""

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

dth_backup_rs = pytest.importorskip("dth_backup_rs")

from dth.backup_tablebase import (  # noqa: E402
    BackupTablebase,
    BackupTablebaseBuilder,
    attempt_support_solution,
    support_of_policy,
    toeplitz_saddle,
)
from dth.solver import SADDLE_GAP_TOLERANCE, solve_matrix  # noqa: E402
from dth.solver import reconstruct_transition_class_matrix  # noqa: E402
from dth.tests.test_backup_sweep_python import make_synthetic_table  # noqa: E402


def test_parity_contract_version() -> None:
    assert dth_backup_rs.PARITY_CONTRACT_VERSION == "dth-backup-parity-v1"


def test_toeplitz_saddle_is_bit_identical() -> None:
    rng = np.random.default_rng(20260730)
    for _ in range(200):
        success = rng.uniform(-1.0, 1.0, size=60)
        failed = float(rng.uniform(-1.0, 1.0))
        gap, maximin, minimax = toeplitz_saddle(success[None, :], np.array([failed]))
        rust_gap, rust_maximin, rust_minimax = dth_backup_rs.toeplitz_saddle_rs(
            success, failed
        )
        assert float(gap[0]) == rust_gap
        assert float(maximin[0]) == rust_maximin
        assert float(minimax[0]) == rust_minimax


def test_support_attempt_is_bit_identical() -> None:
    # Same matrices, same guessed supports: identical accept/decline routing,
    # and bit-identical values and policies on acceptance.
    rng = np.random.default_rng(7)
    accepted = 0
    for _ in range(120):
        success = rng.uniform(-1.0, 1.0, size=60)
        failed = float(rng.uniform(-1.0, 1.0))
        matrix = reconstruct_transition_class_matrix(success, failed)
        _, drop, check = solve_matrix(matrix)
        for rows, cols in (
            (support_of_policy(drop, max_support=12), support_of_policy(check, max_support=12)),
            (tuple(range(60)), tuple(range(60))),
            ((3, 40), (11, 52)),
        ):
            python_solution = attempt_support_solution(success, failed, rows, cols)
            rust_solution = dth_backup_rs.attempt_support_rs(
                success, failed, list(rows), list(cols), SADDLE_GAP_TOLERANCE
            )
            if python_solution is None:
                assert rust_solution is None
                continue
            assert rust_solution is not None
            accepted += 1
            value, drop_policy, check_policy = python_solution
            rust_value, rust_drop, rust_check = rust_solution
            assert value == rust_value
            assert np.array_equal(drop_policy, np.asarray(rust_drop))
            assert np.array_equal(check_policy, np.asarray(rust_check))
    assert accepted >= 40


def test_synthetic_sweep_backends_are_bit_identical(tmp_path) -> None:
    table = make_synthetic_table()
    python_dir = tmp_path / "python"
    rust_dir = tmp_path / "rust"
    BackupTablebaseBuilder(
        output_dir=python_dir, backend="python", table=table
    ).sweep()
    BackupTablebaseBuilder(output_dir=rust_dir, backend="rust", table=table).sweep()

    for name in ("value.npy", "solver_kind.npy"):
        assert (python_dir / name).read_bytes() == (rust_dir / name).read_bytes(), (
            f"{name} differs between backends"
        )
    python_meta = BackupTablebase(python_dir).metadata
    rust_meta = BackupTablebase(rust_dir).metadata
    assert python_meta["execution_backends"] == ["python"]
    assert rust_meta["execution_backends"] == ["rust"]
    for key in (
        "pure_states",
        "support_states",
        "warm_hits",
        "full_support_hits",
        "lp_states",
        "lp_single_dual",
        "lp_highs",
        "warm_attempts",
        "class_count",
        "table_digest",
    ):
        assert python_meta[key] == rust_meta[key], key


def test_rust_resume_is_byte_identical(tmp_path) -> None:
    table = make_synthetic_table(count=32)
    straight = tmp_path / "straight"
    interrupted = tmp_path / "interrupted"
    BackupTablebaseBuilder(output_dir=straight, backend="rust", table=table).sweep()
    first = BackupTablebaseBuilder(output_dir=interrupted, backend="rust", table=table)
    assert first.sweep(stop_after_layers=9) is False
    second = BackupTablebaseBuilder(output_dir=interrupted, backend="rust", table=table)
    assert second.sweep() is True
    for name in ("value.npy", "solver_kind.npy", "tablebase.json"):
        assert (straight / name).read_bytes() == (interrupted / name).read_bytes()


def test_artifact_is_independent_of_rayon_thread_count(tmp_path) -> None:
    # rayon fixes its pool size per process, so the single-thread build runs
    # in a subprocess with RAYON_NUM_THREADS=1 and must reproduce the
    # multi-thread artifact byte for byte.
    table = make_synthetic_table(count=32)
    parallel_dir = tmp_path / "parallel"
    serial_dir = tmp_path / "serial"
    BackupTablebaseBuilder(
        output_dir=parallel_dir, backend="rust", table=table
    ).sweep()

    script = (
        "from dth.backup_tablebase import BackupTablebaseBuilder\n"
        "from dth.tests.test_backup_sweep_python import make_synthetic_table\n"
        f"builder = BackupTablebaseBuilder(output_dir=r'{serial_dir}', "
        "backend='rust', table=make_synthetic_table(count=32))\n"
        "assert builder.sweep() is True\n"
    )
    environment = dict(os.environ)
    environment["RAYON_NUM_THREADS"] = "1"
    root = Path(__file__).resolve().parents[3]
    environment["PYTHONPATH"] = str(root / "src")
    subprocess.run(
        [sys.executable, "-c", script],
        check=True,
        env=environment,
        cwd=root,
        capture_output=True,
    )
    for name in ("value.npy", "solver_kind.npy"):
        assert (parallel_dir / name).read_bytes() == (serial_dir / name).read_bytes()
