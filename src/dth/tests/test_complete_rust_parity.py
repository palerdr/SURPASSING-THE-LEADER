"""Cross-backend parity for the dth_complete_rs kernel.

Python (`dth.complete_tablebase`) is the behavioral authority; the Rust kernel
must reproduce class values and solver routing bit for bit on the same
inputs. Contract: ``src/dth/docs/DTH_COMPLETE_PARITY.md``.
"""

import importlib
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

def _load_rust_extension():
    try:
        return importlib.import_module("dth_complete_rs")
    except ModuleNotFoundError as error:
        if error.name != "dth_complete_rs":
            raise
        if os.environ.get("STL_REQUIRE_RUST_PARITY") == "1":
            raise RuntimeError(
                "dth_complete_rs is required by STL_REQUIRE_RUST_PARITY=1; "
                "build it before running parity tests"
            ) from error
        pytest.skip("dth_complete_rs is not installed", allow_module_level=True)


dth_complete_rs = _load_rust_extension()

from dth.complete_tablebase import (  # noqa: E402
    CompleteTablebase,
    CompleteTablebaseBuilder,
    _rust_source_bundle_digest,
    attempt_support_solution,
    support_of_policy,
    toeplitz_saddle,
)
from dth.solver import SADDLE_GAP_TOLERANCE, solve_matrix  # noqa: E402
from dth.solver import reconstruct_transition_class_matrix  # noqa: E402
from dth.tests.test_complete_sweep_python import make_synthetic_table  # noqa: E402


def test_parity_contract_version() -> None:
    assert dth_complete_rs.PARITY_CONTRACT_VERSION == "dth-complete-parity-v1"
    assert (
        dth_complete_rs.SOURCE_BUNDLE_DIGEST_ALGORITHM
        == "sha256-framed-source-bundle-v1"
    )
    assert dth_complete_rs.SOURCE_BUNDLE_DIGEST == _rust_source_bundle_digest()


def test_rust_public_boundary_rejects_nonfinite_and_uncertified_inputs() -> None:
    success = np.zeros(60, dtype=np.float64)
    invalid = success.copy()
    invalid[7] = np.nan
    with pytest.raises(ValueError, match="finite class values"):
        dth_complete_rs.toeplitz_saddle_rs(invalid, 0.0)
    with pytest.raises(ValueError, match="0..60"):
        dth_complete_rs.attempt_support_rs(success, 0.0, [-1], [0], 1e-6)
    with pytest.raises(ValueError, match="strictly ascending"):
        dth_complete_rs.attempt_support_rs(success, 0.0, [2, 1], [0, 1], 1e-6)
    with pytest.raises(ValueError, match="frozen"):
        dth_complete_rs.attempt_support_rs(success, 0.0, [0], [0], 1e-5)


def test_rust_sweep_rejects_overlapping_work_without_partial_writes() -> None:
    value = np.array([np.nan], dtype=np.float64)
    solver_kind = np.zeros(1, dtype=np.uint8)
    with pytest.raises(ValueError, match="overlap"):
        dth_complete_rs.sweep_layer_rs(
            np.array([0, 1, 0, 1, 0, 1, 0, 1], dtype=np.uint64),
            np.array([0], dtype=np.uint32),
            np.full(60, -1, dtype=np.int32),
            np.array([-1], dtype=np.int32),
            np.array([0.0], dtype=np.float64),
            1,
            np.zeros(0, dtype=np.uint64),
            np.zeros(0, dtype=np.int32),
            np.zeros(0, dtype=np.int32),
            value,
            solver_kind,
            1e-6,
            12,
            False,
        )
    assert np.isnan(value[0])
    assert solver_kind[0] == 0


def test_toeplitz_saddle_is_bit_identical() -> None:
    rng = np.random.default_rng(20260730)
    for _ in range(200):
        success = rng.uniform(-1.0, 1.0, size=60)
        failed = float(rng.uniform(-1.0, 1.0))
        gap, maximin, minimax = toeplitz_saddle(success[None, :], np.array([failed]))
        rust_gap, rust_maximin, rust_minimax = dth_complete_rs.toeplitz_saddle_rs(
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
            rust_solution = dth_complete_rs.attempt_support_rs(
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


@pytest.mark.slow  # 26s: runs the whole sweep on both backends and compares bytes
def test_synthetic_sweep_backends_are_bit_identical(tmp_path) -> None:
    table = make_synthetic_table()
    python_dir = tmp_path / "python"
    rust_dir = tmp_path / "rust"
    CompleteTablebaseBuilder(
        output_dir=python_dir, backend="python", table=table
    ).sweep()
    CompleteTablebaseBuilder(output_dir=rust_dir, backend="rust", table=table).sweep()

    for name in ("value.npy", "solver_kind.npy"):
        assert (python_dir / name).read_bytes() == (rust_dir / name).read_bytes(), (
            f"{name} differs between backends"
        )
    python_meta = CompleteTablebase(python_dir).metadata
    rust_meta = CompleteTablebase(rust_dir).metadata
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
        "lp_ipm",
        "warm_attempts",
        "class_count",
        "table_digest",
    ):
        assert python_meta[key] == rust_meta[key], key


def test_rust_resume_is_byte_identical(tmp_path) -> None:
    table = make_synthetic_table(count=32)
    straight = tmp_path / "straight"
    interrupted = tmp_path / "interrupted"
    CompleteTablebaseBuilder(output_dir=straight, backend="rust", table=table).sweep()
    first = CompleteTablebaseBuilder(output_dir=interrupted, backend="rust", table=table)
    assert first.sweep(stop_after_layers=9) is False
    second = CompleteTablebaseBuilder(output_dir=interrupted, backend="rust", table=table)
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
    CompleteTablebaseBuilder(
        output_dir=parallel_dir, backend="rust", table=table
    ).sweep()

    script = (
        "from dth.complete_tablebase import CompleteTablebaseBuilder\n"
        "from dth.tests.test_complete_sweep_python import make_synthetic_table\n"
        f"builder = CompleteTablebaseBuilder(output_dir=r'{serial_dir}', "
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
