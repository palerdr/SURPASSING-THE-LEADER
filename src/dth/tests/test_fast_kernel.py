"""Recurrence parity, certificate, and artifact lifecycle gates."""

import json
from dataclasses import replace

import numpy as np
import pytest

from dth.complete_tablebase import CompleteTablebase, CompleteTablebaseBuilder
from dth.fast_kernel import recurrence_policy
from dth.solver import reconstruct_transition_class_matrix, solve_matrix
from dth.tests.test_complete_sweep_python import (
    make_synthetic_table,
    independent_class_values,
)


def test_recurrence_policies_against_lp():
    rng = np.random.default_rng(90210)
    accepted = 0
    for _ in range(100):
        success = np.sort(rng.uniform(-1, 1, 60))
        failed = 0.9
        candidate = recurrence_policy(success, failed)
        if candidate is None:
            continue
        value, p, q, gap = candidate
        matrix = reconstruct_transition_class_matrix(success, failed)
        reference, _, _ = solve_matrix(matrix)
        assert gap <= 1e-6
        assert abs(value - reference) <= 1e-6
        assert np.max(matrix @ q) - np.min(p @ matrix) <= 1e-6
        accepted += 1
    assert accepted > 50
    assert recurrence_policy(np.full(60, np.nan), 0) is None
    assert recurrence_policy(np.ones(60), 1) is None


@pytest.mark.slow
def test_fast_complete_resume_and_thread_parity(tmp_path):
    table = make_synthetic_table()
    reference = independent_class_values(table)
    for workers, resume in ((1, False), (4, True)):
        path = tmp_path / str(workers)
        options = dict(
            output_dir=path,
            backend="c",
            warm_start=False,
            table=table,
            kernel_workers=workers,
            checkpoint_every=7,
        )
        if resume:
            assert not CompleteTablebaseBuilder(**options).sweep(stop_after_layers=13)
        assert CompleteTablebaseBuilder(**options).sweep()
        artifact = CompleteTablebase(path)
        values = np.load(path / "value.npy")
        assert np.max(np.abs(values - reference)) <= 1e-6
        assert artifact.metadata["execution_backends"] == ["c"]
    assert (tmp_path / "1/value.npy").read_bytes() == (
        tmp_path / "4/value.npy"
    ).read_bytes()
    assert (tmp_path / "1/solver_kind.npy").read_bytes() == (
        tmp_path / "4/solver_kind.npy"
    ).read_bytes()
    path = tmp_path / "1/tablebase.json"
    manifest = json.loads(path.read_text())
    manifest["metadata"]["ladder"] = "pure/warm-support/full-support/lp-v1"
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="ladder"):
        CompleteTablebase(path.parent)


def test_fast_rejects_unsolved_child(tmp_path):
    table = make_synthetic_table(4)
    success = table.success_child_by_profile.copy()
    success[-1, 0] = 0
    broken = replace(table, success_child_by_profile=success)
    with pytest.raises(RuntimeError, match="invalid child"):
        CompleteTablebaseBuilder(
            tmp_path, backend="c", warm_start=False, table=broken
        ).sweep()


def test_canonical_v3_stage_facade_if_built():
    from pathlib import Path
    from dth.agent import CompleteDTHAgent
    from dth.complete_tablebase import FAST_TABLEBASE_SCHEMA

    artifact = Path("src/dth/artifacts/complete_fast_v1")
    manifest = artifact / "tablebase.json"
    if not manifest.exists():
        pytest.skip("build the canonical recurrence artifact for this gate")
    assert json.loads(manifest.read_text())["schema_version"] == FAST_TABLEBASE_SCHEMA
    agent = CompleteDTHAgent(artifact)
    stage = agent.stage_game((0, 0, 0, 0))
    decision = agent.decide((0, 0, 0, 0))
    assert stage.value == pytest.approx(decision.value, abs=1e-12)
    assert (
        np.max(stage.matrix @ stage.check_policy)
        - np.min(stage.drop_policy @ stage.matrix)
        <= 1e-6
    )
    assert np.sum(stage.drop_policy) == pytest.approx(1)
    assert np.sum(stage.check_policy) == pytest.approx(1)
