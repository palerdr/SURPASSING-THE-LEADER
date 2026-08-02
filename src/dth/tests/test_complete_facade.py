"""The read-only complete facade: digests, schema gates, and class lookups."""

import numpy as np
import pytest

from dth.complete_tablebase import (
    CompleteTablebase,
    CompleteTablebaseBuilder,
    recertify_class,
)
from dth.tests.test_complete_sweep_python import make_synthetic_table


@pytest.fixture(scope="module")
def synthetic_artifact(tmp_path_factory):
    table = make_synthetic_table(count=24)
    target = tmp_path_factory.mktemp("complete") / "artifact"
    CompleteTablebaseBuilder(output_dir=target, backend="python", table=table).sweep()
    return table, target


def test_class_lookup_and_bounds(synthetic_artifact) -> None:
    table, target = synthetic_artifact
    tablebase = CompleteTablebase(target)
    count = len(table.st_by_profile)
    assert np.isfinite(tablebase.value_of_class(0))
    assert np.isfinite(tablebase.value_of_class(count * count - 1))
    with pytest.raises(LookupError):
        tablebase.value_of_class(count * count)


def test_state_lookup_requires_the_canonical_table(synthetic_artifact) -> None:
    _, target = synthetic_artifact
    tablebase = CompleteTablebase(target)
    with pytest.raises(RuntimeError, match="canonical"):
        tablebase.lookup((0, 0, 0, 0))


def test_recertification_agrees_with_stored_values(synthetic_artifact) -> None:
    table, target = synthetic_artifact
    tablebase = CompleteTablebase(target)
    values = np.array(
        [tablebase.value_of_class(index) for index in range(len(table.st_by_profile) ** 2)]
    )
    for class_id in (0, 7, 100, len(values) - 1):
        fresh = recertify_class(table, class_id, values, max_support=12)
        assert abs(fresh - values[class_id]) <= 1e-6


def test_metadata_reports_the_routing_split(synthetic_artifact) -> None:
    table, target = synthetic_artifact
    metadata = CompleteTablebase(target).metadata
    count = len(table.st_by_profile)
    assert metadata["class_count"] == count * count
    assert metadata["profile_count"] == count
    assert metadata["solver_kinds"] == {"pure": 0, "support": 1, "lp": 2}
    assert (
        metadata["pure_states"] + metadata["support_states"] + metadata["lp_states"]
        == count * count
    )
    assert metadata["execution_backends"] == ["python"]


def test_lp_residue_uses_ipm_before_tightened_fallback(monkeypatch) -> None:
    import dth.complete_tablebase as module

    def fail(*args, **kwargs):
        raise RuntimeError("forced")

    monkeypatch.setattr(module, "solve_matrix_single_lp", fail)
    monkeypatch.setattr(module, "solve_matrix", fail)
    monkeypatch.setattr(module, "_solve_matrix_tightened", fail)
    success = np.where(np.arange(60) % 2 == 0, 1.0, -1.0)
    value, drop, check, backend = module._solve_residue(
        success, 0.0, max_support=12
    )
    matrix = module.reconstruct_transition_class_matrix(success, 0.0)
    assert backend == "highs-ipm"
    assert np.isfinite(value)
    expected, full_drop, full_check = module._solve_matrix_ipm(matrix)
    assert value == pytest.approx(expected, abs=1e-9)
    assert max(
        0.0,
        float(np.max(matrix @ full_check) - np.min(matrix.T @ full_drop)),
    ) <= 1e-6
