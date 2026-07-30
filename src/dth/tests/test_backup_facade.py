"""The read-only backup facade: digests, schema gates, and class lookups."""

import numpy as np
import pytest

from dth.backup_tablebase import (
    BackupTablebase,
    BackupTablebaseBuilder,
    recertify_class,
)
from dth.tests.test_backup_sweep_python import make_synthetic_table


@pytest.fixture(scope="module")
def synthetic_artifact(tmp_path_factory):
    table = make_synthetic_table(count=24)
    target = tmp_path_factory.mktemp("backup") / "artifact"
    BackupTablebaseBuilder(output_dir=target, backend="python", table=table).sweep()
    return table, target


def test_class_lookup_and_bounds(synthetic_artifact) -> None:
    table, target = synthetic_artifact
    tablebase = BackupTablebase(target)
    count = len(table.st_by_profile)
    assert np.isfinite(tablebase.value_of_class(0))
    assert np.isfinite(tablebase.value_of_class(count * count - 1))
    with pytest.raises(LookupError):
        tablebase.value_of_class(count * count)


def test_state_lookup_requires_the_canonical_table(synthetic_artifact) -> None:
    _, target = synthetic_artifact
    tablebase = BackupTablebase(target)
    with pytest.raises(RuntimeError, match="canonical"):
        tablebase.lookup((0, 0, 0, 0))


def test_recertification_agrees_with_stored_values(synthetic_artifact) -> None:
    table, target = synthetic_artifact
    tablebase = BackupTablebase(target)
    values = np.array(
        [tablebase.value_of_class(index) for index in range(len(table.st_by_profile) ** 2)]
    )
    for class_id in (0, 7, 100, len(values) - 1):
        fresh = recertify_class(table, class_id, values, max_support=12)
        assert abs(fresh - values[class_id]) <= 1e-6


def test_metadata_reports_the_routing_split(synthetic_artifact) -> None:
    table, target = synthetic_artifact
    metadata = BackupTablebase(target).metadata
    count = len(table.st_by_profile)
    assert metadata["class_count"] == count * count
    assert metadata["profile_count"] == count
    assert metadata["solver_kinds"] == {"pure": 0, "support": 1, "lp": 2}
    assert (
        metadata["pure_states"] + metadata["support_states"] + metadata["lp_states"]
        == count * count
    )
    assert metadata["execution_backends"] == ["python"]
