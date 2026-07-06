import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training.target_merge import merge_duplicate_targets
from training.value_targets import SOURCE_MCTS_BOOTSTRAP, SOURCE_TABLEBASE, ValueTarget


def _target(
    *,
    value: float,
    source: str = SOURCE_MCTS_BOOTSTRAP,
    horizon: int = 30,
    drop_second: int = 1,
):
    features = np.arange(8, dtype=np.float32)
    dropper = np.zeros(61, dtype=np.float32)
    dropper[drop_second - 1] = 1.0
    checker = np.zeros(61, dtype=np.float32)
    checker[59] = 1.0
    mask = np.ones(61, dtype=np.float32)
    return ValueTarget(
        features=features,
        value=value,
        source=source,
        horizon=horizon,
        dropper_dist=dropper,
        checker_dist=checker,
        dropper_legal_mask=mask,
        checker_legal_mask=mask,
        unresolved_probability=0.2,
    )


def test_merge_duplicate_targets_averages_same_contract_records():
    records = [_target(value=-0.4, drop_second=1), _target(value=-0.2, drop_second=2)]

    merged, summary = merge_duplicate_targets(records)

    assert len(merged) == 1
    assert merged[0].source == SOURCE_MCTS_BOOTSTRAP
    assert merged[0].horizon == 30
    assert merged[0].value == -0.30000000000000004
    assert merged[0].dropper_dist[0] == 0.5
    assert merged[0].dropper_dist[1] == 0.5
    assert merged[0].unresolved_probability == 0.2
    assert summary.input_records == 2
    assert summary.output_records == 1
    assert summary.duplicate_groups == 1
    assert summary.merged_groups == 1
    assert summary.conflicting_groups == 0
    assert summary.max_dropper_policy_l1 == 2.0


def test_merge_duplicate_targets_keeps_cross_source_conflicts_separate():
    records = [
        _target(value=-0.4, source=SOURCE_MCTS_BOOTSTRAP),
        _target(value=1.0, source=SOURCE_TABLEBASE, horizon=0),
    ]

    merged, summary = merge_duplicate_targets(records)

    assert len(merged) == 2
    assert summary.duplicate_groups == 1
    assert summary.merged_groups == 0
    assert summary.conflicting_groups == 1


def test_merge_duplicate_targets_respects_source_filter():
    records = [_target(value=-0.4), _target(value=-0.2)]

    merged, summary = merge_duplicate_targets(records, merge_sources={SOURCE_TABLEBASE})

    assert len(merged) == 2
    assert summary.duplicate_groups == 1
    assert summary.merged_groups == 0
    assert summary.skipped_unselected_groups == 1
