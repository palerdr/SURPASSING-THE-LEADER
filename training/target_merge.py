"""Merge and deduplicate saved value-target records.

Trace-driven repair can hit the same public state from several opponent traces.
For MCTS-bootstrap rows that usually means another noisy sample of the same
root value/policy contract, not a distinct training example. This module keeps
the merge conservative: records are averaged only when their feature vector,
source, horizon, and legal masks are identical.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from training.value_targets import ValueTarget, source_breakdown


@dataclass(frozen=True)
class TargetMergeSummary:
    input_records: int
    output_records: int
    duplicate_groups: int
    merged_groups: int
    conflicting_groups: int
    skipped_unselected_groups: int
    max_group_size: int
    max_value_range: float
    max_dropper_policy_l1: float
    max_checker_policy_l1: float
    input_breakdown: dict[str, int]
    output_breakdown: dict[str, int]

    def to_json(self) -> dict:
        return asdict(self)


def _feature_key(record: ValueTarget) -> bytes:
    return np.asarray(record.features, dtype=np.float32).tobytes()


def _same_merge_contract(records: list[ValueTarget]) -> bool:
    first = records[0]
    for record in records[1:]:
        if record.source != first.source or record.horizon != first.horizon:
            return False
        if not np.array_equal(record.dropper_legal_mask, first.dropper_legal_mask):
            return False
        if not np.array_equal(record.checker_legal_mask, first.checker_legal_mask):
            return False
    return True


def _max_pairwise_l1(arrays: list[np.ndarray]) -> float:
    max_l1 = 0.0
    for idx, left in enumerate(arrays):
        for right in arrays[idx + 1 :]:
            max_l1 = max(max_l1, float(np.abs(left - right).sum()))
    return max_l1


def _merge_group(records: list[ValueTarget]) -> ValueTarget:
    first = records[0]
    return ValueTarget(
        features=np.asarray(first.features, dtype=np.float32).copy(),
        value=float(np.mean([record.value for record in records])),
        source=first.source,
        horizon=first.horizon,
        dropper_dist=np.mean(
            [np.asarray(record.dropper_dist, dtype=np.float32) for record in records],
            axis=0,
        ).astype(np.float32),
        checker_dist=np.mean(
            [np.asarray(record.checker_dist, dtype=np.float32) for record in records],
            axis=0,
        ).astype(np.float32),
        dropper_legal_mask=np.asarray(first.dropper_legal_mask, dtype=np.float32).copy(),
        checker_legal_mask=np.asarray(first.checker_legal_mask, dtype=np.float32).copy(),
        unresolved_probability=float(
            np.mean([record.unresolved_probability for record in records])
        ),
    )


def merge_duplicate_targets(
    records: list[ValueTarget],
    *,
    merge_sources: set[str] | None = None,
) -> tuple[list[ValueTarget], TargetMergeSummary]:
    """Average exact duplicate same-contract target records."""
    groups: dict[bytes, list[ValueTarget]] = {}
    order: list[bytes] = []
    for record in records:
        key = _feature_key(record)
        if key not in groups:
            groups[key] = []
            order.append(key)
        groups[key].append(record)

    output: list[ValueTarget] = []
    duplicate_groups = 0
    merged_groups = 0
    conflicting_groups = 0
    skipped_unselected_groups = 0
    max_group_size = 0
    max_value_range = 0.0
    max_dropper_l1 = 0.0
    max_checker_l1 = 0.0

    for key in order:
        group = groups[key]
        if len(group) == 1:
            output.append(group[0])
            continue

        duplicate_groups += 1
        max_group_size = max(max_group_size, len(group))
        values = [float(record.value) for record in group]
        max_value_range = max(max_value_range, max(values) - min(values))
        max_dropper_l1 = max(
            max_dropper_l1,
            _max_pairwise_l1([record.dropper_dist for record in group]),
        )
        max_checker_l1 = max(
            max_checker_l1,
            _max_pairwise_l1([record.checker_dist for record in group]),
        )

        group_sources = {record.source for record in group}
        if merge_sources is not None and not group_sources <= merge_sources:
            output.extend(group)
            skipped_unselected_groups += 1
        elif _same_merge_contract(group):
            output.append(_merge_group(group))
            merged_groups += 1
        else:
            output.extend(group)
            conflicting_groups += 1

    summary = TargetMergeSummary(
        input_records=len(records),
        output_records=len(output),
        duplicate_groups=duplicate_groups,
        merged_groups=merged_groups,
        conflicting_groups=conflicting_groups,
        skipped_unselected_groups=skipped_unselected_groups,
        max_group_size=max_group_size,
        max_value_range=max_value_range,
        max_dropper_policy_l1=max_dropper_l1,
        max_checker_policy_l1=max_checker_l1,
        input_breakdown=source_breakdown(records),
        output_breakdown=source_breakdown(output),
    )
    return output, summary
