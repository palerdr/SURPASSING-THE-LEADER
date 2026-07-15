"""Diagnostic utilities for corpus validation (Phase 9.0).

Three utilities for verifying that a corpus carries usable training signal
*before* paying the compute bill to expand it:

1. ``feature_collision_report``: groups records by hashed feature vector;
   flags groups where multiple distinct target values share identical
   features (true distinct game-states getting smushed by
   ``extract_features``).
2. ``per_axis_coverage``: per known feature axis, counts records per bin.
   Surfaces under-sampled regions of state-space.
3. ``value_distribution_per_axis``: mean/stddev of ``target.value`` per
   axis-bin. Answers "do new axes carry value-signal?" — if a new axis
   has flat value distribution, it's not informing the net.

All three operate on ``list[ValueTarget]`` and have no side effects.
Designed for reuse by both the Phase 9.0 pilot script and the
parameterized REGISTRY acceptance test from Phase F.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass

import numpy as np

from stl.learning.targets import ValueTarget

# Feature index map for the 23-dim extract_features output
# (hal/value_net.py:37). Kept here as a single source of truth; if
# extract_features changes, update this table in lockstep.
FEATURE_INDEX = {
    "hal_cylinder": 0,
    "baku_cylinder": 1,
    "hal_ttd": 2,
    "baku_ttd": 3,
    "hal_deaths": 4,
    "baku_deaths": 5,
    "hal_safe_budget": 6,
    "baku_safe_budget": 7,
    "hal_fail_survival": 8,
    "baku_fail_survival": 9,
    "hal_is_dropper": 10,
    "clock": 11,
    "round_num": 12,
    "half_eq_2": 13,
    "lsr_var_1": 14,
    "lsr_var_2": 15,
    "lsr_var_3": 16,
    "lsr_var_4": 17,
    "is_leap_second_turn": 18,
    "rounds_until_leap_window": 19,
    "cprs": 20,
    "lsr_variation_eq_2": 21,
    "proximity_to_leap": 22,
}

# Per-axis bucket inverse-scaling (multiplier to recover an interpretable
# integer/bin from the normalized feature). Only includes axes where
# bucketing makes physical sense.
AXIS_INVERSE_SCALE = {
    "hal_cylinder": 300.0,
    "baku_cylinder": 300.0,
    "hal_ttd": 300.0,
    "baku_ttd": 300.0,
    "hal_deaths": 4.0,
    "baku_deaths": 4.0,
    "clock": 3600.0,
    "cprs": 12.0,
    "rounds_until_leap_window": 10.0,
    "round_num": 10.0,
}


@dataclass
class CollisionGroup:
    """A group of records with identical feature vectors."""

    feature_hash: str
    record_indices: list[int]
    values: list[float]
    sources: list[str]

    @property
    def has_value_divergence(self) -> bool:
        """True if records in this group disagree on the target value.

        Distinct game states with identical features must (by determinism
        of the LP / MCTS solvers) be revealed by divergent values. If a
        group has tight value agreement, the feature collapse is either
        (a) intended (true game-state equivalence — rare) or (b) silent
        (different states with identical equilibrium values, undetectable
        by this diagnostic).
        """
        if len(self.values) <= 1:
            return False
        return max(self.values) - min(self.values) > 1e-6


def _hash_features(features: np.ndarray, decimals: int = 6) -> str:
    """Stable hash of a feature vector at fixed precision.

    ``decimals=6`` gives 1e-6 quantization — well below the float32
    noise floor — so numerically-identical features hash the same
    while genuinely-different features at machine precision do not.
    """
    return ",".join(f"{x:.{decimals}f}" for x in features)


def feature_collision_report(
    targets: list[ValueTarget],
    *,
    only_divergent: bool = True,
) -> list[CollisionGroup]:
    """Group records by hashed feature vector; return groups of size > 1.

    Args:
        targets: corpus records to inspect.
        only_divergent: if True (default), return only groups whose
            target values disagree across records — those are the true
            "distinct states with identical features" cases. If False,
            return all groups of size > 1 (including duplicates).

    Returns:
        Sorted list of ``CollisionGroup`` records, descending by
        group size.
    """
    buckets: dict[str, list[int]] = {}
    for i, target in enumerate(targets):
        key = _hash_features(target.features)
        buckets.setdefault(key, []).append(i)

    groups: list[CollisionGroup] = []
    for key, indices in buckets.items():
        if len(indices) <= 1:
            continue
        group = CollisionGroup(
            feature_hash=key,
            record_indices=indices,
            values=[float(targets[i].value) for i in indices],
            sources=[targets[i].source for i in indices],
        )
        if only_divergent and not group.has_value_divergence:
            continue
        groups.append(group)

    groups.sort(key=lambda g: -len(g.record_indices))
    return groups


def per_axis_coverage(
    targets: list[ValueTarget],
    *,
    axes: tuple[str, ...] | None = None,
) -> dict[str, dict[int, int]]:
    """Count records per integer bin per axis.

    Bins are recovered from normalized feature values via ``AXIS_INVERSE_SCALE``,
    rounded to int. So a corpus with ``baku_cylinder`` values
    ``{0, 60, 120, 180, 240, 290, 299}`` produces a ``baku_cylinder``
    histogram with those keys.

    Args:
        targets: corpus records to inspect.
        axes: subset of axis names to report. None means all axes
            with an inverse-scale entry.

    Returns:
        ``{axis_name: {bin: count}}``.
    """
    if axes is None:
        axes = tuple(AXIS_INVERSE_SCALE.keys())

    result: dict[str, dict[int, int]] = {axis: {} for axis in axes}
    for target in targets:
        for axis in axes:
            idx = FEATURE_INDEX[axis]
            raw = float(target.features[idx]) * AXIS_INVERSE_SCALE[axis]
            bin_key = int(round(raw))
            result[axis][bin_key] = result[axis].get(bin_key, 0) + 1
    return result


def value_distribution_per_axis(
    targets: list[ValueTarget],
    *,
    axes: tuple[str, ...] | None = None,
) -> dict[str, dict[int, tuple[float, float, int]]]:
    """For each axis bin, report mean / stddev / count of ``target.value``.

    Surfaces "does this axis carry value-signal?" If the mean target
    value is roughly constant across bins, the axis isn't informing
    the net; if mean varies meaningfully across bins, the axis carries
    signal worth training on.

    Args:
        targets: corpus records.
        axes: subset of axis names. None means all known axes.

    Returns:
        ``{axis_name: {bin: (mean_value, stddev_value, count)}}``.
    """
    if axes is None:
        axes = tuple(AXIS_INVERSE_SCALE.keys())

    by_axis_bin: dict[str, dict[int, list[float]]] = {axis: {} for axis in axes}
    for target in targets:
        for axis in axes:
            idx = FEATURE_INDEX[axis]
            raw = float(target.features[idx]) * AXIS_INVERSE_SCALE[axis]
            bin_key = int(round(raw))
            by_axis_bin[axis].setdefault(bin_key, []).append(float(target.value))

    result: dict[str, dict[int, tuple[float, float, int]]] = {}
    for axis, bin_values in by_axis_bin.items():
        result[axis] = {}
        for bin_key, values in bin_values.items():
            mean = statistics.fmean(values)
            stddev = statistics.pstdev(values) if len(values) > 1 else 0.0
            result[axis][bin_key] = (mean, stddev, len(values))
    return result


def axis_carries_signal(
    distribution: dict[int, tuple[float, float, int]],
    *,
    min_bins: int = 2,
    min_records_per_bin: int = 3,
    mean_spread_threshold: float = 0.05,
) -> bool:
    """Heuristic: does this axis's value distribution vary across bins?

    An axis carries signal if its bins' mean values span more than
    ``mean_spread_threshold``. Bins below ``min_records_per_bin`` are
    excluded as underpowered.

    Used by the Phase 9.0 pilot to gate the "should we expand this
    axis in Phase G?" decision.
    """
    populated = [
        (bin_key, mean) for bin_key, (mean, _stddev, count) in distribution.items()
        if count >= min_records_per_bin
    ]
    if len(populated) < min_bins:
        return False
    means = [m for _, m in populated]
    return (max(means) - min(means)) >= mean_spread_threshold


def summarize_corpus(targets: list[ValueTarget]) -> dict[str, object]:
    """One-shot diagnostic summary for a corpus.

    Returns a JSON-serializable dict with source breakdown, axis
    coverage, value-signal verdicts per axis, and divergent collision
    count. Designed for direct inclusion in a pilot run's stdout log.
    """
    from stl.learning.targets import source_breakdown

    coverage = per_axis_coverage(targets)
    distribution = value_distribution_per_axis(targets)
    collisions = feature_collision_report(targets, only_divergent=True)

    signals = {
        axis: axis_carries_signal(distribution[axis])
        for axis in coverage
    }

    return {
        "n_records": len(targets),
        "source_breakdown": source_breakdown(targets),
        "axes_with_signal": [axis for axis, ok in signals.items() if ok],
        "axes_without_signal": [axis for axis, ok in signals.items() if not ok],
        "divergent_collision_count": len(collisions),
        "divergent_collision_largest_group_size": (
            max((len(g.record_indices) for g in collisions), default=0)
        ),
        "per_axis_record_counts": {
            axis: sum(bins.values()) for axis, bins in coverage.items()
        },
    }
