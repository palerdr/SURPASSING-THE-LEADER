"""Tests for training/corpus_diagnostics.py (Phase 9.0).

Synthetic ValueTarget fixtures exercise:
- Feature-vector collision detection (with and without value divergence)
- Per-axis coverage histograms
- Per-axis value-distribution summaries
- The axis_carries_signal heuristic
- The summarize_corpus one-shot report
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.getcwd())

from stl.learning.model import FEATURE_DIM
from stl.learning.corpus_diagnostics import (
    AXIS_INVERSE_SCALE,
    FEATURE_INDEX,
    axis_carries_signal,
    feature_collision_report,
    per_axis_coverage,
    summarize_corpus,
    value_distribution_per_axis,
)
from stl.learning.targets import SOURCE_TERMINAL, ValueTarget


def _make_target(
    *,
    value: float,
    source: str = SOURCE_TERMINAL,
    horizon: int = 0,
    **feature_overrides: float,
) -> ValueTarget:
    """Build a ValueTarget with a fully-controlled 23-dim feature vector.

    Pass any axis name from FEATURE_INDEX as a kwarg to set that
    dimension to a chosen value (in *normalized* [0,1] space; do the
    /CYLINDER_MAX conversion at the call site).
    """
    features = np.zeros(FEATURE_DIM, dtype=np.float32)
    for name, val in feature_overrides.items():
        features[FEATURE_INDEX[name]] = val
    return ValueTarget(
        features=features,
        value=value,
        source=source,
        horizon=horizon,
    )


# ── feature_collision_report ──────────────────────────────────────────────


def test_collision_report_empty_corpus_returns_no_groups():
    assert feature_collision_report([]) == []


def test_collision_report_unique_features_returns_no_groups():
    targets = [
        _make_target(value=0.5, baku_cylinder=0.1),
        _make_target(value=-0.3, baku_cylinder=0.2),
        _make_target(value=0.1, baku_cylinder=0.3),
    ]
    assert feature_collision_report(targets) == []


def test_collision_report_flags_divergent_values_at_same_features():
    # Two distinct states (per construction) mapped to identical features
    # but disagreeing on target value — the diagnostic must surface this.
    targets = [
        _make_target(value=1.0, baku_cylinder=0.5),
        _make_target(value=-1.0, baku_cylinder=0.5),
        _make_target(value=0.0, baku_cylinder=0.7),  # control: unique
    ]
    groups = feature_collision_report(targets, only_divergent=True)
    assert len(groups) == 1
    assert sorted(groups[0].record_indices) == [0, 1]
    assert groups[0].has_value_divergence


def test_collision_report_skips_value_agreeing_duplicates_by_default():
    # Two records with identical features AND identical values: cannot
    # be proven distinct from features alone, so default skips them.
    targets = [
        _make_target(value=0.5, baku_cylinder=0.5),
        _make_target(value=0.5, baku_cylinder=0.5),
    ]
    assert feature_collision_report(targets, only_divergent=True) == []


def test_collision_report_with_only_divergent_false_returns_all_groups():
    targets = [
        _make_target(value=0.5, baku_cylinder=0.5),
        _make_target(value=0.5, baku_cylinder=0.5),
    ]
    groups = feature_collision_report(targets, only_divergent=False)
    assert len(groups) == 1
    assert sorted(groups[0].record_indices) == [0, 1]


def test_collision_report_sorted_largest_first():
    targets = [
        _make_target(value=1.0, baku_cylinder=0.1),
        _make_target(value=-1.0, baku_cylinder=0.1),  # 2-group at 0.1
        _make_target(value=0.5, baku_cylinder=0.2),
        _make_target(value=-0.5, baku_cylinder=0.2),
        _make_target(value=0.7, baku_cylinder=0.2),  # 3-group at 0.2
    ]
    groups = feature_collision_report(targets, only_divergent=True)
    assert len(groups) == 2
    assert len(groups[0].record_indices) == 3
    assert len(groups[1].record_indices) == 2


# ── per_axis_coverage ─────────────────────────────────────────────────────


def test_per_axis_coverage_buckets_by_inverse_scaled_feature():
    # Cylinder 240 → feature 240/300 = 0.8 → bin 240 after rescaling
    targets = [
        _make_target(value=0.0, baku_cylinder=0.8),
        _make_target(value=0.0, baku_cylinder=0.8),
        _make_target(value=0.0, baku_cylinder=240.0 / 300.0),  # also 240
        _make_target(value=0.0, baku_cylinder=290.0 / 300.0),  # 290
    ]
    coverage = per_axis_coverage(targets, axes=("baku_cylinder",))
    assert coverage["baku_cylinder"][240] == 3
    assert coverage["baku_cylinder"][290] == 1


def test_per_axis_coverage_default_covers_all_known_axes():
    targets = [_make_target(value=0.0)]
    coverage = per_axis_coverage(targets)
    # Every axis with an inverse scale should appear, even if all-zero
    for axis in AXIS_INVERSE_SCALE:
        assert axis in coverage


def test_per_axis_coverage_asymmetric_deaths_distinct_bins():
    # The whole point of the bc370f3 fix: (baku=1, hal=0) vs (baku=0, hal=1)
    # must be distinguishable in coverage.
    targets = [
        _make_target(value=0.0, baku_deaths=1.0 / 4.0, hal_deaths=0.0),
        _make_target(value=0.0, baku_deaths=0.0, hal_deaths=1.0 / 4.0),
    ]
    coverage = per_axis_coverage(targets, axes=("baku_deaths", "hal_deaths"))
    assert coverage["baku_deaths"] == {1: 1, 0: 1}
    assert coverage["hal_deaths"] == {0: 1, 1: 1}


# ── value_distribution_per_axis ───────────────────────────────────────────


def test_value_distribution_reports_mean_stddev_count():
    targets = [
        _make_target(value=1.0, baku_cylinder=0.5),
        _make_target(value=0.5, baku_cylinder=0.5),
        _make_target(value=-1.0, baku_cylinder=0.5),
        _make_target(value=0.2, baku_cylinder=0.8),
    ]
    dist = value_distribution_per_axis(targets, axes=("baku_cylinder",))
    # bin 150 (baku_cylinder=0.5 → 150) has 3 values: mean=0.166..., stddev>0
    mean, stddev, count = dist["baku_cylinder"][150]
    assert count == 3
    assert mean == 1.0 / 6.0  # (1.0 + 0.5 - 1.0) / 3
    assert stddev > 0.0
    # bin 240 (baku_cylinder=0.8 → 240) has 1 value: stddev=0 by convention
    mean_240, stddev_240, count_240 = dist["baku_cylinder"][240]
    assert count_240 == 1
    assert mean_240 == 0.2
    assert stddev_240 == 0.0


# ── axis_carries_signal ───────────────────────────────────────────────────


def test_axis_carries_signal_returns_true_when_means_spread():
    distribution = {
        0: (-1.0, 0.0, 5),
        1: (0.0, 0.0, 5),
        2: (1.0, 0.0, 5),
    }
    assert axis_carries_signal(distribution) is True


def test_axis_carries_signal_returns_false_when_flat():
    distribution = {
        0: (0.5, 0.0, 5),
        1: (0.5, 0.0, 5),
        2: (0.5, 0.0, 5),
    }
    assert axis_carries_signal(distribution) is False


def test_axis_carries_signal_skips_underpowered_bins():
    # Single record in each bin → ignored when min_records_per_bin=3
    distribution = {
        0: (-1.0, 0.0, 1),
        1: (1.0, 0.0, 1),
    }
    assert axis_carries_signal(distribution, min_records_per_bin=3) is False


def test_axis_carries_signal_requires_minimum_populated_bins():
    distribution = {
        0: (1.0, 0.0, 5),
    }
    # Only 1 populated bin — can't compare spread
    assert axis_carries_signal(distribution, min_bins=2) is False


# ── summarize_corpus ──────────────────────────────────────────────────────


def test_summarize_corpus_returns_all_expected_keys():
    targets = [
        _make_target(value=1.0, baku_cylinder=0.5),
        _make_target(value=-1.0, baku_cylinder=0.7),
    ]
    summary = summarize_corpus(targets)
    expected_keys = {
        "n_records",
        "source_breakdown",
        "axes_with_signal",
        "axes_without_signal",
        "divergent_collision_count",
        "divergent_collision_largest_group_size",
        "per_axis_record_counts",
    }
    assert set(summary.keys()) == expected_keys
    assert summary["n_records"] == 2


def test_summarize_corpus_zero_divergent_collisions_on_distinct_features():
    targets = [
        _make_target(value=0.5, baku_cylinder=0.1),
        _make_target(value=-0.5, baku_cylinder=0.2),
    ]
    summary = summarize_corpus(targets)
    assert summary["divergent_collision_count"] == 0
    assert summary["divergent_collision_largest_group_size"] == 0


def test_summarize_corpus_reports_collision_when_features_collide_with_value_split():
    targets = [
        _make_target(value=1.0, baku_cylinder=0.5),
        _make_target(value=-1.0, baku_cylinder=0.5),
    ]
    summary = summarize_corpus(targets)
    assert summary["divergent_collision_count"] == 1
    assert summary["divergent_collision_largest_group_size"] == 2
