"""Tests for Phase 5 calibration metrics.

These tests stay framework-free: predictors are plain callables, and
held-out target lists are constructed in-test via ``ValueTarget`` so we
do not have to spin up the full corpus generator. The contract being
verified is the math of the metrics themselves.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hal.value_net import FEATURE_DIM
from training.calibration import (
    CalibrationReport,
    evaluate_value_net,
)
from training.value_targets import ValueTarget


def _features(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.random(FEATURE_DIM, dtype=np.float32)


def _make_targets() -> list[ValueTarget]:
    """A small synthetic held-out set spanning every relevant source."""
    return [
        ValueTarget(features=_features(0), value=1.0, source="terminal", horizon=0),
        ValueTarget(features=_features(1), value=-1.0, source="terminal", horizon=0),
        ValueTarget(features=_features(2), value=1.0, source="tablebase", horizon=1),
        ValueTarget(features=_features(3), value=-1.0, source="tablebase", horizon=1),
        ValueTarget(features=_features(4), value=0.4, source="exact_horizon_2", horizon=2),
        ValueTarget(features=_features(5), value=-0.4, source="exact_horizon_2", horizon=2),
        ValueTarget(features=_features(6), value=0.6, source="exact_horizon_3", horizon=3),
        ValueTarget(features=_features(7), value=-0.6, source="exact_horizon_3", horizon=3),
    ]


def _label_lookup(targets: list[ValueTarget]) -> dict[bytes, float]:
    return {t.features.tobytes(): t.value for t in targets}


def _perfect_predictor(targets: list[ValueTarget]):
    """Returns the true label by hashing the feature bytes."""
    table = _label_lookup(targets)

    def predict(features: np.ndarray) -> float:
        return table[features.tobytes()]

    return predict


def _zero_predictor(_features: np.ndarray) -> float:
    return 0.0


def test_perfect_predictor_zero_mse_per_source():
    targets = _make_targets()
    predict = _perfect_predictor(targets)
    report = evaluate_value_net(predict, targets)

    assert isinstance(report, CalibrationReport)
    assert report.overall_mse == pytest.approx(0.0)
    for source, mse in report.mse_per_source.items():
        assert mse == pytest.approx(0.0), f"source {source!r} has nonzero MSE"


def test_perfect_predictor_zero_exact_target_error():
    targets = _make_targets()
    predict = _perfect_predictor(targets)
    report = evaluate_value_net(predict, targets)
    assert report.exact_target_error == pytest.approx(0.0)


def test_zero_predictor_has_positive_overall_mse():
    targets = _make_targets()
    report = evaluate_value_net(_zero_predictor, targets)
    assert report.overall_mse > 0.0
    # Terminal labels are ±1, so the contribution to MSE is exactly 1.0
    # for a constant-zero predictor.
    assert report.mse_per_source["terminal"] == pytest.approx(1.0)


def test_brier_score_in_unit_interval_for_perfect_predictor():
    targets = _make_targets()
    report = evaluate_value_net(_perfect_predictor(targets), targets)
    assert 0.0 <= report.brier_score <= 1.0
    # The shifted predictions match the labels in sign perfectly, but
    # interior labels (e.g. value=0.4 → unit=0.7 vs. binary=1) still
    # contribute a small Brier penalty. The score should sit well below
    # the chance baseline of 0.25.
    assert report.brier_score < 0.25


def test_brier_score_zero_on_binary_only_corpus():
    # When every label is ±1, a perfect predictor maps to exactly 0 or 1
    # in [0, 1] space, so the Brier score collapses to 0.
    binary_targets = [
        ValueTarget(features=_features(0), value=1.0, source="terminal", horizon=0),
        ValueTarget(features=_features(1), value=-1.0, source="terminal", horizon=0),
        ValueTarget(features=_features(2), value=1.0, source="tablebase", horizon=1),
        ValueTarget(features=_features(3), value=-1.0, source="tablebase", horizon=1),
    ]
    report = evaluate_value_net(_perfect_predictor(binary_targets), binary_targets)
    assert report.brier_score == pytest.approx(0.0)


def test_brier_score_in_unit_interval_for_zero_predictor():
    targets = _make_targets()
    report = evaluate_value_net(_zero_predictor, targets)
    assert 0.0 <= report.brier_score <= 1.0
    # Zero-prediction maps to 0.5 in [0, 1]; binary labels are 0/1, so
    # each squared error is 0.25.
    assert report.brier_score == pytest.approx(0.25)


def test_reliability_bins_sum_to_total_target_count():
    targets = _make_targets()
    report = evaluate_value_net(_perfect_predictor(targets), targets)
    total_count = sum(count for _, _, count in report.reliability_bins)
    assert total_count == len(targets)


def test_reliability_bins_returns_ten_buckets():
    targets = _make_targets()
    report = evaluate_value_net(_zero_predictor, targets)
    assert len(report.reliability_bins) == 10
    centers = [center for center, _, _ in report.reliability_bins]
    # Centers are equally spaced on (0, 1); strict monotonic increase.
    assert all(centers[i] < centers[i + 1] for i in range(len(centers) - 1))


def test_n_targets_reported():
    targets = _make_targets()
    report = evaluate_value_net(_perfect_predictor(targets), targets)
    assert report.n_targets == len(targets)


def test_predictor_clipping_bounds_outputs():
    targets = _make_targets()

    # An adversarial predictor that returns wildly out-of-range values
    # must still be clipped to [-1, 1] before scoring; otherwise the MSE
    # reported here would be dominated by the unclipped error.
    def predict(_features: np.ndarray) -> float:
        return 50.0

    report = evaluate_value_net(predict, targets)
    # Largest possible squared error after clipping to +1 is (+1 - (-1))**2 = 4.
    assert 0.0 <= report.overall_mse <= 4.0


def test_empty_target_list_is_handled():
    report = evaluate_value_net(_zero_predictor, [])
    assert report.n_targets == 0
    assert report.overall_mse == 0.0
    assert report.exact_target_error == 0.0
    assert report.mse_per_source == {}


def test_exact_target_error_only_uses_tablebase_source():
    # Build a corpus where the predictor is right on tablebase and wrong
    # everywhere else; ``exact_target_error`` must remain 0.
    targets = [
        ValueTarget(features=_features(10), value=1.0, source="tablebase", horizon=1),
        ValueTarget(features=_features(11), value=-1.0, source="tablebase", horizon=1),
        ValueTarget(features=_features(12), value=0.5, source="exact_horizon_2", horizon=2),
    ]
    table = {t.features.tobytes(): t.value for t in targets}

    def predict(features: np.ndarray) -> float:
        v = table[features.tobytes()]
        if v == 0.5:
            return -0.5  # wrong, but only on the non-tablebase source
        return v

    report = evaluate_value_net(predict, targets)
    assert report.exact_target_error == pytest.approx(0.0)
    assert report.mse_per_source["exact_horizon_2"] > 0.0
    assert report.mse_per_source["tablebase"] == pytest.approx(0.0)
