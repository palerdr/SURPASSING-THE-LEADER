"""Calibration metrics for the trained value net.

Phase 5 defensibility gate: held-out targets get scored against a
predictor and summarised in a ``CalibrationReport``. The predictor is a
plain callable mapping feature vectors to Hal-perspective predictions in
[-1, 1], so this module deliberately knows nothing about PyTorch or the
specific net architecture — tests can pass a perfect predictor or a
random predictor without dragging in heavy dependencies.

Metrics:

    - ``mse_per_source``: held-out MSE keyed by ``ValueTarget.source``
      (e.g. ``"terminal"``, ``"tablebase"``, ``"exact_horizon_2"``).
    - ``overall_mse``: held-out MSE across every target.
    - ``brier_score``: Brier score (mean squared error) on the binary
      ``sign(value)`` outcome direction. Predictions and labels are
      shifted from [-1, 1] into [0, 1] before scoring so the result is
      bounded in ``[0, 1]``. *Brier (1950)* is the canonical proper
      scoring rule for probabilistic forecasts.
    - ``reliability_bins``: 10 equal-width reliability buckets over the
      shifted-to-[0, 1] predictions. Each entry is
      ``(predicted_bin_center, observed_mean, count)``.
    - ``exact_target_error``: max absolute error on the subset of
      targets sourced from the tablebase (pinned ground-truth states).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from training.value_targets import ValueTarget


_BRIER_BINS = 10


@dataclass(frozen=True)
class CalibrationReport:
    """Aggregate calibration summary for the trained value net."""

    mse_per_source: dict[str, float]
    overall_mse: float
    brier_score: float
    reliability_bins: list[tuple[float, float, int]]
    exact_target_error: float
    n_targets: int
    mean_unresolved_probability_per_source: dict[str, float]


def _predict_value(
    predict_fn: Callable[[np.ndarray], float],
    features: np.ndarray,
) -> float:
    """Call the predictor and clip the result into [-1, 1]."""
    raw = float(predict_fn(features))
    if raw < -1.0:
        return -1.0
    if raw > 1.0:
        return 1.0
    return raw


def _to_unit_interval(value: float) -> float:
    """Map a Hal-perspective value in [-1, 1] to [0, 1]."""
    return 0.5 * (value + 1.0)


def _reliability_bins(
    pred_unit: np.ndarray,
    label_unit: np.ndarray,
    n_bins: int = _BRIER_BINS,
) -> list[tuple[float, float, int]]:
    """Compute equal-width reliability bins on predictions in [0, 1]."""
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bins: list[tuple[float, float, int]] = []
    # Use right-inclusive bucketing for the final bin so prediction == 1.0
    # is not lost. ``np.digitize`` returns indices in ``[1, n_bins]`` for
    # values inside the half-open intervals; clip the upper edge.
    indices = np.digitize(pred_unit, edges[1:-1], right=False)
    for i in range(n_bins):
        center = 0.5 * (edges[i] + edges[i + 1])
        mask = indices == i
        count = int(mask.sum())
        if count == 0:
            observed_mean = 0.0
        else:
            observed_mean = float(label_unit[mask].mean())
        bins.append((float(center), observed_mean, count))
    return bins


def evaluate_value_net(
    predict_fn: Callable[[np.ndarray], float],
    held_out_targets: list[ValueTarget],
) -> CalibrationReport:
    """Score ``predict_fn`` against the held-out targets.

    Args:
        predict_fn: Callable from a 1-D feature vector to a predicted
            Hal-perspective value in [-1, 1]. The implementation may
            return arbitrary floats; results are clipped before scoring.
        held_out_targets: Targets produced by ``training.value_targets``;
            each carries the ground-truth Hal-perspective value plus the
            ``source`` tag describing how the value was derived.

    Returns:
        ``CalibrationReport`` with MSE breakdowns, Brier score,
        reliability bins, and the maximum absolute error on the
        tablebase subset.
    """
    if not held_out_targets:
        return CalibrationReport(
            mse_per_source={},
            overall_mse=0.0,
            brier_score=0.0,
            reliability_bins=[
                (
                    float(0.5 * (i / _BRIER_BINS + (i + 1) / _BRIER_BINS)),
                    0.0,
                    0,
                )
                for i in range(_BRIER_BINS)
            ],
            exact_target_error=0.0,
            n_targets=0,
            mean_unresolved_probability_per_source={},
        )

    n = len(held_out_targets)
    preds = np.empty(n, dtype=np.float64)
    labels = np.empty(n, dtype=np.float64)
    sources: list[str] = []
    unresolved = np.empty(n, dtype=np.float64)
    tablebase_errors: list[float] = []

    for i, target in enumerate(held_out_targets):
        pred = _predict_value(predict_fn, target.features)
        preds[i] = pred
        labels[i] = target.value
        unresolved[i] = float(target.unresolved_probability)
        sources.append(target.source)
        if str(target.source).startswith("tablebase"):
            # Both "tablebase" (±1 boundary) and "tablebase_interior" (Phase F-2)
            # pins are exact by construction, so both feed the exact-target drift
            # diagnostic.
            tablebase_errors.append(abs(pred - target.value))

    sources_array = np.array(sources)
    squared_error = (preds - labels) ** 2
    overall_mse = float(squared_error.mean())

    mse_per_source: dict[str, float] = {}
    mean_unresolved_probability_per_source: dict[str, float] = {}
    for source in np.unique(sources_array):
        mask = sources_array == source
        mse_per_source[str(source)] = float(squared_error[mask].mean())
        mean_unresolved_probability_per_source[str(source)] = float(unresolved[mask].mean())

    pred_unit = np.clip(_to_unit_interval(preds), 0.0, 1.0)
    # Brier score is a binary proper scoring rule. Hal-perspective values
    # of exactly 0.0 are draws / indeterminate equilibria — they have no
    # binary direction and would otherwise be silently mapped to "Hal wins"
    # by ``labels >= 0``. Drop them from the binary scoring set; reliability
    # bins follow the same filter so observed-rate buckets stay coherent.
    nonzero_mask = labels != 0.0
    if nonzero_mask.any():
        brier_pred = pred_unit[nonzero_mask]
        brier_labels = (labels[nonzero_mask] > 0.0).astype(np.float64)
        brier_score = float(((brier_pred - brier_labels) ** 2).mean())
        reliability = _reliability_bins(brier_pred, brier_labels)
    else:
        brier_score = 0.0
        reliability = _reliability_bins(np.empty(0), np.empty(0))

    exact_target_error = float(max(tablebase_errors)) if tablebase_errors else 0.0

    return CalibrationReport(
        mse_per_source=mse_per_source,
        overall_mse=overall_mse,
        brier_score=brier_score,
        reliability_bins=reliability,
        exact_target_error=exact_target_error,
        n_targets=n,
        mean_unresolved_probability_per_source=mean_unresolved_probability_per_source,
    )
