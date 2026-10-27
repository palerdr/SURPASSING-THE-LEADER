"""The exact-matrix response to one Perfect Hal forecast."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from arena.policies.perfect_hal.model import ACTION_COUNT, PerfectHalConfig, _normalize
from dth.agent import CERTIFIED_SADDLE_GAP_TOLERANCE, CertifiedStageGame


def _validate_stage(stage: CertifiedStageGame) -> None:
    matrix = np.asarray(stage.matrix, dtype=np.float64)
    if matrix.shape != (ACTION_COUNT, ACTION_COUNT) or not np.all(np.isfinite(matrix)):
        raise ValueError("certified DTH stage matrix must be finite 60x60")
    for label, raw in (
        ("dropper", stage.drop_policy),
        ("checker", stage.check_policy),
    ):
        policy = np.asarray(raw, dtype=np.float64)
        if (
            policy.shape != (ACTION_COUNT,)
            or not np.all(np.isfinite(policy))
            or np.any(policy < 0.0)
            or abs(float(policy.sum()) - 1.0) > 1e-8
        ):
            raise ValueError(f"certified DTH {label} policy is malformed")
    if (
        not np.isfinite(stage.value)
        or not np.isfinite(stage.saddle_gap)
        or stage.saddle_gap < 0.0
        or stage.saddle_gap > CERTIFIED_SADDLE_GAP_TOLERANCE
    ):
        raise ValueError("certified DTH value or saddle gap is malformed")


def _best_response_policy(
    action_values: np.ndarray,
    config: PerfectHalConfig,
) -> np.ndarray:
    values = np.asarray(action_values, dtype=np.float64)
    if values.shape != (ACTION_COUNT,) or not np.all(np.isfinite(values)):
        raise ValueError("Perfect Hal action values must be finite length 60")
    best = float(np.max(values))
    if config.response_temperature == 0.0:
        winners = values >= best - config.best_response_tolerance
        policy = winners.astype(np.float64)
    else:
        logits = (values - best) / config.response_temperature
        policy = np.exp(np.clip(logits, -700.0, 0.0))
    return _normalize(policy, label="Perfect Hal best response")


@dataclass(frozen=True, slots=True)
class PerfectHalDecision:
    """Public-history and exact-matrix diagnostics for one decision."""

    role: str
    opponent_role: str
    policy: tuple[float, ...]
    opponent_policy: tuple[float, ...]
    action_values: tuple[float, ...]
    expert_weights: tuple[tuple[str, float], ...]
    evidence_count: int
    forecast_entropy: float
    forecast_confidence: float
    expected_payoff: float
    exact_policy_expected_payoff: float
    expected_exploit_gain: float
    best_response_actions: tuple[int, ...]

