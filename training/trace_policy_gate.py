"""Gate trace-state policy probes before expensive live canaries."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class TracePolicyGateConfig:
    max_policy_tv: float = 0.20
    max_abs_value_delta: float = 0.15
    max_trace_champion_action_prob_drop: float = 0.20
    max_trace_candidate_action_prob_rise: float = 0.06
    max_trace_candidate_action_probability: float | None = None


def _checkpoint_by_label(row: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(item["label"]): item for item in row.get("checkpoints", [])}


def _trace_action_probability(
    checkpoint_report: dict[str, Any],
    role: str,
    action: str,
) -> float | None:
    trace_actions = (
        checkpoint_report.get("policies", {})
        .get(role, {})
        .get("trace_actions", {})
    )
    item = trace_actions.get(action)
    if not item:
        return None
    return float(item.get("probability", 0.0))


def evaluate_trace_policy_probe(
    probe_report: dict[str, Any],
    config: TracePolicyGateConfig,
) -> dict[str, Any]:
    """Return pass/fail decision for a trace policy probe report."""
    reasons: list[str] = []
    worst = {
        "policy_tv": 0.0,
        "abs_value_delta": 0.0,
        "trace_champion_action_prob_drop": 0.0,
        "trace_candidate_action_prob_rise": 0.0,
        "trace_candidate_action_probability": 0.0,
    }
    cases = 0

    for row_index, row in enumerate(probe_report.get("rows", [])):
        by_label = _checkpoint_by_label(row)
        if not row.get("checkpoints"):
            continue
        baseline = row["checkpoints"][0]
        for delta in row.get("deltas_vs_first", []):
            label = str(delta["label"])
            candidate = by_label.get(label)
            if candidate is None:
                reasons.append(f"row {row_index} missing checkpoint report for {label}")
                continue
            value_delta = float(delta.get("value_delta", 0.0))
            abs_value_delta = abs(value_delta)
            worst["abs_value_delta"] = max(worst["abs_value_delta"], abs_value_delta)
            if abs_value_delta > config.max_abs_value_delta:
                reasons.append(
                    f"row {row_index} {label} abs value_delta {abs_value_delta:.6f} "
                    f"> {config.max_abs_value_delta:.6f}"
                )

            for role, policy_delta in delta.get("policies", {}).items():
                cases += 1
                tv = float(policy_delta.get("tv", 0.0))
                worst["policy_tv"] = max(worst["policy_tv"], tv)
                if tv > config.max_policy_tv:
                    reasons.append(
                        f"row {row_index} {label}/{role} policy tv {tv:.6f} "
                        f"> {config.max_policy_tv:.6f}"
                    )

                baseline_champion_prob = _trace_action_probability(
                    baseline,
                    role,
                    "champion",
                )
                candidate_champion_prob = _trace_action_probability(
                    candidate,
                    role,
                    "champion",
                )
                if (
                    baseline_champion_prob is not None
                    and candidate_champion_prob is not None
                ):
                    drop = baseline_champion_prob - candidate_champion_prob
                    worst["trace_champion_action_prob_drop"] = max(
                        worst["trace_champion_action_prob_drop"],
                        drop,
                    )
                    if drop > config.max_trace_champion_action_prob_drop:
                        reasons.append(
                            f"row {row_index} {label}/{role} champion trace-action "
                            f"prob drop {drop:.6f} > "
                            f"{config.max_trace_champion_action_prob_drop:.6f}"
                        )

                baseline_candidate_prob = _trace_action_probability(
                    baseline,
                    role,
                    "candidate",
                )
                candidate_candidate_prob = _trace_action_probability(
                    candidate,
                    role,
                    "candidate",
                )
                if (
                    baseline_candidate_prob is not None
                    and candidate_candidate_prob is not None
                ):
                    worst["trace_candidate_action_probability"] = max(
                        worst["trace_candidate_action_probability"],
                        candidate_candidate_prob,
                    )
                    if (
                        config.max_trace_candidate_action_probability is not None
                        and candidate_candidate_prob
                        > config.max_trace_candidate_action_probability
                        and candidate_candidate_prob > baseline_candidate_prob
                    ):
                        reasons.append(
                            f"row {row_index} {label}/{role} candidate trace-action "
                            f"probability {candidate_candidate_prob:.6f} > "
                            f"{config.max_trace_candidate_action_probability:.6f} "
                            f"and baseline {baseline_candidate_prob:.6f}"
                        )
                    rise = candidate_candidate_prob - baseline_candidate_prob
                    worst["trace_candidate_action_prob_rise"] = max(
                        worst["trace_candidate_action_prob_rise"],
                        rise,
                    )
                    if rise > config.max_trace_candidate_action_prob_rise:
                        reasons.append(
                            f"row {row_index} {label}/{role} candidate trace-action "
                            f"prob rise {rise:.6f} > "
                            f"{config.max_trace_candidate_action_prob_rise:.6f}"
                        )

    return {
        "status": "passed" if not reasons else "failed",
        "reasons": reasons,
        "metrics": {
            "cases": cases,
            "worst": worst,
            "thresholds": asdict(config),
        },
    }


def combine_trace_policy_gate_decisions(
    per_report: list[dict[str, Any]],
    config: TracePolicyGateConfig,
) -> dict[str, Any]:
    """Combine per-probe gate decisions into one rolling gate decision."""
    all_reasons: list[str] = []
    total_cases = 0
    worst = {
        "policy_tv": 0.0,
        "abs_value_delta": 0.0,
        "trace_champion_action_prob_drop": 0.0,
        "trace_candidate_action_prob_rise": 0.0,
        "trace_candidate_action_probability": 0.0,
    }
    for item in per_report:
        label = str(item.get("probe_report", item.get("case", "probe")))
        decision = item.get("decision", {})
        all_reasons.extend(f"{label}: {reason}" for reason in decision.get("reasons", []))
        metrics = decision.get("metrics", {})
        total_cases += int(metrics.get("cases", 0))
        for key, value in metrics.get("worst", {}).items():
            worst[key] = max(float(worst.get(key, 0.0)), float(value))

    return {
        "status": "passed" if not all_reasons else "failed",
        "reasons": all_reasons,
        "metrics": {
            "reports": len(per_report),
            "cases": total_cases,
            "worst": worst,
            "thresholds": asdict(config),
        },
    }
