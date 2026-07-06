import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training.trace_policy_gate import (
    TracePolicyGateConfig,
    evaluate_trace_policy_probe,
)
from scripts.check_trace_policy_probe import build_parser
from scripts.run_trace_policy_gate_set import parse_trace_audit_spec


def _probe_report(*, candidate_action_rise=0.03, value_delta=0.02, tv=0.05):
    baseline_candidate_prob = 0.10
    return {
        "rows": [
            {
                "checkpoints": [
                    {
                        "label": "champion",
                        "policies": {
                            "dropper": {
                                "trace_actions": {
                                    "champion": {"second": 2, "probability": 0.50},
                                    "candidate": {
                                        "second": 3,
                                        "probability": baseline_candidate_prob,
                                    },
                                }
                            }
                        },
                    },
                    {
                        "label": "candidate",
                        "policies": {
                            "dropper": {
                                "trace_actions": {
                                    "champion": {"second": 2, "probability": 0.45},
                                    "candidate": {
                                        "second": 3,
                                        "probability": (
                                            baseline_candidate_prob + candidate_action_rise
                                        ),
                                    },
                                }
                            }
                        },
                    },
                ],
                "deltas_vs_first": [
                    {
                        "label": "candidate",
                        "value_delta": value_delta,
                        "policies": {"dropper": {"tv": tv}},
                    }
                ],
            }
        ]
    }


def _probe_report_with_candidate_probabilities(*, baseline_prob, candidate_prob):
    return {
        "rows": [
            {
                "checkpoints": [
                    {
                        "label": "champion",
                        "policies": {
                            "dropper": {
                                "trace_actions": {
                                    "candidate": {
                                        "second": 60,
                                        "probability": baseline_prob,
                                    },
                                }
                            }
                        },
                    },
                    {
                        "label": "candidate",
                        "policies": {
                            "dropper": {
                                "trace_actions": {
                                    "candidate": {
                                        "second": 60,
                                        "probability": candidate_prob,
                                    },
                                }
                            }
                        },
                    },
                ],
                "deltas_vs_first": [
                    {
                        "label": "candidate",
                        "value_delta": 0.0,
                        "policies": {"dropper": {"tv": 0.01}},
                    }
                ],
            }
        ]
    }


def test_trace_policy_gate_passes_small_probe_drift():
    decision = evaluate_trace_policy_probe(
        _probe_report(),
        TracePolicyGateConfig(),
    )

    assert decision["status"] == "passed"
    assert decision["reasons"] == []
    assert decision["metrics"]["cases"] == 1


def test_trace_policy_gate_rejects_candidate_trace_action_rise():
    decision = evaluate_trace_policy_probe(
        _probe_report(candidate_action_rise=0.08),
        TracePolicyGateConfig(max_trace_candidate_action_prob_rise=0.06),
    )

    assert decision["status"] == "failed"
    assert any("candidate trace-action prob rise" in reason for reason in decision["reasons"])


def test_trace_policy_gate_rejects_absolute_candidate_trace_action_probability():
    decision = evaluate_trace_policy_probe(
        _probe_report(candidate_action_rise=0.02),
        TracePolicyGateConfig(max_trace_candidate_action_probability=0.11),
    )

    assert decision["status"] == "failed"
    assert (
        abs(
            decision["metrics"]["worst"]["trace_candidate_action_probability"]
            - 0.12
        )
        < 1e-12
    )
    assert any(
        "candidate trace-action probability" in reason
        for reason in decision["reasons"]
    )


def test_trace_policy_gate_absolute_cap_does_not_reject_below_baseline():
    decision = evaluate_trace_policy_probe(
        _probe_report_with_candidate_probabilities(
            baseline_prob=0.12,
            candidate_prob=0.11,
        ),
        TracePolicyGateConfig(max_trace_candidate_action_probability=0.08),
    )

    assert decision["status"] == "passed"
    assert decision["metrics"]["worst"]["trace_candidate_action_probability"] == 0.11


def test_trace_policy_gate_rejects_value_and_tv_thresholds():
    decision = evaluate_trace_policy_probe(
        _probe_report(value_delta=0.25, tv=0.35),
        TracePolicyGateConfig(max_abs_value_delta=0.15, max_policy_tv=0.20),
    )

    assert decision["status"] == "failed"
    assert any("abs value_delta" in reason for reason in decision["reasons"])
    assert any("policy tv" in reason for reason in decision["reasons"])


def test_trace_policy_gate_cli_accepts_multiple_probe_reports():
    args = build_parser().parse_args(
        [
            "--probe-report", "a.json",
            "--probe-report", "b.json",
            "--out", "gate.json",
            "--max-trace-candidate-action-probability", "0.08",
        ]
    )

    assert args.probe_report == ["a.json", "b.json"]
    assert args.max_trace_candidate_action_probability == 0.08


def test_trace_policy_gate_set_spec_parser_allows_windows_paths():
    spec = parse_trace_audit_spec(
        "lsr|C:/tmp/trace.json|C:/tmp/audit.json"
    )

    assert spec.name == "lsr"
    assert spec.trace_report == "C:/tmp/trace.json"
    assert spec.audit_report == "C:/tmp/audit.json"
