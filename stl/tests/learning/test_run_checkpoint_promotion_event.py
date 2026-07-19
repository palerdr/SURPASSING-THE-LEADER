import os
import sys
from pathlib import Path

sys.path.insert(0, os.getcwd())

from stl.commands.promote import (
    _args_for_tests,
    _required_ladder_opponents,
    build_parser,
    decide_promotion,
)


def _calibration(*, passed=True, beats=True):
    return {
        "checkpoint": "stl/checkpoints/candidate/best.pt",
        "gate_passed": passed,
        "gate_error": None if passed else "tablebase_interior MSE too high",
        "beats_prev_gen": beats,
        "held_out_overall_mse": 0.05,
    }


def _ladder(overall_delta: int, opponent_deltas=None):
    opponent_deltas = opponent_deltas or {
        "safe": overall_delta,
        "pattern_reader": overall_delta,
    }
    return {
        "config": {
            "champion_checkpoint": "stl/checkpoints/champion/best.pt",
            "candidate_checkpoint": "stl/checkpoints/candidate/best.pt",
            "agent_iterations": 30,
        },
        "aggregate": {
            "delta": {"wins_delta": overall_delta},
            "opponents": {
                name: {"delta": {"wins_delta": delta}}
                for name, delta in opponent_deltas.items()
            },
        }
    }


def _exploitability(median_delta: float, *, worse=0, overlap=0):
    verdicts = {}
    if worse:
        verdicts["candidate_certified_worse"] = worse
    if overlap:
        verdicts["intervals_overlap"] = overlap
    return {
        "config": {
            "champion_checkpoint": "stl/checkpoints/champion/best.pt",
            "candidate_checkpoint": "stl/checkpoints/candidate/best.pt",
            "iterations": 30,
        },
        "summary": {
            "median_midpoint_delta": median_delta,
            "verdicts": verdicts,
        }
    }


def _policy_drift(
    *,
    max_tv=0.08,
    mean_tv=None,
    max_entropy_drop=0.0,
    min_champion_entropy=0.2,
    min_candidate_entropy=0.2,
):
    mean_tv = max_tv / 2 if mean_tv is None else mean_tv
    return {
        "config": {
            "champion_checkpoint": "stl/checkpoints/champion/best.pt",
            "candidate_checkpoint": "stl/checkpoints/candidate/best.pt",
            "iterations": 30,
        },
        "summary": {
            "cases": 2,
            "max_tv": max_tv,
            "mean_tv": mean_tv,
            "max_entropy_drop": max_entropy_drop,
            "min_champion_entropy": min_champion_entropy,
            "min_candidate_entropy": min_candidate_entropy,
        },
    }


def _trace_policy_gate(*, status="passed", reasons=None):
    return {
        "config": {
            "champion_checkpoint": "stl/checkpoints/champion/best.pt",
            "candidate_checkpoint": "stl/checkpoints/candidate/best.pt",
            "iterations": 30,
        },
        "decision": {
            "status": status,
            "reasons": reasons or [],
            "metrics": {"cases": 1},
        }
    }


def test_decide_promotion_accepts_when_all_gates_are_non_regressing():
    decision = decide_promotion(
        _calibration(),
        _ladder(15, {"random": 1, "safe": 1, "pattern_reader": 1}),
        _exploitability(0.02, overlap=3),
        _args_for_tests(),
        _policy_drift(),
    )

    assert decision["status"] == "accepted"
    assert decision["reasons"] == []
    assert decision["warnings"] == ["3 exploitability case(s) overlap; not proof of improvement"]


def test_decide_promotion_includes_policy_drift_summary_when_supplied():
    decision = decide_promotion(
        _calibration(),
        _ladder(15, {"random": 1, "safe": 1, "pattern_reader": 1}),
        _exploitability(0.02),
        _args_for_tests(),
        _policy_drift(max_tv=0.08),
    )

    assert decision["status"] == "accepted"
    assert decision["metrics"]["policy_drift"]["max_tv"] == 0.08


def test_decide_promotion_includes_trace_policy_gate_when_supplied():
    decision = decide_promotion(
        _calibration(),
        _ladder(15, {"random": 1, "safe": 1, "pattern_reader": 1}),
        _exploitability(0.02),
        _args_for_tests(),
        _policy_drift(max_tv=0.08),
        _trace_policy_gate(),
    )

    assert decision["status"] == "accepted"
    assert decision["metrics"]["trace_policy_gate"]["status"] == "passed"


def test_decide_promotion_rejects_failed_trace_policy_gate():
    decision = decide_promotion(
        _calibration(),
        _ladder(15, {"random": 1, "safe": 1, "pattern_reader": 1}),
        _exploitability(0.02),
        _args_for_tests(),
        _policy_drift(max_tv=0.08),
        _trace_policy_gate(
            status="failed",
            reasons=["row 0 candidate/dropper candidate trace-action prob rise 0.08"],
        ),
    )

    assert decision["status"] == "rejected"
    assert any("trace policy gate failed" in reason for reason in decision["reasons"])


def test_decide_promotion_rejects_trace_policy_gate_checkpoint_mismatch():
    trace_gate = _trace_policy_gate()
    trace_gate["config"]["candidate_checkpoint"] = "stl/checkpoints/other/best.pt"

    decision = decide_promotion(
        _calibration(),
        _ladder(15, {"random": 1, "safe": 1, "pattern_reader": 1}),
        _exploitability(0.02),
        _args_for_tests(),
        _policy_drift(max_tv=0.08),
        trace_gate,
    )

    assert decision["status"] == "rejected"
    assert any("candidate checkpoint mismatch" in reason for reason in decision["reasons"])


def test_decide_promotion_rejects_missing_required_policy_drift_report():
    decision = decide_promotion(
        _calibration(),
        _ladder(15, {"random": 1, "safe": 1, "pattern_reader": 1}),
        _exploitability(0.02),
        _args_for_tests(),
    )

    assert decision["status"] == "rejected"
    assert "policy drift report is required" in decision["reasons"]


def test_decide_promotion_rejects_missing_required_trace_policy_gate_report():
    decision = decide_promotion(
        _calibration(),
        _ladder(15, {"random": 1, "safe": 1, "pattern_reader": 1}),
        _exploitability(0.02),
        _args_for_tests(require_trace_policy_gate_report=True),
        _policy_drift(max_tv=0.08),
    )

    assert decision["status"] == "rejected"
    assert "trace policy gate report is required" in decision["reasons"]


def test_promotion_decision_defaults_pattern_reader_as_required_ladder_opponent():
    args = build_parser().parse_args(
        [
            "--calibration-report", "calibration.json",
            "--ladder-report", "ladder.json",
            "--exploitability-report", "exploitability.json",
        ]
    )

    assert _required_ladder_opponents(args) == ("pattern_reader",)


def test_required_ladder_opponent_flag_adds_to_pattern_reader_default():
    args = build_parser().parse_args(
        [
            "--calibration-report", "calibration.json",
            "--ladder-report", "ladder.json",
            "--exploitability-report", "exploitability.json",
            "--required-ladder-opponent", "random",
        ]
    )

    assert _required_ladder_opponents(args) == ("pattern_reader", "random")


def test_decide_promotion_rejects_missing_required_pattern_reader_rung():
    decision = decide_promotion(
        _calibration(),
        _ladder(15, {"random": 2, "safe": 2}),
        _exploitability(0.02),
        _args_for_tests(),
        _policy_drift(max_tv=0.08),
    )

    assert decision["status"] == "rejected"
    assert "required ladder opponent(s) missing: pattern_reader" in decision["reasons"]


def test_decide_promotion_rejects_policy_setting_mismatch():
    ladder = _ladder(15, {"random": 1, "safe": 1, "pattern_reader": 1})
    ladder["config"]["policy_ensemble_size"] = 5
    ladder["config"]["policy_uniform_mix"] = 0.4
    drift = _policy_drift(max_tv=0.08)
    drift["config"]["policy_ensemble_size"] = 5
    drift["config"]["policy_uniform_mix"] = 0.4

    decision = decide_promotion(
        _calibration(),
        ladder,
        _exploitability(0.02),
        _args_for_tests(),
        drift,
    )

    assert decision["status"] == "rejected"
    assert any("policy setting mismatch" in reason for reason in decision["reasons"])


def test_decide_promotion_rejects_policy_budget_mismatch():
    drift = _policy_drift(max_tv=0.08)
    drift["config"]["iterations"] = 200

    decision = decide_promotion(
        _calibration(),
        _ladder(15, {"random": 1, "safe": 1, "pattern_reader": 1}),
        _exploitability(0.02),
        _args_for_tests(),
        drift,
    )

    assert decision["status"] == "rejected"
    assert any("policy budget mismatch" in reason for reason in decision["reasons"])


def test_decide_promotion_rejects_resolve_setting_mismatch():
    ladder = _ladder(15, {"random": 1, "safe": 1, "pattern_reader": 1})
    ladder["config"]["resolve_at_critical"] = True
    ladder["config"]["resolve_horizon"] = 3

    decision = decide_promotion(
        _calibration(),
        ladder,
        _exploitability(0.02),
        _args_for_tests(),
        _policy_drift(max_tv=0.08),
    )

    assert decision["status"] == "rejected"
    assert any("policy setting mismatch" in reason for reason in decision["reasons"])


def test_decide_promotion_rejects_excessive_policy_drift():
    decision = decide_promotion(
        _calibration(),
        _ladder(15, {"random": 1, "safe": 1, "pattern_reader": 1}),
        _exploitability(0.02),
        _args_for_tests(max_policy_drift_tv=0.5),
        _policy_drift(max_tv=0.8),
    )

    assert decision["status"] == "rejected"
    assert any("policy drift max_tv" in reason for reason in decision["reasons"])


def test_decide_promotion_rejects_excessive_mean_policy_drift():
    decision = decide_promotion(
        _calibration(),
        _ladder(15, {"random": 1, "safe": 1, "pattern_reader": 1}),
        _exploitability(0.02),
        _args_for_tests(max_policy_drift_mean_tv=0.10),
        _policy_drift(max_tv=0.2, mean_tv=0.18),
    )

    assert decision["status"] == "rejected"
    assert any("policy drift mean_tv" in reason for reason in decision["reasons"])


def test_decide_promotion_rejects_policy_entropy_collapse():
    decision = decide_promotion(
        _calibration(),
        _ladder(15, {"random": 1, "safe": 1, "pattern_reader": 1}),
        _exploitability(0.02),
        _args_for_tests(max_policy_entropy_drop=0.25),
        _policy_drift(max_tv=0.2, max_entropy_drop=0.5),
    )

    assert decision["status"] == "rejected"
    assert any("policy entropy drop" in reason for reason in decision["reasons"])


def test_decide_promotion_rejects_low_candidate_policy_entropy():
    decision = decide_promotion(
        _calibration(),
        _ladder(15, {"random": 1, "safe": 1, "pattern_reader": 1}),
        _exploitability(0.02),
        _args_for_tests(min_policy_candidate_entropy=0.05),
        _policy_drift(max_tv=0.08, min_candidate_entropy=0.03),
    )

    assert decision["status"] == "rejected"
    assert any("policy candidate entropy" in reason for reason in decision["reasons"])


def test_decide_promotion_allows_champion_equivalent_low_policy_entropy():
    decision = decide_promotion(
        _calibration(),
        _ladder(15, {"random": 1, "safe": 1, "pattern_reader": 1}),
        _exploitability(0.02),
        _args_for_tests(min_policy_candidate_entropy=0.05),
        _policy_drift(
            max_tv=0.08,
            min_champion_entropy=0.03,
            min_candidate_entropy=0.03,
        ),
    )

    assert decision["status"] == "accepted"
    assert decision["reasons"] == []


def test_decide_promotion_rejects_low_candidate_policy_entropy_below_champion_floor():
    decision = decide_promotion(
        _calibration(),
        _ladder(15, {"random": 1, "safe": 1, "pattern_reader": 1}),
        _exploitability(0.02),
        _args_for_tests(min_policy_candidate_entropy=0.05),
        _policy_drift(
            max_tv=0.08,
            min_champion_entropy=0.04,
            min_candidate_entropy=0.03,
        ),
    )

    assert decision["status"] == "rejected"
    assert any("below champion floor" in reason for reason in decision["reasons"])


def test_decide_promotion_rejects_ladder_and_exploitability_regressions():
    decision = decide_promotion(
        _calibration(),
        _ladder(-11, {"random": 3, "safe": -3, "pattern_reader": -4}),
        _exploitability(0.01, worse=1),
        _args_for_tests(),
        _policy_drift(),
    )

    assert decision["status"] == "rejected"
    assert any("ladder wins_delta -11" in reason for reason in decision["reasons"])
    assert any("safe -3" in reason for reason in decision["reasons"])
    assert any("certified-worse" in reason for reason in decision["reasons"])


def test_decide_promotion_rejects_calibration_regression():
    decision = decide_promotion(
        _calibration(passed=True, beats=False),
        _ladder(5),
        _exploitability(0.01),
        _args_for_tests(),
        _policy_drift(),
    )

    assert decision["status"] == "rejected"
    assert "held-out calibration did not strictly beat previous generation" in decision["reasons"]


def test_decide_promotion_rejects_candidate_checkpoint_mismatch():
    ladder = _ladder(5)
    ladder["config"]["candidate_checkpoint"] = "stl/checkpoints/other/best.pt"

    decision = decide_promotion(
        _calibration(),
        ladder,
        _exploitability(0.01),
        _args_for_tests(),
        _policy_drift(),
    )

    assert decision["status"] == "rejected"
    assert any("candidate checkpoint mismatch" in reason for reason in decision["reasons"])


def test_decide_promotion_rejects_champion_checkpoint_mismatch():
    exploitability = _exploitability(0.01)
    exploitability["config"]["champion_checkpoint"] = "stl/checkpoints/other_champion/best.pt"

    decision = decide_promotion(
        _calibration(),
        _ladder(5),
        exploitability,
        _args_for_tests(),
        _policy_drift(),
    )

    assert decision["status"] == "rejected"
    assert any("champion checkpoint mismatch" in reason for reason in decision["reasons"])


def test_decide_promotion_accepts_relative_and_absolute_equivalent_paths():
    calibration = _calibration()
    calibration["checkpoint"] = str(Path("stl/checkpoints/candidate/best.pt").resolve(strict=False))

    decision = decide_promotion(
        calibration,
        _ladder(5),
        _exploitability(0.01),
        _args_for_tests(),
        _policy_drift(),
    )

    assert decision["status"] == "accepted"
    assert decision["reasons"] == []


def test_decide_promotion_rejects_policy_drift_checkpoint_mismatch():
    policy_drift = _policy_drift()
    policy_drift["config"]["candidate_checkpoint"] = "stl/checkpoints/other/best.pt"

    decision = decide_promotion(
        _calibration(),
        _ladder(5),
        _exploitability(0.01),
        _args_for_tests(),
        policy_drift,
    )

    assert decision["status"] == "rejected"
    assert any("candidate checkpoint mismatch" in reason for reason in decision["reasons"])
