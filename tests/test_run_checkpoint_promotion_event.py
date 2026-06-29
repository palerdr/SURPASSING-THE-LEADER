import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.run_checkpoint_promotion_event import _args_for_tests, decide_promotion


def _calibration(*, passed=True, beats=True):
    return {
        "checkpoint": "checkpoints/candidate/best.pt",
        "gate_passed": passed,
        "gate_error": None if passed else "tablebase_interior MSE too high",
        "beats_prev_gen": beats,
        "held_out_overall_mse": 0.05,
    }


def _ladder(overall_delta: int, opponent_deltas=None):
    opponent_deltas = opponent_deltas or {"safe": overall_delta}
    return {
        "config": {
            "champion_checkpoint": "checkpoints/champion/best.pt",
            "candidate_checkpoint": "checkpoints/candidate/best.pt",
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
            "champion_checkpoint": "checkpoints/champion/best.pt",
            "candidate_checkpoint": "checkpoints/candidate/best.pt",
        },
        "summary": {
            "median_midpoint_delta": median_delta,
            "verdicts": verdicts,
        }
    }


def _policy_drift(max_tv=0.2):
    return {
        "config": {
            "champion_checkpoint": "checkpoints/champion/best.pt",
            "candidate_checkpoint": "checkpoints/candidate/best.pt",
        },
        "summary": {
            "cases": 2,
            "max_tv": max_tv,
            "mean_tv": max_tv / 2,
        },
    }


def test_decide_promotion_accepts_when_all_gates_are_non_regressing():
    decision = decide_promotion(
        _calibration(),
        _ladder(15, {"random": 1, "safe": 1, "pattern_reader": 1}),
        _exploitability(0.02, overlap=3),
        _args_for_tests(),
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
        _policy_drift(max_tv=0.3),
    )

    assert decision["status"] == "accepted"
    assert decision["metrics"]["policy_drift"]["max_tv"] == 0.3


def test_decide_promotion_rejects_ladder_and_exploitability_regressions():
    decision = decide_promotion(
        _calibration(),
        _ladder(-11, {"random": 3, "safe": -3, "pattern_reader": -4}),
        _exploitability(0.01, worse=1),
        _args_for_tests(),
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
    )

    assert decision["status"] == "rejected"
    assert "held-out calibration did not strictly beat previous generation" in decision["reasons"]


def test_decide_promotion_rejects_candidate_checkpoint_mismatch():
    ladder = _ladder(5)
    ladder["config"]["candidate_checkpoint"] = "checkpoints/other/best.pt"

    decision = decide_promotion(
        _calibration(),
        ladder,
        _exploitability(0.01),
        _args_for_tests(),
    )

    assert decision["status"] == "rejected"
    assert any("candidate checkpoint mismatch" in reason for reason in decision["reasons"])


def test_decide_promotion_rejects_champion_checkpoint_mismatch():
    exploitability = _exploitability(0.01)
    exploitability["config"]["champion_checkpoint"] = "checkpoints/other_champion/best.pt"

    decision = decide_promotion(
        _calibration(),
        _ladder(5),
        exploitability,
        _args_for_tests(),
    )

    assert decision["status"] == "rejected"
    assert any("champion checkpoint mismatch" in reason for reason in decision["reasons"])


def test_decide_promotion_accepts_relative_and_absolute_equivalent_paths():
    calibration = _calibration()
    calibration["checkpoint"] = str(Path("checkpoints/candidate/best.pt").resolve(strict=False))

    decision = decide_promotion(
        calibration,
        _ladder(5),
        _exploitability(0.01),
        _args_for_tests(),
    )

    assert decision["status"] == "accepted"
    assert decision["reasons"] == []


def test_decide_promotion_rejects_policy_drift_checkpoint_mismatch():
    policy_drift = _policy_drift()
    policy_drift["config"]["candidate_checkpoint"] = "checkpoints/other/best.pt"

    decision = decide_promotion(
        _calibration(),
        _ladder(5),
        _exploitability(0.01),
        _args_for_tests(),
        policy_drift,
    )

    assert decision["status"] == "rejected"
    assert any("candidate checkpoint mismatch" in reason for reason in decision["reasons"])
