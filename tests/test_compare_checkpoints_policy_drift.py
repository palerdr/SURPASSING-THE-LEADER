import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.compare_checkpoints_policy_drift import (
    _policy_vector,
    drift_gate_decision,
    policy_delta,
    summarize,
)


def test_policy_vector_projects_seconds_to_length_61_distribution():
    vec = _policy_vector((1, 3), np.array([0.25, 0.75]))

    assert vec.shape == (61,)
    assert vec[0] == 0.25
    assert vec[2] == 0.75
    assert vec.sum() == 1.0


def test_policy_delta_identical_distributions_are_zero():
    p = _policy_vector((1, 2), np.array([0.5, 0.5]))

    delta = policy_delta(p, p.copy())

    assert delta["tv"] == 0.0
    assert delta["l1"] == 0.0
    assert delta["l_inf"] == 0.0
    assert delta["js"] == 0.0


def test_policy_delta_total_variation_for_disjoint_pure_actions():
    p = _policy_vector((1,), np.array([1.0]))
    q = _policy_vector((2,), np.array([1.0]))

    delta = policy_delta(p, q)

    assert delta["tv"] == 1.0
    assert delta["l1"] == 2.0
    assert delta["l_inf"] == 1.0
    assert delta["js"] > 0.0


def test_summarize_reports_worst_case():
    rows = [
        {"seed": 0, "scenario": "a", "role": "dropper", "delta": {"tv": 0.1, "js": 0.01}},
        {"seed": 1, "scenario": "b", "role": "checker", "delta": {"tv": 0.4, "js": 0.02}},
    ]

    summary = summarize(rows)

    assert summary["cases"] == 2
    assert summary["max_tv"] == 0.4
    assert summary["mean_tv"] == 0.25
    assert summary["worst_case"]["scenario"] == "b"


def test_summarize_empty_rows_has_entropy_keys():
    summary = summarize([])

    assert summary["min_champion_entropy"] == 0.0
    assert summary["min_candidate_entropy"] == 0.0


def test_summarize_reports_entropy_readability_diagnostics():
    rows = [
        {
            "seed": 0,
            "scenario": "a",
            "role": "dropper",
            "delta": {
                "tv": 0.1,
                "js": 0.01,
                "champion_entropy": 2.0,
                "candidate_entropy": 1.2,
            },
        },
        {
            "seed": 1,
            "scenario": "b",
            "role": "checker",
            "delta": {
                "tv": 0.2,
                "js": 0.02,
                "champion_entropy": 1.0,
                "candidate_entropy": 1.3,
            },
        },
    ]

    summary = summarize(rows)

    assert np.isclose(summary["max_entropy_drop"], 0.8)
    assert np.isclose(summary["mean_entropy_delta"], -0.25)
    assert np.isclose(summary["min_champion_entropy"], 1.0)
    assert np.isclose(summary["min_candidate_entropy"], 1.2)


def test_drift_gate_is_optional_without_thresholds():
    assert drift_gate_decision({"max_tv": 0.5}) is None


def test_drift_gate_passes_when_all_thresholds_hold():
    decision = drift_gate_decision(
        {
            "max_tv": 0.05,
            "max_entropy_drop": 0.1,
            "min_candidate_entropy": 1.5,
        },
        max_tv=0.1,
        max_entropy_drop=0.2,
        min_candidate_entropy=1.0,
    )

    assert decision["status"] == "passed"
    assert decision["reasons"] == []


def test_drift_gate_fails_closed_on_policy_or_entropy_regression():
    decision = drift_gate_decision(
        {
            "max_tv": 0.12,
            "max_entropy_drop": 0.3,
            "min_candidate_entropy": 0.9,
        },
        max_tv=0.1,
        max_entropy_drop=0.2,
        min_candidate_entropy=1.0,
    )

    assert decision["status"] == "failed"
    assert len(decision["reasons"]) == 3
    assert "max_tv" in decision["reasons"][0]
    assert "max_entropy_drop" in decision["reasons"][1]
    assert "min_candidate_entropy" in decision["reasons"][2]
