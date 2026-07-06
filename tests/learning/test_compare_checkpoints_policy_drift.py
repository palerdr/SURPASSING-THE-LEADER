import os
import sys

import numpy as np

sys.path.insert(0, os.getcwd())

from stl.commands.compare_policy_drift import _policy_vector, policy_delta, summarize


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
