import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.audit_utils import (
    masked_entropy,
    progression_tuple,
    summarize_turn_support,
    summarize_values,
    top_action_probs,
)


def test_masked_entropy_ignores_zero_probability_mass():
    probs = np.array([0.5, 0.5, 0.0], dtype=np.float32)
    mask = np.array([True, True, False])

    entropy = masked_entropy(probs, mask)

    assert 0.69 < entropy < 0.70


def test_top_action_probs_respects_mask_and_sorting():
    probs = np.array([0.05, 0.60, 0.20, 0.15], dtype=np.float32)
    mask = np.array([True, False, True, True])

    top_actions = top_action_probs(probs, mask, top_k=2)

    assert top_actions == [
        {"second": 3, "prob": 0.20000000298023224},
        {"second": 4, "prob": 0.15000000596046448},
    ]


def test_summarize_turn_support_reports_collapse():
    rows = [
        {"turn_index": 0, "action_second": 60, "entropy": 0.1, "value_estimate": 0.5},
        {"turn_index": 0, "action_second": 60, "entropy": 0.2, "value_estimate": 0.4},
        {"turn_index": 0, "action_second": 59, "entropy": 0.3, "value_estimate": 0.6},
    ]

    summary = summarize_turn_support(rows, max_turns=2)

    assert summary == [
        {
            "turn_index": 0,
            "samples": 3,
            "unique_actions": 2,
            "dominant_second": 60,
            "dominant_share": 2 / 3,
            "mean_entropy": 0.20000000000000004,
            "mean_value": 0.5,
        }
    ]


def test_summarize_values_handles_empty_and_nonempty_inputs():
    assert summarize_values([])["count"] == 0

    summary = summarize_values([0.1, 0.2, 0.3])

    assert summary["count"] == 3
    assert round(summary["mean"], 3) == 0.2
    assert round(summary["min"], 3) == 0.1
    assert round(summary["max"], 3) == 0.3


def test_progression_tuple_prioritizes_bridge_progression():
    summary = {
        "reached_round7_pressure": True,
        "reached_round8_bridge": True,
        "reached_round9_pre_leap": False,
        "reached_leap_window": True,
        "reached_leap_turn": False,
        "won": True,
    }

    assert progression_tuple(summary) == (0, 1, 1, 1, 0, 1)
