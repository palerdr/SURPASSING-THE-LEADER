from __future__ import annotations

from collections import Counter, defaultdict

import numpy as np


def masked_entropy(probs: np.ndarray, mask: np.ndarray) -> float:
    legal_probs = probs[mask.astype(bool)]
    legal_probs = legal_probs[legal_probs > 0]
    if legal_probs.size == 0:
        return 0.0
    return float(-(legal_probs * np.log(legal_probs)).sum())


def top_action_probs(probs: np.ndarray, mask: np.ndarray, top_k: int = 5) -> list[dict[str, float | int]]:
    legal_indices = np.flatnonzero(mask)
    ranked = sorted(legal_indices, key=lambda idx: float(probs[idx]), reverse=True)
    return [
        {"second": int(idx) + 1, "prob": float(probs[idx])}
        for idx in ranked[:top_k]
    ]


def summarize_turn_support(turn_rows: list[dict], max_turns: int) -> list[dict[str, float | int]]:
    grouped: dict[int, list[dict]] = defaultdict(list)
    for row in turn_rows:
        grouped[int(row["turn_index"])].append(row)

    summaries: list[dict[str, float | int]] = []
    for turn_index in range(max_turns):
        rows = grouped.get(turn_index, [])
        if not rows:
            continue

        chosen_counts = Counter(int(row["action_second"]) for row in rows)
        dominant_second, dominant_count = chosen_counts.most_common(1)[0]
        summaries.append(
            {
                "turn_index": turn_index,
                "samples": len(rows),
                "unique_actions": len(chosen_counts),
                "dominant_second": dominant_second,
                "dominant_share": dominant_count / len(rows),
                "mean_entropy": float(np.mean([row["entropy"] for row in rows])),
                "mean_value": float(np.mean([row["value_estimate"] for row in rows])),
            }
        )

    return summaries


def summarize_values(values: list[float]) -> dict[str, float | int]:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return {
            "count": 0,
            "mean": 0.0,
            "std": 0.0,
            "min": 0.0,
            "max": 0.0,
        }

    return {
        "count": int(arr.size),
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "min": float(arr.min()),
        "max": float(arr.max()),
    }


def progression_tuple(summary: dict) -> tuple[int, int, int, int, int, int]:
    return (
        int(bool(summary.get("reached_round9_pre_leap", False))),
        int(bool(summary.get("reached_round8_bridge", False))),
        int(bool(summary.get("reached_round7_pressure", False))),
        int(bool(summary.get("reached_leap_window", False))),
        int(bool(summary.get("reached_leap_turn", False))),
        int(bool(summary.get("won", False))),
    )
