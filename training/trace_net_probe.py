"""Probe raw ValueNet value/policy heads at audited trace states."""

from __future__ import annotations

from typing import Any, Callable

import numpy as np

from training.trace_bootstrap import replay_trace_state
from training.trace_policy_probe import CheckpointProbeSpec


PredictFn = Callable[[Any], tuple[float, np.ndarray, np.ndarray]]


def _game_key(row: dict[str, Any]) -> tuple[int, int]:
    return int(row["seed"]), int(row["game_index"])


def _games_by_trajectory(trace_report: dict[str, Any]) -> dict[str, dict[tuple[int, int], dict]]:
    return {
        "champion": {
            _game_key(row): row
            for row in trace_report.get("games", {}).get("champion", [])
        },
        "candidate": {
            _game_key(row): row
            for row in trace_report.get("games", {}).get("candidate", [])
        },
    }


def _paired_divergence_by_hint(audit_report: dict[str, Any]) -> dict[tuple[int, int, int], dict]:
    indexed = {}
    for row in audit_report.get("paired_games", []):
        divergence = row.get("first_divergence")
        if not divergence:
            continue
        indexed[
            (
                int(row["key"]["seed"]),
                int(row["key"]["game_index"]),
                int(divergence["history_index"]),
            )
        ] = divergence
    return indexed


def _roles_for_hint(hint: dict[str, Any], roles: tuple[str, ...]) -> tuple[str, ...]:
    if roles != ("trace",):
        return roles
    role = hint.get("hal_role")
    if role in ("dropper", "checker"):
        return (str(role),)
    return ("dropper", "checker")


def _entropy(vector: np.ndarray) -> float:
    probs = vector[vector > 0.0]
    if probs.size == 0:
        return 0.0
    return float(-(probs * np.log(probs)).sum())


def _top_policy(vector: np.ndarray, top_k: int) -> list[dict[str, float | int]]:
    order = sorted(range(vector.size), key=lambda idx: (-float(vector[idx]), idx))
    return [
        {"second": int(idx + 1), "probability": float(vector[idx])}
        for idx in order[:top_k]
        if float(vector[idx]) > 0.0
    ]


def _action_second(record: dict[str, Any] | None, role: str) -> int | None:
    if not record:
        return None
    field = "drop_time" if role == "dropper" else "check_time"
    value = record.get(field)
    return None if value is None else int(value)


def _trace_action_probabilities(
    vector: np.ndarray,
    divergence: dict[str, Any] | None,
    role: str,
) -> dict[str, Any]:
    if not divergence:
        return {}
    out = {}
    for name, record_key in (
        ("champion", "champion_record"),
        ("candidate", "candidate_record"),
    ):
        second = _action_second(divergence.get(record_key), role)
        if second is None:
            continue
        out[name] = {
            "second": second,
            "probability": float(vector[second - 1]) if 1 <= second <= 61 else 0.0,
        }
    return out


def _default_predict_fns(
    checkpoints: list[CheckpointProbeSpec],
    device: str,
) -> list[tuple[CheckpointProbeSpec, PredictFn]]:
    from training.train_value_net import load_checkpoint, make_predict_fn

    out = []
    for spec in checkpoints:
        model = load_checkpoint(spec.checkpoint, device=device)
        out.append((spec, make_predict_fn(model, device=device)))
    return out


def probe_trace_net_outputs(
    *,
    trace_report: dict[str, Any],
    audit_report: dict[str, Any],
    checkpoints: list[CheckpointProbeSpec],
    device: str = "cpu",
    roles: tuple[str, ...] = ("trace",),
    top_k: int = 5,
    predict_fns: list[tuple[CheckpointProbeSpec, PredictFn]] | None = None,
) -> dict[str, Any]:
    """Probe raw net value/policy outputs at audit target hints."""
    if not checkpoints and not predict_fns:
        raise ValueError("at least one checkpoint or predict_fn is required")
    invalid_roles = [role for role in roles if role not in ("trace", "dropper", "checker")]
    if invalid_roles:
        raise ValueError(f"invalid roles: {invalid_roles}")

    games = _games_by_trajectory(trace_report)
    divergences = _paired_divergence_by_hint(audit_report)
    predictors = predict_fns or _default_predict_fns(checkpoints, device)

    rows = []
    max_value_delta = 0.0
    max_policy_tv = 0.0
    for hint in audit_report.get("target_hints", []):
        trajectory = str(hint.get("trajectory", "candidate"))
        key = (int(hint["seed"]), int(hint["game_index"]))
        row = games.get(trajectory, {}).get(key)
        if row is None:
            continue
        history_index = int(hint["history_index"])
        game = replay_trace_state(row, history_index)
        probe_roles = _roles_for_hint(hint, roles)
        divergence = divergences.get((key[0], key[1], history_index))
        reports = []
        vectors_by_label = {}
        for spec, predict_fn in predictors:
            value, dropper_dist, checker_dist = predict_fn(game)
            policies = {}
            vectors_by_label[spec.label] = {}
            for role in probe_roles:
                vector = np.asarray(
                    dropper_dist if role == "dropper" else checker_dist,
                    dtype=np.float64,
                )
                total = float(vector.sum())
                if total > 0.0:
                    vector = vector / total
                vectors_by_label[spec.label][role] = vector
                policies[role] = {
                    "entropy": _entropy(vector),
                    "top": _top_policy(vector, top_k),
                    "trace_actions": _trace_action_probabilities(vector, divergence, role),
                }
            reports.append(
                {
                    "label": spec.label,
                    "checkpoint": spec.checkpoint,
                    "net_value_for_hal": float(value),
                    "policies": policies,
                }
            )

        baseline = reports[0]
        deltas = []
        for report in reports[1:]:
            value_delta = float(report["net_value_for_hal"]) - float(
                baseline["net_value_for_hal"]
            )
            max_value_delta = max(max_value_delta, abs(value_delta))
            policy_deltas = {}
            for role in probe_roles:
                base_vec = vectors_by_label[baseline["label"]][role]
                vec = vectors_by_label[report["label"]][role]
                tv = float(0.5 * np.abs(vec - base_vec).sum())
                max_policy_tv = max(max_policy_tv, tv)
                policy_deltas[role] = {
                    "tv": tv,
                    "entropy_delta": float(
                        report["policies"][role]["entropy"]
                        - baseline["policies"][role]["entropy"]
                    ),
                }
            deltas.append(
                {
                    "label": report["label"],
                    "value_delta": value_delta,
                    "policies": policy_deltas,
                }
            )

        rows.append(
            {
                "hint": hint,
                "roles": list(probe_roles),
                "trace_divergence": divergence,
                "checkpoints": reports,
                "deltas_vs_first": deltas,
            }
        )

    return {
        "config": {"device": device, "roles": list(roles), "top_k": int(top_k)},
        "checkpoints": [spec.__dict__ for spec, _ in predictors],
        "summary": {
            "states": len(rows),
            "max_abs_value_delta": max_value_delta,
            "max_policy_tv": max_policy_tv,
        },
        "rows": rows,
    }
