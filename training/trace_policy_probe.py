"""Probe deployed SolverAgent policy/value at audited trace states."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from environment.cfr.exact import exact_public_state
from environment.cfr.subgame_resolve import is_critical
from hal.agent import SolverAgent
from src.Game import Game
from training.trace_bootstrap import replay_trace_state


AgentFactory = Callable[..., Any]


@dataclass(frozen=True)
class CheckpointProbeSpec:
    label: str
    checkpoint: str
    search_prior_uniform_mix: float | None = None


def parse_checkpoint_spec(raw: str) -> CheckpointProbeSpec:
    """Parse ``label:path`` CLI checkpoint specs."""
    if ":" not in raw:
        raise ValueError(f"checkpoint spec must be 'label:path', got {raw!r}")
    label, checkpoint = raw.split(":", 1)
    label = label.strip()
    checkpoint = checkpoint.strip()
    if not label or not checkpoint:
        raise ValueError(f"checkpoint spec must be 'label:path', got {raw!r}")
    return CheckpointProbeSpec(label=label, checkpoint=checkpoint)


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


def _vector(seconds: tuple[int, ...], probabilities: np.ndarray) -> np.ndarray:
    vector = np.zeros(61, dtype=np.float64)
    for second, probability in zip(seconds, probabilities):
        vector[int(second) - 1] = float(probability)
    total = float(vector.sum())
    if total > 0.0:
        vector /= total
    return vector


def _entropy(vector: np.ndarray) -> float:
    probs = vector[vector > 0.0]
    if probs.size == 0:
        return 0.0
    return float(-(probs * np.log(probs)).sum())


def _top_policy(vector: np.ndarray, top_k: int) -> list[dict[str, float | int]]:
    if top_k <= 0:
        return []
    order = sorted(
        range(vector.size),
        key=lambda idx: (-float(vector[idx]), idx),
    )
    return [
        {"second": int(idx + 1), "probability": float(vector[idx])}
        for idx in order[:top_k]
        if float(vector[idx]) > 0.0
    ]


def _policy_report(agent: Any, game: Game, role: str, top_k: int) -> dict[str, Any]:
    seconds, probabilities = agent.policy(game, role)
    vector = _vector(tuple(int(second) for second in seconds), np.asarray(probabilities))
    return {
        "seconds": [int(second) for second in seconds],
        "entropy": _entropy(vector),
        "top": _top_policy(vector, top_k),
        "vector": vector,
    }


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
    actions = {}
    for name, record_key in (
        ("champion", "champion_record"),
        ("candidate", "candidate_record"),
    ):
        second = _action_second(divergence.get(record_key), role)
        if second is None:
            continue
        actions[name] = {
            "second": second,
            "probability": float(vector[second - 1]) if 1 <= second <= 61 else 0.0,
        }
    return actions


def _public_state_summary(game: Game) -> dict[str, Any]:
    dropper, checker = game.get_roles_for_half(game.current_half)
    return {
        "round_num": int(game.round_num),
        "current_half": int(game.current_half),
        "game_clock": float(game.game_clock),
        "game_clock_display": game.format_game_clock(),
        "turn_duration": int(game.get_turn_duration()),
        "roles": {"dropper": dropper.name, "checker": checker.name},
        "critical": bool(is_critical(game)),
        "public_state": repr(exact_public_state(game)),
    }


def _paired_divergence_by_hint(audit_report: dict[str, Any]) -> dict[tuple[int, int, int], dict]:
    indexed = {}
    for row in audit_report.get("paired_games", []):
        divergence = row.get("first_divergence")
        if not divergence:
            continue
        key = (
            int(row["key"]["seed"]),
            int(row["key"]["game_index"]),
            int(divergence["history_index"]),
        )
        indexed[key] = divergence
    return indexed


def _roles_for_hint(hint: dict[str, Any], roles: tuple[str, ...]) -> tuple[str, ...]:
    if roles != ("trace",):
        return roles
    role = hint.get("hal_role")
    if role in ("dropper", "checker"):
        return (str(role),)
    return ("dropper", "checker")


def probe_trace_policies(
    *,
    trace_report: dict[str, Any],
    audit_report: dict[str, Any],
    checkpoints: list[CheckpointProbeSpec],
    iterations: int,
    seed: int = 0,
    policy_ensemble_size: int = 1,
    policy_uniform_mix: float = 0.0,
    search_prior_uniform_mix: float = 0.0,
    resolve_at_critical: bool = False,
    resolve_horizon: int = 3,
    roles: tuple[str, ...] = ("trace",),
    top_k: int = 5,
    agent_factory: AgentFactory = SolverAgent,
) -> dict[str, Any]:
    """Probe deployed search values and policies at audit target hints."""
    if not checkpoints:
        raise ValueError("at least one checkpoint is required")
    invalid_roles = [role for role in roles if role not in ("trace", "dropper", "checker")]
    if invalid_roles:
        raise ValueError(f"invalid roles: {invalid_roles}")

    games = _games_by_trajectory(trace_report)
    divergences = _paired_divergence_by_hint(audit_report)
    agents = [
        (
            spec,
            agent_factory(
                spec.checkpoint,
                player_name="Hal",
                iterations=iterations,
                seed=seed,
                policy_ensemble_size=policy_ensemble_size,
                policy_uniform_mix=policy_uniform_mix,
                search_prior_uniform_mix=(
                    search_prior_uniform_mix
                    if spec.search_prior_uniform_mix is None
                    else spec.search_prior_uniform_mix
                ),
                resolve_at_critical=resolve_at_critical,
                resolve_horizon=resolve_horizon,
            ),
        )
        for spec in checkpoints
    ]

    rows = []
    max_value_delta = 0.0
    max_tv = 0.0
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
        checkpoint_reports = []
        vectors_by_label: dict[str, dict[str, np.ndarray]] = {}

        for spec, agent in agents:
            search_result = agent.search(game)
            policies: dict[str, dict[str, Any]] = {}
            vectors_by_label[spec.label] = {}
            for role in probe_roles:
                policy = _policy_report(agent, game, role, top_k)
                vectors_by_label[spec.label][role] = policy.pop("vector")
                policy["trace_actions"] = _trace_action_probabilities(
                    vectors_by_label[spec.label][role],
                    divergence,
                    role,
                )
                policies[role] = policy
            checkpoint_reports.append(
                {
                    "label": spec.label,
                    "checkpoint": spec.checkpoint,
                    "root_value_for_hal": float(search_result.root_value_for_hal),
                    "root_visits": int(search_result.root_visits),
                    "cells_used": int(search_result.cells_used),
                    "policies": policies,
                }
            )

        baseline = checkpoint_reports[0]
        baseline_label = str(baseline["label"])
        deltas = []
        for report in checkpoint_reports[1:]:
            label = str(report["label"])
            value_delta = float(report["root_value_for_hal"]) - float(
                baseline["root_value_for_hal"]
            )
            max_value_delta = max(max_value_delta, abs(value_delta))
            policy_deltas = {}
            for role in probe_roles:
                baseline_vector = vectors_by_label[baseline_label][role]
                vector = vectors_by_label[label][role]
                tv = float(0.5 * np.abs(vector - baseline_vector).sum())
                max_tv = max(max_tv, tv)
                policy_deltas[role] = {
                    "tv": tv,
                    "entropy_delta": float(
                        report["policies"][role]["entropy"]
                        - baseline["policies"][role]["entropy"]
                    ),
                    "top_probability_delta": (
                        float(vector[int(baseline["policies"][role]["top"][0]["second"]) - 1])
                        - float(
                            baseline_vector[
                                int(baseline["policies"][role]["top"][0]["second"]) - 1
                            ]
                        )
                        if baseline["policies"][role]["top"]
                        else 0.0
                    ),
                }
            deltas.append(
                {
                    "label": label,
                    "value_delta": value_delta,
                    "policies": policy_deltas,
                }
            )

        rows.append(
            {
                "hint": hint,
                "state": _public_state_summary(game),
                "trace_divergence": divergence,
                "roles": list(probe_roles),
                "checkpoints": checkpoint_reports,
                "deltas_vs_first": deltas,
            }
        )

    return {
        "config": {
            "iterations": int(iterations),
            "seed": int(seed),
            "policy_ensemble_size": int(policy_ensemble_size),
            "policy_uniform_mix": float(policy_uniform_mix),
            "search_prior_uniform_mix": float(search_prior_uniform_mix),
            "resolve_at_critical": bool(resolve_at_critical),
            "resolve_horizon": int(resolve_horizon),
            "roles": list(roles),
            "top_k": int(top_k),
        },
        "checkpoints": [spec.__dict__ for spec in checkpoints],
        "summary": {
            "states": len(rows),
            "max_abs_value_delta": max_value_delta,
            "max_policy_tv": max_tv,
        },
        "rows": rows,
    }
