"""Probe bounded selective resolves at audited trace states.

This module is diagnostic glue: it replays public states from trace reports,
runs ``resolve_subgame`` directly even when the state is not marked critical,
and records the local exact/selective strategy evidence. It does not generate
training rows by itself; use it to decide whether a trace repair should become
an MCTS-bootstrap target.
"""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Any

import numpy as np

from environment.cfr.evaluator import LeafEvaluator
from environment.cfr.exact import ExactGameSnapshot, ExactSearchConfig, exact_public_state
from environment.cfr.selective import generate_candidates
from environment.cfr.subgame_resolve import is_critical, resolve_subgame
from src.Game import Game
from training.trace_bootstrap import replay_trace_state


@dataclass(frozen=True)
class ResolveProbeSpec:
    label: str
    evaluator: LeafEvaluator | None


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
        if 1 <= int(second) <= 61:
            vector[int(second) - 1] = float(probability)
    total = float(vector.sum())
    if total > 0.0:
        vector /= total
    return vector


def _top_policy(vector: np.ndarray, top_k: int) -> list[dict[str, float | int]]:
    if top_k <= 0:
        return []
    order = sorted(range(vector.size), key=lambda idx: (-float(vector[idx]), idx))
    return [
        {"second": int(idx + 1), "probability": float(vector[idx])}
        for idx in order[:top_k]
        if float(vector[idx]) > 0.0
    ]


def _strategy_report(
    seconds: tuple[int, ...],
    strategy: np.ndarray,
    *,
    top_k: int,
    trace_divergence: dict[str, Any] | None,
    role: str,
) -> dict[str, Any]:
    strategy = np.asarray(strategy, dtype=np.float64)
    vector = _vector(seconds, strategy)
    return {
        "seconds": [int(second) for second in seconds],
        "probabilities": [float(probability) for probability in strategy],
        "top": _top_policy(vector, top_k),
        "trace_actions": _trace_action_probabilities(vector, trace_divergence, role),
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
            "second": int(second),
            "probability": float(vector[second - 1]) if 1 <= second <= 61 else 0.0,
        }
    return actions


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
        "hal_cylinder": float(game.player1.cylinder),
        "baku_cylinder": float(game.player2.cylinder),
        "hal_safe_strategies_remaining": int(game.player1.safe_strategies_remaining),
        "baku_safe_strategies_remaining": int(game.player2.safe_strategies_remaining),
        "public_state": repr(exact_public_state(game)),
    }


def probe_trace_resolves(
    *,
    trace_report: dict[str, Any],
    audit_report: dict[str, Any],
    evaluators: list[ResolveProbeSpec],
    horizons: tuple[int, ...] = (3,),
    top_k: int = 5,
    config: ExactSearchConfig | None = None,
) -> dict[str, Any]:
    """Run bounded selective resolves for each target hint in an audit report."""
    if not evaluators:
        raise ValueError("at least one evaluator spec is required")
    if not horizons:
        raise ValueError("at least one horizon is required")
    invalid_horizons = [horizon for horizon in horizons if int(horizon) < 0]
    if invalid_horizons:
        raise ValueError(f"horizons must be non-negative: {invalid_horizons}")

    config = config or ExactSearchConfig()
    games = _games_by_trajectory(trace_report)
    divergences = _paired_divergence_by_hint(audit_report)
    rows = []
    skipped_missing_games = 0
    max_candidate_count = 0
    max_runtime_seconds = 0.0
    resolve_count = 0

    for hint in audit_report.get("target_hints", []):
        trajectory = str(hint.get("trajectory", "candidate"))
        key = (int(hint["seed"]), int(hint["game_index"]))
        row = games.get(trajectory, {}).get(key)
        if row is None:
            skipped_missing_games += 1
            continue

        history_index = int(hint["history_index"])
        game = replay_trace_state(row, history_index)
        candidates = generate_candidates(game, config)
        max_candidate_count = max(max_candidate_count, int(candidates.joint_count))
        divergence = divergences.get((key[0], key[1], history_index))
        snapshot = ExactGameSnapshot(game)
        resolve_reports = []

        for spec in evaluators:
            for horizon in horizons:
                snapshot.restore(game)
                start = time.perf_counter()
                result = resolve_subgame(
                    game,
                    horizon=int(horizon),
                    config=config,
                    evaluator=spec.evaluator,
                )
                runtime_seconds = time.perf_counter() - start
                snapshot.restore(game)
                max_runtime_seconds = max(max_runtime_seconds, float(runtime_seconds))
                resolve_count += 1
                resolve_reports.append(
                    {
                        "label": spec.label,
                        "horizon": int(horizon),
                        "runtime_seconds": float(runtime_seconds),
                        "value_for_hal": float(result.value_for_hal),
                        "unresolved_probability": float(result.unresolved_probability),
                        "candidate_count": int(result.candidate_count),
                        "dropper": _strategy_report(
                            tuple(int(second) for second in result.drop_seconds),
                            np.asarray(result.dropper_strategy, dtype=np.float64),
                            top_k=top_k,
                            trace_divergence=divergence,
                            role="dropper",
                        ),
                        "checker": _strategy_report(
                            tuple(int(second) for second in result.check_seconds),
                            np.asarray(result.checker_strategy, dtype=np.float64),
                            top_k=top_k,
                            trace_divergence=divergence,
                            role="checker",
                        ),
                    }
                )

        rows.append(
            {
                "hint": hint,
                "state": _public_state_summary(game),
                "candidate_actions": {
                    "drop_seconds": [int(second) for second in candidates.drop_seconds],
                    "check_seconds": [int(second) for second in candidates.check_seconds],
                    "joint_count": int(candidates.joint_count),
                },
                "trace_divergence": divergence,
                "resolves": resolve_reports,
            }
        )

    return {
        "config": {
            "horizons": [int(horizon) for horizon in horizons],
            "top_k": int(top_k),
            "evaluators": [spec.label for spec in evaluators],
        },
        "summary": {
            "states": len(rows),
            "resolves": resolve_count,
            "skipped_missing_games": skipped_missing_games,
            "max_candidate_count": max_candidate_count,
            "max_runtime_seconds": max_runtime_seconds,
        },
        "rows": rows,
    }
