"""Diagnostics for matched checkpoint trace-game divergences.

The trace files emitted by ``scripts/trace_checkpoint_games.py`` are useful
only after we can quickly see where champion and candidate play first split.
This module keeps that analysis separate from training: it reads public traces,
replays only the pre-divergence states, and reports compact hints for follow-up
bootstrap or policy-guard data.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict
from typing import Any

from environment.cfr.exact import exact_public_state
from environment.cfr.subgame_resolve import is_critical
from src.Game import Game
from training.trace_bootstrap import replay_trace_state


_COMPARE_RECORD_FIELDS = (
    "round_num",
    "half",
    "dropper",
    "checker",
    "drop_time",
    "check_time",
    "turn_duration",
    "result",
    "survived",
)

_REPORT_RECORD_FIELDS = (
    "round_num",
    "half",
    "dropper",
    "checker",
    "drop_time",
    "check_time",
    "turn_duration",
    "result",
    "survived",
    "st_gained",
    "death_duration",
    "survival_probability",
)


def _game_key(row: dict[str, Any]) -> tuple[int, int]:
    return int(row["seed"]), int(row["game_index"])


def _winner(row: dict[str, Any]) -> str | None:
    winner = row.get("winner")
    return None if winner is None else str(winner)


def _score_for_hal(row: dict[str, Any]) -> int:
    winner = _winner(row)
    if winner == "Hal":
        return 1
    if winner == "Baku":
        return -1
    return 0


def _outcome_class(champion: dict[str, Any], candidate: dict[str, Any]) -> str:
    champion_score = _score_for_hal(champion)
    candidate_score = _score_for_hal(candidate)
    if champion_score > candidate_score:
        return "candidate_regression"
    if candidate_score > champion_score:
        return "candidate_improvement"
    return "same_outcome"


def _record_view(record: dict[str, Any] | None) -> dict[str, Any] | None:
    if record is None:
        return None
    return {field: record.get(field) for field in _REPORT_RECORD_FIELDS if field in record}


def _records_equal(left: dict[str, Any], right: dict[str, Any]) -> bool:
    return all(left.get(field) == right.get(field) for field in _COMPARE_RECORD_FIELDS)


def _record_diff_fields(left: dict[str, Any] | None, right: dict[str, Any] | None) -> list[str]:
    if left is None or right is None:
        return ["history_length"]
    return [
        field
        for field in _COMPARE_RECORD_FIELDS
        if left.get(field) != right.get(field)
    ]


def _hal_role(record: dict[str, Any] | None) -> str | None:
    if record is None:
        return None
    if str(record.get("dropper", "")).lower() == "hal":
        return "dropper"
    if str(record.get("checker", "")).lower() == "hal":
        return "checker"
    return None


def _is_clean_hal_action_divergence(divergence: dict[str, Any]) -> bool:
    """Return true when only Hal's action differs at the first divergence."""
    diff_fields = set(divergence.get("diff_fields", []))
    hal_role = divergence.get("hal_role")
    if hal_role == "dropper":
        return "drop_time" in diff_fields and "check_time" not in diff_fields
    if hal_role == "checker":
        return "check_time" in diff_fields and "drop_time" not in diff_fields
    return False


def _non_actionable_reason(divergence: dict[str, Any] | None) -> str:
    if divergence is None:
        return "no_divergence"
    if _is_clean_hal_action_divergence(divergence):
        return "actionable"
    diff_fields = set(divergence.get("diff_fields", []))
    if diff_fields and diff_fields <= {"result", "survived"}:
        return "chance_only"
    if {"drop_time", "check_time"} <= diff_fields:
        return "mixed_actor_action"
    if "history_length" in diff_fields:
        return "history_length"
    return "non_hal_action_or_state"


def _state_report(game: Game) -> dict[str, Any]:
    dropper, checker = game.get_roles_for_half(game.current_half)
    summary = game.get_state_summary()
    summary["roles"] = {
        "dropper": dropper.name,
        "checker": checker.name,
    }
    summary["turn_duration"] = game.get_turn_duration()
    summary["critical"] = is_critical(game)
    summary["exact_public_state"] = asdict(exact_public_state(game))
    return summary


def _safe_replay_report(row: dict[str, Any], history_index: int) -> dict[str, Any]:
    try:
        game = replay_trace_state(row, history_index)
    except Exception as exc:  # pragma: no cover - defensive report path
        return {
            "replay_ok": False,
            "error": str(exc),
        }
    return {
        "replay_ok": True,
        "state": _state_report(game),
    }


def first_divergence_report(
    champion: dict[str, Any],
    candidate: dict[str, Any],
) -> dict[str, Any] | None:
    """Return the first half-round where two matched trace histories differ."""
    champion_history = list(champion.get("history", []))
    candidate_history = list(candidate.get("history", []))
    min_len = min(len(champion_history), len(candidate_history))

    divergence_index: int | None = None
    for idx in range(min_len):
        if not _records_equal(champion_history[idx], candidate_history[idx]):
            divergence_index = idx
            break
    if divergence_index is None and len(champion_history) != len(candidate_history):
        divergence_index = min_len
    if divergence_index is None:
        return None

    champion_record = (
        champion_history[divergence_index]
        if divergence_index < len(champion_history)
        else None
    )
    candidate_record = (
        candidate_history[divergence_index]
        if divergence_index < len(candidate_history)
        else None
    )
    champion_replay = _safe_replay_report(champion, divergence_index)
    candidate_replay = _safe_replay_report(candidate, divergence_index)
    shared_public_state = False
    critical = False
    if champion_replay.get("replay_ok") and candidate_replay.get("replay_ok"):
        champion_state = champion_replay["state"]["exact_public_state"]
        candidate_state = candidate_replay["state"]["exact_public_state"]
        shared_public_state = champion_state == candidate_state
        critical = bool(champion_replay["state"]["critical"])

    return {
        "history_index": divergence_index,
        "kind": "history_length" if champion_record is None or candidate_record is None else "record",
        "diff_fields": _record_diff_fields(champion_record, candidate_record),
        "hal_role": _hal_role(champion_record) or _hal_role(candidate_record),
        "critical": critical,
        "shared_public_state": shared_public_state,
        "champion_record": _record_view(champion_record),
        "candidate_record": _record_view(candidate_record),
        "champion_pre_state": champion_replay,
        "candidate_pre_state": candidate_replay,
    }


def _counter_dict(counter: Counter[str]) -> dict[str, int]:
    return {key: int(counter[key]) for key in sorted(counter)}


def _target_hint(
    *,
    pair: dict[str, Any],
    divergence: dict[str, Any],
    reason: str,
) -> dict[str, Any]:
    key = pair["key"]
    return {
        "reason": reason,
        "trajectory": "candidate",
        "seed": int(key["seed"]),
        "game_index": int(key["game_index"]),
        "game_seed": int(pair["game_seed"]),
        "history_index": int(divergence["history_index"]),
        "critical": bool(divergence["critical"]),
        "hal_role": divergence.get("hal_role"),
    }


def analyze_trace_divergences(trace_report: dict[str, Any]) -> dict[str, Any]:
    """Summarize outcome swings and first divergences in a checkpoint trace."""
    champion_games = {
        _game_key(row): row for row in trace_report.get("games", {}).get("champion", [])
    }
    candidate_games = {
        _game_key(row): row for row in trace_report.get("games", {}).get("candidate", [])
    }
    paired_keys = sorted(champion_games.keys() & candidate_games.keys())

    outcome_counts: Counter[str] = Counter()
    cause_delta_counts: Counter[str] = Counter()
    first_divergence_buckets: Counter[str] = Counter()
    candidate_loss_causes: Counter[str] = Counter()
    target_hints: list[dict[str, Any]] = []
    paired_reports: list[dict[str, Any]] = []
    non_actionable_regressions = 0
    non_actionable_reasons: Counter[str] = Counter()
    critical_chance_only_regressions = 0
    score_delta_total = 0

    for seed, game_index in paired_keys:
        champion = champion_games[(seed, game_index)]
        candidate = candidate_games[(seed, game_index)]
        outcome = _outcome_class(champion, candidate)
        outcome_counts[outcome] += 1
        champion_score = _score_for_hal(champion)
        candidate_score = _score_for_hal(candidate)
        score_delta = candidate_score - champion_score
        score_delta_total += score_delta
        divergence = first_divergence_report(champion, candidate)

        if _winner(candidate) == "Baku":
            candidate_loss_causes[str(candidate.get("cause", "unknown"))] += 1
        cause_delta_counts[
            f"{champion.get('cause', 'unknown')} -> {candidate.get('cause', 'unknown')}"
        ] += 1
        if divergence is None:
            first_divergence_buckets["none"] += 1
        else:
            bucket = (
                f"h{divergence['history_index']}|"
                f"critical={divergence['critical']}|"
                f"hal_role={divergence.get('hal_role')}"
            )
            first_divergence_buckets[bucket] += 1

        actionability = None
        if outcome == "candidate_regression":
            reason = _non_actionable_reason(divergence)
            actionability = {
                "actionable": reason == "actionable",
                "reason": reason,
            }

        pair_report = {
            "key": {"seed": seed, "game_index": game_index},
            "game_seed": int(champion.get("game_seed", candidate.get("game_seed", 0))),
            "outcome_class": outcome,
            "score_delta": score_delta,
            "champion": {
                "winner": champion.get("winner"),
                "cause": champion.get("cause"),
                "half_rounds": champion.get("half_rounds", len(champion.get("history", []))),
                "score": champion_score,
            },
            "candidate": {
                "winner": candidate.get("winner"),
                "cause": candidate.get("cause"),
                "half_rounds": candidate.get("half_rounds", len(candidate.get("history", []))),
                "score": candidate_score,
            },
            "first_divergence": divergence,
            "actionability": actionability,
        }
        if outcome == "candidate_regression" and divergence is not None:
            if _is_clean_hal_action_divergence(divergence):
                target_hints.append(
                    _target_hint(
                        pair=pair_report,
                        divergence=divergence,
                        reason="candidate_regression_first_action_divergence",
                    )
                )
            else:
                non_actionable_regressions += 1
                reason = _non_actionable_reason(divergence)
                non_actionable_reasons[reason] += 1
                if reason == "chance_only" and bool(divergence.get("critical")):
                    critical_chance_only_regressions += 1
        elif outcome == "candidate_regression":
            non_actionable_regressions += 1
            non_actionable_reasons[_non_actionable_reason(divergence)] += 1
        paired_reports.append(pair_report)

    missing_champion = sorted(set(candidate_games) - set(champion_games))
    missing_candidate = sorted(set(champion_games) - set(candidate_games))
    return {
        "config": trace_report.get("config", {}),
        "source_summaries": trace_report.get("summaries", {}),
        "summary": {
            "paired_games": len(paired_keys),
            "missing_champion_games": len(missing_champion),
            "missing_candidate_games": len(missing_candidate),
            "candidate_score_delta_total": score_delta_total,
            "outcomes": _counter_dict(outcome_counts),
            "candidate_loss_causes": _counter_dict(candidate_loss_causes),
            "cause_transitions": _counter_dict(cause_delta_counts),
            "first_divergence_buckets": _counter_dict(first_divergence_buckets),
            "target_hints": len(target_hints),
            "non_actionable_regressions": non_actionable_regressions,
            "non_actionable_regression_reasons": _counter_dict(non_actionable_reasons),
            "critical_chance_only_regressions": critical_chance_only_regressions,
        },
        "target_hints": target_hints,
        "paired_games": paired_reports,
    }
