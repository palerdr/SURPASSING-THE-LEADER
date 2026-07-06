"""Policy-guard targets from audited checkpoint trace divergences."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from environment.cfr.exact import exact_public_state
from environment.cfr.subgame_resolve import is_critical
from hal.agent import SolverAgent
from training.policy_guard import PolicyGuardRecord, policy_guard_target_for_game
from training.trace_bootstrap import replay_trace_state
from training.value_targets import SOURCE_POLICY_GUARD


@dataclass(frozen=True)
class TracePolicyGuardSummary:
    records: int
    states_considered: int
    skipped_duplicates: int
    skipped_missing_games: int
    skipped_noncritical: int


def _game_key(row: dict) -> tuple[int, int]:
    return int(row["seed"]), int(row["game_index"])


def _games_by_trajectory(trace_report: dict) -> dict[str, dict[tuple[int, int], dict]]:
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


def generate_trace_policy_guard_records(
    *,
    trace_report: dict,
    audit_report: dict,
    label_agent: SolverAgent,
    label_checkpoint: str | Path,
    source: str = SOURCE_POLICY_GUARD,
    max_states: int | None = None,
    critical_only: bool = False,
) -> tuple[list[PolicyGuardRecord], TracePolicyGuardSummary]:
    """Champion-label audited trace target hints as source-tagged guard rows."""
    games = _games_by_trajectory(trace_report)
    records: list[PolicyGuardRecord] = []
    seen_states = set()
    states_considered = 0
    skipped_duplicates = 0
    skipped_missing_games = 0
    skipped_noncritical = 0
    checkpoint_id = str(label_checkpoint)

    for hint in audit_report.get("target_hints", []):
        if max_states is not None and len(records) >= max_states:
            break
        trajectory = str(hint.get("trajectory", "candidate"))
        row = games.get(trajectory, {}).get((int(hint["seed"]), int(hint["game_index"])))
        if row is None:
            skipped_missing_games += 1
            continue
        states_considered += 1
        history_index = int(hint["history_index"])
        game = replay_trace_state(row, history_index)
        critical = is_critical(game)
        if critical_only and not critical:
            skipped_noncritical += 1
            continue
        state_key = exact_public_state(game)
        if state_key in seen_states:
            skipped_duplicates += 1
            continue
        seen_states.add(state_key)

        scenario = (
            f"trace_policy_guard_{trajectory}_"
            f"s{row['seed']}_g{row['game_index']}_h{history_index}_"
            f"{hint.get('reason', 'target_hint')}"
        )
        records.append(
            policy_guard_target_for_game(
                game=game,
                agent=label_agent,
                scenario=scenario,
                seed=int(row["seed"]),
                checkpoint=checkpoint_id,
                source=source,
            )
        )

    return records, TracePolicyGuardSummary(
        records=len(records),
        states_considered=states_considered,
        skipped_duplicates=skipped_duplicates,
        skipped_missing_games=skipped_missing_games,
        skipped_noncritical=skipped_noncritical,
    )
