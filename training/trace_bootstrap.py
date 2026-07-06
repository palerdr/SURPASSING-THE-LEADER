"""MCTS-bootstrap targets from traced checkpoint regressions.

This module turns matched trace-game histories into ordinary
``mcts_bootstrap`` rows. It is intentionally data-plumbing only: labels still
come from matrix-game MCTS, policy targets are root average strategies, and
critical states can use the existing horizon-bounded subgame resolve hook.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np

from environment.cfr.exact import ExactSearchConfig, exact_public_state
from environment.cfr.subgame_resolve import is_critical
from hal.value_net import extract_features
from src.Game import Game, HalfRoundResult
from training.tournament import _default_starting_game
from training.value_targets import (
    SOURCE_MCTS_BOOTSTRAP,
    ValueTarget,
    _legal_policy_vectors,
    _strategy_vectors,
    save_targets,
)


OutcomeFilter = Literal[
    "champion_win_candidate_loss",
    "candidate_win_champion_loss",
    "candidate_losses",
    "all",
]
Trajectory = Literal["candidate", "champion", "both"]


@dataclass(frozen=True)
class TraceBootstrapMetadata:
    scenario: str
    trajectory: str
    seed: int
    game_index: int
    game_seed: int
    history_index: int
    checkpoint: str
    iterations: int
    critical: bool
    subgame_resolve_at_critical: bool
    subgame_resolve_horizon: int


@dataclass(frozen=True)
class TraceBootstrapRecord:
    target: ValueTarget
    metadata: TraceBootstrapMetadata


@dataclass(frozen=True)
class TraceBootstrapSummary:
    records: int
    matched_games: int
    states_considered: int
    skipped_duplicates: int
    skipped_noncritical: int
    skipped_missing_games: int = 0


def _game_key(row: dict) -> tuple[int, int]:
    return int(row["seed"]), int(row["game_index"])


def _winner(row: dict) -> str | None:
    winner = row.get("winner")
    return None if winner is None else str(winner)


def _pair_matches(champion: dict, candidate: dict, outcome_filter: OutcomeFilter) -> bool:
    champion_winner = _winner(champion)
    candidate_winner = _winner(candidate)
    if outcome_filter == "champion_win_candidate_loss":
        return champion_winner == "Hal" and candidate_winner == "Baku"
    if outcome_filter == "candidate_win_champion_loss":
        return champion_winner == "Baku" and candidate_winner == "Hal"
    if outcome_filter == "candidate_losses":
        return candidate_winner == "Baku"
    if outcome_filter == "all":
        return True
    raise ValueError(f"unknown outcome_filter: {outcome_filter}")


def _trajectory_rows(
    champion: dict,
    candidate: dict,
    trajectory: Trajectory,
) -> tuple[tuple[str, dict], ...]:
    if trajectory == "candidate":
        return (("candidate", candidate),)
    if trajectory == "champion":
        return (("champion", champion),)
    if trajectory == "both":
        return (("champion", champion), ("candidate", candidate))
    raise ValueError(f"unknown trajectory: {trajectory}")


def _survived_outcome(row: dict) -> bool | None:
    survived = row.get("survived")
    if survived is None:
        return None
    return bool(survived)


def replay_trace_state(row: dict, history_index: int) -> Game:
    """Rebuild the pre-decision state before ``history[history_index]``."""
    if history_index < 0 or history_index > len(row["history"]):
        raise IndexError("history_index is outside the trace history")

    game = _default_starting_game(int(row["game_seed"]))
    for step in row["history"][:history_index]:
        record = game.resolve_half_round(
            int(step["drop_time"]),
            int(step["check_time"]),
            survived_outcome=_survived_outcome(step),
        )
        expected = str(step["result"])
        if record.result != HalfRoundResult(expected):
            raise ValueError(
                "trace replay diverged: "
                f"expected result {expected!r}, got {record.result.value!r}"
            )
    return game


def _bootstrap_target_for_game(
    *,
    game: Game,
    predict_fn,
    iterations: int,
    exploration_c: float,
    rng: np.random.Generator,
    config: ExactSearchConfig,
    subgame_resolve_at_critical: bool,
    subgame_resolve_horizon: int,
) -> ValueTarget:
    from environment.cfr.evaluator import ValueNetEvaluator
    from environment.cfr.mcts import MCTSConfig, mcts_search

    result = mcts_search(
        game=game,
        config=MCTSConfig(
            iterations=iterations,
            exploration_c=exploration_c,
            evaluator=None,
            use_tablebase=False,
        ),
        evaluator=ValueNetEvaluator(model_fn=predict_fn),
        rng=rng,
        exact_config=config,
        subgame_resolve_at_critical=subgame_resolve_at_critical,
        subgame_resolve_horizon=subgame_resolve_horizon,
    )
    drop_dist, check_dist = _strategy_vectors(
        drop_seconds=result.root_drop_seconds,
        check_seconds=result.root_check_seconds,
        dropper_strategy=result.root_strategy_dropper_avg,
        checker_strategy=result.root_strategy_checker_avg,
    )
    _, _, drop_mask, check_mask = _legal_policy_vectors(game, config)
    return ValueTarget(
        features=extract_features(game),
        value=float(result.root_value_for_hal),
        source=SOURCE_MCTS_BOOTSTRAP,
        horizon=int(iterations),
        dropper_dist=drop_dist,
        checker_dist=check_dist,
        dropper_legal_mask=drop_mask,
        checker_legal_mask=check_mask,
        unresolved_probability=0.0,
    )


def _history_window_indices(
    row: dict,
    *,
    state_window: int,
    min_history: int,
) -> range:
    if state_window <= 0:
        raise ValueError("state_window must be positive")
    history_len = len(row["history"])
    start = max(int(min_history), history_len - int(state_window))
    return range(start, history_len)


def generate_trace_bootstrap_records(
    *,
    trace_report: dict,
    predict_fn,
    checkpoint: str | Path,
    iterations: int,
    seed: int = 0,
    outcome_filter: OutcomeFilter = "champion_win_candidate_loss",
    trajectory: Trajectory = "candidate",
    state_window: int = 3,
    min_history: int = 0,
    max_states: int | None = None,
    critical_only: bool = True,
    subgame_resolve_at_critical: bool = True,
    subgame_resolve_horizon: int = 3,
    exploration_c: float = 1.0,
    config: ExactSearchConfig | None = None,
) -> tuple[list[TraceBootstrapRecord], TraceBootstrapSummary]:
    """Generate targeted MCTS-bootstrap rows from matched trace games."""
    config = config or ExactSearchConfig()
    champion_games = {
        _game_key(row): row for row in trace_report.get("games", {}).get("champion", [])
    }
    candidate_games = {
        _game_key(row): row for row in trace_report.get("games", {}).get("candidate", [])
    }
    rng_root = np.random.default_rng(seed)
    checkpoint_id = str(checkpoint)
    records: list[TraceBootstrapRecord] = []
    seen_states = set()
    matched_games = 0
    states_considered = 0
    skipped_duplicates = 0
    skipped_noncritical = 0

    for key in sorted(champion_games.keys() & candidate_games.keys()):
        champion = champion_games[key]
        candidate = candidate_games[key]
        if not _pair_matches(champion, candidate, outcome_filter):
            continue
        matched_games += 1

        for trajectory_name, row in _trajectory_rows(champion, candidate, trajectory):
            for history_index in _history_window_indices(
                row,
                state_window=state_window,
                min_history=min_history,
            ):
                if max_states is not None and len(records) >= max_states:
                    break
                states_considered += 1
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

                state_seed = int(rng_root.integers(0, 1 << 31))
                target = _bootstrap_target_for_game(
                    game=game,
                    predict_fn=predict_fn,
                    iterations=iterations,
                    exploration_c=exploration_c,
                    rng=np.random.default_rng(state_seed),
                    config=config,
                    subgame_resolve_at_critical=subgame_resolve_at_critical,
                    subgame_resolve_horizon=subgame_resolve_horizon,
                )
                scenario = (
                    f"{outcome_filter}_{trajectory_name}_"
                    f"s{row['seed']}_g{row['game_index']}_h{history_index}"
                )
                records.append(
                    TraceBootstrapRecord(
                        target=target,
                        metadata=TraceBootstrapMetadata(
                            scenario=scenario,
                            trajectory=trajectory_name,
                            seed=int(row["seed"]),
                            game_index=int(row["game_index"]),
                            game_seed=int(row["game_seed"]),
                            history_index=int(history_index),
                            checkpoint=checkpoint_id,
                            iterations=int(iterations),
                            critical=critical,
                            subgame_resolve_at_critical=bool(subgame_resolve_at_critical),
                            subgame_resolve_horizon=int(subgame_resolve_horizon),
                        ),
                    )
                )
            if max_states is not None and len(records) >= max_states:
                break
        if max_states is not None and len(records) >= max_states:
            break

    return records, TraceBootstrapSummary(
        records=len(records),
        matched_games=matched_games,
        states_considered=states_considered,
        skipped_duplicates=skipped_duplicates,
        skipped_noncritical=skipped_noncritical,
    )


def generate_trace_bootstrap_records_from_hints(
    *,
    trace_report: dict,
    target_hints: list[dict],
    predict_fn,
    checkpoint: str | Path,
    iterations: int,
    seed: int = 0,
    max_states: int | None = None,
    critical_only: bool = False,
    subgame_resolve_at_critical: bool = True,
    subgame_resolve_horizon: int = 3,
    exploration_c: float = 1.0,
    config: ExactSearchConfig | None = None,
) -> tuple[list[TraceBootstrapRecord], TraceBootstrapSummary]:
    """Generate MCTS-bootstrap rows from explicit trace target hints."""
    config = config or ExactSearchConfig()
    games_by_trajectory = {
        "champion": {
            _game_key(row): row
            for row in trace_report.get("games", {}).get("champion", [])
        },
        "candidate": {
            _game_key(row): row
            for row in trace_report.get("games", {}).get("candidate", [])
        },
    }
    rng_root = np.random.default_rng(seed)
    checkpoint_id = str(checkpoint)
    records: list[TraceBootstrapRecord] = []
    seen_states = set()
    states_considered = 0
    skipped_duplicates = 0
    skipped_noncritical = 0
    skipped_missing_games = 0
    matched_keys = set()

    for hint in target_hints:
        if max_states is not None and len(records) >= max_states:
            break
        trajectory_name = str(hint.get("trajectory", "candidate"))
        if trajectory_name not in games_by_trajectory:
            skipped_missing_games += 1
            continue
        key = (int(hint["seed"]), int(hint["game_index"]))
        row = games_by_trajectory[trajectory_name].get(key)
        if row is None:
            skipped_missing_games += 1
            continue
        matched_keys.add((trajectory_name, key))
        history_index = int(hint["history_index"])
        states_considered += 1
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

        state_seed = int(rng_root.integers(0, 1 << 31))
        target = _bootstrap_target_for_game(
            game=game,
            predict_fn=predict_fn,
            iterations=iterations,
            exploration_c=exploration_c,
            rng=np.random.default_rng(state_seed),
            config=config,
            subgame_resolve_at_critical=subgame_resolve_at_critical,
            subgame_resolve_horizon=subgame_resolve_horizon,
        )
        reason = str(hint.get("reason", "trace_hint"))
        scenario = (
            f"{reason}_{trajectory_name}_"
            f"s{row['seed']}_g{row['game_index']}_h{history_index}"
        )
        records.append(
            TraceBootstrapRecord(
                target=target,
                metadata=TraceBootstrapMetadata(
                    scenario=scenario,
                    trajectory=trajectory_name,
                    seed=int(row["seed"]),
                    game_index=int(row["game_index"]),
                    game_seed=int(row["game_seed"]),
                    history_index=history_index,
                    checkpoint=checkpoint_id,
                    iterations=int(iterations),
                    critical=critical,
                    subgame_resolve_at_critical=bool(subgame_resolve_at_critical),
                    subgame_resolve_horizon=int(subgame_resolve_horizon),
                ),
            )
        )

    return records, TraceBootstrapSummary(
        records=len(records),
        matched_games=len(matched_keys),
        states_considered=states_considered,
        skipped_duplicates=skipped_duplicates,
        skipped_noncritical=skipped_noncritical,
        skipped_missing_games=skipped_missing_games,
    )


def save_trace_bootstrap_records(
    records: list[TraceBootstrapRecord],
    path: str | Path,
) -> None:
    """Save targets plus trace metadata arrays for auditability."""
    save_targets([record.target for record in records], path)
    if not records:
        return

    with np.load(path, allow_pickle=True) as data:
        existing = {key: data[key] for key in data.files}
    existing.update(
        {
            "trace_bootstrap_scenarios": np.array(
                [record.metadata.scenario for record in records]
            ),
            "trace_bootstrap_trajectories": np.array(
                [record.metadata.trajectory for record in records]
            ),
            "trace_bootstrap_seeds": np.array(
                [record.metadata.seed for record in records],
                dtype=np.int64,
            ),
            "trace_bootstrap_game_indices": np.array(
                [record.metadata.game_index for record in records],
                dtype=np.int32,
            ),
            "trace_bootstrap_game_seeds": np.array(
                [record.metadata.game_seed for record in records],
                dtype=np.int64,
            ),
            "trace_bootstrap_history_indices": np.array(
                [record.metadata.history_index for record in records],
                dtype=np.int32,
            ),
            "trace_bootstrap_checkpoints": np.array(
                [record.metadata.checkpoint for record in records]
            ),
            "trace_bootstrap_iterations": np.array(
                [record.metadata.iterations for record in records],
                dtype=np.int32,
            ),
            "trace_bootstrap_critical": np.array(
                [record.metadata.critical for record in records],
                dtype=np.bool_,
            ),
            "trace_bootstrap_subgame_resolve_at_critical": np.array(
                [record.metadata.subgame_resolve_at_critical for record in records],
                dtype=np.bool_,
            ),
            "trace_bootstrap_subgame_resolve_horizons": np.array(
                [record.metadata.subgame_resolve_horizon for record in records],
                dtype=np.int32,
            ),
        }
    )
    np.savez(path, **existing)
