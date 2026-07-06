"""Policy-guard rows from pattern_reader canary traces.

These targets are not outcome RL. They collect public states reached while a
probe checkpoint plays against the pattern_reader canary, then label those
states with the champion SolverAgent search contract.
"""

from __future__ import annotations

import random
from collections import Counter
from dataclasses import dataclass

from environment.cfr.exact import exact_public_state
from environment.legal_actions import validate_action
from environment.opponents.pattern_reader import PatternReaderBaku
from hal.agent import SolverAgent
from src.Game import Game
from training.policy_guard import PolicyGuardRecord, policy_guard_target_for_game
from training.tournament import _HALF_ROUND_SAFETY_LIMIT, _default_starting_game


@dataclass(frozen=True)
class PatternReaderTraceSummary:
    records: int
    games: int
    hal_wins: int
    baku_wins: int
    draws: int


def _modal_frequency(samples: list[int]) -> float:
    if not samples:
        return 0.0
    counts = Counter(samples)
    return max(counts.values()) / len(samples)


def _hal_check_seconds(game: Game) -> list[int]:
    return [rec.check_time for rec in game.history if rec.checker.lower() == "hal"]


def _hal_drop_seconds(game: Game) -> list[int]:
    return [rec.drop_time for rec in game.history if rec.dropper.lower() == "hal"]


def pattern_reader_active_for_state(game: Game) -> bool:
    """Return whether pattern_reader would exploit Hal in this half-round."""
    if game.game_over:
        return False
    dropper, checker = game.get_roles_for_half(game.current_half)
    if dropper.name.lower() == "baku":
        samples = _hal_check_seconds(game)
    elif checker.name.lower() == "baku":
        samples = _hal_drop_seconds(game)
    else:
        return False
    return len(samples) >= 3 and _modal_frequency(samples) >= 0.5


def _checked_action(agent, game: Game, *, actor: str, role: str, turn_duration: int) -> int:
    second = int(agent.choose_action(game, role, turn_duration))
    validate_action(second, actor=actor, role=role, turn_duration=turn_duration)
    return second


def generate_pattern_reader_guard_records(
    *,
    probe_agent,
    label_agent: SolverAgent,
    label_checkpoint: str,
    seeds: list[int],
    games_per_seed: int,
    max_states: int,
    active_only: bool = True,
    min_history: int = 0,
) -> tuple[list[PolicyGuardRecord], PatternReaderTraceSummary]:
    """Trace probe-vs-pattern_reader games and champion-label selected states."""
    records: list[PolicyGuardRecord] = []
    seen_states = set()
    hal_wins = 0
    baku_wins = 0
    draws = 0
    games_played = 0

    for seed in seeds:
        rng = random.Random(seed)
        for game_idx in range(games_per_seed):
            if len(records) >= max_states:
                break
            games_played += 1
            game_seed = rng.randrange(1 << 31)
            game = _default_starting_game(game_seed)
            reader = PatternReaderBaku(target_name="Hal")
            safety_counter = 0

            while not game.game_over and safety_counter < _HALF_ROUND_SAFETY_LIMIT:
                safety_counter += 1
                should_collect = len(game.history) >= min_history
                if active_only:
                    should_collect = should_collect and pattern_reader_active_for_state(game)
                state_key = exact_public_state(game)
                if should_collect and state_key not in seen_states and len(records) < max_states:
                    records.append(
                        policy_guard_target_for_game(
                            game=game,
                            agent=label_agent,
                            scenario=f"pattern_reader_s{seed}_g{game_idx}_h{len(game.history)}",
                            seed=seed,
                            checkpoint=label_checkpoint,
                        )
                    )
                    seen_states.add(state_key)

                dropper, checker = game.get_roles_for_half(game.current_half)
                turn_duration = game.get_turn_duration()
                if dropper.name.lower() == "hal":
                    drop_time = _checked_action(
                        probe_agent,
                        game,
                        actor="hal",
                        role="dropper",
                        turn_duration=turn_duration,
                    )
                    check_time = _checked_action(
                        reader,
                        game,
                        actor="baku",
                        role="checker",
                        turn_duration=turn_duration,
                    )
                else:
                    drop_time = _checked_action(
                        reader,
                        game,
                        actor="baku",
                        role="dropper",
                        turn_duration=turn_duration,
                    )
                    check_time = _checked_action(
                        probe_agent,
                        game,
                        actor="hal",
                        role="checker",
                        turn_duration=turn_duration,
                    )
                game.play_half_round(drop_time, check_time)

            if game.game_over and game.winner is not None:
                if game.winner.name.lower() == "hal":
                    hal_wins += 1
                else:
                    baku_wins += 1
            else:
                draws += 1
        if len(records) >= max_states:
            break

    return records, PatternReaderTraceSummary(
        records=len(records),
        games=games_played,
        hal_wins=hal_wins,
        baku_wins=baku_wins,
        draws=draws,
    )
