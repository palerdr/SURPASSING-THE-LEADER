"""Policy-guard rows from scripted-opponent trace states."""

from __future__ import annotations

import random
from collections.abc import Callable
from dataclasses import dataclass

from environment.cfr.exact import exact_public_state
from environment.legal_actions import validate_action
from environment.opponents.base import Opponent
from hal.agent import SolverAgent
from src.Game import Game
from training.policy_guard import PolicyGuardRecord, policy_guard_target_for_game
from training.value_targets import SOURCE_OPPONENT_TRACE_GUARD
from training.tournament import _HALF_ROUND_SAFETY_LIMIT, _default_starting_game


@dataclass(frozen=True)
class OpponentTraceSummary:
    opponent: str
    records: int
    games: int
    hal_wins: int
    baku_wins: int
    draws: int


def _checked_action(agent, game: Game, *, actor: str, role: str, turn_duration: int) -> int:
    second = int(agent.choose_action(game, role, turn_duration))
    validate_action(second, actor=actor, role=role, turn_duration=turn_duration)
    return second


def generate_opponent_trace_guard_records(
    *,
    probe_agent,
    label_agent: SolverAgent,
    label_checkpoint: str,
    opponent_name: str,
    opponent_factory: Callable[[int], Opponent],
    seeds: list[int],
    games_per_seed: int,
    max_states: int,
    min_history: int = 0,
) -> tuple[list[PolicyGuardRecord], OpponentTraceSummary]:
    """Trace probe-vs-opponent games and champion-label selected public states.

    The opponent lifecycle mirrors ``training.strength.run_ladder``: one
    opponent instance is created per ladder seed and reset at each fresh game.
    This matters for RNG-backed scripted opponents, whose random streams should
    not be re-seeded from the per-game engine seed.
    """
    records: list[PolicyGuardRecord] = []
    seen_states = set()
    hal_wins = 0
    baku_wins = 0
    draws = 0
    games_played = 0

    for seed in seeds:
        rng = random.Random(seed)
        opponent = opponent_factory(seed)
        for game_idx in range(games_per_seed):
            if len(records) >= max_states:
                break
            games_played += 1
            game_seed = rng.randrange(1 << 31)
            game = _default_starting_game(game_seed)
            opponent.reset()
            safety_counter = 0

            while not game.game_over and safety_counter < _HALF_ROUND_SAFETY_LIMIT:
                safety_counter += 1
                state_key = exact_public_state(game)
                if (
                    len(game.history) >= min_history
                    and state_key not in seen_states
                    and len(records) < max_states
                ):
                    records.append(
                        policy_guard_target_for_game(
                            game=game,
                            agent=label_agent,
                            scenario=(
                                f"{opponent_name}_trace_s{seed}_g{game_idx}_h{len(game.history)}"
                            ),
                            seed=seed,
                            checkpoint=label_checkpoint,
                            source=SOURCE_OPPONENT_TRACE_GUARD,
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
                        opponent,
                        game,
                        actor="baku",
                        role="checker",
                        turn_duration=turn_duration,
                    )
                else:
                    drop_time = _checked_action(
                        opponent,
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

    return records, OpponentTraceSummary(
        opponent=opponent_name,
        records=len(records),
        games=games_played,
        hal_wins=hal_wins,
        baku_wins=baku_wins,
        draws=draws,
    )
