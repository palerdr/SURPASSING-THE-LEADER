"""Tournament harness for value-net evaluation.

Phase 5 defensibility gate: pit two action-choosing callables against
each other for ``n_games`` and aggregate winrate, average length, and
the distribution over termination causes.

The action callables are deliberately untyped beyond the
``Opponent.choose_action`` signature
(``(game, role, turn_duration) -> int``) so the harness composes
cleanly with both scripted opponents (``environment.opponents.*``) and
arbitrary ad-hoc callables (e.g. an MCTS-with-trained-net policy).

This module is framework-free: it only imports the engine and (for
optional starting-state cloning) the tablebase scenario type.
"""

from __future__ import annotations

import random
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Callable

from environment.cfr.tactical_scenarios import TacticalScenario
from src.Constants import (
    OPENING_START_CLOCK,
    PHYSICALITY_BAKU,
    PHYSICALITY_HAL,
)
from src.Game import Game, HalfRoundResult
from src.Player import Player
from src.Referee import Referee


ChooseAction = Callable[[Game, str, int], int]


_HALF_ROUND_SAFETY_LIMIT = 200


@dataclass(frozen=True)
class MatchResult:
    """Aggregate outcome of an ``n_games`` tournament."""

    games_played: int
    hal_wins: int
    baku_wins: int
    draws: int
    avg_game_length_half_rounds: float
    cause_of_termination: dict[str, int] = field(default_factory=dict)


def _default_starting_game(seed: int) -> Game:
    """Construct a fresh canonical R1T1 starting game.

    Hal is ``player1`` and the first dropper, mirroring the convention
    used throughout the rigorous core (``tactical_scenarios._base_game``,
    ``training.value_targets._build_game``).
    """
    hal = Player(name="Hal", physicality=PHYSICALITY_HAL)
    baku = Player(name="Baku", physicality=PHYSICALITY_BAKU)
    referee = Referee()
    game = Game(player1=hal, player2=baku, referee=referee, first_dropper=hal)
    game.seed(seed)
    game.game_clock = OPENING_START_CLOCK
    game.current_half = 1
    return game


def _starting_game_from_scenario(
    scenario: TacticalScenario, seed: int
) -> Game:
    """Deep-copy a scenario's game so each match starts from a pristine state."""
    game = deepcopy(scenario.game)
    game.seed(seed)
    return game


def _classify_termination(game: Game) -> str:
    """Map a finished game to a coarse termination-cause bucket."""
    if not game.game_over:
        return "unfinished"
    if not game.history:
        return "unknown"

    last = game.history[-1]
    loser_name = game.loser.name.lower() if game.loser is not None else None

    if last.result == HalfRoundResult.CHECK_FAIL_DIED:
        if loser_name == "hal":
            return "hal_failed_check"
        if loser_name == "baku":
            return "baku_failed_check"
        return "failed_check"
    if last.result == HalfRoundResult.CYLINDER_OVERFLOW_DIED:
        if loser_name == "hal":
            return "hal_overflow"
        if loser_name == "baku":
            return "baku_overflow"
        return "overflow"
    return last.result.value


def _half_round_count_for(game: Game) -> int:
    """Number of half-round records appended during this match."""
    return len(game.history)


def play_match(
    hal_choose_action: ChooseAction,
    baku_choose_action: ChooseAction,
    n_games: int,
    seed: int,
    starting_scenario: TacticalScenario | None = None,
) -> MatchResult:
    """Play ``n_games`` matches between two action-choosing callables.

    Each game is initialised from a fresh copy of either
    ``starting_scenario.game`` (if provided) or a canonical R1T1 board.
    The per-game RNG is seeded from a deterministic mix of ``seed`` and
    the game index, so the entire match is reproducible from
    ``(seed, n_games, scenario)``.

    Within each game we loop half-rounds, asking each callable for its
    drop or check second based on whether it controls the dropper or
    the checker. Hal is ``player1`` in both the default and scenario
    paths (mirroring the rigorous-core convention), so the lookup is
    name-based rather than role-based.
    """
    if n_games <= 0:
        return MatchResult(
            games_played=0,
            hal_wins=0,
            baku_wins=0,
            draws=0,
            avg_game_length_half_rounds=0.0,
            cause_of_termination={},
        )

    rng = random.Random(seed)
    hal_wins = 0
    baku_wins = 0
    draws = 0
    total_half_rounds = 0
    cause_counts: dict[str, int] = {}

    for game_idx in range(n_games):
        game_seed = rng.randrange(1 << 31)

        if starting_scenario is None:
            game = _default_starting_game(game_seed)
        else:
            game = _starting_game_from_scenario(starting_scenario, game_seed)

        safety_counter = 0
        while not game.game_over:
            if safety_counter >= _HALF_ROUND_SAFETY_LIMIT:
                break
            safety_counter += 1

            dropper, checker = game.get_roles_for_half(game.current_half)
            turn_duration = game.get_turn_duration()

            if dropper.name.lower() == "hal":
                drop_time = hal_choose_action(game, "dropper", turn_duration)
                check_time = baku_choose_action(game, "checker", turn_duration)
            else:
                drop_time = baku_choose_action(game, "dropper", turn_duration)
                check_time = hal_choose_action(game, "checker", turn_duration)

            game.play_half_round(drop_time, check_time)

        if game.game_over:
            if game.winner is None:
                draws += 1
            elif game.winner.name.lower() == "hal":
                hal_wins += 1
            else:
                baku_wins += 1
        else:
            draws += 1

        cause = _classify_termination(game)
        cause_counts[cause] = cause_counts.get(cause, 0) + 1
        total_half_rounds += _half_round_count_for(game)

    avg_length = total_half_rounds / n_games
    return MatchResult(
        games_played=n_games,
        hal_wins=hal_wins,
        baku_wins=baku_wins,
        draws=draws,
        avg_game_length_half_rounds=avg_length,
        cause_of_termination=cause_counts,
    )
