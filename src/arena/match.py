"""One deterministic agent-versus-agent game.

The referee and the seats match the interactive ones: the player-one seat is
named Hal and drops first, while the player-two Baku seat holds the leap-window
action asymmetry. ``hal_lab.harness.series`` plays each base seed in both
seatings and records results per agent.
"""

from __future__ import annotations

import random

from arena.agent import PolicyDrivenAgent, public_state_from_game
from arena.contracts import (
    PublicGameOutcome,
    PublicHalfRound,
    end_provider_game,
    observe_provider,
    reset_provider_game,
)
from arena.variants import PureDTHGame
from stl.engine.actions import validate_action
from stl.engine.game import PHYSICALITY_BAKU, PHYSICALITY_HAL, Game, Player, Referee


def play_match_game(
    provider_one,
    provider_two,
    *,
    seed: int,
    start_clock: int,
    max_half_rounds: int,
    game_index: int = 0,
    pure_dth: bool = False,
) -> tuple[str | None, int]:
    """Play one full game; provider_one holds the Hal seat.

    ``pure_dth=True`` removes only STL's leap-window action 61 while retaining
    the shared canonical engine mechanics.
    """

    seat_one = Player(name="Hal", physicality=PHYSICALITY_HAL)
    seat_two = Player(name="Baku", physicality=PHYSICALITY_BAKU)
    game_type = PureDTHGame if pure_dth else Game
    game = game_type(
        player1=seat_one,
        player2=seat_two,
        referee=Referee(),
        rng=random.Random(seed),
    )
    game.game_clock = start_clock
    agents = {
        "Hal": PolicyDrivenAgent(provider_one, player_name="Hal", seed=seed * 2 + 1),
        "Baku": PolicyDrivenAgent(provider_two, player_name="Baku", seed=seed * 2 + 2),
    }
    reset_provider_game(provider_one)
    reset_provider_game(provider_two)
    half_rounds = 0
    while not game.game_over and half_rounds < max_half_rounds:
        dropper, checker = game.get_roles_for_half(game.current_half)
        turn_duration = game.get_turn_duration()
        public_state = public_state_from_game(game, turn_duration=turn_duration)
        drop = agents[dropper.name].choose_action(game, "dropper", turn_duration)
        check = agents[checker.name].choose_action(game, "checker", turn_duration)
        validate_action(
            drop, actor=dropper.name, role="dropper", turn_duration=turn_duration
        )
        validate_action(
            check, actor=checker.name, role="checker", turn_duration=turn_duration
        )
        result = game.play_half_round(drop, check)
        public_record = PublicHalfRound(
            game_index=game_index,
            half_round_index=half_rounds,
            pre_decision_state=public_state,
            dropper_name=result.dropper,
            checker_name=result.checker,
            drop_time=int(result.drop_time),
            check_time=int(result.check_time),
            outcome=result.result.value,
            game_over=bool(game.game_over),
            winner_name=game.winner.name if game.winner is not None else None,
        )
        observe_provider(provider_one, public_record)
        observe_provider(provider_two, public_record)
        half_rounds += 1
    winner_name = (
        game.winner.name if game.game_over and game.winner is not None else None
    )
    outcome = PublicGameOutcome(
        game_index=game_index,
        winner_name=winner_name,
        half_rounds=half_rounds,
    )
    end_provider_game(provider_one, outcome)
    end_provider_game(provider_two, outcome)
    return winner_name, half_rounds
