#!/usr/bin/env python3
"""Run CanonicalHal against scripted Baku opponents and print diagnostics."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.Player import Player
from src.Referee import Referee
from src.Game import Game
from src.Constants import PHYSICALITY_HAL, PHYSICALITY_BAKU
from hal.hal_opponent import CanonicalHal
from hal.train import load_checkpoint
from hal.evaluate import set_nn_evaluator
from environment.opponents.factory import create_scripted_opponent

BAKU_OPPONENTS = [
    "baku_teacher",
    "baku_route",
    "baku_leap",
    "baku_preserve",
    "baku_resilience",
    "random",
    "safe",
]

NUM_GAMES = 50
SEARCH_DEPTH = 2
MAX_TURNS = 30


def play_game(hal_ai, baku_bot, seed, verbose=False):
    hal = Player(name="Hal", physicality=PHYSICALITY_HAL)
    baku = Player(name="Baku", physicality=PHYSICALITY_BAKU)
    ref = Referee()
    game = Game(player1=hal, player2=baku, referee=ref, first_dropper=hal)
    game.seed(seed)
    game.game_clock = 720.0

    hal_ai.reset()
    if hasattr(baku_bot, 'reset'):
        baku_bot.reset()

    turn = 0
    while not game.game_over and turn < MAX_TURNS:
        dropper, checker = game.get_roles_for_half(game.current_half)
        td = game.get_turn_duration()

        if dropper.name.lower() == "hal":
            hal_action = hal_ai.choose_action(game, "dropper", td)
            baku_action = baku_bot.choose_action(game, "checker", td)
            record = game.play_half_round(hal_action, baku_action)
        else:
            baku_action = baku_bot.choose_action(game, "dropper", td)
            hal_action = hal_ai.choose_action(game, "checker", td)
            record = game.play_half_round(baku_action, hal_action)

        if verbose:
            role = "D" if dropper.name.lower() == "hal" else "C"
            print(f"  R{record.round_num}H{record.half} Hal={role} "
                  f"hal_action={hal_action:2d} baku_action={baku_action:2d} "
                  f"result={record.result.name} "
                  f"hal_cyl={game.player1.cylinder:.0f} baku_cyl={game.player2.cylinder:.0f}")

        turn += 1

    if game.game_over:
        return game.winner.name.lower() == "hal"
    return None  # draw / timeout


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--opponent", default=None, help="Single opponent to test")
    parser.add_argument("--games", type=int, default=NUM_GAMES)
    parser.add_argument("--depth", type=int, default=SEARCH_DEPTH)
    parser.add_argument("--checkpoint", default=None, help="Path to value net checkpoint (.pt)")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    if args.checkpoint:
        net = load_checkpoint(args.checkpoint)
        set_nn_evaluator(net)
        print(f"loaded checkpoint: {args.checkpoint}")
    else:
        set_nn_evaluator(None)
        print("using handcrafted eval")

    opponents = [args.opponent] if args.opponent else BAKU_OPPONENTS
    hal_ai = CanonicalHal(seed=0, depth=args.depth)

    for opp_name in opponents:
        baku_bot = create_scripted_opponent(opp_name)
        if baku_bot is None:
            continue

        wins = 0
        losses = 0
        draws = 0

        for i in range(args.games):
            if args.verbose:
                print(f"\n--- {opp_name} game {i+1} ---")
            result = play_game(hal_ai, baku_bot, seed=i, verbose=args.verbose)
            if result is True:
                wins += 1
            elif result is False:
                losses += 1
            else:
                draws += 1

        total = wins + losses + draws
        wr = wins / total * 100 if total > 0 else 0
        print(f"{opp_name:20s}  wins={wins}  losses={losses}  draws={draws}  "
              f"winrate={wr:.1f}%  ({args.games} games, depth={args.depth})")


if __name__ == "__main__":
    main()
