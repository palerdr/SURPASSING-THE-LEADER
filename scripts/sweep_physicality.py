#!/usr/bin/env python3
"""Sweep Baku physicality to find the balanced value.

At the balanced physicality, Hal and Baku should win ~50/50 when both
play competently — the leap second becomes the only tie-breaker.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import src.Constants as Constants
from src.Player import Player
from src.Referee import Referee
from src.Game import Game
from hal.hal_opponent import CanonicalHal
from hal.evaluate import set_nn_evaluator
from hal.train import load_checkpoint
from environment.opponents.factory import create_scripted_opponent


def run_games(baku_phys, opponent_name, n_games, depth, use_nn):
    original = Constants.PHYSICALITY_BAKU
    Constants.PHYSICALITY_BAKU = baku_phys

    wins = 0
    for i in range(n_games):
        hal = Player(name="Hal", physicality=Constants.PHYSICALITY_HAL)
        baku = Player(name="Baku", physicality=baku_phys)
        game = Game(player1=hal, player2=baku, referee=Referee(), first_dropper=hal)
        game.seed(i)
        game.game_clock = 720.0

        hal_ai = CanonicalHal(seed=i, depth=depth)
        opp = create_scripted_opponent(opponent_name, seed=i)
        hal_ai.reset()
        opp.reset()

        turn = 0
        while not game.game_over and turn < 30:
            dropper, _ = game.get_roles_for_half(game.current_half)
            td = game.get_turn_duration()
            if dropper.name.lower() == "hal":
                h = hal_ai.choose_action(game, "dropper", td)
                b = opp.choose_action(game, "checker", td)
                game.play_half_round(h, b)
            else:
                b = opp.choose_action(game, "dropper", td)
                h = hal_ai.choose_action(game, "checker", td)
                game.play_half_round(b, h)
            turn += 1
        if game.game_over and game.winner.name.lower() == "hal":
            wins += 1

    Constants.PHYSICALITY_BAKU = original
    return wins / n_games


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--games", type=int, default=50)
    parser.add_argument("--depth", type=int, default=1)
    parser.add_argument("--nn", type=str, default=None)
    parser.add_argument("--opponent", type=str, default="baku_teacher")
    args = parser.parse_args()

    if args.nn:
        net = load_checkpoint(args.nn)
        set_nn_evaluator(net)
        print(f"Using NN eval: {args.nn}")
    else:
        set_nn_evaluator(None)
        print("Using handcrafted eval")

    print(f"Opponent: {args.opponent}, Games: {args.games}, Depth: {args.depth}")
    print(f"{'Physicality':>12s} {'Hal WR':>8s} {'Note':>20s}")
    print("-" * 44)

    for phys in [0.80, 0.85, 0.88, 0.90, 0.92, 0.94, 0.96, 0.98, 1.00]:
        wr = run_games(phys, args.opponent, args.games, args.depth, args.nn)
        note = ""
        if abs(wr - 0.50) < 0.06:
            note = "<-- near balanced"
        elif wr > 0.70:
            note = "Hal dominant"
        elif wr < 0.30:
            note = "Baku dominant"
        print(f"{phys:12.2f} {wr:7.0%} {note:>20s}")


if __name__ == "__main__":
    main()
