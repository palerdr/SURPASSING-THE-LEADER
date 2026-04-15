#!/usr/bin/env python3
"""Deep AlphaZero training with best-checkpoint selection."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import src.Constants as Constants
from hal.value_net import ValueNet, FEATURE_DIM
from hal.self_play import generate_dataset
from hal.train import train_value_net, save_checkpoint, load_checkpoint
from hal.evaluate import set_nn_evaluator
from hal.hal_opponent import CanonicalHal
from environment.opponents.factory import create_scripted_opponent
from src.Player import Player
from src.Referee import Referee
from src.Game import Game

OPPONENTS = ["baku_teacher", "baku_leap", "baku_route", "baku_preserve", "safe", "random"]
EVAL_OPPONENTS = ["baku_teacher", "baku_leap"]
CHECKPOINT_DIR = "models/checkpoints"


def evaluate_net(net, n_games=50, depth=1):
    set_nn_evaluator(net)
    total_wins = 0
    total_games = 0
    for opp_name in EVAL_OPPONENTS:
        wins = 0
        for i in range(n_games):
            hal = Player(name="Hal", physicality=Constants.PHYSICALITY_HAL)
            baku = Player(name="Baku", physicality=Constants.PHYSICALITY_BAKU)
            game = Game(player1=hal, player2=baku, referee=Referee(), first_dropper=hal)
            game.seed(i + 10000)
            game.game_clock = 720.0
            hal_ai = CanonicalHal(seed=i, depth=depth)
            opp = create_scripted_opponent(opp_name, seed=i)
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
        total_wins += wins
        total_games += n_games
        print(f"    vs {opp_name:20s}: {wins}/{n_games} = {wins/n_games*100:.0f}%")
    return total_wins / total_games


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--games", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--depth", type=int, default=1)
    parser.add_argument("--eval-games", type=int, default=50)
    parser.add_argument("--baku-phys", type=float, default=None)
    parser.add_argument("--resume", type=str, default=None)
    args = parser.parse_args()

    if args.baku_phys is not None:
        Constants.PHYSICALITY_BAKU = args.baku_phys
        print(f"Baku physicality: {args.baku_phys}")

    net = None
    replay_buffer = []
    best_wr = 0.0
    best_iter = -1

    if args.resume:
        net = load_checkpoint(args.resume)
        print(f"Resumed from {args.resume}")

    for i in range(args.iterations):
        print(f"\n{'='*60}")
        print(f"ITERATION {i}")
        print(f"{'='*60}")

        if net is not None and i > 0:
            set_nn_evaluator(net)
        else:
            set_nn_evaluator(None)

        print(f"  Generating {args.games} games...")
        new_data = generate_dataset(
            n_games=args.games,
            opponent_names=OPPONENTS,
            search_depth=args.depth,
            base_seed=i * args.games,
        )
        replay_buffer.extend(new_data)
        wins = sum(1 for e in new_data if e.outcome > 0)
        print(f"  New: {len(new_data)} exp ({wins}W/{len(new_data)-wins}L)")
        print(f"  Buffer: {len(replay_buffer)} total")

        if net is None:
            net = ValueNet(FEATURE_DIM)

        print(f"  Training {args.epochs} epochs (lr={args.lr})...")
        history = train_value_net(net, replay_buffer, epochs=args.epochs,
                                  batch_size=256, lr=args.lr)
        print(f"  Loss: {history[0]:.4f} → {history[-1]:.4f}")

        path = os.path.join(CHECKPOINT_DIR, f"hal_deep_iter{i}.pt")
        os.makedirs(CHECKPOINT_DIR, exist_ok=True)
        save_checkpoint(net, path)

        print(f"  Evaluating:")
        wr = evaluate_net(net, n_games=args.eval_games, depth=args.depth)
        print(f"  Combined winrate: {wr:.0%}")

        if wr > best_wr:
            best_wr = wr
            best_iter = i
            save_checkpoint(net, os.path.join(CHECKPOINT_DIR, "hal_deep_best.pt"))
            print(f"  *** New best! (iter {i}, {wr:.0%})")

    print(f"\nBest iteration: {best_iter} ({best_wr:.0%})")
    print(f"Best checkpoint: models/checkpoints/hal_deep_best.pt")


if __name__ == "__main__":
    main()
