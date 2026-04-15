#!/usr/bin/env python3
"""AlphaZero-style training loop: generate → train → reload → repeat."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hal.value_net import ValueNet, FEATURE_DIM
from hal.self_play import generate_dataset
from hal.train import train_value_net, save_checkpoint, load_checkpoint
from hal.evaluate import set_nn_evaluator

OPPONENTS = ["baku_teacher", "baku_leap", "baku_route", "safe", "random"]
CHECKPOINT_DIR = "models/checkpoints"


def run_iteration(iteration, net, replay_buffer, games_per_iter, epochs, depth, base_seed):
    print(f"\n{'='*60}")
    print(f"ITERATION {iteration}")
    print(f"{'='*60}")

    if net is not None and iteration > 0:
        set_nn_evaluator(net)
        print(f"  Using NN eval from iteration {iteration - 1}")
    else:
        set_nn_evaluator(None)
        print(f"  Using handcrafted eval (iteration 0)")

    print(f"  Generating {games_per_iter} games at depth={depth}...")
    new_data = generate_dataset(
        n_games=games_per_iter,
        opponent_names=OPPONENTS,
        search_depth=depth,
        base_seed=base_seed + iteration * games_per_iter,
    )
    replay_buffer.extend(new_data)
    wins = sum(1 for e in new_data if e.outcome > 0)
    print(f"  New data: {len(new_data)} experiences ({wins} win, {len(new_data)-wins} loss)")
    print(f"  Replay buffer: {len(replay_buffer)} total experiences")

    if net is None:
        net = ValueNet(FEATURE_DIM)

    print(f"  Training {epochs} epochs on full buffer...")
    history = train_value_net(net, replay_buffer, epochs=epochs, batch_size=256, lr=1e-3)
    print(f"  Loss: {history[0]:.4f} → {history[-1]:.4f}")

    path = os.path.join(CHECKPOINT_DIR, f"hal_value_net_iter{iteration}.pt")
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    save_checkpoint(net, path)
    print(f"  Saved: {path}")

    return net


def evaluate_checkpoint(net, depth=1, n_games=20):
    from hal.hal_opponent import CanonicalHal
    from environment.opponents.factory import create_scripted_opponent
    from src.Player import Player
    from src.Referee import Referee
    from src.Game import Game
    from src.Constants import PHYSICALITY_HAL, PHYSICALITY_BAKU

    set_nn_evaluator(net)
    results = {}

    for opp_name in ["random", "safe", "baku_teacher", "baku_leap"]:
        wins = 0
        for i in range(n_games):
            hal = Player(name="Hal", physicality=PHYSICALITY_HAL)
            baku = Player(name="Baku", physicality=PHYSICALITY_BAKU)
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
        results[opp_name] = wins / n_games
        print(f"    vs {opp_name:20s}: {wins}/{n_games} = {wins/n_games*100:.0f}%")

    return results


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--games", type=int, default=500)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--depth", type=int, default=1)
    parser.add_argument("--eval-games", type=int, default=20)
    parser.add_argument("--resume", type=str, default=None)
    args = parser.parse_args()

    net = None
    replay_buffer = []
    if args.resume:
        net = load_checkpoint(args.resume)
        print(f"Resumed from {args.resume}")

    for i in range(args.iterations):
        net = run_iteration(i, net, replay_buffer, args.games, args.epochs, args.depth, base_seed=0)

        print(f"\n  Evaluating iteration {i}:")
        evaluate_checkpoint(net, depth=args.depth, n_games=args.eval_games)

    final_path = os.path.join(CHECKPOINT_DIR, "hal_value_net.pt")
    save_checkpoint(net, final_path)
    print(f"\nFinal checkpoint: {final_path}")


if __name__ == "__main__":
    main()
