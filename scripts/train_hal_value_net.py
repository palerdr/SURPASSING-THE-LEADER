#!/usr/bin/env python3
"""Quick training run for the Hal value network."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hal.value_net import ValueNet, FEATURE_DIM
from hal.self_play import generate_dataset
from hal.train import train_value_net, save_checkpoint

OPPONENTS = ["baku_teacher", "baku_leap", "baku_route", "safe", "random"]

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--games", type=int, default=100)
    parser.add_argument("--depth", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--out", type=str, default="models/checkpoints/hal_value_net.pt")
    args = parser.parse_args()

    print(f"Generating data: {args.games} games, depth={args.depth}")
    dataset = generate_dataset(
        n_games=args.games,
        opponent_names=OPPONENTS,
        search_depth=args.depth,
        base_seed=0,
    )
    print(f"Dataset: {len(dataset)} experiences")

    wins = sum(1 for e in dataset if e.outcome > 0)
    losses = len(dataset) - wins
    print(f"Outcomes: {wins} win-positions, {losses} loss-positions")

    net = ValueNet(FEATURE_DIM)
    print(f"\nTraining: {args.epochs} epochs, batch_size={args.batch_size}, lr={args.lr}")
    history = train_value_net(net, dataset, epochs=args.epochs,
                              batch_size=args.batch_size, lr=args.lr)

    for i, loss in enumerate(history):
        print(f"  epoch {i+1:2d}: loss={loss:.4f}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    save_checkpoint(net, args.out)
    print(f"\nSaved to {args.out}")


if __name__ == "__main__":
    main()
