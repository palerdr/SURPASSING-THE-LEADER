"""Self-play training loop.

Alternates training Hal and Baku agents against each other's checkpoints.
Each generation, the opponent is sampled from a frozen pool of past versions.

Usage:
    python training/train_self_play.py --generations 10 --timesteps 100000
"""

import sys
import os
import random
import argparse
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sb3_contrib import MaskablePPO
from environment.dth_env import DTHEnv
from environment.opponents.random_bot import RandomBot
from environment.opponents.model_opponent import ModelOpponent
from training.curriculum import CURRICULA, make_curriculum_sampler


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Self-play training")
    parser.add_argument("--generations", type=int, default=10)
    parser.add_argument("--timesteps", type=int, default=100_000,
                        help="Timesteps per generation")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--curriculum", choices=sorted(CURRICULA.keys()), default="mixed")
    parser.add_argument("--max-steps", type=int, default=None,
                        help="Optional emergency episode cap. Disabled by default.")
    return parser


def format_max_steps(max_steps: int | None) -> str:
    return "disabled" if max_steps is None else str(max_steps)


def self_play_loop(
    generations: int,
    timesteps_per_gen: int,
    seed: int | None,
    curriculum: str,
    max_steps: int | None,
):
    """Run the self-play training loop.

    Protocol:
    1. Bootstrap: Train Hal vs RandomBot (generation 0).
    2. For each generation:
       a. Pick which role to train this generation (alternates hal/baku).
       b. Sample an opponent from the frozen pool for the OTHER role.
       c. Train for timesteps_per_gen steps.
       d. Save checkpoint and add to the pool.
    """
    save_dir = Path("models/self_play")
    save_dir.mkdir(parents=True, exist_ok=True)
    scenario_sampler = make_curriculum_sampler(curriculum, seed=seed)

    # Frozen opponent pools — lists of checkpoint paths
    hal_pool: list[str] = []
    baku_pool: list[str] = []

    # ── Generation 0: Bootstrap both agents vs RandomBot ──
    print("=== Generation 0: Bootstrap ===")
    print(
        "Self-play config: "
        f"curriculum={curriculum} max_steps={format_max_steps(max_steps)} "
        f"timesteps_per_gen={timesteps_per_gen} seed={seed}"
    )

    # Bootstrap Hal
    print("Training Hal vs RandomBot...")
    hal_env = DTHEnv(
        opponent=RandomBot(),
        agent_role="hal",
        seed=seed,
        scenario_sampler=scenario_sampler,
        max_steps=max_steps,
    )
    hal_model = MaskablePPO("MlpPolicy", hal_env, learning_rate=3e-4,
                            n_steps=2048, batch_size=64, n_epochs=10,
                            gamma=0.99, verbose=1, seed=seed)
    hal_model.learn(total_timesteps=timesteps_per_gen)
    hal_path = str(save_dir / "hal_gen0")
    hal_model.save(hal_path)
    hal_pool.append(hal_path)
    print(f"Saved: {hal_path}")

    # Bootstrap Baku
    print("Training Baku vs RandomBot...")
    baku_env = DTHEnv(
        opponent=RandomBot(),
        agent_role="baku",
        seed=seed,
        scenario_sampler=scenario_sampler,
        max_steps=max_steps,
    )
    baku_model = MaskablePPO("MlpPolicy", baku_env, learning_rate=3e-4,
                             n_steps=2048, batch_size=64, n_epochs=10,
                             gamma=0.99, verbose=1, seed=seed)
    baku_model.learn(total_timesteps=timesteps_per_gen)
    baku_path = str(save_dir / "baku_gen0")
    baku_model.save(baku_path)
    baku_pool.append(baku_path)
    print(f"Saved: {baku_path}")

    # ── Generations 1..N: Alternate training against opponent pool ──
    for gen in range(1, generations + 1):
        train_hal = (gen % 2 == 1)
        role = "hal" if train_hal else "baku"
        opp_role = "baku" if train_hal else "hal"
        opp_pool = baku_pool if train_hal else hal_pool

        # Sample opponent from frozen pool
        opp_path = random.choice(opp_pool)
        opp_bot = ModelOpponent(opp_path, role=opp_role)

        print(f"\n=== Generation {gen}: Train {role} vs {opp_role} (from {Path(opp_path).name}) ===")
        print(
            "Generation config: "
            f"curriculum={curriculum} max_steps={format_max_steps(max_steps)} "
            f"timesteps={timesteps_per_gen}"
        )

        env = DTHEnv(
            opponent=opp_bot,
            agent_role=role,
            seed=seed,
            scenario_sampler=scenario_sampler,
            max_steps=max_steps,
        )

        # Continue training from last checkpoint of this role
        prev_pool = hal_pool if train_hal else baku_pool
        prev_path = prev_pool[-1]
        model = MaskablePPO.load(prev_path, env=env)
        model.learn(total_timesteps=timesteps_per_gen)

        # Save and add to pool
        new_path = str(save_dir / f"{role}_gen{gen}")
        model.save(new_path)
        prev_pool.append(new_path)
        print(f"Saved: {new_path}")

    print("\n=== Self-play complete ===")
    print(f"Hal pool: {[Path(p).name for p in hal_pool]}")
    print(f"Baku pool: {[Path(p).name for p in baku_pool]}")


def main():
    parser = build_parser()
    args = parser.parse_args()

    self_play_loop(args.generations, args.timesteps, args.seed, args.curriculum, args.max_steps)


if __name__ == "__main__":
    main()
