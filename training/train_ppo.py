"""Train a Hal or Baku agent using MaskablePPO.

Usage:
    python training/train_ppo.py --role hal --opponent random --timesteps 100000
    python training/train_ppo.py --role baku --opponent safe --timesteps 200000
"""

import sys
import os
import argparse
from pathlib import Path

# Project root on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sb3_contrib import MaskablePPO
from environment.dth_env import DTHEnv
from environment.opponents.factory import (
    build_opponent_league,
    create_scripted_opponent,
    opponent_config_label,
    opponent_role_for_agent,
    parse_weighted_model_spec,
    scripted_opponent_names,
)
from environment.reward import ROUTE_SHAPING_PRESETS
from training.behavior_clone import run_behavior_cloning
from training.bridge_traces import BRIDGE_TRACE_SETS
from training.curriculum import CURRICULA, make_curriculum_sampler


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train DTH agent")
    parser.add_argument("--role", choices=["hal", "baku"], default="hal")
    parser.add_argument("--opponent", choices=scripted_opponent_names(), default="random",
                        help="Opponent bot or teacher")
    parser.add_argument("--opponent-weight", type=float, default=1.0,
                        help="League weight for the scripted --opponent entry when mixing in learned opponents.")
    parser.add_argument("--opponent-model", action="append", default=[],
                        help="Optional learned opponent checkpoint path, with optional :weight suffix. Repeat to form a league.")
    parser.add_argument("--init-model", default=None,
                        help="Optional checkpoint to continue training from.")
    parser.add_argument("--timesteps", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--shaping", action="store_true",
                         help="Enable sparse route-progression shaping on top of terminal reward.")
    parser.add_argument("--shaping-preset", choices=sorted(ROUTE_SHAPING_PRESETS.keys()), default="light",
                        help="Route shaping preset. Use bridge to make stage milestones dominate shortcut wins.")
    parser.add_argument("--curriculum", choices=sorted(CURRICULA.keys()), default="none")
    parser.add_argument("--max-steps", type=int, default=None,
                        help="Optional emergency episode cap. Disabled by default.")
    parser.add_argument("--bridge-trace-set", choices=sorted(BRIDGE_TRACE_SETS.keys()), default=None,
                        help="Optional exact bridge trace set for a small behavior-cloning warm start.")
    parser.add_argument("--bridge-trace-file", default=None,
                        help="Optional JSON trace file to add to the behavior-cloning warm start.")
    parser.add_argument("--teacher-demo-file", action="append", default=[],
                        help="Optional serialized teacher-demo dataset to add to the behavior-cloning warm start.")
    parser.add_argument("--bc-epochs", type=int, default=0,
                        help="Number of behavior-cloning epochs to run before PPO.")
    parser.add_argument("--bc-batch-size", type=int, default=32,
                        help="Mini-batch size for behavior-cloning warm start.")
    parser.add_argument("--bc-learning-rate", type=float, default=1e-4,
                        help="Learning rate for behavior-cloning warm start.")
    parser.add_argument("--bc-entropy-coeff", type=float, default=0.0,
                        help="Entropy regularization coefficient for BC (higher = more diverse).")
    parser.add_argument("--bc-family-balanced", action="store_true",
                        help="Weight BC samples so each opening family contributes equally.")
    parser.add_argument("--bc-quality-scheme", choices=["stage_linear", "stage_exp", "stage_extreme"], default=None,
                        help="Quality-weighted BC: weight samples by route stage reached.")
    parser.add_argument("--skip-ppo", action="store_true",
                        help="Run warm start only and skip PPO updates.")
    parser.add_argument("--run-tag", default=None,
                        help="Optional suffix to keep checkpoint names distinct across matrix runs.")
    return parser


def make_opponent(name: str):
    return create_scripted_opponent(name)


def format_max_steps(max_steps: int | None) -> str:
    return "disabled" if max_steps is None else str(max_steps)


def parse_weighted_model_spec(spec: str) -> tuple[str, float]:
    model_path = spec
    weight = 1.0
    if ":" in spec:
        maybe_path, maybe_weight = spec.rsplit(":", 1)
        try:
            weight = float(maybe_weight)
            model_path = maybe_path
        except ValueError:
            model_path = spec

    if weight <= 0:
        raise ValueError(f"Opponent model weight must be > 0, got {weight} for {spec}")
    return model_path, weight


def opponent_role_for_agent(agent_role: str) -> str:
    return "baku" if agent_role == "hal" else "hal"


def build_training_opponent(
    *,
    agent_role: str,
    opponent_name: str,
    opponent_weight: float,
    opponent_model_specs: list[str],
    seed: int | None,
):
    return build_opponent_league(
        agent_role=agent_role,
        opponent_name=opponent_name,
        opponent_weight=opponent_weight,
        opponent_model_specs=opponent_model_specs,
        seed=seed,
    )


def build_or_load_model(env: DTHEnv, init_model: str | None, seed: int | None):
    if init_model:
        return MaskablePPO.load(init_model, env=env)

    return MaskablePPO(
        "MlpPolicy",
        env,
        learning_rate = 3e-4,
        n_steps = 2048,
        batch_size = 64,
        n_epochs = 10,
        gamma = 0.99,
        verbose = 1,
        seed = seed
        )

def main():
    parser = build_parser()
    args = parser.parse_args()

    #build
    opponent = build_training_opponent(
        agent_role=args.role,
        opponent_name=args.opponent,
        opponent_weight=args.opponent_weight,
        opponent_model_specs=args.opponent_model,
        seed=args.seed,
    )
    scenario_sampler = make_curriculum_sampler(args.curriculum, seed=args.seed)
    env = DTHEnv(opponent=opponent, agent_role=args.role, seed=args.seed,
                 use_shaping=args.shaping, shaping_preset=args.shaping_preset,
                 scenario_sampler=scenario_sampler,
                 max_steps=args.max_steps)

    model = build_or_load_model(env, args.init_model, args.seed)

    if (args.bridge_trace_set is not None or args.bridge_trace_file is not None or args.teacher_demo_file) and args.bc_epochs > 0:
        num_samples = run_behavior_cloning(
            model,
            trace_set_name=args.bridge_trace_set,
            trace_file=args.bridge_trace_file,
            teacher_demo_files=args.teacher_demo_file,
            opponent_factory=make_opponent,
            epochs=args.bc_epochs,
            batch_size=args.bc_batch_size,
            learning_rate=args.bc_learning_rate,
            entropy_coeff=args.bc_entropy_coeff,
            family_balanced=args.bc_family_balanced,
            quality_scheme=args.bc_quality_scheme,
        )
        print(
            "Behavior cloning warm start: "
            f"trace_set={args.bridge_trace_set or 'none'} "
            f"trace_file={args.bridge_trace_file or 'none'} "
            f"teacher_demo_files={len(args.teacher_demo_file)} "
            f"samples={num_samples} "
            f"epochs={args.bc_epochs} batch_size={args.bc_batch_size} "
            f"lr={args.bc_learning_rate}"
        )

    #train
    print(
        "Training config: "
        f"role={args.role} opponent={opponent_config_label(args.opponent, args.opponent_model, args.opponent_weight)} "
        f"shaping={'on' if args.shaping else 'off'} shaping_preset={args.shaping_preset} "
        f"max_steps={format_max_steps(args.max_steps)} "
        f"curriculum={args.curriculum} init_model={args.init_model or 'none'} "
        f"timesteps={args.timesteps} seed={args.seed} skip_ppo={args.skip_ppo}"
    )
    if not args.skip_ppo:
        model.learn(total_timesteps=args.timesteps)

    # ── Save ──
    save_dir = Path("models/checkpoints")
    save_dir.mkdir(parents=True, exist_ok=True)
    opponent_label = opponent_config_label(args.opponent, args.opponent_model, args.opponent_weight)
    suffix = "bc_only" if args.skip_ppo else str(args.timesteps)
    run_suffix = f"_{args.run_tag}" if args.run_tag else ""
    save_path = save_dir / f"{args.role}_vs_{opponent_label}_{args.curriculum}_{suffix}{run_suffix}"
    model.save(str(save_path))
    print(f"Saved to {save_path}")


if __name__ == "__main__":
    main()
