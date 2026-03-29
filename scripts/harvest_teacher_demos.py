from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from environment.opponents.factory import create_model_opponent, create_scripted_opponent, scripted_opponent_names
from training.curriculum import SCENARIOS
from training.teacher_demos import rollout_teacher_episode, save_teacher_demo_file


STAGE_ORDER = {
    None: -1,
    "round7_pressure": 0,
    "round8_bridge": 1,
    "round9_pre_leap": 2,
    "leap_window": 3,
    "leap_turn": 4,
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Harvest direct teacher-policy demonstration samples.")
    parser.add_argument("--teacher", choices=scripted_opponent_names(), required=True)
    parser.add_argument("--role", choices=["hal", "baku"], required=True)
    parser.add_argument("--opponent", choices=scripted_opponent_names(), default="bridge_pressure")
    parser.add_argument("--opponent-model", default=None)
    parser.add_argument("--scenario", choices=["opening", *sorted(SCENARIOS.keys())], default="opening")
    parser.add_argument("--games", type=int, default=64)
    parser.add_argument("--max-steps", type=int, default=120)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--target-stage", choices=["round7_pressure", "round8_bridge", "round9_pre_leap", "leap_window", "leap_turn"], default=None)
    parser.add_argument("--max-samples", type=int, default=0, help="Optional cap; 0 means keep all samples.")
    parser.add_argument("--output", required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    teacher = create_scripted_opponent(args.teacher)
    opponent = create_model_opponent(args.opponent_model, agent_role=args.role) if args.opponent_model else create_scripted_opponent(args.opponent)

    harvested = []
    kept_games = 0
    for game_index in range(args.games):
        samples, reached_stage, _won = rollout_teacher_episode(
            teacher=teacher,
            teacher_name=args.teacher,
            opponent=opponent,
            opponent_name=None if args.opponent_model else args.opponent,
            opponent_model_path=args.opponent_model,
            agent_role=args.role,
            seed=args.seed + game_index,
            game_index=game_index,
            scenario_name=args.scenario,
            max_steps=args.max_steps,
        )
        if args.target_stage is not None and STAGE_ORDER[reached_stage] < STAGE_ORDER[args.target_stage]:
            continue
        harvested.extend(samples)
        kept_games += 1
        if args.max_samples > 0 and len(harvested) >= args.max_samples:
            harvested = harvested[: args.max_samples]
            break

    save_teacher_demo_file(args.output, harvested)
    print(
        "Teacher demos harvested: "
        f"samples={len(harvested)} kept_games={kept_games}/{args.games} teacher={args.teacher} "
        f"role={args.role} opponent={args.opponent_model or args.opponent} scenario={args.scenario} output={args.output}"
    )


if __name__ == "__main__":
    main()
