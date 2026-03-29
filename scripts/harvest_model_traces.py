"""Harvest replayable bridge traces from a trained model.

This is used to capture novel successful opening trajectories from stable
retained checkpoints so they can be folded back into BC warm starts.
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sb3_contrib import MaskablePPO

from environment.dth_env import DTHEnv
from environment.opponents.factory import create_model_opponent, create_scripted_opponent, scripted_opponent_names
from environment.route_stages import current_route_stage_flags
from training.bridge_traces import BridgeTraceSpec, save_trace_file


STAGE_ORDER = {
    None: -1,
    "round7_pressure": 0,
    "round8_bridge": 1,
    "round9_pre_leap": 2,
    "leap_window": 3,
    "leap_turn": 4,
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Harvest successful model rollouts as bridge traces.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--role", choices=["hal", "baku"], default="baku")
    parser.add_argument("--opponent", choices=scripted_opponent_names(), default="bridge_pressure")
    parser.add_argument("--opponent-model", default=None)
    parser.add_argument("--games", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-steps", type=int, default=80)
    parser.add_argument("--target-stage", choices=["round7_pressure", "round8_bridge", "round9_pre_leap", "leap_window", "leap_turn"], default="round7_pressure")
    parser.add_argument("--max-traces", type=int, default=64)
    parser.add_argument("--stochastic", action="store_true",
                        help="Use stochastic policy rollout instead of deterministic.")
    parser.add_argument("--output", required=True)
    return parser


def highest_stage(flags: dict[str, bool]) -> str | None:
    for stage_name in ("leap_turn", "leap_window", "round9_pre_leap", "round8_bridge", "round7_pressure"):
        if flags.get(stage_name, False):
            return stage_name
    return None


def main() -> None:
    args = build_parser().parse_args()
    model = MaskablePPO.load(args.model)
    opponent = create_model_opponent(args.opponent_model, agent_role=args.role) if args.opponent_model else create_scripted_opponent(args.opponent)
    assert opponent is not None

    harvested: list[BridgeTraceSpec] = []
    seen: set[tuple[int, tuple[int, ...]]] = set()
    for game_index in range(args.games):
        env = DTHEnv(opponent=opponent, agent_role=args.role, seed=args.seed + game_index)
        obs, _info = env.reset()
        actions: list[int] = []
        reached_stage = highest_stage(current_route_stage_flags(env.game))

        for _step in range(args.max_steps):
            action, _ = model.predict(obs, action_masks=env.action_masks(), deterministic=not args.stochastic)
            action_second = int(action) + 1
            actions.append(action_second)
            obs, _reward, terminated, truncated, _info = env.step(int(action))
            reached_stage = highest_stage(current_route_stage_flags(env.game)) or reached_stage
            if terminated or truncated:
                break

        if STAGE_ORDER[reached_stage] < STAGE_ORDER[args.target_stage]:
            continue
        key = (args.seed + game_index, tuple(actions))
        if key in seen:
            continue
        seen.add(key)
        harvested.append(
            BridgeTraceSpec(
                name=f"model_{os.path.basename(args.model).replace('.zip','')}_{args.seed + game_index}",
                agent_role=args.role,
                opponent_name=None if args.opponent_model else args.opponent,
                opponent_model_path=args.opponent_model,
                scenario_name="opening",
                actions=tuple(actions),
                seed=args.seed + game_index,
            )
        )
        if len(harvested) >= args.max_traces:
            break

    save_trace_file(args.output, tuple(harvested))
    print(
        "Model trace harvest complete: "
        f"model={args.model} opponent={args.opponent_model or args.opponent} target={args.target_stage} "
        f"traces={len(harvested)} output={args.output}"
    )


if __name__ == "__main__":
    main()
