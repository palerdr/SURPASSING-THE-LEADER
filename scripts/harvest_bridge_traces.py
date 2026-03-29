from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sb3_contrib import MaskablePPO

from environment.dth_env import DTHEnv
from environment.route_stages import current_route_stage_flags
from scripts.evaluate import make_opponent
from training.bridge_traces import BridgeTraceSpec, save_trace_file
from training.curriculum import SCENARIOS, get_scenario


OPENING_SCENARIO = "opening"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Harvest successful live bridge traces from stochastic rollouts.")
    parser.add_argument("--model", required=True, help="Path to trained model")
    parser.add_argument("--role", choices=["hal", "baku"], default="baku")
    parser.add_argument("--opponent", default="bridge_pressure", help="Scripted opponent name")
    parser.add_argument("--opponent-model", default=None, help="Optional frozen model opponent path")
    parser.add_argument("--scenario", choices=[OPENING_SCENARIO, *sorted(SCENARIOS.keys())], required=True)
    parser.add_argument("--target-stage", choices=["round7_pressure", "round8_bridge", "round9_pre_leap"], required=True)
    parser.add_argument("--games", type=int, default=200)
    parser.add_argument("--max-traces", type=int, default=32)
    parser.add_argument("--max-steps", type=int, default=120)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--deterministic", action="store_true", help="Use deterministic actions instead of stochastic rollouts.")
    parser.add_argument("--output", required=True, help="Output JSON path for harvested traces")
    return parser


def main() -> None:
    args = build_parser().parse_args()

    model = MaskablePPO.load(args.model)
    harvested: list[BridgeTraceSpec] = []

    for game_index in range(args.games):
        if len(harvested) >= args.max_traces:
            break

        env = DTHEnv(
            opponent=make_opponent(args.opponent if args.opponent_model is None else args.role, args.opponent_model),
            agent_role=args.role,
            seed=args.seed + game_index,
        )
        reset_options = None if args.scenario == OPENING_SCENARIO else {"scenario": get_scenario(args.scenario)}
        obs, _ = env.reset(options=reset_options)
        assert env.game is not None
        assert env.agent is not None
        actions: list[int] = []

        for _step in range(args.max_steps):
            mask = env.action_masks()
            action, _ = model.predict(obs, action_masks=mask, deterministic=args.deterministic)
            second = int(action) + 1
            actions.append(second)
            obs, _reward, terminated, truncated, _info = env.step(int(action))
            flags = current_route_stage_flags(env.game)

            if flags[args.target_stage] and not env.game.game_over and env.agent.alive:
                harvested.append(
                    BridgeTraceSpec(
                        name=f"{args.scenario}_to_{args.target_stage}_{args.seed + game_index}",
                        agent_role=args.role,
                        opponent_name=None if args.opponent_model is not None else args.opponent,
                        opponent_model_path=args.opponent_model,
                        scenario_name=args.scenario,
                        seed=args.seed + game_index,
                        actions=tuple(actions),
                    )
                )
                break

            if terminated or truncated:
                break

    save_trace_file(args.output, tuple(harvested))
    print(
        "Harvested traces: "
        f"{len(harvested)}/{args.max_traces} saved to {args.output} "
        f"from scenario={args.scenario} target={args.target_stage}"
    )


if __name__ == "__main__":
    main()
