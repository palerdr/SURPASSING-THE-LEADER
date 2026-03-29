from __future__ import annotations

import argparse
import os
import sys

import numpy as np
from sb3_contrib import MaskablePPO

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from environment.dth_env import DTHEnv
from scripts.audit_opening import policy_probe
from scripts.audit_utils import summarize_values
from scripts.evaluate import make_opponent
from training.curriculum import get_scenario


DEFAULT_SCENARIOS = [
    "opening",
    "round7_pressure",
    "round8_bridge",
    "round9_pre_leap",
    "round9_leap_deduced",
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit critic values across opening and bridge states.")
    parser.add_argument("--model", required=True, help="Path to trained model")
    parser.add_argument("--role", choices=["hal", "baku"], default="baku")
    parser.add_argument("--opponent", default="safe", help="Bot name: random, safe, leap_safe")
    parser.add_argument("--opponent-model", default=None, help="Path to trained opponent model")
    parser.add_argument("--games", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--scenarios", nargs="+", default=DEFAULT_SCENARIOS)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    opponent = make_opponent(
        args.opponent if not args.opponent_model else args.role,
        args.opponent_model,
    )
    model = MaskablePPO.load(args.model)

    print(
        "Value audit config: "
        f"model={args.model} role={args.role} opponent={args.opponent_model or args.opponent} "
        f"games={args.games} scenarios={args.scenarios}"
    )

    for scenario_name in args.scenarios:
        values: list[float] = []
        entropies: list[float] = []
        top_actions: list[int] = []

        for game_index in range(args.games):
            env = DTHEnv(
                opponent=opponent,
                agent_role=args.role,
                seed=args.seed + game_index,
            )
            options = None
            if scenario_name != "opening":
                options = {"scenario": get_scenario(scenario_name)}
            obs, _info = env.reset(options=options)
            mask = env.action_masks()
            probe = policy_probe(model, obs, mask, top_k=args.top_k)
            values.append(probe["value_estimate"])
            entropies.append(probe["entropy"])
            top_actions.append(int(probe["top_actions"][0]["second"]))

        summary = summarize_values(values)
        dominant_action = max(set(top_actions), key=top_actions.count)
        dominant_share = top_actions.count(dominant_action) / len(top_actions)
        print(
            f"  {scenario_name}: "
            f"value_mean={summary['mean']:.3f} "
            f"value_std={summary['std']:.3f} "
            f"value_min={summary['min']:.3f} "
            f"value_max={summary['max']:.3f} "
            f"entropy_mean={float(np.mean(entropies)):.3f} "
            f"mode_action={dominant_action} "
            f"mode_share={100 * dominant_share:.0f}%"
        )


if __name__ == "__main__":
    main()
