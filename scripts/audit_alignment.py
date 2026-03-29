from __future__ import annotations

import argparse
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sb3_contrib import MaskablePPO

from environment.dth_env import DTHEnv
from environment.observation import compute_lsr_variation
from environment.route_stages import current_route_stage_flags
from scripts.evaluate import make_opponent


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit route alignment drift for opening trajectories.")
    parser.add_argument("--model", required=True, help="Path to trained model")
    parser.add_argument("--role", choices=["hal", "baku"], default="baku")
    parser.add_argument("--opponent", default="safe", help="Bot name: random, safe, leap_safe, bridge_pressure")
    parser.add_argument("--opponent-model", default=None, help="Path to trained opponent model")
    parser.add_argument("--games", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-steps", type=int, default=400)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    opponent = make_opponent(
        args.opponent if not args.opponent_model else args.role,
        args.opponent_model,
    )
    model = MaskablePPO.load(args.model)

    reached_leap_window = 0
    reached_round7 = 0
    reached_round8 = 0
    reached_round9 = 0
    misaligned_records: list[dict] = []

    for game_index in range(args.games):
        env = DTHEnv(
            opponent=opponent,
            agent_role=args.role,
            seed=args.seed + game_index,
        )
        obs, _ = env.reset()
        flags = current_route_stage_flags(env.game)
        reached = dict(flags)
        leap_snapshot = None

        for _step in range(args.max_steps):
            mask = env.action_masks()
            action, _ = model.predict(obs, action_masks=mask, deterministic=True)
            obs, reward, terminated, truncated, _info = env.step(int(action))
            flags = current_route_stage_flags(env.game)
            for key, value in flags.items():
                reached[key] = reached.get(key, False) or value

            if leap_snapshot is None and flags["leap_window"]:
                dropper, checker = env.game.get_roles_for_half(env.game.current_half)
                agent_turn_role = "dropper" if env.agent is dropper else "checker"
                leap_snapshot = {
                    "seed": args.seed + game_index,
                    "round_num": env.game.round_num,
                    "current_half": env.game.current_half,
                    "game_clock": env.game.game_clock,
                    "variation": compute_lsr_variation(env.game),
                    "agent_role_at_leap": agent_turn_role,
                    "agent_cylinder": env.agent.cylinder,
                    "opp_cylinder": env.opp_player.cylinder,
                    "agent_ttd": env.agent.ttd,
                    "opp_ttd": env.opp_player.ttd,
                    "referee_cprs": env.game.referee.cprs_performed,
                }

            if terminated or truncated:
                break

        reached_leap_window += int(reached["leap_window"])
        reached_round7 += int(reached["round7_pressure"])
        reached_round8 += int(reached["round8_bridge"])
        reached_round9 += int(reached["round9_pre_leap"])

        if reached["leap_window"] and not reached["round9_pre_leap"] and leap_snapshot is not None:
            misaligned_records.append(leap_snapshot)

    signature_counts = Counter(
        (
            record["round_num"],
            record["current_half"],
            record["variation"],
            record["agent_role_at_leap"],
        )
        for record in misaligned_records
    )

    print(
        "Alignment audit config: "
        f"model={args.model} role={args.role} opponent={args.opponent_model or args.opponent} "
        f"games={args.games} max_steps={args.max_steps}"
    )
    print(f"opening->round7 reach: {reached_round7}/{args.games}")
    print(f"opening->round8 reach: {reached_round8}/{args.games}")
    print(f"opening->round9_pre_leap reach: {reached_round9}/{args.games}")
    print(f"opening->leap-window reach: {reached_leap_window}/{args.games}")
    print(f"leap-window without round9_pre_leap: {len(misaligned_records)}/{args.games}")

    print("\nMisalignment signatures:")
    for signature, count in signature_counts.most_common(10):
        round_num, current_half, variation, agent_role_at_leap = signature
        print(
            f"  round={round_num} half={current_half} variation={variation} "
            f"agent_role={agent_role_at_leap} x {count}"
        )

    if misaligned_records:
        mean_agent_cyl = sum(record["agent_cylinder"] for record in misaligned_records) / len(misaligned_records)
        mean_opp_cyl = sum(record["opp_cylinder"] for record in misaligned_records) / len(misaligned_records)
        mean_agent_ttd = sum(record["agent_ttd"] for record in misaligned_records) / len(misaligned_records)
        mean_opp_ttd = sum(record["opp_ttd"] for record in misaligned_records) / len(misaligned_records)
        mean_cprs = sum(record["referee_cprs"] for record in misaligned_records) / len(misaligned_records)
        print("\nAverage leap-window misalignment state:")
        print(
            f"  agent_cylinder={mean_agent_cyl:.1f} opp_cylinder={mean_opp_cyl:.1f} "
            f"agent_ttd={mean_agent_ttd:.1f} opp_ttd={mean_opp_ttd:.1f} "
            f"referee_cprs={mean_cprs:.1f}"
        )


if __name__ == "__main__":
    main()
