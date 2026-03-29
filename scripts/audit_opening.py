from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter

import numpy as np
from sb3_contrib import MaskablePPO

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from environment.dth_env import DTHEnv
from environment.observation import compute_lsr_variation
from environment.route_stages import current_route_stage_flags
from scripts.audit_utils import masked_entropy, summarize_turn_support, top_action_probs
from scripts.evaluate import make_opponent


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit opening trajectories and action support.")
    parser.add_argument("--model", required=True, help="Path to trained model")
    parser.add_argument("--role", choices=["hal", "baku"], default="baku")
    parser.add_argument("--opponent", default="safe", help="Bot name: random, safe, leap_safe")
    parser.add_argument("--opponent-model", default=None, help="Path to trained opponent model")
    parser.add_argument("--games", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-turns", type=int, default=20)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--prefix-turns", type=int, default=8)
    parser.add_argument("--stochastic", action="store_true", help="Sample from the policy instead of deterministic actions.")
    parser.add_argument("--output-jsonl", default=None, help="Optional path for full per-turn audit rows.")
    return parser


def policy_probe(model: MaskablePPO, obs: np.ndarray, mask: np.ndarray, top_k: int) -> dict:
    obs_tensor, _ = model.policy.obs_to_tensor(obs)
    distribution = model.policy.get_distribution(obs_tensor, action_masks=mask)
    probs = distribution.distribution.probs.detach().cpu().numpy()[0]
    values = model.policy.predict_values(obs_tensor).detach().cpu().numpy().reshape(-1)

    return {
        "probs": probs,
        "value_estimate": float(values[0]),
        "entropy": masked_entropy(probs, mask),
        "top_actions": top_action_probs(probs, mask, top_k=top_k),
    }


def first_route_stage(flags: dict[str, bool]) -> str | None:
    for stage_name in ("round7_pressure", "round8_bridge", "round9_pre_leap", "leap_window", "leap_turn"):
        if flags.get(stage_name, False):
            return stage_name
    return None


def write_jsonl(path: str, rows: list[dict]) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    opponent = make_opponent(
        args.opponent if not args.opponent_model else args.role,
        args.opponent_model,
    )
    model = MaskablePPO.load(args.model)

    turn_rows: list[dict] = []
    episode_summaries: list[dict] = []

    for game_index in range(args.games):
        env = DTHEnv(
            opponent=opponent,
            agent_role=args.role,
            seed=args.seed + game_index,
        )
        obs, _info = env.reset()
        chosen_prefix: list[int] = []
        reached_stage = None
        won = False

        for turn_index in range(args.max_turns):
            pre_flags = current_route_stage_flags(env.game)
            if reached_stage is None:
                reached_stage = first_route_stage(pre_flags)

            mask = env.action_masks()
            dropper, checker = env.game.get_roles_for_half(env.game.current_half)
            agent_turn_role = "dropper" if env.agent is dropper else "checker"
            probe = policy_probe(model, obs, mask, top_k=args.top_k)
            action, _ = model.predict(
                obs,
                action_masks=mask,
                deterministic=not args.stochastic,
            )
            action_second = int(action) + 1
            chosen_prefix.append(action_second)

            pre_my_cylinder = env.agent.cylinder
            pre_opp_cylinder = env.opp_player.cylinder
            pre_my_ttd = env.agent.ttd
            pre_opp_ttd = env.opp_player.ttd
            pre_round = env.game.round_num
            pre_half = env.game.current_half
            pre_clock = env.game.game_clock
            pre_variation = compute_lsr_variation(env.game)

            obs, reward, terminated, truncated, info = env.step(int(action))
            rec = env.game.history[-1]
            post_flags = current_route_stage_flags(env.game)
            if reached_stage is None:
                reached_stage = first_route_stage(post_flags)

            row = {
                "game_index": game_index,
                "seed": args.seed + game_index,
                "turn_index": turn_index,
                "pre_round_num": pre_round,
                "pre_half": pre_half,
                "pre_clock": pre_clock,
                "pre_lsr_variation": pre_variation,
                "pre_route_flags": pre_flags,
                "agent_turn_role": agent_turn_role,
                "action_second": action_second,
                "value_estimate": probe["value_estimate"],
                "entropy": probe["entropy"],
                "top_actions": probe["top_actions"],
                "pre_my_cylinder": pre_my_cylinder,
                "pre_opp_cylinder": pre_opp_cylinder,
                "pre_my_ttd": pre_my_ttd,
                "pre_opp_ttd": pre_opp_ttd,
                "drop_time": rec.drop_time,
                "check_time": rec.check_time,
                "turn_duration": rec.turn_duration,
                "result": rec.result.value,
                "st_gained": rec.st_gained,
                "death_duration": rec.death_duration,
                "post_round_num": env.game.round_num,
                "post_half": env.game.current_half,
                "post_clock": env.game.game_clock,
                "post_route_flags": post_flags,
                "post_my_cylinder": env.agent.cylinder,
                "post_opp_cylinder": env.opp_player.cylinder,
                "post_my_ttd": env.agent.ttd,
                "post_opp_ttd": env.opp_player.ttd,
                "awareness": info["awareness"],
                "terminated": terminated,
                "truncated": truncated,
                "reward": reward,
            }
            turn_rows.append(row)

            if terminated or truncated:
                won = bool(terminated and env.game.winner is env.agent and reward > 0)
                break

        final_flags = current_route_stage_flags(env.game)
        if reached_stage is None:
            reached_stage = first_route_stage(final_flags)

        episode_summaries.append(
            {
                "game_index": game_index,
                "seed": args.seed + game_index,
                "turns_played": sum(1 for row in turn_rows if row["game_index"] == game_index),
                "won": won,
                "reached_stage": reached_stage,
                "prefix": chosen_prefix[: args.prefix_turns],
            }
        )

    if args.output_jsonl:
        write_jsonl(args.output_jsonl, turn_rows)

    turn_support = summarize_turn_support(turn_rows, args.max_turns)
    prefix_counts = Counter(tuple(summary["prefix"]) for summary in episode_summaries)
    stage_counts = Counter(summary["reached_stage"] or "none" for summary in episode_summaries)

    print(
        "Opening audit config: "
        f"model={args.model} role={args.role} opponent={args.opponent_model or args.opponent} "
        f"games={args.games} max_turns={args.max_turns} stochastic={'on' if args.stochastic else 'off'} "
        f"output_jsonl={args.output_jsonl or 'none'}"
    )
    print(f"Games: {args.games}")
    print(f"Wins: {sum(summary['won'] for summary in episode_summaries)}/{args.games}")
    print(f"Reached stages: {dict(stage_counts)}")

    print("\nTurn support:")
    for summary in turn_support:
        print(
            f"  T{summary['turn_index'] + 1}: "
            f"unique={summary['unique_actions']} "
            f"mode={summary['dominant_second']} "
            f"mode_share={100 * summary['dominant_share']:.0f}% "
            f"entropy={summary['mean_entropy']:.3f} "
            f"value={summary['mean_value']:.3f}"
        )

    print("\nCommon prefixes:")
    for prefix, count in prefix_counts.most_common(5):
        print(f"  {list(prefix)} x {count}")

    print("\nSample trajectories:")
    for summary in episode_summaries[:3]:
        print(
            f"  seed={summary['seed']} turns={summary['turns_played']} "
            f"stage={summary['reached_stage']} prefix={summary['prefix']}"
        )


if __name__ == "__main__":
    main()
