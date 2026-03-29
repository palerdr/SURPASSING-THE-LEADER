from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict

from sb3_contrib import MaskablePPO

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from environment.dth_env import DTHEnv
from environment.route_stages import current_route_stage_flags
from scripts.audit_utils import progression_tuple
from scripts.evaluate import make_opponent


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Counterfactual one-step audit for recurring opening states.")
    parser.add_argument("--model", required=True, help="Path to trained model")
    parser.add_argument("--role", choices=["hal", "baku"], default="baku")
    parser.add_argument("--opponent", default="safe", help="Bot name: random, safe, leap_safe")
    parser.add_argument("--opponent-model", default=None, help="Path to trained opponent model")
    parser.add_argument("--games", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--inspect-turns", type=int, default=8)
    parser.add_argument("--max-steps", type=int, default=400)
    parser.add_argument("--output-json", default=None, help="Optional path for structured counterfactual results.")
    return parser


def rollout_episode_summary(model: MaskablePPO, env: DTHEnv, max_steps: int) -> dict:
    flags = current_route_stage_flags(env.game)
    reached = dict(flags)
    terminated = False
    truncated = False
    reward = 0.0

    for _ in range(max_steps):
        mask = env.action_masks()
        action, _ = model.predict(env._get_obs() if hasattr(env, "_get_obs") else env.unwrapped._get_obs(), action_masks=mask, deterministic=True)  # pragma: no cover
        raise RuntimeError("rollout_episode_summary should not use hidden env APIs")

    return {
        "won": False,
        "terminated": terminated,
        "truncated": truncated,
        "reached_round7_pressure": reached["round7_pressure"],
        "reached_round8_bridge": reached["round8_bridge"],
        "reached_round9_pre_leap": reached["round9_pre_leap"],
        "reached_leap_window": reached["leap_window"],
        "reached_leap_turn": reached["leap_turn"],
        "half_rounds": len(env.game.history),
        "final_reward": reward,
    }


def step_with_model(model: MaskablePPO, env: DTHEnv, obs):
    mask = env.action_masks()
    action, _ = model.predict(obs, action_masks=mask, deterministic=True)
    next_obs, reward, terminated, truncated, info = env.step(int(action))
    return int(action) + 1, next_obs, reward, terminated, truncated, info


def rollout_from_obs(model: MaskablePPO, env: DTHEnv, obs, max_steps: int) -> dict:
    reached = dict(current_route_stage_flags(env.game))
    reward = 0.0
    terminated = False
    truncated = False

    while len(env.game.history) < max_steps:
        _action_second, obs, reward, terminated, truncated, _info = step_with_model(model, env, obs)
        flags = current_route_stage_flags(env.game)
        for key, value in flags.items():
            reached[key] = reached.get(key, False) or value
        if terminated or truncated:
            break

    return {
        "won": bool(terminated and env.game.winner is env.agent and reward > 0),
        "terminated": terminated,
        "truncated": truncated,
        "reached_round7_pressure": reached["round7_pressure"],
        "reached_round8_bridge": reached["round8_bridge"],
        "reached_round9_pre_leap": reached["round9_pre_leap"],
        "reached_leap_window": reached["leap_window"],
        "reached_leap_turn": reached["leap_turn"],
        "half_rounds": len(env.game.history),
        "final_reward": reward,
    }


def aggregate_summaries(summaries: list[dict]) -> dict:
    total = len(summaries)

    def rate(key: str) -> float:
        return sum(bool(summary[key]) for summary in summaries) / total

    return {
        "games": total,
        "won": rate("won"),
        "reached_round7_pressure": rate("reached_round7_pressure"),
        "reached_round8_bridge": rate("reached_round8_bridge"),
        "reached_round9_pre_leap": rate("reached_round9_pre_leap"),
        "reached_leap_window": rate("reached_leap_window"),
        "reached_leap_turn": rate("reached_leap_turn"),
        "average_half_rounds": sum(summary["half_rounds"] for summary in summaries) / total,
        "progression_tuple": progression_tuple({
            "reached_round7_pressure": rate("reached_round7_pressure") > 0,
            "reached_round8_bridge": rate("reached_round8_bridge") > 0,
            "reached_round9_pre_leap": rate("reached_round9_pre_leap") > 0,
            "reached_leap_window": rate("reached_leap_window") > 0,
            "reached_leap_turn": rate("reached_leap_turn") > 0,
            "won": rate("won") > 0,
        }),
    }


def baseline_prefix(model: MaskablePPO, env: DTHEnv, inspect_turns: int) -> tuple[list[int], list[dict]]:
    obs, _ = env.reset()
    actions: list[int] = []
    state_records: list[dict] = []

    for turn_index in range(inspect_turns):
        mask = env.action_masks()
        legal_seconds = [int(idx) + 1 for idx in mask.nonzero()[0]]
        dropper, checker = env.game.get_roles_for_half(env.game.current_half)
        turn_role = "dropper" if env.agent is dropper else "checker"
        action, _ = model.predict(obs, action_masks=mask, deterministic=True)
        action_second = int(action) + 1
        state_records.append(
            {
                "turn_index": turn_index,
                "turn_role": turn_role,
                "chosen_action": action_second,
                "legal_seconds": legal_seconds,
            }
        )
        actions.append(action_second)
        obs, _reward, terminated, truncated, _info = env.step(int(action))
        if terminated or truncated:
            break

    return actions, state_records


def replay_to_turn(
    *,
    model: MaskablePPO,
    opponent_name: str,
    opponent_model: str | None,
    role: str,
    seed: int,
    prefix_actions: list[int],
) -> tuple[DTHEnv, object]:
    opponent = make_opponent(
        opponent_name if not opponent_model else role,
        opponent_model,
    )
    env = DTHEnv(opponent=opponent, agent_role=role, seed=seed)
    obs, _ = env.reset()
    for action_second in prefix_actions:
        obs, _reward, terminated, truncated, _info = env.step(action_second - 1)
        if terminated or truncated:
            raise RuntimeError("Baseline terminated before requested turn during replay")
    return env, obs


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    model = MaskablePPO.load(args.model)

    baseline_by_seed: dict[int, list[int]] = {}
    template_records: list[dict] | None = None
    for game_index in range(args.games):
        seed = args.seed + game_index
        env = DTHEnv(
            opponent=make_opponent(
                args.opponent if not args.opponent_model else args.role,
                args.opponent_model,
            ),
            agent_role=args.role,
            seed=seed,
        )
        prefix_actions, state_records = baseline_prefix(model, env, args.inspect_turns)
        baseline_by_seed[seed] = prefix_actions
        if template_records is None:
            template_records = state_records

    assert template_records is not None

    turn_results: list[dict] = []
    for record in template_records:
        turn_index = record["turn_index"]
        legal_seconds = record["legal_seconds"]
        chosen_action = record["chosen_action"]
        per_action_summaries: dict[int, dict] = {}

        for action_second in legal_seconds:
            summaries = []
            for seed, prefix_actions in baseline_by_seed.items():
                if turn_index >= len(prefix_actions):
                    continue
                env, obs = replay_to_turn(
                    model=model,
                    opponent_name=args.opponent,
                    opponent_model=args.opponent_model,
                    role=args.role,
                    seed=seed,
                    prefix_actions=prefix_actions[:turn_index],
                )
                obs, reward, terminated, truncated, _info = env.step(action_second - 1)
                reached = dict(current_route_stage_flags(env.game))
                if terminated or truncated:
                    summaries.append(
                        {
                            "won": bool(terminated and env.game.winner is env.agent and reward > 0),
                            "terminated": terminated,
                            "truncated": truncated,
                            "reached_round7_pressure": reached["round7_pressure"],
                            "reached_round8_bridge": reached["round8_bridge"],
                            "reached_round9_pre_leap": reached["round9_pre_leap"],
                            "reached_leap_window": reached["leap_window"],
                            "reached_leap_turn": reached["leap_turn"],
                            "half_rounds": len(env.game.history),
                            "final_reward": reward,
                        }
                    )
                else:
                    summaries.append(rollout_from_obs(model, env, obs, max_steps=args.max_steps))

            per_action_summaries[action_second] = aggregate_summaries(summaries)

        chosen_summary = per_action_summaries[chosen_action]
        better_actions = []
        for action_second, summary in per_action_summaries.items():
            if action_second == chosen_action:
                continue
            if summary["progression_tuple"] > chosen_summary["progression_tuple"]:
                better_actions.append((action_second, summary))

        ranked_actions = sorted(
            per_action_summaries.items(),
            key=lambda item: (
                item[1]["progression_tuple"],
                item[1]["average_half_rounds"],
                item[1]["won"],
            ),
            reverse=True,
        )

        turn_results.append(
            {
                "turn_index": turn_index,
                "turn_role": record["turn_role"],
                "chosen_action": chosen_action,
                "chosen_summary": chosen_summary,
                "better_action_count": len(better_actions),
                "better_actions": [
                    {"action_second": action_second, "summary": summary}
                    for action_second, summary in better_actions[:10]
                ],
                "top_actions": [
                    {"action_second": action_second, "summary": summary}
                    for action_second, summary in ranked_actions[:10]
                ],
            }
        )

    if args.output_json:
        with open(args.output_json, "w", encoding="utf-8") as handle:
            json.dump(turn_results, handle, indent=2, sort_keys=True)

    print(
        "Counterfactual audit config: "
        f"model={args.model} role={args.role} opponent={args.opponent_model or args.opponent} "
        f"games={args.games} inspect_turns={args.inspect_turns} max_steps={args.max_steps} "
        f"output_json={args.output_json or 'none'}"
    )
    for result in turn_results:
        chosen = result["chosen_summary"]
        print(
            f"\nT{result['turn_index'] + 1} {result['turn_role']}: "
            f"chosen={result['chosen_action']} "
            f"chosen_progress={chosen['progression_tuple']} "
            f"chosen_round7={100 * chosen['reached_round7_pressure']:.0f}% "
            f"chosen_round8={100 * chosen['reached_round8_bridge']:.0f}% "
            f"chosen_round9={100 * chosen['reached_round9_pre_leap']:.0f}% "
            f"chosen_leap_window={100 * chosen['reached_leap_window']:.0f}% "
            f"better_actions={result['better_action_count']}"
        )
        for option in result["top_actions"][:5]:
            summary = option["summary"]
            print(
                f"  a={option['action_second']}: "
                f"progress={summary['progression_tuple']} "
                f"r7={100 * summary['reached_round7_pressure']:.0f}% "
                f"r8={100 * summary['reached_round8_bridge']:.0f}% "
                f"r9={100 * summary['reached_round9_pre_leap']:.0f}% "
                f"lw={100 * summary['reached_leap_window']:.0f}% "
                f"lt={100 * summary['reached_leap_turn']:.0f}% "
                f"win={100 * summary['won']:.0f}% "
                f"len={summary['average_half_rounds']:.1f}"
            )


if __name__ == "__main__":
    main()
