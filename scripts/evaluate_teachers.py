from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from environment.dth_env import DTHEnv
from environment.opponents.factory import create_model_opponent, create_scripted_opponent, scripted_opponent_names
from environment.route_stages import current_route_stage_flags, stage_is_eligible_from_start
from scripts.evaluate import GameMetrics, format_max_steps, make_scenario_sampler, print_summary, summarize_games
from training.curriculum import CURRICULA, SCENARIOS


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate scripted teachers from opening or seeded states.")
    parser.add_argument("--teacher", choices=scripted_opponent_names(), required=True)
    parser.add_argument("--role", choices=["hal", "baku"], required=True)
    parser.add_argument("--opponent", choices=scripted_opponent_names(), default="bridge_pressure")
    parser.add_argument("--opponent-model", default=None)
    parser.add_argument("--games", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--scenario", choices=["opening", *sorted(SCENARIOS.keys())], default="opening")
    parser.add_argument("--curriculum", choices=sorted(CURRICULA.keys()), default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    return parser


def play_teacher_game(teacher, env: DTHEnv) -> GameMetrics:
    obs, info = env.reset()
    del obs
    awareness = info["awareness"]
    stage_flags = current_route_stage_flags(env.game)
    reached_round7_pressure = stage_flags["round7_pressure"]
    reached_round8_bridge = stage_flags["round8_bridge"]
    reached_round9_pre_leap = stage_flags["round9_pre_leap"]
    reached_leap_window = stage_flags["leap_window"]
    reached_leap_turn = stage_flags["leap_turn"]
    started_before_round7_pressure = stage_is_eligible_from_start(env.game, "round7_pressure")
    started_before_round8_bridge = stage_is_eligible_from_start(env.game, "round8_bridge")
    started_before_round9_pre_leap = stage_is_eligible_from_start(env.game, "round9_pre_leap")
    started_before_leap_window = stage_is_eligible_from_start(env.game, "leap_window")
    started_before_leap_turn = stage_is_eligible_from_start(env.game, "leap_turn")
    awareness_transitions = 0
    checker_61_count = 0
    checker_actions = 0
    dropper_61_count = 0
    dropper_actions = 0

    while True:
        mask = env.action_masks()
        dropper, checker = env.game.get_roles_for_half(env.game.current_half)
        role = "dropper" if env.agent is dropper else "checker"
        action_second = teacher.choose_action(env.game, role, env.game.get_turn_duration())
        action_index = action_second - 1
        if not mask[action_index]:
            raise ValueError(f"Illegal scripted action second={action_second} role={role}")

        if role == "checker":
            checker_actions += 1
            if action_second == 61:
                checker_61_count += 1
        else:
            dropper_actions += 1
            if action_second == 61:
                dropper_61_count += 1

        _obs, reward, terminated, truncated, info = env.step(action_index)
        rec = env.game.history[-1]
        stage_flags = current_route_stage_flags(env.game)
        reached_round7_pressure = reached_round7_pressure or stage_flags["round7_pressure"]
        reached_round8_bridge = reached_round8_bridge or stage_flags["round8_bridge"]
        reached_round9_pre_leap = reached_round9_pre_leap or stage_flags["round9_pre_leap"]
        reached_leap_window = reached_leap_window or stage_flags["leap_window"]
        reached_leap_turn = reached_leap_turn or stage_flags["leap_turn"] or (rec.turn_duration == 61)
        if info["awareness"] != awareness:
            awareness_transitions += 1
            awareness = info["awareness"]
        if terminated or truncated:
            return GameMetrics(
                won=bool(terminated and env.game.winner is env.agent and reward > 0),
                half_rounds=len(env.game.history),
                reached_round7_pressure=reached_round7_pressure,
                reached_round8_bridge=reached_round8_bridge,
                reached_round9_pre_leap=reached_round9_pre_leap,
                reached_leap_window=reached_leap_window,
                reached_leap_turn=reached_leap_turn,
                started_before_round7_pressure=started_before_round7_pressure,
                started_before_round8_bridge=started_before_round8_bridge,
                started_before_round9_pre_leap=started_before_round9_pre_leap,
                started_before_leap_window=started_before_leap_window,
                started_before_leap_turn=started_before_leap_turn,
                awareness_transitions=awareness_transitions,
                checker_61_count=checker_61_count,
                checker_actions=checker_actions,
                dropper_61_count=dropper_61_count,
                dropper_actions=dropper_actions,
                truncated=truncated,
            )


def main() -> None:
    args = build_parser().parse_args()
    teacher = create_scripted_opponent(args.teacher)
    opponent = create_model_opponent(args.opponent_model, agent_role=args.role) if args.opponent_model else create_scripted_opponent(args.opponent)
    scenario_name = None if args.scenario == "opening" else args.scenario
    scenario_sampler = make_scenario_sampler(scenario_name=scenario_name, curriculum_name=args.curriculum, seed=args.seed)

    print(
        "Teacher eval config: "
        f"teacher={args.teacher} role={args.role} opponent={args.opponent_model or args.opponent} "
        f"scenario={args.scenario} curriculum={args.curriculum or 'none'} games={args.games} "
        f"max_steps={format_max_steps(args.max_steps)} seed={args.seed}"
    )

    results: list[GameMetrics] = []
    for game_index in range(args.games):
        env = DTHEnv(
            opponent=opponent,
            agent_role=args.role,
            seed=args.seed + game_index,
            scenario_sampler=scenario_sampler,
            max_steps=args.max_steps,
        )
        results.append(play_teacher_game(teacher, env))

    print_summary(summarize_games(results))


if __name__ == "__main__":
    main()
