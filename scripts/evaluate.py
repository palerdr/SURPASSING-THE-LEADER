"""Evaluate a trained agent against an opponent.

Usage:
    python scripts/evaluate.py --model models/self_play/hal_gen3 --role hal --opponent safe --games 5
    python scripts/evaluate.py --model models/self_play/hal_gen3 --role hal --opponent-model models/self_play/baku_gen2 --games 10
"""

from __future__ import annotations

import sys
import os
import argparse
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sb3_contrib import MaskablePPO

from environment.dth_env import DTHEnv
from environment.opponents.factory import create_model_opponent, create_scripted_opponent, scripted_opponent_names
from environment.route_stages import current_route_stage_flags, stage_is_eligible_from_start
from src.Constants import LS_WINDOW_START
from training.curriculum import CURRICULA, SCENARIOS, get_scenario, make_curriculum_sampler


@dataclass(frozen=True)
class GameMetrics:
    won: bool
    half_rounds: int
    reached_round7_pressure: bool
    reached_round8_bridge: bool
    reached_round9_pre_leap: bool
    reached_leap_window: bool
    reached_leap_turn: bool
    started_before_round7_pressure: bool
    started_before_round8_bridge: bool
    started_before_round9_pre_leap: bool
    started_before_leap_window: bool
    started_before_leap_turn: bool
    awareness_transitions: int
    checker_61_count: int
    checker_actions: int
    dropper_61_count: int
    dropper_actions: int
    truncated: bool


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate trained agent")
    parser.add_argument("--model", required=True, help="Path to trained model")
    parser.add_argument("--role", choices=["hal", "baku"], default="hal")
    parser.add_argument("--opponent", choices=scripted_opponent_names(), default="safe",
                        help="Bot or teacher name")
    parser.add_argument("--opponent-model", default=None,
                        help="Path to trained opponent model (overrides --opponent)")
    parser.add_argument("--games", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--scenario", choices=sorted(SCENARIOS.keys()), default=None,
                        help="Evaluate from a fixed named scenario start.")
    parser.add_argument("--curriculum", choices=sorted(CURRICULA.keys()), default=None,
                        help="Evaluate from a curriculum sampler instead of the opening state.")
    parser.add_argument("--max-steps", type=int, default=None,
                        help="Optional emergency episode cap. Disabled by default.")
    parser.add_argument("--verbose", action="store_true",
                        help="Print the full half-round trace for each game.")
    return parser


def make_opponent(name: str, model_path: str | None = None):
    if model_path:
        return create_model_opponent(model_path, agent_role=name)
    return create_scripted_opponent(name)


def format_max_steps(max_steps: int | None) -> str:
    return "disabled" if max_steps is None else str(max_steps)


def make_scenario_sampler(
    *,
    scenario_name: str | None,
    curriculum_name: str | None,
    seed: int | None,
):
    if scenario_name is not None and curriculum_name is not None:
        raise ValueError("Use either --scenario or --curriculum, not both")

    if scenario_name is not None:
        return lambda _rng: get_scenario(scenario_name)

    if curriculum_name is not None:
        return make_curriculum_sampler(curriculum_name, seed=seed)

    return None


def play_game(model, env: DTHEnv, verbose: bool = False) -> GameMetrics:
    """Play one game and return aggregate metrics for the controlled agent."""
    obs, info = env.reset()
    awareness = info["awareness"]
    stage_flags = current_route_stage_flags(env.game)

    reached_round7_pressure = stage_flags["round7_pressure"]
    reached_round8_bridge = stage_flags["round8_bridge"]
    reached_round9_pre_leap = stage_flags["round9_pre_leap"]
    reached_leap_window = env.game.game_clock >= LS_WINDOW_START
    reached_leap_turn = env.game.is_leap_second_turn()
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
        agent_role = "dropper" if env.agent is dropper else "checker"

        action, _ = model.predict(obs, action_masks=mask, deterministic=True)
        action_second = int(action) + 1

        if agent_role == "checker":
            checker_actions += 1
            if action_second == 61:
                checker_61_count += 1
        else:
            dropper_actions += 1
            if action_second == 61:
                dropper_61_count += 1

        obs, reward, terminated, truncated, info = env.step(int(action))
        rec = env.game.history[-1]
        stage_flags = current_route_stage_flags(env.game)

        reached_round7_pressure = reached_round7_pressure or stage_flags["round7_pressure"]
        reached_round8_bridge = reached_round8_bridge or stage_flags["round8_bridge"]
        reached_round9_pre_leap = reached_round9_pre_leap or stage_flags["round9_pre_leap"]
        reached_leap_window = reached_leap_window or (env.game.game_clock >= LS_WINDOW_START)
        reached_leap_turn = reached_leap_turn or stage_flags["leap_turn"] or (rec.turn_duration == 61)

        if info["awareness"] != awareness:
            awareness_transitions += 1
            awareness = info["awareness"]

        if verbose:
            clock = env.game.format_game_clock()
            p1_cyl = env.game.player1.cylinder
            p2_cyl = env.game.player2.cylinder
            print(
                f"  R{rec.round_num + 1}T{rec.half} [{clock}] "
                f"D:{rec.dropper}@{rec.drop_time} C:{rec.checker}@{rec.check_time} "
                f"-> {rec.result.value} ST={rec.st_gained:.0f} "
                f"| Hal cyl={p1_cyl:.0f} Baku cyl={p2_cyl:.0f} "
                f"| awareness={info['awareness']}"
            )

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


def summarize_games(results: list[GameMetrics]) -> dict[str, float | int]:
    total_games = len(results)
    if total_games == 0:
        raise ValueError("No evaluation results to summarize")

    wins = sum(result.won for result in results)
    total_half_rounds = sum(result.half_rounds for result in results)
    round7_games = sum(
        result.started_before_round7_pressure and result.reached_round7_pressure
        for result in results
    )
    round8_games = sum(
        result.started_before_round8_bridge and result.reached_round8_bridge
        for result in results
    )
    round9_games = sum(
        result.started_before_round9_pre_leap and result.reached_round9_pre_leap
        for result in results
    )
    leap_window_games = sum(result.reached_leap_window for result in results)
    leap_turn_games = sum(result.reached_leap_turn for result in results)
    opening_to_round7_eligible = sum(result.started_before_round7_pressure for result in results)
    opening_to_round8_eligible = sum(result.started_before_round8_bridge for result in results)
    opening_to_round9_eligible = sum(result.started_before_round9_pre_leap for result in results)
    opening_to_leap_window_eligible = sum(result.started_before_leap_window for result in results)
    opening_to_leap_window_count = sum(
        result.started_before_leap_window and result.reached_leap_window
        for result in results
    )
    leap_window_to_leap_turn_eligible = sum(
        result.reached_leap_window and result.started_before_leap_turn
        for result in results
    )
    leap_window_to_leap_turn_count = sum(
        result.reached_leap_window and result.started_before_leap_turn and result.reached_leap_turn
        for result in results
    )
    leap_turn_to_win_count = sum(result.reached_leap_turn and result.won for result in results)
    awareness_transition_count = sum(result.awareness_transitions for result in results)
    awareness_transition_games = sum(result.awareness_transitions > 0 for result in results)
    checker_61_count = sum(result.checker_61_count for result in results)
    checker_actions = sum(result.checker_actions for result in results)
    dropper_61_count = sum(result.dropper_61_count for result in results)
    dropper_actions = sum(result.dropper_actions for result in results)
    truncation_count = sum(result.truncated for result in results)

    def safe_rate(numerator: int, denominator: int) -> float:
        if denominator == 0:
            return 0.0
        return numerator / denominator

    return {
        "games": total_games,
        "wins": wins,
        "win_rate": safe_rate(wins, total_games),
        "average_half_rounds": total_half_rounds / total_games,
        "opening_to_round7_pressure_count": round7_games,
        "opening_to_round7_pressure_eligible": opening_to_round7_eligible,
        "opening_to_round7_pressure_reach_rate": safe_rate(round7_games, opening_to_round7_eligible),
        "opening_to_round8_bridge_count": round8_games,
        "opening_to_round8_bridge_eligible": opening_to_round8_eligible,
        "opening_to_round8_bridge_reach_rate": safe_rate(round8_games, opening_to_round8_eligible),
        "opening_to_round9_pre_leap_count": round9_games,
        "opening_to_round9_pre_leap_eligible": opening_to_round9_eligible,
        "opening_to_round9_pre_leap_reach_rate": safe_rate(round9_games, opening_to_round9_eligible),
        "opening_to_leap_window_count": opening_to_leap_window_count,
        "opening_to_leap_window_eligible": opening_to_leap_window_eligible,
        "opening_to_leap_window_reach_rate": safe_rate(opening_to_leap_window_count, opening_to_leap_window_eligible),
        "leap_window_reach_rate": safe_rate(leap_window_games, total_games),
        "leap_turn_reach_rate": safe_rate(leap_turn_games, total_games),
        "leap_window_to_leap_turn_count": leap_window_to_leap_turn_count,
        "leap_window_to_leap_turn_eligible": leap_window_to_leap_turn_eligible,
        "leap_window_to_leap_turn_reach_rate": safe_rate(leap_window_to_leap_turn_count, leap_window_to_leap_turn_eligible),
        "leap_turn_to_win_count": leap_turn_to_win_count,
        "leap_turn_to_win_eligible": leap_turn_games,
        "leap_turn_to_win_rate": safe_rate(leap_turn_to_win_count, leap_turn_games),
        "awareness_transition_count": awareness_transition_count,
        "awareness_transition_rate": safe_rate(awareness_transition_games, total_games),
        "checker_61_count": checker_61_count,
        "checker_61_rate": safe_rate(checker_61_count, checker_actions),
        "dropper_61_count": dropper_61_count,
        "dropper_61_rate": safe_rate(dropper_61_count, dropper_actions),
        "truncation_count": truncation_count,
        "truncation_rate": safe_rate(truncation_count, total_games),
    }


def format_stage_rate(summary: dict[str, float | int], prefix: str) -> str:
    count = int(summary[f"{prefix}_count"])
    eligible = int(summary[f"{prefix}_eligible"])
    if eligible == 0:
        return f"{count}/{eligible} (n/a)"
    rate = 100 * float(summary[f"{prefix}_reach_rate" if prefix != "leap_turn_to_win" else f"{prefix}_rate"])
    return f"{count}/{eligible} ({rate:.0f}%)"


def print_summary(summary: dict[str, float | int]) -> None:
    print(f"\n{'=' * 40}")
    print(f"Games: {summary['games']}")
    print(f"Wins: {summary['wins']} ({100 * summary['win_rate']:.0f}%)")
    print(f"Average half-rounds: {summary['average_half_rounds']:.2f}")
    print(
        "Route stages: "
        f"opening->round7 {format_stage_rate(summary, 'opening_to_round7_pressure')} "
        f"| opening->round8 {format_stage_rate(summary, 'opening_to_round8_bridge')} "
        f"| opening->round9_pre_leap {format_stage_rate(summary, 'opening_to_round9_pre_leap')}"
    )
    print(
        "Route conversions: "
        f"opening->leap-window {format_stage_rate(summary, 'opening_to_leap_window')} "
        f"| leap-window->leap-turn {format_stage_rate(summary, 'leap_window_to_leap_turn')} "
        f"| leap-turn->win {format_stage_rate(summary, 'leap_turn_to_win')}"
    )
    print(
        f"Leap-window reach: {100 * summary['leap_window_reach_rate']:.0f}% "
        f"| leap-turn reach: {100 * summary['leap_turn_reach_rate']:.0f}%"
    )
    print(
        f"Awareness transitions: {summary['awareness_transition_count']} total "
        f"({100 * summary['awareness_transition_rate']:.0f}% of games)"
    )
    print(
        f"Checker 61 usage: {summary['checker_61_count']} "
        f"({100 * summary['checker_61_rate']:.0f}% of checker actions)"
    )
    print(
        f"Dropper 61 usage: {summary['dropper_61_count']} "
        f"({100 * summary['dropper_61_rate']:.0f}% of dropper actions)"
    )
    print(
        f"Truncations: {summary['truncation_count']} "
        f"({100 * summary['truncation_rate']:.0f}% of games)"
    )


def main():
    parser = build_parser()
    args = parser.parse_args()

    opponent = make_opponent(
        args.opponent if not args.opponent_model else args.role,
        args.opponent_model,
    )
    model = MaskablePPO.load(args.model)
    scenario_sampler = make_scenario_sampler(
        scenario_name=args.scenario,
        curriculum_name=args.curriculum,
        seed=args.seed,
    )

    print(
        "Evaluation config: "
        f"role={args.role} opponent_model={'yes' if args.opponent_model else 'no'} "
        f"opponent={args.opponent if not args.opponent_model else args.opponent_model} "
        f"scenario={args.scenario or 'opening'} curriculum={args.curriculum or 'none'} "
        f"games={args.games} max_steps={format_max_steps(args.max_steps)} "
        f"verbose={'on' if args.verbose else 'off'} seed={args.seed}"
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
        if args.verbose:
            print(f"\n-- Game {game_index + 1} --")
        results.append(play_game(model, env, verbose=args.verbose))

    print_summary(summarize_games(results))


if __name__ == "__main__":
    main()
