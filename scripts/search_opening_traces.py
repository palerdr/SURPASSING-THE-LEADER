"""Search for opening-positive bridge traces and export them as replayable specs.

The output format matches `BridgeTraceSpec` JSON so searched traces can be
fed directly into the existing behavior-cloning warm start.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from collections import Counter
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from environment.dth_env import DTHEnv
from environment.opponents.factory import create_model_opponent, create_scripted_opponent, scripted_opponent_names
from environment.route_math import is_active_lsr
from environment.route_stages import ROUTE_STAGE_SPECS, current_route_stage_flags
from training.bridge_traces import BridgeTraceSpec, load_trace_file, save_trace_file
from training.curriculum import SCENARIOS, get_scenario


CANDIDATE_SETS = {
    "opening_small": {
        "checker": (1, 2, 5, 8, 10, 15, 24, 30, 34, 45, 60),
        "dropper": (1, 2, 3, 5, 10, 24, 35, 56, 60, 61),
    }
}


@dataclass(frozen=True)
class SearchResult:
    trace: BridgeTraceSpec
    rank: tuple
    state_key: tuple


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Search opening traces that reach a target bridge stage.")
    parser.add_argument("--role", choices=["hal", "baku"], default="baku")
    parser.add_argument("--opponent", choices=scripted_opponent_names(), default="bridge_pressure")
    parser.add_argument("--opponent-model", default=None)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--seed-count", type=int, default=256)
    parser.add_argument("--beam-width", type=int, default=128)
    parser.add_argument("--max-depth", type=int, default=16)
    parser.add_argument("--target-stage", choices=[*ROUTE_STAGE_SPECS.keys(), "leap_window", "leap_turn"], default="round7_pressure")
    parser.add_argument("--candidate-set", choices=sorted(CANDIDATE_SETS.keys()), default="opening_small")
    parser.add_argument("--max-traces", type=int, default=64)
    parser.add_argument("--max-traces-per-seed", type=int, default=4)
    parser.add_argument("--max-traces-per-family", type=int, default=4)
    parser.add_argument("--family-prefix-turns", type=int, default=8)
    parser.add_argument("--seed-trace-file", action="append", default=[])
    parser.add_argument("--seed-trace-prefix-turns", type=int, default=8)
    parser.add_argument("--keep-near-misses", type=int, default=0)
    parser.add_argument("--output", required=True)
    parser.add_argument("--verbose", action="store_true")
    return parser


def make_opponent(name: str, model_path: str | None, role: str):
    if model_path is not None:
        return create_model_opponent(model_path, agent_role=role)
    return create_scripted_opponent(name)


def replay_prefix(
    *,
    role: str,
    opponent_name: str,
    opponent_model_path: str | None,
    seed: int,
    prefix_actions: tuple[int, ...],
):
    opponent = make_opponent(opponent_name, opponent_model_path, role)
    env = DTHEnv(opponent=opponent, agent_role=role, seed=seed)
    obs, _ = env.reset()
    terminated = False
    truncated = False
    for second in prefix_actions:
        action_index = second - 1
        mask = env.action_masks()
        if action_index < 0 or action_index >= len(mask) or not mask[action_index]:
            raise ValueError(f"Illegal replay action second={second} seed={seed} prefix={prefix_actions}")
        obs, _reward, terminated, truncated, _info = env.step(action_index)
        if terminated or truncated:
            break
    return env, obs, terminated, truncated


def current_role(env: DTHEnv) -> str:
    dropper, _checker = env.game.get_roles_for_half(env.game.current_half)
    return "dropper" if env.agent is dropper else "checker"


def candidate_seconds_for_turn(env: DTHEnv, role: str, candidate_set_name: str) -> list[int]:
    mask = env.action_masks()
    seconds = []
    for second in CANDIDATE_SETS[candidate_set_name][role]:
        idx = second - 1
        if 0 <= idx < len(mask) and bool(mask[idx]):
            seconds.append(second)
    return seconds


def compact_state_key(env: DTHEnv) -> tuple:
    game = env.game
    return (
        game.round_num,
        game.current_half,
        int(game.game_clock),
        game.player1.cylinder,
        game.player1.ttd,
        game.player1.deaths,
        game.player2.cylinder,
        game.player2.ttd,
        game.player2.deaths,
        game.referee.cprs_performed,
        env.awareness.value,
        bool(game.game_over),
        bool(env.agent.alive),
    )


def reached_target(env: DTHEnv, target_stage: str) -> bool:
    return current_route_stage_flags(env.game).get(target_stage, False) and env.agent.alive and not env.game.game_over


def overshot_target(env: DTHEnv, target_stage: str) -> bool:
    if target_stage in ROUTE_STAGE_SPECS:
        stage = ROUTE_STAGE_SPECS[target_stage]
        return (env.game.round_num, env.game.current_half) > (stage.round_num, stage.current_half)
    if target_stage == "leap_window":
        return current_route_stage_flags(env.game)["leap_window"] and not reached_target(env, target_stage)
    if target_stage == "leap_turn":
        return current_route_stage_flags(env.game)["leap_turn"] and not reached_target(env, target_stage)
    raise ValueError(f"Unknown target stage: {target_stage}")


def target_state_for_stage(target_stage: str) -> dict | None:
    if target_stage in SCENARIOS:
        return get_scenario(target_stage)
    return None


def rank_branch(env: DTHEnv, prefix_len: int, target_stage: str) -> tuple:
    game = env.game
    hal = game.player1
    baku = game.player2
    target = target_state_for_stage(target_stage)

    if target is None:
        target_clock = 0.0
        target_hal_cyl = hal.cylinder
        target_hal_ttd = hal.ttd
        target_baku_cyl = baku.cylinder
        target_baku_ttd = baku.ttd
        target_cprs = game.referee.cprs_performed
    else:
        target_clock = float(target["game_clock"])
        target_hal_cyl = float(target["hal"]["cylinder"])
        target_hal_ttd = float(target["hal"]["ttd"])
        target_baku_cyl = float(target["baku"]["cylinder"])
        target_baku_ttd = float(target["baku"]["ttd"])
        target_cprs = int(target["referee_cprs"])

    return (
        int(reached_target(env, target_stage)),
        int(env.agent.alive and not game.game_over),
        int(is_active_lsr(game)),
        game.round_num,
        int(game.current_half == 1),
        -abs(game.game_clock - target_clock),
        -abs(hal.cylinder - target_hal_cyl),
        -abs(hal.ttd - target_hal_ttd),
        -abs(baku.cylinder - target_baku_cyl),
        -abs(baku.ttd - target_baku_ttd),
        -abs(game.referee.cprs_performed - target_cprs),
        -prefix_len,
    )


def trace_name(opponent_name: str, seed: int, index: int) -> str:
    return f"search_opening_to_round7_{opponent_name}_{seed}_{index}"


def family_key(actions: tuple[int, ...], prefix_turns: int) -> tuple[int, ...]:
    return actions[:prefix_turns]


def load_seed_prefixes(paths: list[str], prefix_turns: int) -> tuple[tuple[int, ...], ...]:
    prefixes: list[tuple[int, ...]] = []
    seen: set[tuple[int, ...]] = set()
    for path in paths:
        for spec in load_trace_file(path):
            if spec.scenario_name != "opening":
                continue
            prefix = spec.actions[:prefix_turns]
            if prefix and prefix not in seen:
                seen.add(prefix)
                prefixes.append(prefix)
    return tuple(prefixes)


def search_seed(
    *,
    role: str,
    opponent_name: str,
    opponent_model_path: str | None,
    seed: int,
    beam_width: int,
    max_depth: int,
    target_stage: str,
    candidate_set_name: str,
    max_results: int,
    initial_prefixes: tuple[tuple[int, ...], ...] = (),
) -> tuple[list[SearchResult], list[tuple[tuple, tuple[int, ...], tuple]]]:
    beam_prefixes = [tuple()]
    beam_prefixes.extend(prefix for prefix in initial_prefixes if len(prefix) < max_depth)
    beam = list(dict.fromkeys(beam_prefixes))
    seen_depth_by_key: dict[tuple, int] = {}
    successes: list[SearchResult] = []
    near_misses: list[tuple[tuple, tuple[int, ...], tuple]] = []

    for _depth in range(max_depth):
        expanded: list[tuple[tuple, tuple[int, ...], tuple]] = []
        for prefix in beam:
            env, _obs, terminated, truncated = replay_prefix(
                role=role,
                opponent_name=opponent_name,
                opponent_model_path=opponent_model_path,
                seed=seed,
                prefix_actions=prefix,
            )
            if terminated or truncated or env.game.game_over or not env.agent.alive or overshot_target(env, target_stage):
                continue

            role_name = current_role(env)
            for second in candidate_seconds_for_turn(env, role_name, candidate_set_name):
                child_prefix = prefix + (second,)
                child_env, _child_obs, child_terminated, child_truncated = replay_prefix(
                    role=role,
                    opponent_name=opponent_name,
                    opponent_model_path=opponent_model_path,
                    seed=seed,
                    prefix_actions=child_prefix,
                )
                child_rank = rank_branch(child_env, len(child_prefix), target_stage)
                child_key = compact_state_key(child_env)

                if reached_target(child_env, target_stage):
                    successes.append(
                        SearchResult(
                            trace=BridgeTraceSpec(
                                name=trace_name(opponent_name, seed, len(successes)),
                                agent_role=role,
                                opponent_name=opponent_name if opponent_model_path is None else None,
                                opponent_model_path=opponent_model_path,
                                scenario_name="opening",
                                actions=child_prefix,
                                seed=seed,
                            ),
                            rank=child_rank,
                            state_key=child_key,
                        )
                    )
                    if len(successes) >= max_results:
                        return successes, near_misses
                    continue

                if child_terminated or child_truncated or child_env.game.game_over or not child_env.agent.alive or overshot_target(child_env, target_stage):
                    continue

                prev_depth = seen_depth_by_key.get(child_key)
                if prev_depth is not None and prev_depth <= len(child_prefix):
                    continue
                seen_depth_by_key[child_key] = len(child_prefix)
                expanded.append((child_rank, child_prefix, child_key))

        expanded.sort(key=lambda item: item[0], reverse=True)
        near_misses = expanded
        beam = [prefix for _rank, prefix, _key in expanded[:beam_width]]
        if not beam:
            break

    return successes, near_misses


def search_many_seeds(
    *,
    role: str,
    opponent_name: str,
    opponent_model_path: str | None,
    seed_start: int,
    seed_count: int,
    beam_width: int,
    max_depth: int,
    target_stage: str,
    candidate_set_name: str,
    max_traces: int,
    max_traces_per_seed: int,
    max_traces_per_family: int,
    family_prefix_turns: int,
    seed_prefixes: tuple[tuple[int, ...], ...],
    verbose: bool,
):
    found: list[SearchResult] = []
    near_misses: list[tuple[int, tuple, tuple[int, ...], tuple]] = []
    family_counts: Counter[tuple[int, ...]] = Counter()
    for seed in range(seed_start, seed_start + seed_count):
        remaining = max_traces - len(found)
        if remaining <= 0:
            break
        seed_results, seed_near_misses = search_seed(
            role=role,
            opponent_name=opponent_name,
            opponent_model_path=opponent_model_path,
            seed=seed,
            beam_width=beam_width,
            max_depth=max_depth,
            target_stage=target_stage,
            candidate_set_name=candidate_set_name,
            max_results=max(remaining, max_traces_per_seed),
            initial_prefixes=seed_prefixes,
        )
        accepted_for_seed = 0
        for result in seed_results:
            fam = family_key(result.trace.actions, family_prefix_turns)
            if family_counts[fam] >= max_traces_per_family:
                continue
            found.append(result)
            family_counts[fam] += 1
            accepted_for_seed += 1
            if accepted_for_seed >= max_traces_per_seed or len(found) >= max_traces:
                break
        near_misses.extend((seed, rank, prefix, key) for rank, prefix, key in seed_near_misses[:3])
        if verbose and seed_results:
            print(f"seed={seed} found={len(seed_results)} accepted={accepted_for_seed} prefix={list(seed_results[0].trace.actions)}")

    near_misses.sort(key=lambda item: item[1], reverse=True)
    return tuple(result.trace for result in found[:max_traces]), near_misses


def main() -> None:
    args = build_parser().parse_args()
    seed_prefixes = load_seed_prefixes(args.seed_trace_file, args.seed_trace_prefix_turns)
    traces, near_misses = search_many_seeds(
        role=args.role,
        opponent_name=args.opponent,
        opponent_model_path=args.opponent_model,
        seed_start=args.seed_start,
        seed_count=args.seed_count,
        beam_width=args.beam_width,
        max_depth=args.max_depth,
        target_stage=args.target_stage,
        candidate_set_name=args.candidate_set,
        max_traces=args.max_traces,
        max_traces_per_seed=args.max_traces_per_seed,
        max_traces_per_family=args.max_traces_per_family,
        family_prefix_turns=args.family_prefix_turns,
        seed_prefixes=seed_prefixes,
        verbose=args.verbose,
    )
    save_trace_file(args.output, traces)

    print(
        "Opening search complete: "
        f"role={args.role} opponent={args.opponent_model or args.opponent} target={args.target_stage} "
        f"seed_range=[{args.seed_start}, {args.seed_start + args.seed_count}) traces={len(traces)} "
        f"seed_prefixes={len(seed_prefixes)} output={args.output}"
    )
    if args.keep_near_misses > 0 and near_misses:
        print("Top near misses:")
        for seed, rank, prefix, _key in near_misses[: args.keep_near_misses]:
            print(f"  seed={seed} prefix={list(prefix)} rank={rank}")


if __name__ == "__main__":
    main()
