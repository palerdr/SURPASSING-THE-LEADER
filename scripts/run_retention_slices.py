"""Run PPO retention in small slices with automatic eval gates.

This script starts from a frozen BC anchor, trains in small PPO chunks, and
stops as soon as opening transfer, alignment, or late seeded competence fall
outside configured bounds.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sb3_contrib import MaskablePPO

from environment.dth_env import DTHEnv
from environment.opponents.factory import create_scripted_opponent
from environment.route_stages import current_route_stage_flags
from scripts.evaluate import make_opponent, play_game, summarize_games
from training.behavior_clone import run_behavior_cloning
from training.curriculum import make_curriculum_sampler
from training.train_ppo import build_training_opponent, build_or_load_model


@dataclass(frozen=True)
class SliceEval:
    opening: dict[str, dict[str, float | int]]
    alignment: dict[str, int]
    seeded_round9: dict[str, float | int]


@dataclass(frozen=True)
class GateDecision:
    passed: bool
    reasons: tuple[str, ...]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run retention PPO in small gated slices.")
    parser.add_argument("--anchor-model", required=True)
    parser.add_argument("--role", choices=["hal", "baku"], default="baku")
    parser.add_argument("--opponent", default="bridge_pressure")
    parser.add_argument("--opponent-weight", type=float, default=1.0)
    parser.add_argument("--opponent-model", action="append", default=[])
    parser.add_argument("--curriculum", default="opening_to_round7")
    parser.add_argument("--shaping", action="store_true")
    parser.add_argument("--shaping-preset", default="exact_bridge")
    parser.add_argument("--slice-timesteps", type=int, default=2048)
    parser.add_argument("--max-slices", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--eval-games", type=int, default=50)
    parser.add_argument("--run-tag", default="retention")
    parser.add_argument(
        "--eval-opponent",
        action="append",
        default=["bridge_pressure", "hal_death_trade", "hal_pressure"],
        help="Repeat to set opening eval opponents.",
    )
    parser.add_argument("--max-misaligned", type=int, default=5)
    parser.add_argument("--min-seeded-round9-win", type=float, default=100.0)
    parser.add_argument("--report-path", default=None)
    parser.add_argument("--anchor-trace-set", default=None)
    parser.add_argument("--anchor-trace-file", default=None)
    parser.add_argument("--anchor-teacher-demo-file", action="append", default=[])
    parser.add_argument("--anchor-bc-epochs", type=int, default=0)
    parser.add_argument("--anchor-bc-batch-size", type=int, default=32)
    parser.add_argument("--anchor-bc-learning-rate", type=float, default=1e-4)
    return parser


def scripted_opponent_factory(name: str):
    return create_scripted_opponent(name)


def evaluate_model(model_path: str, *, role: str, opening_opponents: list[str], eval_games: int, seed: int) -> SliceEval:
    opening: dict[str, dict[str, float | int]] = {}
    for opponent_name in opening_opponents:
        opponent = make_opponent(opponent_name)
        assert opponent is not None
        model = MaskablePPO.load(model_path)
        results = []
        for game_index in range(eval_games):
            env = DTHEnv(opponent=opponent, agent_role=role, seed=seed + game_index)
            results.append(play_game(model, env, verbose=False))
        summary = summarize_games(results)
        opening[opponent_name] = {
            "wins": summary["wins"],
            "win_rate": summary["win_rate"],
            "round7_rate": summary["opening_to_round7_pressure_reach_rate"],
            "round8_rate": summary["opening_to_round8_bridge_reach_rate"],
            "round9_rate": summary["opening_to_round9_pre_leap_reach_rate"],
            "leap_window_rate": summary["opening_to_leap_window_reach_rate"],
        }

    alignment_model = MaskablePPO.load(model_path)
    alignment_opponent = make_opponent("bridge_pressure")
    assert alignment_opponent is not None
    reached_round7 = 0
    reached_round8 = 0
    reached_round9 = 0
    reached_leap_window = 0
    misaligned = 0
    for game_index in range(eval_games):
        env = DTHEnv(opponent=alignment_opponent, agent_role=role, seed=seed + game_index)
        obs, _ = env.reset()
        reached = dict(current_route_stage_flags(env.game))
        for _step in range(80):
            action, _ = alignment_model.predict(obs, action_masks=env.action_masks(), deterministic=True)
            obs, _reward, terminated, truncated, _info = env.step(int(action))
            flags = current_route_stage_flags(env.game)
            for key, value in flags.items():
                reached[key] = reached.get(key, False) or value
            if terminated or truncated:
                break
        reached_round7 += int(reached["round7_pressure"])
        reached_round8 += int(reached["round8_bridge"])
        reached_round9 += int(reached["round9_pre_leap"])
        reached_leap_window += int(reached["leap_window"])
        if reached["leap_window"] and not reached["round9_pre_leap"]:
            misaligned += 1

    seeded_model = MaskablePPO.load(model_path)
    seeded_opponent = make_opponent("bridge_pressure")
    assert seeded_opponent is not None
    seeded_results = []
    for game_index in range(eval_games):
        env = DTHEnv(
            opponent=seeded_opponent,
            agent_role=role,
            seed=seed + game_index,
            scenario_sampler=lambda _rng: {
                "name": "round9_pre_leap",
                **{
                    "game_clock": 3420.0,
                    "round_num": 8,
                    "current_half": 1,
                    "first_dropper": "hal",
                    "hal": {"cylinder": 34.0, "ttd": 204.0, "deaths": 3, "alive": True},
                    "baku": {"cylinder": 0.0, "ttd": 121.0, "deaths": 2, "alive": True},
                    "referee_cprs": 5,
                    "awareness": "deduced",
                },
            },
        )
        seeded_results.append(play_game(seeded_model, env, verbose=False))
    seeded_summary = summarize_games(seeded_results)

    return SliceEval(
        opening=opening,
        alignment={
            "round7_count": reached_round7,
            "round8_count": reached_round8,
            "round9_count": reached_round9,
            "leap_window_count": reached_leap_window,
            "misaligned_leap_window_count": misaligned,
        },
        seeded_round9={
            "wins": seeded_summary["wins"],
            "win_rate": seeded_summary["win_rate"],
        },
    )


def evaluate_gate(anchor_eval: SliceEval, candidate_eval: SliceEval, *, max_misaligned: int, min_seeded_round9_win: float) -> GateDecision:
    reasons: list[str] = []
    anchor_bridge = float(anchor_eval.opening["bridge_pressure"]["round7_rate"])
    candidate_bridge = float(candidate_eval.opening["bridge_pressure"]["round7_rate"])
    if candidate_bridge < anchor_bridge:
        reasons.append(
            f"bridge round7 regressed {candidate_bridge:.2%} < anchor {anchor_bridge:.2%}"
        )

    for held_out in ("hal_death_trade", "hal_pressure"):
        anchor_rate = float(anchor_eval.opening[held_out]["round7_rate"])
        candidate_rate = float(candidate_eval.opening[held_out]["round7_rate"])
        if candidate_rate < anchor_rate:
            reasons.append(
                f"{held_out} round7 regressed {candidate_rate:.2%} < anchor {anchor_rate:.2%}"
            )

    if candidate_eval.alignment["misaligned_leap_window_count"] > max_misaligned:
        reasons.append(
            f"misaligned leap-window count {candidate_eval.alignment['misaligned_leap_window_count']} > {max_misaligned}"
        )

    seeded_win_pct = 100.0 * float(candidate_eval.seeded_round9["win_rate"])
    if seeded_win_pct < min_seeded_round9_win:
        reasons.append(
            f"seeded round9 win {seeded_win_pct:.1f}% < {min_seeded_round9_win:.1f}%"
        )

    return GateDecision(passed=not reasons, reasons=tuple(reasons))


def save_report(path: str, report: dict) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2))


def main() -> None:
    args = build_parser().parse_args()
    opening_opponents = list(dict.fromkeys(args.eval_opponent))

    anchor_eval = evaluate_model(
        args.anchor_model,
        role=args.role,
        opening_opponents=opening_opponents,
        eval_games=args.eval_games,
        seed=args.seed,
    )

    opponent = build_training_opponent(
        agent_role=args.role,
        opponent_name=args.opponent,
        opponent_weight=args.opponent_weight,
        opponent_model_specs=args.opponent_model,
        seed=args.seed,
    )
    scenario_sampler = make_curriculum_sampler(args.curriculum, seed=args.seed)
    env = DTHEnv(
        opponent=opponent,
        agent_role=args.role,
        seed=args.seed,
        use_shaping=args.shaping,
        shaping_preset=args.shaping_preset,
        scenario_sampler=scenario_sampler,
        max_steps=args.max_steps,
    )
    model = build_or_load_model(env, args.anchor_model, args.seed)

    save_dir = Path("models/checkpoints")
    save_dir.mkdir(parents=True, exist_ok=True)

    slices = []
    best_path = args.anchor_model
    stopped = False
    for slice_index in range(1, args.max_slices + 1):
        model.learn(total_timesteps=args.slice_timesteps, reset_num_timesteps=False)
        rehearsal_samples = 0
        if args.anchor_bc_epochs > 0 and (
            args.anchor_trace_set is not None
            or args.anchor_trace_file is not None
            or args.anchor_teacher_demo_file
        ):
            rehearsal_samples = run_behavior_cloning(
                model,
                trace_set_name=args.anchor_trace_set,
                trace_file=args.anchor_trace_file,
                teacher_demo_files=args.anchor_teacher_demo_file,
                opponent_factory=scripted_opponent_factory,
                epochs=args.anchor_bc_epochs,
                batch_size=args.anchor_bc_batch_size,
                learning_rate=args.anchor_bc_learning_rate,
            )
        total_timesteps = slice_index * args.slice_timesteps
        checkpoint = save_dir / f"{args.role}_retention_{args.run_tag}_{total_timesteps}"
        model.save(str(checkpoint))
        candidate_eval = evaluate_model(
            f"{checkpoint}.zip",
            role=args.role,
            opening_opponents=opening_opponents,
            eval_games=args.eval_games,
            seed=args.seed,
        )
        decision = evaluate_gate(
            anchor_eval,
            candidate_eval,
            max_misaligned=args.max_misaligned,
            min_seeded_round9_win=args.min_seeded_round9_win,
        )
        slices.append(
            {
                "slice_index": slice_index,
                "timesteps": total_timesteps,
                "checkpoint": f"{checkpoint}.zip",
                "anchor_rehearsal_samples": rehearsal_samples,
                "eval": asdict(candidate_eval),
                "gate": asdict(decision),
            }
        )
        print(
            f"slice={slice_index} timesteps={total_timesteps} passed={decision.passed} "
            f"bridge_r7={candidate_eval.opening['bridge_pressure']['round7_rate']:.2%} "
            f"hal_trade_r7={candidate_eval.opening['hal_death_trade']['round7_rate']:.2%} "
            f"misaligned={candidate_eval.alignment['misaligned_leap_window_count']} "
            f"seeded_r9_win={100.0 * candidate_eval.seeded_round9['win_rate']:.1f}%"
        )
        if not decision.passed:
            stopped = True
            break
        best_path = f"{checkpoint}.zip"

    report = {
        "anchor_model": args.anchor_model,
        "best_path": best_path,
        "stopped_early": stopped,
        "anchor_eval": asdict(anchor_eval),
        "slices": slices,
    }
    if args.report_path:
        save_report(args.report_path, report)
    else:
        print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
