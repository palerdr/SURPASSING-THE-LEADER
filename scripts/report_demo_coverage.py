from __future__ import annotations

import argparse
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training.teacher_demos import load_teacher_demo_file


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Summarize teacher-demo dataset coverage.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--prefix-turns", type=int, default=8)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    samples = load_teacher_demo_file(args.input)
    by_game: dict[tuple[int, int], list[int]] = {}
    teacher_counts = Counter()
    opponent_counts = Counter()
    scenario_counts = Counter()
    stage_counts = Counter()

    for sample in samples:
        game_key = (sample.seed, sample.game_index)
        by_game.setdefault(game_key, []).append(sample.action_index + 1)
        teacher_counts[sample.teacher_name] += 1
        opponent_counts[sample.opponent_name or sample.opponent_model_path or "model"] += 1
        scenario_counts[sample.scenario_name] += 1
        stage_counts[sample.reached_stage or "none"] += 1

    prefix_counts = Counter(tuple(actions[: args.prefix_turns]) for actions in by_game.values())

    print(f"Teacher demo file: {args.input}")
    print(f"Samples: {len(samples)}")
    print(f"Episodes: {len(by_game)}")
    print(f"Teachers: {dict(teacher_counts)}")
    print(f"Opponents: {dict(opponent_counts)}")
    print(f"Scenarios: {dict(scenario_counts)}")
    print(f"Reached stages: {dict(stage_counts)}")
    print("Common prefixes:")
    for prefix, count in prefix_counts.most_common(5):
        print(f"  {list(prefix)} x {count}")


if __name__ == "__main__":
    main()
