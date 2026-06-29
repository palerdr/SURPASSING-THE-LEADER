#!/usr/bin/env python3
"""Compare two SolverAgent checkpoints on the same frozen opponent ladder."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hal.agent import DEFAULT_CHECKPOINT, SolverAgent, make_choose_action
from training.strength import gate_report, run_ladder


def _empty_counts() -> dict:
    return {"games": 0, "wins": 0, "draws": 0, "losses": 0}


def _counts_summary(counts: dict) -> dict:
    games = max(1, int(counts["games"]))
    return {
        **counts,
        "win_rate": counts["wins"] / games,
        "score_rate": (counts["wins"] + 0.5 * counts["draws"]) / games,
    }


def _copy_counts(stats: dict) -> dict:
    return {
        "games": int(stats["games"]),
        "wins": int(stats["wins"]),
        "draws": int(stats["draws"]),
        "losses": int(stats["losses"]),
    }


def _add_counts(total: dict, stats: dict) -> None:
    total["games"] += int(stats["games"])
    total["wins"] += int(stats["wins"])
    total["draws"] += int(stats["draws"])
    total["losses"] += int(stats["losses"])


def _delta(champion: dict, candidate: dict) -> dict:
    return {
        "wins_delta": candidate["wins"] - champion["wins"],
        "losses_delta": candidate["losses"] - champion["losses"],
        "win_rate_delta": candidate["win_rate"] - champion["win_rate"],
        "score_rate_delta": candidate["score_rate"] - champion["score_rate"],
    }


def _json_safe(value):
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return "inf" if value > 0 else "-inf"
    return value


def _summarize_delta(champion: dict, candidate: dict, opponents: list[str]) -> dict:
    return {
        "overall": _delta(champion["overall"], candidate["overall"]),
        "opponents": {
            name: _delta(champion["opponents"][name], candidate["opponents"][name])
            for name in opponents
        },
    }


def _aggregate_reports(seed_reports: list[dict], opponents: list[str]) -> dict:
    champion_total = _empty_counts()
    candidate_total = _empty_counts()
    per_opponent: dict[str, dict] = {
        name: {"champion": _empty_counts(), "candidate": _empty_counts()}
        for name in opponents
    }

    for report in seed_reports:
        _add_counts(champion_total, report["champion"]["overall"])
        _add_counts(candidate_total, report["candidate"]["overall"])
        for name in opponents:
            _add_counts(per_opponent[name]["champion"], report["champion"]["opponents"][name])
            _add_counts(per_opponent[name]["candidate"], report["candidate"]["opponents"][name])

    champion = _counts_summary(champion_total)
    candidate = _counts_summary(candidate_total)
    opponent_summary = {}
    for name, counts in per_opponent.items():
        c0 = _counts_summary(counts["champion"])
        c1 = _counts_summary(counts["candidate"])
        opponent_summary[name] = {
            "champion": c0,
            "candidate": c1,
            "delta": _delta(c0, c1),
        }
    return {
        "champion": champion,
        "candidate": candidate,
        "delta": _delta(champion, candidate),
        "opponents": opponent_summary,
    }


def _make_agent(checkpoint: str, *, iterations: int, seed: int) -> SolverAgent:
    return SolverAgent(
        checkpoint,
        player_name="Hal",
        iterations=iterations,
        seed=seed,
    )


def _run_one_seed(args, opponents: list[str], seed: int) -> dict:
    champion_agent = _make_agent(
        args.champion_checkpoint,
        iterations=args.agent_iterations,
        seed=seed,
    )
    candidate_agent = _make_agent(
        args.candidate_checkpoint,
        iterations=args.agent_iterations,
        seed=seed,
    )
    champion_report = gate_report(
        run_ladder(
            make_choose_action(champion_agent),
            opponents,
            n_games=args.games,
            seed=seed,
        )
    )
    candidate_report = gate_report(
        run_ladder(
            make_choose_action(candidate_agent),
            opponents,
            n_games=args.games,
            seed=seed,
        )
    )
    return {
        "seed": seed,
        "champion": champion_report,
        "candidate": candidate_report,
        "delta": _summarize_delta(champion_report, candidate_report, opponents),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--champion-checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--candidate-checkpoint", required=True)
    parser.add_argument("--agent-iterations", type=int, default=50)
    parser.add_argument("--games", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--seeds",
        default=None,
        help="Comma-separated seeds. Overrides --seed and aggregates all runs.",
    )
    parser.add_argument(
        "--opponents",
        default="random,safe,baku_lsr_engineering,pattern_reader",
    )
    parser.add_argument(
        "--out",
        default=str(Path("checkpoints") / "checkpoint_ladder_compare" / "report.json"),
    )
    args = parser.parse_args()

    opponents = [name.strip() for name in args.opponents.split(",") if name.strip()]
    seeds = (
        [int(item.strip()) for item in args.seeds.split(",") if item.strip()]
        if args.seeds
        else [args.seed]
    )
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(
        f"Comparing champion={args.champion_checkpoint} candidate={args.candidate_checkpoint} "
        f"iterations={args.agent_iterations} games={args.games}/opponent seeds={seeds}"
    )
    print(f"Opponents: {', '.join(opponents)}")
    start = time.time()
    seed_reports = []
    for seed in seeds:
        print(f"  seed {seed}...", flush=True)
        report = _run_one_seed(args, opponents, seed)
        seed_reports.append(report)
        print(f"    delta {report['delta']['overall']}", flush=True)

    aggregate = _aggregate_reports(seed_reports, opponents)
    report = {
        "config": vars(args),
        "seeds": seeds,
        "opponents": opponents,
        "seed_reports": seed_reports,
        "aggregate": aggregate,
        "elapsed_seconds": round(time.time() - start, 2),
    }
    with out_path.open("w", encoding="utf-8") as fh:
        json.dump(_json_safe(report), fh, indent=2, allow_nan=False)

    print(f"Aggregate champion overall : {aggregate['champion']}")
    print(f"Aggregate candidate overall: {aggregate['candidate']}")
    print(f"Aggregate delta            : {aggregate['delta']}")
    print(f"Report: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
