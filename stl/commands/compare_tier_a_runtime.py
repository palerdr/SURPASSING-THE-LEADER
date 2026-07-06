#!/usr/bin/env python3
"""Compare SolverAgent with and without Tier A runtime lookup on one ladder."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stl.commands.compare_ladder import _json_safe
from stl.play.agent import DEFAULT_CHECKPOINT, SolverAgent, make_choose_action
from stl.learning.strength import gate_report, run_ladder


def _make_agent(*, checkpoint: str, iterations: int, seed: int, use_tier_a: bool, tier_a_width: float):
    return SolverAgent(
        checkpoint,
        player_name="Hal",
        iterations=iterations,
        seed=seed,
        use_tier_a=use_tier_a,
        tier_a_width=tier_a_width,
    )


def _tier_a_stats(agent: SolverAgent) -> dict:
    evaluator = getattr(agent, "evaluator", None)
    return {
        "hits": int(getattr(evaluator, "hits", 0)),
        "wide_hits": int(getattr(evaluator, "wide_hits", 0)),
        "misses": dict(getattr(evaluator, "misses", {})),
    }


def _empty_tier_a_stats() -> dict:
    return {"hits": 0, "wide_hits": 0, "misses": {}}


def _add_tier_a_stats(total: dict, stats: dict) -> None:
    total["hits"] += int(stats.get("hits", 0))
    total["wide_hits"] += int(stats.get("wide_hits", 0))
    misses = stats.get("misses", {})
    for reason, count in misses.items():
        total["misses"][reason] = total["misses"].get(reason, 0) + int(count)


def _summarize_delta(baseline: dict, tier_a: dict, opponents: list[str]) -> dict:
    per_opponent: dict[str, dict] = {}
    for name in opponents:
        b = baseline["opponents"][name]
        t = tier_a["opponents"][name]
        per_opponent[name] = {
            "win_rate_delta": t["win_rate"] - b["win_rate"],
            "score_rate_delta": t["score_rate"] - b["score_rate"],
            "wins_delta": t["wins"] - b["wins"],
            "losses_delta": t["losses"] - b["losses"],
            "avg_length_delta": (
                t["avg_game_length_half_rounds"] - b["avg_game_length_half_rounds"]
            ),
        }
    return {
        "overall": {
            "win_rate_delta": (
                tier_a["overall"]["win_rate"] - baseline["overall"]["win_rate"]
            ),
            "score_rate_delta": (
                tier_a["overall"]["score_rate"] - baseline["overall"]["score_rate"]
            ),
            "wins_delta": tier_a["overall"]["wins"] - baseline["overall"]["wins"],
            "losses_delta": tier_a["overall"]["losses"] - baseline["overall"]["losses"],
        },
        "opponents": per_opponent,
    }


def _empty_counts() -> dict:
    return {"games": 0, "wins": 0, "draws": 0, "losses": 0}


def _add_counts(total: dict, report: dict) -> None:
    overall = report["overall"]
    total["games"] += int(overall["games"])
    total["wins"] += int(overall["wins"])
    total["draws"] += int(overall["draws"])
    total["losses"] += int(overall["losses"])


def _counts_summary(counts: dict) -> dict:
    games = max(1, int(counts["games"]))
    return {
        **counts,
        "win_rate": counts["wins"] / games,
        "score_rate": (counts["wins"] + 0.5 * counts["draws"]) / games,
    }


def _aggregate_reports(seed_reports: list[dict], opponents: list[str]) -> dict:
    baseline_total = _empty_counts()
    tier_a_total = _empty_counts()
    tier_a_stats_total = _empty_tier_a_stats()
    per_opponent: dict[str, dict] = {
        name: {"baseline": _empty_counts(), "tier_a": _empty_counts()}
        for name in opponents
    }

    for report in seed_reports:
        _add_counts(baseline_total, report["baseline"])
        _add_counts(tier_a_total, report["tier_a"])
        _add_tier_a_stats(tier_a_stats_total, report.get("tier_a_evaluator_stats", {}))
        for name in opponents:
            b = report["baseline"]["opponents"][name]
            t = report["tier_a"]["opponents"][name]
            per_opponent[name]["baseline"]["games"] += int(b["games"])
            per_opponent[name]["baseline"]["wins"] += int(b["wins"])
            per_opponent[name]["baseline"]["draws"] += int(b["draws"])
            per_opponent[name]["baseline"]["losses"] += int(b["losses"])
            per_opponent[name]["tier_a"]["games"] += int(t["games"])
            per_opponent[name]["tier_a"]["wins"] += int(t["wins"])
            per_opponent[name]["tier_a"]["draws"] += int(t["draws"])
            per_opponent[name]["tier_a"]["losses"] += int(t["losses"])

    baseline = _counts_summary(baseline_total)
    tier_a = _counts_summary(tier_a_total)
    opponent_summary = {}
    for name, counts in per_opponent.items():
        b = _counts_summary(counts["baseline"])
        t = _counts_summary(counts["tier_a"])
        opponent_summary[name] = {
            "baseline": b,
            "tier_a": t,
            "delta": {
                "wins_delta": t["wins"] - b["wins"],
                "losses_delta": t["losses"] - b["losses"],
                "win_rate_delta": t["win_rate"] - b["win_rate"],
                "score_rate_delta": t["score_rate"] - b["score_rate"],
            },
        }
    return {
        "baseline": baseline,
        "tier_a": tier_a,
        "delta": {
            "wins_delta": tier_a["wins"] - baseline["wins"],
            "losses_delta": tier_a["losses"] - baseline["losses"],
            "win_rate_delta": tier_a["win_rate"] - baseline["win_rate"],
            "score_rate_delta": tier_a["score_rate"] - baseline["score_rate"],
        },
        "tier_a_evaluator_stats": tier_a_stats_total,
        "opponents": opponent_summary,
    }


def _run_one_seed(args, opponents: list[str], seed: int) -> dict:
    baseline_agent = _make_agent(
        checkpoint=args.checkpoint,
        iterations=args.agent_iterations,
        seed=seed,
        use_tier_a=False,
        tier_a_width=args.tier_a_width,
    )
    tier_a_agent = _make_agent(
        checkpoint=args.checkpoint,
        iterations=args.agent_iterations,
        seed=seed,
        use_tier_a=True,
        tier_a_width=args.tier_a_width,
    )
    baseline_results = run_ladder(
        make_choose_action(baseline_agent),
        opponents,
        n_games=args.games,
        seed=seed,
    )
    tier_a_results = run_ladder(
        make_choose_action(tier_a_agent),
        opponents,
        n_games=args.games,
        seed=seed,
    )
    baseline_report = gate_report(baseline_results)
    tier_a_report = gate_report(tier_a_results)
    return {
        "seed": seed,
        "baseline": baseline_report,
        "tier_a": tier_a_report,
        "delta": _summarize_delta(baseline_report, tier_a_report, opponents),
        "tier_a_evaluator_stats": _tier_a_stats(tier_a_agent),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--agent-iterations", type=int, default=50)
    parser.add_argument("--games", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--seeds",
        default=None,
        help="Comma-separated seeds. Overrides --seed and aggregates all runs.",
    )
    parser.add_argument("--tier-a-width", type=float, default=0.0)
    parser.add_argument(
        "--opponents",
        default="random,safe,baku_lsr_engineering,pattern_reader",
    )
    parser.add_argument(
        "--out",
        default=str(Path("checkpoints") / "tier_a_runtime_compare" / "report.json"),
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
        f"Comparing checkpoint={args.checkpoint} iterations={args.agent_iterations} "
        f"games={args.games}/opponent seeds={seeds}"
    )
    print(f"Opponents: {', '.join(opponents)}")

    start = time.time()
    seed_reports = []
    for seed in seeds:
        print(f"  seed {seed}...", flush=True)
        seed_reports.append(_run_one_seed(args, opponents, seed))
        print(f"    delta {seed_reports[-1]['delta']['overall']}", flush=True)
    aggregate = _aggregate_reports(seed_reports, opponents)

    report = {
        "config": vars(args),
        "seeds": seeds,
        "opponents": opponents,
        "seed_reports": seed_reports,
        "aggregate": aggregate,
        # Backward-compatible top-level aliases for one-seed reports.
        "baseline": seed_reports[0]["baseline"],
        "tier_a": seed_reports[0]["tier_a"],
        "delta": seed_reports[0]["delta"],
        "elapsed_seconds": round(time.time() - start, 2),
    }
    with out_path.open("w", encoding="utf-8") as fh:
        json.dump(_json_safe(report), fh, indent=2, allow_nan=False)

    print(f"Aggregate baseline overall: {aggregate['baseline']}")
    print(f"Aggregate Tier A overall  : {aggregate['tier_a']}")
    print(f"Aggregate delta           : {aggregate['delta']}")
    print(f"Report: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
