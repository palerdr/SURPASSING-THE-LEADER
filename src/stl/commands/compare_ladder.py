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

from stl.play.agent import DEFAULT_CHECKPOINT, SolverAgent, make_choose_action
from stl.learning.strength import gate_report, run_ladder


def _empty_counts() -> dict:
    return {"games": 0, "wins": 0, "draws": 0, "losses": 0}


def _counts_summary(counts: dict) -> dict:
    games = max(1, int(counts["games"]))
    return {
        **counts,
        "win_rate": counts["wins"] / games,
        "score_rate": (counts["wins"] + 0.5 * counts["draws"]) / games,
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


def _make_agent(
    checkpoint: str,
    *,
    iterations: int,
    seed: int,
    policy_ensemble_size: int = 1,
    policy_uniform_mix: float = 0.0,
    search_prior_uniform_mix: float = 0.0,
    resolve_at_critical: bool = False,
    resolve_horizon: int = 3,
) -> SolverAgent:
    return SolverAgent(
        checkpoint,
        player_name="Hal",
        iterations=iterations,
        seed=seed,
        policy_ensemble_size=policy_ensemble_size,
        policy_uniform_mix=policy_uniform_mix,
        search_prior_uniform_mix=search_prior_uniform_mix,
        resolve_at_critical=resolve_at_critical,
        resolve_horizon=resolve_horizon,
    )


def _run_checkpoint_report(
    args,
    checkpoint: str,
    opponents: list[str],
    seed: int,
    *,
    search_prior_uniform_mix: float = 0.0,
) -> dict:
    # Each opponent rung gets a fresh SolverAgent. Otherwise the stochastic
    # action stream consumed by earlier rungs changes later-rung results,
    # making diagnostics depend on opponent order.
    results = {}
    for opponent in opponents:
        agent = _make_agent(
            checkpoint,
            iterations=args.agent_iterations,
            seed=seed,
            policy_ensemble_size=args.policy_ensemble_size,
            policy_uniform_mix=args.policy_uniform_mix,
            search_prior_uniform_mix=search_prior_uniform_mix,
            resolve_at_critical=args.resolve_at_critical,
            resolve_horizon=args.resolve_horizon,
        )
        results.update(
            run_ladder(
                make_choose_action(agent),
                [opponent],
                n_games=args.games,
                seed=seed,
            )
        )
    return gate_report(results)


def _run_one_seed(args, opponents: list[str], seed: int) -> dict:
    champion_report = _run_checkpoint_report(
        args,
        args.champion_checkpoint,
        opponents,
        seed,
        search_prior_uniform_mix=(
            args.search_prior_uniform_mix
            if args.champion_search_prior_uniform_mix is None
            else args.champion_search_prior_uniform_mix
        ),
    )
    candidate_report = _run_checkpoint_report(
        args,
        args.candidate_checkpoint,
        opponents,
        seed,
        search_prior_uniform_mix=(
            args.search_prior_uniform_mix
            if args.candidate_search_prior_uniform_mix is None
            else args.candidate_search_prior_uniform_mix
        ),
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
    parser.add_argument(
        "--policy-ensemble-size",
        type=int,
        default=1,
        help="Average this many independent state-seeded MCTS root policies per policy query.",
    )
    parser.add_argument(
        "--policy-uniform-mix",
        type=float,
        default=0.0,
        help="Blend each deployed root policy with this much uniform mass over its support.",
    )
    parser.add_argument("--search-prior-uniform-mix", type=float, default=0.0)
    parser.add_argument("--champion-search-prior-uniform-mix", type=float, default=None)
    parser.add_argument("--candidate-search-prior-uniform-mix", type=float, default=None)
    parser.add_argument(
        "--resolve-at-critical",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use SolverAgent critical-state subgame resolve during deployed search.",
    )
    parser.add_argument(
        "--resolve-horizon",
        type=int,
        default=3,
        help="Selective-solve horizon for --resolve-at-critical.",
    )
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
        f"iterations={args.agent_iterations} policy_ensemble_size={args.policy_ensemble_size} "
        f"policy_uniform_mix={args.policy_uniform_mix:g} "
        f"search_prior_uniform_mix={args.search_prior_uniform_mix:g} "
        f"champion_search_prior_uniform_mix="
        f"{args.champion_search_prior_uniform_mix if args.champion_search_prior_uniform_mix is not None else args.search_prior_uniform_mix:g} "
        f"candidate_search_prior_uniform_mix="
        f"{args.candidate_search_prior_uniform_mix if args.candidate_search_prior_uniform_mix is not None else args.search_prior_uniform_mix:g} "
        f"resolve_at_critical={args.resolve_at_critical} "
        f"resolve_horizon={args.resolve_horizon} "
        f"games={args.games}/opponent seeds={seeds}"
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
