#!/usr/bin/env python3
"""Generate policy_guard targets from scripted-opponent trace states."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from environment.opponents.factory import create_scripted_opponent
from hal.agent import SolverAgent
from scripts.run_tier_a_frontier_event import _split_csv
from training.strength.match_gate import _opponent_seed
from training.opponent_trace_guard import generate_opponent_trace_guard_records
from training.policy_guard import save_policy_guard_records
from training.value_targets import source_breakdown


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe-checkpoint", required=True)
    parser.add_argument("--label-checkpoint", required=True)
    parser.add_argument("--opponent", required=True)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--policy-ensemble-size", type=int, default=1)
    parser.add_argument("--policy-uniform-mix", type=float, default=0.0)
    parser.add_argument(
        "--resolve-at-critical",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use SolverAgent critical-state subgame resolve while labeling policies.",
    )
    parser.add_argument("--resolve-horizon", type=int, default=3)
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--games", type=int, default=10)
    parser.add_argument("--max-states", type=int, default=128)
    parser.add_argument("--min-history", type=int, default=0)
    parser.add_argument("--out", required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    seeds = [int(item) for item in _split_csv(args.seeds)]
    records = []
    total_games = 0
    total_hal_wins = 0
    total_baku_wins = 0
    total_draws = 0
    for seed in seeds:
        if len(records) >= args.max_states:
            break
        probe_agent = SolverAgent(
            args.probe_checkpoint,
            player_name="Hal",
            iterations=args.iterations,
            seed=seed,
            policy_ensemble_size=args.policy_ensemble_size,
            policy_uniform_mix=args.policy_uniform_mix,
            resolve_at_critical=args.resolve_at_critical,
            resolve_horizon=args.resolve_horizon,
        )
        label_agent = SolverAgent(
            args.label_checkpoint,
            player_name="Hal",
            iterations=args.iterations,
            seed=seed,
            policy_ensemble_size=args.policy_ensemble_size,
            policy_uniform_mix=args.policy_uniform_mix,
            resolve_at_critical=args.resolve_at_critical,
            resolve_horizon=args.resolve_horizon,
        )
        seed_records, summary = generate_opponent_trace_guard_records(
            probe_agent=probe_agent,
            label_agent=label_agent,
            label_checkpoint=args.label_checkpoint,
            opponent_name=args.opponent,
            opponent_factory=lambda base_seed, name=args.opponent: create_scripted_opponent(
                name,
                seed=_opponent_seed(base_seed, name),
            ),
            seeds=[seed],
            games_per_seed=args.games,
            max_states=args.max_states - len(records),
            min_history=args.min_history,
        )
        records.extend(seed_records)
        total_games += summary.games
        total_hal_wins += summary.hal_wins
        total_baku_wins += summary.baku_wins
        total_draws += summary.draws
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    save_policy_guard_records(records, out)
    print(
        f"Wrote {len(records)} policy_guard targets to {out}; "
        f"breakdown={source_breakdown([record.target for record in records])}; "
        f"trace={{'opponent': {args.opponent!r}, 'games': {total_games}, "
        f"'hal_wins': {total_hal_wins}, 'baku_wins': {total_baku_wins}, "
        f"'draws': {total_draws}}}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
