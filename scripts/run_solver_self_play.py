#!/usr/bin/env python3
"""Generate solver-MCTS self-play targets compatible with train_value_net.

The legacy ``hal.self_play`` path is intentionally not used here. This script
labels visited public states with ``SolverAgent`` root average policies and
root values, then saves the rows under source ``self_play_mcts``.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training.solver_self_play import (
    generate_solver_self_play_records,
    save_self_play_records,
)
from training.value_targets import source_breakdown


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hal-checkpoint", required=True)
    parser.add_argument(
        "--opponent-checkpoint",
        default=None,
        help="Optional Baku SolverAgent checkpoint. If omitted, --opponent names a scripted opponent.",
    )
    parser.add_argument("--opponent", default="safe")
    parser.add_argument("--games", type=int, default=4)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-half-rounds", type=int, default=200)
    parser.add_argument("--out", required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    records = generate_solver_self_play_records(
        hal_checkpoint=args.hal_checkpoint,
        opponent_checkpoint=args.opponent_checkpoint,
        opponent_name=args.opponent,
        games=args.games,
        iterations=args.iterations,
        seed=args.seed,
        max_half_rounds=args.max_half_rounds,
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    save_self_play_records(records, out)
    print(
        f"Wrote {len(records)} solver self-play rows to {out}; "
        f"breakdown={source_breakdown([record.target for record in records])}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
