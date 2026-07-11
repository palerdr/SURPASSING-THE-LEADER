"""Generate reconstructable V2 Generation-Zero anchor and ruler shards."""

from __future__ import annotations

import argparse
from pathlib import Path

from stl.learning.contracts import config_digest
from stl.learning.replay import ShardRole, save_replay_shard
from stl.learning.targets import (
    HOLDOUT_TERMINAL_CONFIGS,
    TRAINING_TERMINAL_CONFIGS,
    _generate_tablebase_targets,
    _generate_terminal_targets,
    generate_holdout_targets,
    generate_targets,
    source_breakdown,
    to_training_record_v2,
)
from stl.solver.exact import ExactSearchConfig


def _search_digest() -> str:
    return config_digest(
        {
            "perspective": "Hal",
            "matrix_solver": "lp",
            "action_mode": "full_width",
            "target_contract": "terminal/tablebase/exact-horizon-v2",
        }
    )


def build_gen0_targets(*, holdout: bool, workers: int, smoke: bool):
    config = ExactSearchConfig()
    if holdout and not smoke:
        return generate_holdout_targets(config=config, workers=workers)

    if smoke:
        anchors = generate_targets(
            baku_cylinder_grid=(299.0,),
            hal_cylinder_grid=(0.0,),
            clock_grid=(3540.0,),
            half_grid=(1, 2),
            deaths_grid=(0,),
            cpr_grid=(0,),
            config=config,
            workers=workers,
        )
        terminal_configs = (
            HOLDOUT_TERMINAL_CONFIGS[:2]
            if holdout
            else TRAINING_TERMINAL_CONFIGS[:2]
        )
        anchors.extend(_generate_terminal_targets(config, terminal_configs))
        anchors.extend(_generate_tablebase_targets(config, split_interior=holdout))
        return anchors

    anchors = generate_targets(config=config, workers=workers)
    anchors.extend(_generate_terminal_targets(config, TRAINING_TERMINAL_CONFIGS))
    anchors.extend(_generate_tablebase_targets(config, split_interior=False))
    return anchors


def write_gen0_shard(
    path: str | Path,
    *,
    holdout: bool,
    workers: int,
    smoke: bool,
) -> dict:
    targets = build_gen0_targets(holdout=holdout, workers=workers, smoke=smoke)
    digest = _search_digest()
    records = [
        to_training_record_v2(
            target,
            search_config_digest=digest,
            rng_seeds={"generation": 0},
        )
        for target in targets
    ]
    manifest = save_replay_shard(
        records,
        path,
        shard_role=ShardRole.EXTERNAL_RULER if holdout else ShardRole.REPLAY,
    )
    return {
        "path": str(path),
        "records": len(records),
        "sources": source_breakdown(targets),
        "manifest": manifest,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--train-out",
        default="outputs/regen2rl/gen0_train_v2.npz",
    )
    parser.add_argument(
        "--ruler-out",
        default="outputs/regen2rl/gen0_external_ruler_v2.npz",
    )
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument(
        "--smoke",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use a two-state structural fixture instead of the full anchor grids.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.workers <= 0:
        raise ValueError("workers must be positive")
    train = write_gen0_shard(
        args.train_out,
        holdout=False,
        workers=args.workers,
        smoke=args.smoke,
    )
    ruler = write_gen0_shard(
        args.ruler_out,
        holdout=True,
        workers=args.workers,
        smoke=args.smoke,
    )
    print(f"Gen-0 train: {train['records']} records -> {train['path']}")
    print(f"External ruler: {ruler['records']} records -> {ruler['path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
