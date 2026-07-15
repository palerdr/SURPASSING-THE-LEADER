"""Narrow command-line entry points for the standalone ToySTL pipeline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from stl.toy.artifacts import canonical_json
from stl.toy.mcts import (
    conformance_report,
    conformance_summary,
    make_tablebase_evaluator,
    run_conformance,
)
from stl.toy.rules import ruleset_for_name
from stl.toy.self_play import ToySelfPlayConfig, generate_self_play, write_self_play
from stl.toy.tablebase import build_tablebase, load_tablebase, write_tablebase
from stl.toy.targets import build_exact_targets, write_exact_targets
from stl.toy.train import ToyTrainConfig, load_toy_checkpoint, train_exact_targets


def _rules(args: argparse.Namespace):
    return ruleset_for_name(args.ruleset, max_half_rounds=args.max_half_rounds)


def _output_dir(args: argparse.Namespace, ruleset_id: str) -> Path:
    return Path(args.output_dir) if args.output_dir else Path("outputs") / "toy" / ruleset_id


def _parse_ints(value: str) -> tuple[int, ...]:
    result = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not result:
        raise argparse.ArgumentTypeError("expected a comma-separated integer list")
    return result


def _audit_roots(rules) -> tuple:
    if rules.ruleset_id != "bucket12_fixed50":
        return (rules.initial_state(),)
    levels = (0, 15, 30, 45, 59)
    return tuple(
        rules.initial_state().with_updates(
            hal_load=hal_load,
            baku_load=baku_load,
            role_phase=phase,
        )
        for phase in (0, 1)
        for hal_load, baku_load in (
            (0, 0),
            (levels[1], levels[1]),
            (levels[2], levels[3]),
            (levels[4], levels[2]),
        )
    )


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(canonical_json(payload) + "\n", encoding="utf-8")


def command_exact(args: argparse.Namespace) -> int:
    rules = _rules(args)
    tablebase = build_tablebase(rules, max_horizon=args.max_horizon)
    npz_path, manifest_path, manifest = write_tablebase(
        tablebase,
        _output_dir(args, rules.ruleset_id),
    )
    print(canonical_json({"npz": str(npz_path), "manifest": str(manifest_path), "sha256": manifest["npz_sha256"]}))
    return 0


def command_targets(args: argparse.Namespace) -> int:
    rules = _rules(args)
    output_dir = _output_dir(args, rules.ruleset_id)
    tablebase_npz = output_dir / "tablebase.npz"
    tablebase_manifest = output_dir / "tablebase.json"
    tablebase = (
        load_tablebase(tablebase_npz, tablebase_manifest)
        if tablebase_npz.exists() and tablebase_manifest.exists()
        else None
    )
    targets = build_exact_targets(rules, tablebase=tablebase)
    npz_path, manifest_path, manifest = write_exact_targets(targets, output_dir)
    print(canonical_json({"npz": str(npz_path), "manifest": str(manifest_path), "sha256": manifest["npz_sha256"]}))
    return 0


def command_train(args: argparse.Namespace) -> int:
    rules = _rules(args)
    output_dir = Path(args.output_dir) if args.output_dir else Path("outputs") / "toy" / rules.ruleset_id / "checkpoint"
    result = train_exact_targets(
        args.targets,
        args.target_manifest,
        rules,
        output_dir,
        config=ToyTrainConfig(
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            seed=args.seed,
            hidden_dim=args.hidden_dim,
        ),
    )
    print(canonical_json({
        "checkpoint": str(result.checkpoint_path),
        "manifest": str(result.checkpoint_manifest_path),
        "best_epoch": result.best_epoch,
        "best_validation_loss": result.best_validation_loss,
    }))
    return 0


def command_mcts_audit(args: argparse.Namespace) -> int:
    rules = _rules(args)
    output_dir = _output_dir(args, rules.ruleset_id)
    tablebase_npz = output_dir / "tablebase.npz"
    tablebase_manifest = output_dir / "tablebase.json"
    evaluator = None
    if tablebase_npz.exists() and tablebase_manifest.exists():
        evaluator = make_tablebase_evaluator(
            load_tablebase(tablebase_npz, tablebase_manifest),
            rules,
        )
    records = run_conformance(
        rules,
        _audit_roots(rules),
        horizons=args.horizons,
        budgets=args.budgets,
        seeds=tuple(range(args.seed_start, args.seed_start + args.seed_count)),
        evaluator=evaluator,
    )
    report = conformance_report(records)
    report_path = output_dir / "mcts_conformance.json"
    _write_json(report_path, report)
    summary = conformance_summary(records)
    print(canonical_json({
        "report": str(report_path),
        "records": len(records),
        "median_value_error": summary["median_value_error"],
        "p95_value_error": summary["p95_value_error"],
        "gates": summary["gates"],
        "report_sha256": report["report_sha256"],
    }))
    return 0


def command_self_play(args: argparse.Namespace) -> int:
    rules = _rules(args)
    model = load_toy_checkpoint(args.checkpoint, rules)
    config = ToySelfPlayConfig(
        games=args.games,
        seed=args.seed,
        action_temperature=args.action_temperature,
        root_noise_epsilon=args.root_noise_epsilon,
        root_dirichlet_alpha_scale=args.root_dirichlet_alpha_scale,
        mcts_iterations=args.mcts_iterations,
        exploration_c=args.exploration_c,
    )
    result = generate_self_play(
        rules,
        model,
        config=config,
        checkpoint_path=args.checkpoint,
    )
    npz_path, manifest_path, manifest = write_self_play(result, _output_dir(args, rules.ruleset_id))
    print(canonical_json({"npz": str(npz_path), "manifest": str(manifest_path), "sha256": manifest["npz_sha256"], "trajectory_sha256": result.metadata["trajectory_sha256"]}))
    return 0


def _common_rules(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--ruleset", default="bucket12_fixed50")
    parser.add_argument("--max-half-rounds", type=int, default=8)
    parser.add_argument("--output-dir")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m stl.toy.cli")
    subparsers = parser.add_subparsers(dest="command", required=True)

    exact = subparsers.add_parser("exact", help="build the exhaustive v0 tablebase")
    _common_rules(exact)
    exact.add_argument("--max-horizon", type=int)
    exact.set_defaults(function=command_exact)

    targets = subparsers.add_parser("targets", help="write exact supervised targets")
    _common_rules(targets)
    targets.set_defaults(function=command_targets)

    train = subparsers.add_parser("train", help="train the tiny policy/value network")
    _common_rules(train)
    train.add_argument("--targets", required=True)
    train.add_argument("--target-manifest", required=True)
    train.add_argument("--epochs", type=int, default=100)
    train.add_argument("--batch-size", type=int, default=256)
    train.add_argument("--learning-rate", type=float, default=1e-3)
    train.add_argument("--hidden-dim", type=int, default=64)
    train.add_argument("--seed", type=int, default=4)
    train.set_defaults(function=command_train)

    audit = subparsers.add_parser("mcts-audit", help="compare MCTS with exact leaf values")
    _common_rules(audit)
    audit.add_argument("--horizons", type=_parse_ints, default=(1, 4, 8))
    audit.add_argument("--budgets", type=_parse_ints, default=(64, 256, 1024))
    audit.add_argument("--seed-start", type=int, default=0)
    audit.add_argument("--seed-count", type=int, default=10)
    audit.set_defaults(function=command_mcts_audit)

    self_play = subparsers.add_parser("self-play", help="generate MCTS replay rows")
    _common_rules(self_play)
    self_play.add_argument("--checkpoint", required=True)
    self_play.add_argument("--games", type=int, default=4)
    self_play.add_argument("--seed", type=int, default=4)
    self_play.add_argument("--action-temperature", type=float, default=1.0)
    self_play.add_argument("--root-noise-epsilon", type=float, default=0.25)
    self_play.add_argument("--root-dirichlet-alpha-scale", type=float, default=10.0)
    self_play.add_argument("--mcts-iterations", type=int, default=256)
    self_play.add_argument("--exploration-c", type=float, default=1.0)
    self_play.set_defaults(function=command_self_play)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.function(args))


if __name__ == "__main__":
    raise SystemExit(main())
