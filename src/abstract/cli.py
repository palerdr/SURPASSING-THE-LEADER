"""Exact-only command-line entry point for the abstract project."""

from __future__ import annotations

import argparse

from pathlib import Path

from abstract.artifacts import canonical_json
from abstract.packed_tablebase import PackedTablebase, build_packed_tablebase
from abstract.rules import ruleset_for_name
from abstract.tablebase import build_tablebase, write_tablebase


def _output_dir(args: argparse.Namespace, ruleset_id: str) -> Path:
    return (
        Path(args.output_dir)
        if args.output_dir
        else Path("src") / "abstract" / "outputs" / ruleset_id
    )


def command_exact(args: argparse.Namespace) -> int:
    rules = ruleset_for_name(args.ruleset)
    packed = (
        args.packed
        or rules.ruleset_id
        in {
            "bucket6_unified80",
            "bucket12_unified80",
            "bucket6_frozen95",
            "bucket12_frozen95",
        }
        or rules.action_size > 6
    )
    if packed:
        output_dir = _output_dir(args, rules.ruleset_id)
        print(
            "Building or resuming the packed exhaustive abstract tablebase. "
            "Progress is checkpointed; Press Control-C to stop safely.",
            flush=True,
        )
        manifest_path = build_packed_tablebase(
            rules,
            output_dir,
            checkpoint_states=args.checkpoint_states,
            backend=args.backend,
        )
        loaded = PackedTablebase(output_dir, verify_hashes=False)
        print(
            canonical_json(
                {
                    "artifact_dir": str(output_dir),
                    "manifest": str(manifest_path),
                    "reachable_state_count": loaded.manifest["metadata"][
                        "reachable_state_count"
                    ],
                    "schema_version": loaded.manifest["schema_version"],
                }
            )
        )
        return 0

    print(
        "Building the exhaustive abstract tablebase. This is a one-time operation and can take a few minutes. "
        "Press Control-C to cancel.",
        flush=True,
    )
    tablebase = build_tablebase(rules)
    npz_path, manifest_path, manifest = write_tablebase(tablebase, _output_dir(args, rules.ruleset_id))
    print(canonical_json({
        "npz": str(npz_path),
        "manifest": str(manifest_path),
        "sha256": manifest["npz_sha256"],
        "reachable_state_count": tablebase["metadata"]["reachable_state_count"],
    }))
    return 0


def command_lookup(args: argparse.Namespace) -> int:
    tablebase = PackedTablebase(Path(args.artifact_dir))
    row = tablebase.lookup(tuple(args.state))
    row["drop_policy"] = row["drop_policy"].tolist()
    row["check_policy"] = row["check_policy"].tolist()
    print(canonical_json(row))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m abstract")
    subparsers = parser.add_subparsers(dest="command", required=True)
    exact = subparsers.add_parser("exact", help="build the exhaustive role-relative terminal tablebase")
    exact.add_argument("--ruleset", default="bucket6_frozen95")
    exact.add_argument("--output-dir")
    exact.add_argument(
        "--packed",
        action="store_true",
        help="use the resumable packed v3 builder (automatic for production rulesets)",
    )
    exact.add_argument(
        "--checkpoint-states",
        type=int,
        default=10_000,
        help="number of reachability/backup rows between durable checkpoints",
    )
    exact.add_argument(
        "--backend",
        choices=("auto", "python", "rust"),
        default="auto",
        help="packed hot-loop backend; auto uses Rust when the parity-versioned extension is installed",
    )
    exact.set_defaults(function=command_exact)

    lookup = subparsers.add_parser(
        "lookup",
        help="look up a packed tablebase row and derive its state SHA ID",
    )
    lookup.add_argument("artifact_dir")
    lookup.add_argument(
        "state",
        nargs=4,
        type=int,
        metavar=("CHECKER_LOAD", "CHECKER_TTD", "DROPPER_LOAD", "DROPPER_TTD"),
    )
    lookup.set_defaults(function=command_lookup)
    return parser


def main(argv: list[str] | None = None) -> int:
    try:
        args = build_parser().parse_args(argv)
        return int(args.function(args))
    except KeyboardInterrupt:
        print("\nCancelled.", flush=True)
        return 130
