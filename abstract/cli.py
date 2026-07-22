"""Exact-only command-line entry point for the abstract project."""

from __future__ import annotations

import argparse

from pathlib import Path

from abstract.artifacts import canonical_json
from abstract.rules import ruleset_for_name
from abstract.tablebase import build_tablebase, write_tablebase


def _output_dir(args: argparse.Namespace, ruleset_id: str) -> Path:
    return Path(args.output_dir) if args.output_dir else Path("abstract") / "outputs" / ruleset_id


def command_exact(args: argparse.Namespace) -> int:
    rules = ruleset_for_name(args.ruleset)
    tablebase = build_tablebase(rules)
    npz_path, manifest_path, manifest = write_tablebase(tablebase, _output_dir(args, rules.ruleset_id))
    print(canonical_json({
        "npz": str(npz_path),
        "manifest": str(manifest_path),
        "sha256": manifest["npz_sha256"],
        "reachable_state_count": tablebase["metadata"]["reachable_state_count"],
    }))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m abstract")
    subparsers = parser.add_subparsers(dest="command", required=True)
    exact = subparsers.add_parser("exact", help="build the exhaustive role-relative terminal tablebase")
    exact.add_argument("--ruleset", default="bucket6_ttd_curve95")
    exact.add_argument("--output-dir")
    exact.set_defaults(function=command_exact)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.function(args))
