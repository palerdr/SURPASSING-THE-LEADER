"""Research match command: ``python -m hal_lab match``.

``match`` plays a paired-seat agent-versus-agent series with one predeclared
SPRT through :mod:`hal_lab.harness.series`. It offers every play agent and the
research-only Aggro Hal. This module owns the Aggro Hal flags and factory, and
the factory calls the public ``arena.policies.aggro_hal.make_live_provider``.
The terminal and the browser build Hal through ``arena.policies.registry``, and
this command builds every play agent through the same registry.
"""

from __future__ import annotations

import argparse

from arena.policies import registry
from hal_lab.harness.series import run_paired_series, write_report
from stl.engine.game import OPENING_START_CLOCK

# Agent-versus-agent series offer every play agent and the research-only Aggro
# Hal. make_aggro_hal_provider builds Aggro Hal from the flags in
# add_research_arguments.
MATCH_AGENTS = (
    "abstract",
    "dth",
    "adaptive-dth",
    "exploit-hal",
    "aggro-hal",
    "perfect-hal",
    "pm-hal",
)


def make_aggro_hal_provider(args: argparse.Namespace):
    if not args.aggro_hal_checkpoint:
        raise ValueError("--aggro-hal-checkpoint is required for --hal-agent aggro-hal")
    from arena.policies.aggro_hal import make_live_provider

    return make_live_provider(
        artifact_dir=registry.dth_artifact_dir(args),
        checkpoint=args.aggro_hal_checkpoint,
        device=args.aggro_hal_device,
        fast_adaptation=args.aggro_hal_fast_adaptation,
    )


def make_provider(kind: str, args: argparse.Namespace):
    """Build Aggro Hal here and every play agent through the registry."""

    if kind == "aggro-hal":
        return make_aggro_hal_provider(args)
    return registry.make_provider(kind, args)


# The tests use this name; command_match looks it up here at call time, so a
# monkeypatch of this module still intercepts it.
_make_provider = make_provider


def add_research_arguments(parser: argparse.ArgumentParser) -> None:
    """Add the options of Aggro Hal, which only agent-versus-agent series offer."""

    parser.add_argument(
        "--aggro-hal-checkpoint",
        default=None,
        help="required direct recurrent Aggro Hal checkpoint",
    )
    parser.add_argument(
        "--aggro-hal-device",
        choices=("cpu", "cuda"),
        default="cpu",
        help="explicit Aggro Hal inference device; defaults to CPU",
    )
    parser.add_argument(
        "--aggro-hal-fast-adaptation",
        action="store_true",
        help="blend concentrated public action evidence into Aggro Hal's forecast",
    )


def command_match(args: argparse.Namespace) -> int:
    if (
        any(
            registry.requires_pure_dth(kind, args)
            for kind in (args.candidate, args.opponent)
        )
        and not args.pure_dth
    ):
        raise ValueError(
            "aggro-hal, perfect-hal, and pm-hal are pure-DTH policies; pass "
            "--pure-dth so action 61 is impossible"
        )
    report = run_paired_series(
        args.candidate,
        args.opponent,
        make_candidate=lambda: _make_provider(args.candidate, args),
        make_opponent=lambda: _make_provider(args.opponent, args),
        base_seeds=args.games,
        seed_start=args.seed,
        start_clock=args.start_clock,
        max_half_rounds=args.max_half_rounds,
        pure_dth=args.pure_dth,
    )
    destination = write_report(report, args.output)
    sprt = report["sprt"]
    print(
        f"{args.candidate} vs {args.opponent}: "
        f"{sprt['wins']}-{sprt['losses']} decisive "
        f"({report['stopped_games']} stopped), SPRT {sprt['decision']}; "
        f"report {destination}"
    )
    for line in report["candidate_summaries"]:
        print(line)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m hal_lab")
    commands = parser.add_subparsers(dest="command", required=True)
    match = commands.add_parser(
        "match",
        help="paired-seat agent-versus-agent series with a predeclared SPRT",
    )
    match.add_argument("--candidate", choices=MATCH_AGENTS, required=True)
    match.add_argument("--opponent", choices=MATCH_AGENTS, required=True)
    registry.add_play_arguments(match)
    add_research_arguments(match)
    match.add_argument(
        "--games",
        type=int,
        default=50,
        help="maximum base seeds; each is played in both seatings",
    )
    match.add_argument("--seed", type=int, default=0)
    match.add_argument("--start-clock", type=int, default=OPENING_START_CLOCK)
    match.add_argument("--max-half-rounds", type=int, default=200)
    match.add_argument(
        "--pure-dth",
        action="store_true",
        help=(
            "run the pure 1..60 DTH action contract "
            "(required for aggro-hal, perfect-hal, and pm-hal)"
        ),
    )
    match.add_argument(
        "--output",
        required=True,
        help="JSON report path; keep it under the candidate project's artifacts",
    )
    match.set_defaults(function=command_match)
    return parser


def main(argv: list[str] | None = None) -> int:
    try:
        args = build_parser().parse_args(argv)
        return int(args.function(args))
    except KeyboardInterrupt:
        print("\nExited.", flush=True)
        return 130
