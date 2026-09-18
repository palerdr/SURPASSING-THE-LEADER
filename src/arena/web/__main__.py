"""Run the local browser server: ``uv run python -m arena.web``.

Hal is built once here, at startup, using the same provider construction and
the same agent options the terminal CLI uses. The ``abstract`` provider is
refused because it can build a tablebase from scratch, which must never happen
behind an HTTP request.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import uvicorn

from arena import cli
from arena.session import validate_human_display_name
from arena.web.app import DEFAULT_WEBCLIENT_DIST, SeriesConfig, SessionConfig, create_app
from stl.engine.game import OPENING_START_CLOCK

# Perfect and PM Hal are pure-DTH policies with no action-61 contract.
PURE_DTH_ONLY = frozenset({"perfect-hal", "pm-hal"})


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m arena.web")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--hal-agent",
        choices=("dth", "adaptive-dth", "exploit-hal", "perfect-hal", "pm-hal"),
        default="dth",
        help="'abstract' is unavailable here: it may build a tablebase on first use",
    )
    cli._add_agent_arguments(parser)
    parser.add_argument("--human-name", default="Baku")
    parser.add_argument(
        "--public-hal-label",
        default=None,
        help="optional display label that conceals the provider implementation",
    )
    parser.add_argument(
        "--conceal-hal-details",
        action="store_true",
        help="record provider summary and diagnostics without serving them",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="base RNG seed; game N of the series uses seed + N, as arena play does",
    )
    parser.add_argument("--start-clock", type=int, default=OPENING_START_CLOCK)
    parser.add_argument("--max-half-rounds", type=int, default=None)
    parser.add_argument(
        "--pure-dth",
        action="store_true",
        help="permanent literal actions 1..60; required for Perfect and PM Hal",
    )
    parser.add_argument(
        "--transcript",
        default=None,
        help="optional JSON path, rewritten after every finished game",
    )
    parser.add_argument(
        "--webclient-dist",
        default=str(DEFAULT_WEBCLIENT_DIST),
        help="built browser client to serve at /; run npm run build to create it",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    args.human_name = validate_human_display_name(args.human_name)
    if args.hal_agent in PURE_DTH_ONLY and not args.pure_dth:
        raise SystemExit(
            f"{args.hal_agent} is a pure-DTH policy; pass --pure-dth so action "
            "61 is impossible"
        )

    app = create_app(
        hal_factory=lambda: cli._make_hal(args),
        config=SessionConfig(
            human_name=args.human_name,
            seed=args.seed,
            start_clock=args.start_clock,
            max_half_rounds=args.max_half_rounds,
            pure_dth=bool(args.pure_dth),
        ),
        series=SeriesConfig(
            hal_agent=args.hal_agent,
            public_hal_label=args.public_hal_label,
            conceal_hal_details=bool(args.conceal_hal_details),
            transcript_path=Path(args.transcript) if args.transcript else None,
        ),
        webclient_dist=Path(args.webclient_dist),
    )
    print(f"Surpassing The Leader — http://{args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
