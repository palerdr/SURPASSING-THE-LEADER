"""Run the local browser server: ``uv run python -m arena.web``.

Hal is built once here, at startup, using the same provider construction the
terminal CLI uses. The ``abstract`` provider is refused because it can build a
tablebase from scratch, which must never happen behind an HTTP request.
"""

from __future__ import annotations

import argparse

import uvicorn

from arena.web.app import SessionConfig, create_app
from stl.engine.game import OPENING_START_CLOCK


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m arena.web")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--hal-agent",
        choices=("dth", "adaptive-dth", "exploit-hal", "stl-mcts"),
        default="dth",
        help="'abstract' is unavailable here: it may build a tablebase on first use",
    )
    parser.add_argument("--human-name", default="Baku")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--start-clock", type=int, default=OPENING_START_CLOCK)
    parser.add_argument("--max-half-rounds", type=int, default=None)
    parser.add_argument("--dth-complete-tablebase", default=None)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--exploit-hal-config", default=None)
    parser.add_argument("--exploit-hal-checkpoint", default=None)
    parser.add_argument("--adaptive-prior-json", default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    def hal_factory():
        from arena import cli

        play_args = cli.build_parser().parse_args(
            ["play", "--hal-agent", args.hal_agent, "--skip-rules"]
        )
        for name in (
            "dth_complete_tablebase",
            "checkpoint",
            "iterations",
            "exploit_hal_config",
            "exploit_hal_checkpoint",
            "adaptive_prior_json",
        ):
            value = getattr(args, name, None)
            if value is not None:
                setattr(play_args, name, value)
        play_args.seed = args.seed
        return cli._make_hal(play_args)

    app = create_app(
        hal_factory=hal_factory,
        config=SessionConfig(
            human_name=args.human_name,
            seed=args.seed,
            start_clock=args.start_clock,
            max_half_rounds=args.max_half_rounds,
        ),
    )
    print(f"Surpassing The Leader — http://{args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
