"""Interactive canonical STL referee with pluggable Hal policy providers."""

from __future__ import annotations

import argparse
from pathlib import Path

from arena.abstract_adapter import AbstractTablebasePolicyProvider
from arena.agent import PolicyDrivenAgent
from stl.engine.actions import validate_action
from stl.engine.game import OPENING_START_CLOCK, PHYSICALITY_BAKU, PHYSICALITY_HAL, Game, Player, Referee


def _abstract_artifact(args: argparse.Namespace) -> tuple[Path, str]:
    if args.buckets == 5:
        ruleset_id = "bucket12_unified80"
        default = Path("abstract/outputs") / ruleset_id
    else:
        ruleset_id = "bucket6_unified80"
        default = Path("abstract/outputs") / ruleset_id
    return (
        Path(args.abstract_tablebase) if args.abstract_tablebase else default,
        ruleset_id,
    )


def _human_action(*, actor: str, role: str, legal: tuple[int, ...]) -> int:
    allowed = f"{legal[0]}-{legal[-1]}" if legal == tuple(range(legal[0], legal[-1] + 1)) else str(legal)
    while True:
        try:
            action = int(input(f"{actor} ({role}) choose second [{allowed}]: "))
        except ValueError:
            print("Enter a legal integer second.")
            continue
        except (EOFError, KeyboardInterrupt):
            raise KeyboardInterrupt from None
        if action in legal:
            return action
        print(f"Legal seconds: {allowed}")


def _make_hal(args: argparse.Namespace) -> PolicyDrivenAgent:
    if args.hal_agent == "abstract":
        tablebase_path, ruleset_id = _abstract_artifact(args)
        packed_artifact = tablebase_path.is_dir() or tablebase_path.suffix != ".npz"
        if packed_artifact:
            artifact_dir = (
                tablebase_path.parent
                if tablebase_path.name == "tablebase.json"
                else tablebase_path
            )
            artifact_ready = (artifact_dir / "tablebase.json").is_file()
            output_dir = artifact_dir
            manifest_path = None
        else:
            manifest_path = tablebase_path.with_suffix(".json")
            artifact_ready = tablebase_path.is_file() and manifest_path.is_file()
            output_dir = tablebase_path.parent
        if not artifact_ready:
            print(
                f"{args.buckets}-second abstract tablebase is missing; building or resuming it now. "
                "Press Control-C to cancel.",
                flush=True,
            )
            from abstract.cli import main as abstract_main

            command = [
                "exact",
                "--ruleset",
                ruleset_id,
                "--output-dir",
                str(output_dir),
            ]
            if packed_artifact:
                command.extend(("--backend", args.abstract_backend))
            result = abstract_main(command)
            if result != 0:
                raise RuntimeError(f"abstract tablebase build exited with status {result}")
        provider = AbstractTablebasePolicyProvider(
            tablebase_path,
            bucket_seconds=args.buckets,
            tablebase_manifest=manifest_path,
        )
    else:
        if not args.checkpoint:
            raise ValueError("--checkpoint is required for --hal-agent stl-mcts")
        from arena.stl_adapter import STLSolverPolicyProvider
        from stl.play.agent import SolverAgent

        provider = STLSolverPolicyProvider(
            SolverAgent(args.checkpoint, player_name="Hal", iterations=args.iterations, seed=args.seed)
        )
    return PolicyDrivenAgent(provider, player_name="Hal", seed=args.seed)


def _print_state(game: Game) -> None:
    print(f"\nClock {game.format_game_clock()} | round {game.round_num + 1}")
    for player in (game.player1, game.player2):
        print(f"  {player.name}: cylinder={player.cylinder:.0f}s TTD={player.ttd:.0f}s deaths={player.deaths}")


def command_play(args: argparse.Namespace) -> int:
    hal_agent = _make_hal(args)
    hal = Player(name="Hal", physicality=PHYSICALITY_HAL)
    human = Player(name=args.human_name, physicality=PHYSICALITY_BAKU)
    game = Game(player1=hal, player2=human, referee=Referee(), rng=__import__("random").Random(args.seed))
    game.game_clock = args.start_clock
    print(f"Playing canonical STL: Hal uses {args.hal_agent}; you are {human.name}.")
    half_rounds = 0
    while not game.game_over and (args.max_half_rounds is None or half_rounds < args.max_half_rounds):
        _print_state(game)
        dropper, checker = game.get_roles_for_half(game.current_half)
        turn_duration = game.get_turn_duration()
        if dropper is hal:
            drop = hal_agent.choose_action(game, "dropper", turn_duration)
            print("Hal has selected a drop time.")
        else:
            from stl.engine.actions import legal_seconds
            drop = _human_action(actor=dropper.name, role="dropper", legal=legal_seconds(dropper.name, "dropper", turn_duration))
        if checker is hal:
            check = hal_agent.choose_action(game, "checker", turn_duration)
            print(f"Hal checks at second {check}.")
        else:
            from stl.engine.actions import legal_seconds
            check = _human_action(actor=checker.name, role="checker", legal=legal_seconds(checker.name, "checker", turn_duration))
        validate_action(drop, actor=dropper.name, role="dropper", turn_duration=turn_duration)
        validate_action(check, actor=checker.name, role="checker", turn_duration=turn_duration)
        result = game.play_half_round(drop, check)
        print(f"{dropper.name} dropped at {drop}; {checker.name} checked at {check}; {result.result.value}.")
        half_rounds += 1
    if game.game_over:
        print(f"Game over: {game.winner.name} wins.")
    else:
        print(f"Session stopped after {half_rounds} half-rounds.")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m arena")
    commands = parser.add_subparsers(dest="command", required=True)
    play = commands.add_parser("play", help="play canonical STL against a pluggable Hal policy")
    play.add_argument("--hal-agent", choices=("abstract", "stl-mcts"), default="abstract")
    play.add_argument(
        "--buckets",
        type=int,
        choices=(5, 10),
        default=10,
        help="abstract tablebase bucket width in seconds",
    )
    play.add_argument(
        "--abstract-tablebase",
        default=None,
        help="override the bucket-specific abstract artifact path",
    )
    play.add_argument(
        "--abstract-backend",
        choices=("auto", "python", "rust"),
        default="auto",
        help="backend used when a missing abstract tablebase must be built",
    )
    play.add_argument("--checkpoint")
    play.add_argument("--iterations", type=int, default=200)
    play.add_argument("--human-name", default="Baku")
    play.add_argument("--seed", type=int, default=0)
    play.add_argument("--start-clock", type=int, default=OPENING_START_CLOCK)
    play.add_argument("--max-half-rounds", type=int, default=None, help="stop after this many half-rounds")
    play.set_defaults(function=command_play)
    return parser


def main(argv: list[str] | None = None) -> int:
    try:
        args = build_parser().parse_args(argv)
        return int(args.function(args))
    except KeyboardInterrupt:
        print("\nExited.", flush=True)
        return 130
