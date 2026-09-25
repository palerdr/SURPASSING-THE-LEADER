"""Terminal play: ``python -m terminal play``.

A human plays canonical STL, or pure DTH with ``--pure-dth``, against one Hal
policy provider. ``arena.policies.registry`` builds Hal and holds the agent
flags, and ``arena.session.PlaySession`` sequences the referee calls. This
module owns the terminal input and output and the public transcript.
"""

from __future__ import annotations

import argparse
import sys

from arena.agent import PolicyDrivenAgent
from arena.contracts import reset_provider_game
from arena.policies import registry
from arena.session import (
    CANONICAL_HAL_NAME,
    CANONICAL_HUMAN_NAME,
    Phase,
    PlaySession,
    validate_human_display_name,
)
from arena.transcript import write_play_transcript
from stl.engine.game import (
    OPENING_START_CLOCK,
    PHYSICALITY_BAKU,
    PHYSICALITY_HAL,
    Game,
    Player,
    Referee,
)

# The provider registry lives in arena.policies.registry. These aliases keep the
# names that the tests use; command_play looks up _make_hal here at call time,
# so a monkeypatch of this module still intercepts it.
DEFAULT_DTH_COMPLETE_TABLEBASE = registry.DEFAULT_DTH_COMPLETE_TABLEBASE
_abstract_artifact = registry.abstract_artifact
_make_adaptive_dth_provider = registry.make_adaptive_dth_provider
_make_perfect_hal_provider = registry.make_perfect_hal_provider
_make_provider = registry.make_provider
_make_hal = registry.make_hal


def _human_action(*, actor: str, role: str, legal: tuple[int, ...]) -> int:
    allowed = (
        f"{legal[0]}-{legal[-1]}"
        if legal == tuple(range(legal[0], legal[-1] + 1))
        else str(legal)
    )
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


def _print_state(game: Game, *, human_display_name: str = CANONICAL_HUMAN_NAME) -> None:
    print(f"\nClock {game.format_game_clock()} | round {game.round_num + 1}")
    for player in (game.player1, game.player2):
        name = (
            human_display_name if player.name == CANONICAL_HUMAN_NAME else player.name
        )
        print(
            f"  {name}: cylinder={player.cylinder:.0f}s TTD={player.ttd:.0f}s deaths={player.deaths}"
        )


def _show_rules(args: argparse.Namespace) -> None:
    """Show one rules screen per play session and gate interactive play."""
    if args.skip_rules:
        return

    from arena.presentation.rules_text import rules_body

    hal_label = args.public_hal_label or args.hal_agent
    if args.tui:
        from terminal.tui import Layout, draw, enable_ansi, render_rules

        enable_ansi()
        layout = Layout.detect(args.frame_width, args.frame_height)
        draw(
            render_rules(
                human_name=args.human_name,
                hal_label=hal_label,
                layout=layout,
            )
        )
        prompt = ""
    else:
        print("\nSURPASSING THE LEADER — GAME RULES")
        print(f"You: {args.human_name} | Opponent: Hal ({hal_label})\n")
        print("\n".join(rules_body()))
        prompt = "\nPress Enter to begin: "

    # Piped input contains game actions, not a disposable acknowledgement.
    if not sys.stdin.isatty():
        return
    try:
        input(prompt)
    except EOFError:
        return


def _play_one_game(
    args: argparse.Namespace,
    hal_agent: PolicyDrivenAgent,
    *,
    game_index: int,
) -> dict[str, object]:
    game_seed = None if args.seed is None else args.seed + game_index
    start_clock = (
        args.start_clock_sequence[game_index]
        if args.start_clock_sequence is not None
        else args.start_clock
    )
    hal = Player(name=CANONICAL_HAL_NAME, physicality=PHYSICALITY_HAL)
    human = Player(name=CANONICAL_HUMAN_NAME, physicality=PHYSICALITY_BAKU)
    game_type = Game
    if args.pure_dth:
        from arena.variants import PureDTHGame

        game_type = PureDTHGame
    game = game_type(
        player1=hal,
        player2=human,
        referee=Referee(),
        rng=__import__("random").Random(game_seed),
    )
    game.game_clock = start_clock
    reset_provider_game(hal_agent.provider)

    view = None
    show_outcome = None
    show_victory = None
    if args.tui:
        from arena.presentation.scene_art import SceneArt
        from terminal.tui import (
            Layout,
            draw,
            enable_ansi,
            render_frame,
            render_outcome,
            render_victory,
        )

        enable_ansi()
        layout = Layout.detect(args.frame_width, args.frame_height)
        art = SceneArt.load()
        colour = not args.no_colour

        def view():  # noqa: F811
            draw(
                render_frame(
                    game,
                    art=art,
                    human_name=human.name,
                    human_label=args.human_name,
                    frame=view.frame,
                    layout=layout,
                    colour=colour,
                    glyphs=args.glyphs,
                )
            )
            view.frame += 1

        view.frame = 0

        def show_outcome(record):  # noqa: F811
            draw(
                render_outcome(
                    record,
                    game,
                    human_name=human.name,
                    human_label=args.human_name,
                    layout=layout,
                    colour=colour,
                )
            )
            # Pausing reads stdin, which in a scripted run holds the next
            # action. Only wait when a human is actually at the terminal.
            if args.no_pause or not sys.stdin.isatty():
                return
            try:
                input()
            except (EOFError, KeyboardInterrupt):
                pass

        def show_victory():  # noqa: F811
            # One still frame of the winner — the first of the idle sheet.
            draw(
                render_victory(
                    game,
                    art=art,
                    human_name=human.name,
                    human_label=args.human_name,
                    layout=layout,
                    colour=colour,
                    glyphs=args.glyphs,
                )
            )
    elif args.games > 1:
        print(f"\nGame {game_index + 1}/{args.games}")

    session = PlaySession(
        game=game,
        hal_agent=hal_agent,
        hal=hal,
        human=human,
        human_display_name=args.human_name,
        game_index=game_index,
        game_seed=game_seed,
        start_clock=start_clock,
        max_half_rounds=args.max_half_rounds,
    )
    session.begin()
    while session.phase is Phase.AWAITING_ACTION:
        if view is not None:
            view()
        else:
            _print_state(game, human_display_name=args.human_name)
        # Hal acts inside submit(), after this returns, so nothing about its
        # choice exists while the human is deciding.
        record = session.submit(
            _human_action(
                actor=session.human_display_name,
                role=session.human_role(),
                legal=session.legal_actions(),
            )
        )
        if show_outcome is not None:
            show_outcome(record)
        else:
            dropper = session.display_canonical_name(record.dropper)
            checker = session.display_canonical_name(record.checker)
            print(
                f"{dropper} dropped at {record.drop_time}; "
                f"{checker} checked at {record.check_time}; "
                f"{record.result.value}."
            )
        session.acknowledge()
    if game.game_over:
        if show_victory is not None and game.winner is not None:
            show_victory()
        if game.winner is not None:
            print(f"Game over: {session.display_name(game.winner)} wins.")
        else:
            print("Game over: no surviving winner.")
    else:
        print(f"Session stopped after {session.half_rounds} half-rounds.")
    return session.finish()


def command_play(args: argparse.Namespace) -> int:
    args.human_name = validate_human_display_name(args.human_name)
    if registry.requires_pure_dth(args.hal_agent, args) and not args.pure_dth:
        raise ValueError(
            f"{args.hal_agent} is a pure-DTH policy; pass --pure-dth so action "
            "61 is impossible"
        )
    if args.games <= 0:
        raise ValueError("--games must be positive")
    if args.tui and args.games != 1:
        raise ValueError("--tui supports one game per invocation")
    if (
        args.start_clock_sequence is not None
        and len(args.start_clock_sequence) != args.games
    ):
        raise ValueError("--start-clock-sequence must contain one value per game")
    hal_agent = _make_hal(args)
    _show_rules(args)
    transcript: dict[str, object] = {
        "schema_version": "arena-public-play-session-v1",
        "hal_agent": args.hal_agent,
        "public_hal_label": args.public_hal_label,
        "human_name": args.human_name,
        "base_seed": args.seed,
        "start_clock": args.start_clock,
        "start_clock_sequence": args.start_clock_sequence,
        "requested_games": args.games,
        "pure_dth": bool(args.pure_dth),
        "games": [],
    }
    games = transcript["games"]
    assert isinstance(games, list)
    for game_index in range(args.games):
        games.append(_play_one_game(args, hal_agent, game_index=game_index))
        if args.transcript:
            write_play_transcript(args.transcript, transcript)
    match_summary = getattr(hal_agent.provider, "match_summary", None)
    if callable(match_summary):
        summary = match_summary()
        if not args.conceal_hal_details:
            print(summary)
        transcript["hal_summary"] = summary
    experiment_diagnostics = getattr(hal_agent.provider, "experiment_diagnostics", None)
    if callable(experiment_diagnostics):
        transcript["hal_diagnostics"] = experiment_diagnostics()
    if args.transcript:
        destination = write_play_transcript(args.transcript, transcript)
        print(f"Public session transcript: {destination}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m terminal")
    commands = parser.add_subparsers(dest="command", required=True)
    play = commands.add_parser(
        "play", help="play STL or explicit pure DTH against a pluggable Hal policy"
    )
    play.add_argument("--hal-agent", choices=registry.PLAY_AGENTS, default="dth")
    registry.add_play_arguments(play)
    play.add_argument("--human-name", default="Baku")
    play.add_argument(
        "--public-hal-label",
        default=None,
        help="optional display label that conceals the provider implementation",
    )
    play.add_argument(
        "--conceal-hal-details",
        action="store_true",
        help="record provider summary and diagnostics without printing them",
    )
    play.add_argument(
        "--games",
        type=int,
        default=1,
        help="games in one repeated-opponent session; Hal retains its opponent model",
    )
    play.add_argument(
        "--transcript",
        default=None,
        help="optional JSON path for public states, revealed actions, and outcomes",
    )
    play.add_argument(
        "--seed",
        type=int,
        default=None,
        help="RNG seed for Hal's sampling and the revival rolls; omit for "
        "fresh randomness each match, set for a reproducible replay",
    )
    play.add_argument("--start-clock", type=int, default=OPENING_START_CLOCK)
    play.add_argument(
        "--pure-dth",
        action="store_true",
        help=(
            "use permanent literal actions 1..60; required for Aggro, Perfect, "
            "and PM Hal"
        ),
    )
    play.add_argument(
        "--start-clock-sequence",
        type=int,
        nargs="+",
        default=None,
        help="optional per-game start clocks; length must equal --games",
    )
    play.add_argument(
        "--max-half-rounds",
        type=int,
        default=None,
        help="stop after this many half-rounds",
    )
    play.add_argument(
        "--tui",
        action="store_true",
        help="render the terminal interface instead of plain text",
    )
    play.add_argument(
        "--no-colour",
        action="store_true",
        help="render sprites as ASCII density instead of truecolor glyph cells",
    )
    play.add_argument(
        "--glyphs",
        choices=("sextant", "quadrant"),
        default="sextant",
        help="sprite glyph set: sextant (2x3 pixels per cell, needs Symbols for "
        "Legacy Computing — Windows Terminal and current Cascadia fonts have it) "
        "or quadrant (2x2, universal Block Elements)",
    )
    play.add_argument(
        "--frame-width",
        type=int,
        default=None,
        help="override the auto-detected frame width in columns",
    )
    play.add_argument(
        "--frame-height",
        type=int,
        default=None,
        help="override the auto-detected terminal height in lines",
    )
    play.add_argument(
        "--no-pause",
        action="store_true",
        help="do not wait for input on the half-round outcome screen",
    )
    play.add_argument(
        "--skip-rules",
        action="store_true",
        help="start immediately without the opening rules screen",
    )
    play.set_defaults(function=command_play)
    return parser


def main(argv: list[str] | None = None) -> int:
    try:
        args = build_parser().parse_args(argv)
        return int(args.function(args))
    except KeyboardInterrupt:
        print("\nExited.", flush=True)
        return 130
