"""Interactive canonical STL referee with pluggable Hal policy providers."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from arena.abstract_adapter import AbstractTablebasePolicyProvider
from arena.agent import PolicyDrivenAgent
from arena.contracts import reset_provider_game
from arena.session import (
    CANONICAL_HAL_NAME,
    CANONICAL_HUMAN_NAME,
    Phase,
    PlaySession,
    validate_human_display_name,
)
from stl.engine.game import (
    OPENING_START_CLOCK,
    PHYSICALITY_BAKU,
    PHYSICALITY_HAL,
    Game,
    Player,
    Referee,
)

DEFAULT_DTH_COMPLETE_TABLEBASE = "src/dth/artifacts/complete_full_v1"


def _abstract_artifact(args: argparse.Namespace) -> tuple[Path, str]:
    if args.buckets == 5:
        ruleset_id = "bucket12_frozen95"
        default = Path("src/abstract/outputs") / ruleset_id
    else:
        ruleset_id = "bucket6_frozen95"
        default = Path("src/abstract/outputs") / ruleset_id
    return (
        Path(args.abstract_tablebase) if args.abstract_tablebase else default,
        ruleset_id,
    )


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


def _make_dth_provider(args: argparse.Namespace):
    from arena.dth_adapter import DTHCompletePolicyProvider

    return DTHCompletePolicyProvider(artifact_dir=_dth_artifact_dir(args))


def _dth_artifact_dir(args: argparse.Namespace) -> Path:
    artifact_dir = Path(args.dth_complete_tablebase)
    manifest = artifact_dir / "tablebase.json"
    if not manifest.is_file():
        raise FileNotFoundError(
            f"complete DTH tablebase is required at {artifact_dir}; "
            "build it with: uv run python -m dth complete"
        )
    return artifact_dir


def _make_adaptive_dth_provider(args: argparse.Namespace):
    from arena.adaptive_dth import (
        AdaptiveDTHPolicyProvider,
        ExploitationConfig,
        load_opponent_model,
    )

    opponent = load_opponent_model(
        args.adaptive_prior_json,
        default_strength=args.adaptive_prior_strength,
        decay=args.adaptive_decay,
    )

    return AdaptiveDTHPolicyProvider(
        artifact_dir=_dth_artifact_dir(args),
        opponent=opponent,
        config=ExploitationConfig(
            epsilon_grid=tuple(args.adaptive_epsilon_grid),
            match_epsilon_budget=args.adaptive_match_epsilon_budget,
            confidence=args.adaptive_confidence,
            posterior_samples=args.adaptive_posterior_samples,
        ),
        seed=args.seed,
    )


def _make_exploit_hal_provider(args: argparse.Namespace):
    if not args.exploit_hal_checkpoint:
        raise ValueError(
            "--exploit-hal-checkpoint is required for --hal-agent exploit-hal"
        )
    from arena.policies.adaptive import load_opponent_model
    from arena.policies.exploit_hal import make_live_provider
    from arena.policies.train_exploit_hal import (
        load_training_config,
        exploit_config_from_mapping,
    )

    tracked = load_training_config(args.exploit_hal_config)
    config = exploit_config_from_mapping(tracked)
    opponent = load_opponent_model(
        args.adaptive_prior_json,
        default_strength=args.adaptive_prior_strength,
        decay=args.adaptive_decay,
    )
    return make_live_provider(
        artifact_dir=_dth_artifact_dir(args),
        checkpoint=args.exploit_hal_checkpoint,
        opponent=opponent,
        config=config,
        seed=args.seed,
        stochastic=args.exploit_hal_stochastic,
    )


def _make_aggro_hal_provider(args: argparse.Namespace):
    if not args.aggro_hal_checkpoint:
        raise ValueError("--aggro-hal-checkpoint is required for --hal-agent aggro-hal")
    from arena.policies.aggro_hal import make_live_provider

    return make_live_provider(
        artifact_dir=_dth_artifact_dir(args),
        checkpoint=args.aggro_hal_checkpoint,
        device=args.aggro_hal_device,
        fast_adaptation=args.aggro_hal_fast_adaptation,
    )


def _make_provider(kind: str, args: argparse.Namespace):
    if kind == "abstract":
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
                raise RuntimeError(
                    f"abstract tablebase build exited with status {result}"
                )
        return AbstractTablebasePolicyProvider(
            tablebase_path,
            bucket_seconds=args.buckets,
            tablebase_manifest=manifest_path,
        )
    if kind == "dth":
        return _make_dth_provider(args)
    if kind == "adaptive-dth":
        return _make_adaptive_dth_provider(args)
    if kind == "exploit-hal":
        return _make_exploit_hal_provider(args)
    if kind == "aggro-hal":
        return _make_aggro_hal_provider(args)
    if kind == "stl-mcts":
        raise ValueError(
            "stl-mcts is retired: the STL play/solver stack it depended on no "
            "longer exists"
        )
    raise ValueError(f"unknown agent kind {kind!r}")


def _make_hal(args: argparse.Namespace) -> PolicyDrivenAgent:
    return PolicyDrivenAgent(
        _make_provider(args.hal_agent, args), player_name="Hal", seed=args.seed
    )


def _print_state(game: Game, *, human_display_name: str = CANONICAL_HUMAN_NAME) -> None:
    print(f"\nClock {game.format_game_clock()} | round {game.round_num + 1}")
    for player in (game.player1, game.player2):
        name = human_display_name if player.name == CANONICAL_HUMAN_NAME else player.name
        print(
            f"  {name}: cylinder={player.cylinder:.0f}s TTD={player.ttd:.0f}s deaths={player.deaths}"
        )


def _show_rules(args: argparse.Namespace) -> None:
    """Show one rules screen per play session and gate interactive play."""
    if args.skip_rules:
        return

    from arena.tui import Layout, draw, enable_ansi, render_rules, rules_body

    hal_label = args.public_hal_label or args.hal_agent
    if args.tui:
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


def _write_play_transcript(
    destination: str | Path, transcript: dict[str, object]
) -> Path:
    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(transcript, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)
    return path


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
    game = Game(
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
        from arena.tui import (
            Layout,
            SceneArt,
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
        "games": [],
    }
    games = transcript["games"]
    assert isinstance(games, list)
    for game_index in range(args.games):
        games.append(_play_one_game(args, hal_agent, game_index=game_index))
        if args.transcript:
            _write_play_transcript(args.transcript, transcript)
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
        destination = _write_play_transcript(args.transcript, transcript)
        print(f"Public session transcript: {destination}")
    return 0


def _add_agent_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--buckets",
        type=int,
        choices=(5, 10),
        default=10,
        help="abstract tablebase bucket width in seconds",
    )
    parser.add_argument(
        "--abstract-tablebase",
        default=None,
        help="override the bucket-specific abstract artifact path",
    )
    parser.add_argument(
        "--abstract-backend",
        choices=("auto", "python", "rust"),
        default="auto",
        help="backend used when a missing abstract tablebase must be built",
    )
    parser.add_argument(
        "--dth-complete-tablebase",
        default=DEFAULT_DTH_COMPLETE_TABLEBASE,
        metavar="DTH_COMPLETE_TABLEBASE",
        help="completed exact DTH quotient tablebase directory",
    )
    parser.add_argument(
        "--adaptive-prior-json",
        default=None,
        help="optional versioned role or role-mixture population prior",
    )
    parser.add_argument(
        "--adaptive-prior-strength",
        type=float,
        default=1.0,
        help="uniform role-prior pseudo-observation count for adaptive DTH",
    )
    parser.add_argument(
        "--adaptive-decay",
        type=float,
        default=0.9,
        help="adaptive DTH evidence retention after each same-role observation",
    )
    parser.add_argument(
        "--adaptive-epsilon-grid",
        type=float,
        nargs="+",
        default=(0.0, 0.0025, 0.005, 0.01, 0.02),
        help="candidate one-step safety losses for adaptive DTH",
    )
    parser.add_argument(
        "--adaptive-match-epsilon-budget",
        type=float,
        default=0.05,
        help="maximum cumulative one-step safety loss per game",
    )
    parser.add_argument(
        "--adaptive-confidence",
        type=float,
        default=0.95,
        help="posterior improvement-probability gate for adaptive DTH",
    )
    parser.add_argument(
        "--adaptive-posterior-samples",
        type=int,
        default=512,
        help="Dirichlet draws per adaptive DTH epsilon candidate",
    )
    parser.add_argument(
        "--exploit-hal-checkpoint",
        default=None,
        help="required versioned actor-critic checkpoint for Exploit Hal",
    )
    parser.add_argument(
        "--exploit-hal-config",
        default="src/arena/config/exploit_hal_v2.yaml",
        help="tracked Exploit Hal configuration used for checkpoint validation",
    )
    parser.add_argument(
        "--exploit-hal-stochastic",
        action="store_true",
        help="sample candidates during evaluation instead of deterministic argmax",
    )
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
    from arena.match import run_paired_series, write_report

    if "aggro-hal" in {args.candidate, args.opponent} and not args.pure_dth:
        raise ValueError(
            "aggro-hal is a pure-DTH policy; pass --pure-dth so action 61 is impossible"
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
    parser = argparse.ArgumentParser(prog="python -m arena")
    commands = parser.add_subparsers(dest="command", required=True)
    play = commands.add_parser(
        "play", help="play canonical STL against a pluggable Hal policy"
    )
    play.add_argument(
        "--hal-agent",
        choices=("abstract", "dth", "adaptive-dth", "exploit-hal"),
        default="dth",
    )
    _add_agent_arguments(play)
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

    match = commands.add_parser(
        "match",
        help="paired-seat agent-versus-agent series with a predeclared SPRT",
    )
    match.add_argument(
        "--candidate",
        choices=(
            "abstract",
            "dth",
            "adaptive-dth",
            "exploit-hal",
            "aggro-hal",
        ),
        required=True,
    )
    match.add_argument(
        "--opponent",
        choices=(
            "abstract",
            "dth",
            "adaptive-dth",
            "exploit-hal",
            "aggro-hal",
        ),
        required=True,
    )
    _add_agent_arguments(match)
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
        help="run the pure 1..60 DTH action contract (required for aggro-hal)",
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
