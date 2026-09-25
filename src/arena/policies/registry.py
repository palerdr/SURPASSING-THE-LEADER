"""One registry of Hal policy providers for every play surface.

The terminal app and the local browser server build Hal through this module:
the play choices, the play flags, the provider factories, and the one pure-DTH
gate. ``hal_lab``'s match command builds every play agent here too. It owns the
research-only Aggro Hal choice, its flags, and its factory.
Each factory imports its provider when it runs, so importing the registry loads
no solver peer, torch, or training code.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from arena.agent import PolicyDrivenAgent

DEFAULT_DTH_COMPLETE_TABLEBASE = "src/dth/artifacts/complete_full_v1"

# Providers a human can face in `python -m terminal play`.
PLAY_AGENTS = (
    "abstract",
    "dth",
    "adaptive-dth",
    "exploit-hal",
    "perfect-hal",
    "pm-hal",
)
# The browser refuses `abstract`, because its provider can build a tablebase on
# first use, and that must never happen behind an HTTP request.
BROWSER_AGENTS = tuple(kind for kind in PLAY_AGENTS if kind != "abstract")

# Policies that assume the permanent 1..60 action set. They need --pure-dth so
# action 61 cannot occur. aggro-hal is the research-only Aggro Hal that
# hal_lab builds; the one gate lists it too.
PURE_DTH_ONLY = frozenset({"aggro-hal", "perfect-hal", "pm-hal"})


def requires_pure_dth(kind: str, args: argparse.Namespace) -> bool:
    """Return whether ``kind`` may play only the pure-DTH surface.

    The translated Perfect variant adds a canonical leap fallback, so it may
    also play canonical STL.
    """

    if kind == "perfect-hal" and args.perfect_hal_model == "translated-v1":
        return False
    return kind in PURE_DTH_ONLY


def abstract_artifact(args: argparse.Namespace) -> tuple[Path, str]:
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


def make_dth_provider(args: argparse.Namespace):
    from arena.dth_adapter import DTHCompletePolicyProvider

    return DTHCompletePolicyProvider(artifact_dir=dth_artifact_dir(args))


def dth_artifact_dir(args: argparse.Namespace) -> Path:
    artifact_dir = Path(args.dth_complete_tablebase)
    manifest = artifact_dir / "tablebase.json"
    if not manifest.is_file():
        raise FileNotFoundError(
            f"complete DTH tablebase is required at {artifact_dir}; "
            "build it with: uv run python -m dth complete"
        )
    return artifact_dir


def make_adaptive_dth_provider(args: argparse.Namespace):
    from arena.policies.adaptive import (
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
        artifact_dir=dth_artifact_dir(args),
        opponent=opponent,
        config=ExploitationConfig(
            epsilon_grid=tuple(args.adaptive_epsilon_grid),
            match_epsilon_budget=args.adaptive_match_epsilon_budget,
            confidence=args.adaptive_confidence,
            posterior_samples=args.adaptive_posterior_samples,
        ),
        seed=args.seed,
    )


def make_exploit_hal_provider(args: argparse.Namespace):
    if not args.exploit_hal_checkpoint:
        raise ValueError(
            "--exploit-hal-checkpoint is required for --hal-agent exploit-hal"
        )
    from arena.policies.adaptive import load_opponent_model
    from arena.policies.exploit_hal import make_live_provider
    from arena.policies.exploit_hal_config import (
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
        artifact_dir=dth_artifact_dir(args),
        checkpoint=args.exploit_hal_checkpoint,
        opponent=opponent,
        config=config,
        seed=args.seed,
        stochastic=args.exploit_hal_stochastic,
    )


def make_perfect_hal_provider(args: argparse.Namespace):
    if args.perfect_hal_model == "translated-v1":
        from arena.translated_hal_adapter import TranslatedHalPolicyProvider

        return TranslatedHalPolicyProvider(dth_artifact_dir(args))
    from arena.policies.perfect_hal import (
        PerfectHalConfig,
        PerfectHalOpponentModel,
        PerfectHalPolicyProvider,
    )

    config = PerfectHalConfig(
        prior_strength=args.perfect_hal_prior_strength,
        expert_learning_rate=args.perfect_hal_expert_learning_rate,
        expert_weight_retention=args.perfect_hal_expert_weight_retention,
        response_temperature=args.perfect_hal_response_temperature,
    )
    if args.perfect_hal_model == "ensemble":
        from arena.policies.ensemble_hal import EnsembleHalPolicyProvider

        return EnsembleHalPolicyProvider(dth_artifact_dir(args), config)
    # Keep the v1 play default after the v2 human-emulator promotion failure.
    model = PerfectHalOpponentModel(config) if args.perfect_hal_model == "v1" else None
    return PerfectHalPolicyProvider(
        artifact_dir=dth_artifact_dir(args),
        config=config,
        opponent_model=model,
    )


def make_pm_hal_provider(args: argparse.Namespace):
    from dataclasses import replace

    from arena.policies.pm_hal import load_pm_hal_config, make_live_provider

    base = load_pm_hal_config(args.pm_hal_config)
    game_budget = (
        base.game_epsilon_budget
        if args.pm_hal_game_epsilon_budget is None
        else args.pm_hal_game_epsilon_budget
    )
    press_cap = (
        min(base.press_epsilon_cap, game_budget)
        if args.pm_hal_press_epsilon_cap is None
        else args.pm_hal_press_epsilon_cap
    )
    dominate_cap = (
        min(base.dominate_epsilon_cap, game_budget)
        if args.pm_hal_dominate_epsilon_cap is None
        else args.pm_hal_dominate_epsilon_cap
    )

    return make_live_provider(
        artifact_dir=dth_artifact_dir(args),
        config=replace(
            base,
            game_epsilon_budget=game_budget,
            probe_epsilon_cap=min(base.probe_epsilon_cap, press_cap),
            press_epsilon_cap=press_cap,
            dominate_epsilon_cap=dominate_cap,
            posterior_samples=(
                base.posterior_samples
                if args.pm_hal_posterior_samples is None
                else args.pm_hal_posterior_samples
            ),
        ),
        aggro_checkpoint=args.pm_hal_aggro_checkpoint,
        device=args.pm_hal_device,
        seed=args.seed,
    )


def make_provider(kind: str, args: argparse.Namespace):
    if kind == "abstract":
        from arena.abstract_adapter import AbstractTablebasePolicyProvider

        tablebase_path, ruleset_id = abstract_artifact(args)
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
        return make_dth_provider(args)
    if kind == "adaptive-dth":
        return make_adaptive_dth_provider(args)
    if kind == "exploit-hal":
        return make_exploit_hal_provider(args)
    if kind == "perfect-hal":
        return make_perfect_hal_provider(args)
    if kind == "pm-hal":
        return make_pm_hal_provider(args)
    if kind == "stl-mcts":
        raise ValueError(
            "stl-mcts is retired: the STL play/solver stack it depended on no "
            "longer exists"
        )
    raise ValueError(f"unknown agent kind {kind!r}")


def make_hal(args: argparse.Namespace) -> PolicyDrivenAgent:
    return PolicyDrivenAgent(
        make_provider(args.hal_agent, args), player_name="Hal", seed=args.seed
    )


def add_play_arguments(parser: argparse.ArgumentParser) -> None:
    """Add the options of every provider in :data:`PLAY_AGENTS`."""

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
        "--perfect-hal-model",
        choices=("v1", "bayesian-v2", "ensemble", "translated-v1"),
        default="v1",
        help="v1 remains the default; translated-v1 uses frozen parameters and a leap fallback",
    )
    parser.add_argument(
        "--perfect-hal-prior-strength",
        type=float,
        default=0.02,
        help="v1 public pseudo-observation mass before Perfect Hal sees an action",
    )
    parser.add_argument(
        "--perfect-hal-expert-learning-rate",
        type=float,
        default=1.25,
        help="v1 Perfect Hal prequential expert-score learning rate",
    )
    parser.add_argument(
        "--perfect-hal-expert-weight-retention",
        type=float,
        default=0.92,
        help="v1 Perfect Hal expert-score retention across public reveals",
    )
    parser.add_argument(
        "--perfect-hal-response-temperature",
        type=float,
        default=0.0,
        help="Perfect Hal best-response temperature; zero is a hard response",
    )
    parser.add_argument(
        "--pm-hal-config",
        default="src/arena/config/pm_hal_controller_v3.json",
        help="frozen PM Hal controller configuration",
    )
    parser.add_argument(
        "--pm-hal-aggro-checkpoint",
        default=None,
        help=(
            "optional compatible Aggro checkpoint; PM Hal remains checkpoint-free "
            "when omitted"
        ),
    )
    parser.add_argument(
        "--pm-hal-device",
        choices=("cpu", "cuda"),
        default="cpu",
        help="explicit device for the optional PM Hal recurrent expert",
    )
    parser.add_argument(
        "--pm-hal-game-epsilon-budget",
        type=float,
        default=None,
        help="maximum cumulative PM Hal local worst-case-loss charge per game",
    )
    parser.add_argument(
        "--pm-hal-press-epsilon-cap",
        type=float,
        default=None,
        help="maximum local loss admitted in PM Hal press mode",
    )
    parser.add_argument(
        "--pm-hal-dominate-epsilon-cap",
        type=float,
        default=None,
        help="maximum local loss admitted in PM Hal dominate mode",
    )
    parser.add_argument(
        "--pm-hal-posterior-samples",
        type=int,
        default=None,
        help="forecast-ensemble posterior draws per PM Hal frontier candidate",
    )
