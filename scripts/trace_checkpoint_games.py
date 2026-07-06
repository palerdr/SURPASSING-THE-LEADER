#!/usr/bin/env python3
"""Trace checkpoint-vs-opponent games with full public histories."""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from environment.legal_actions import validate_action
from environment.opponents.base import Opponent
from environment.opponents.factory import create_scripted_opponent
from hal.agent import DEFAULT_CHECKPOINT, SolverAgent
from scripts.compare_checkpoints_ladder import _json_safe
from scripts.run_tier_a_frontier_event import _split_csv
from src.Game import Game, HalfRoundResult
from training.strength.match_gate import _opponent_seed
from training.tournament import _HALF_ROUND_SAFETY_LIMIT, _default_starting_game


def _checked_action(agent, game: Game, *, actor: str, role: str, turn_duration: int) -> int:
    second = int(agent.choose_action(game, role, turn_duration))
    validate_action(second, actor=actor, role=role, turn_duration=turn_duration)
    return second


def _classify_termination(game: Game) -> str:
    if not game.game_over:
        return "unfinished"
    if not game.history:
        return "unknown"
    last = game.history[-1]
    loser_name = game.loser.name.lower() if game.loser is not None else None
    if last.result == HalfRoundResult.CHECK_FAIL_DIED:
        return f"{loser_name}_failed_check" if loser_name else "failed_check"
    if last.result == HalfRoundResult.CYLINDER_OVERFLOW_DIED:
        return f"{loser_name}_overflow" if loser_name else "overflow"
    return last.result.value


def _record_to_json(record) -> dict:
    row = asdict(record)
    row["result"] = record.result.value
    return row


def _make_agent(args, checkpoint: str, seed: int) -> SolverAgent:
    return SolverAgent(
        checkpoint,
        player_name="Hal",
        iterations=args.iterations,
        seed=seed,
        policy_ensemble_size=args.policy_ensemble_size,
        policy_uniform_mix=args.policy_uniform_mix,
        resolve_at_critical=args.resolve_at_critical,
        resolve_horizon=args.resolve_horizon,
    )


def _game_seed_for_index(seed: int, game_index: int) -> int:
    rng = random.Random(seed)
    for _ in range(game_index + 1):
        game_seed = rng.randrange(1 << 31)
    return game_seed


def _make_ladder_opponent(opponent: str, seed: int) -> Opponent:
    return create_scripted_opponent(opponent, seed=_opponent_seed(seed, opponent))


def _play_one(
    args,
    *,
    agent: SolverAgent,
    baku: Opponent,
    opponent: str,
    seed: int,
    game_index: int,
) -> dict:
    game_seed = _game_seed_for_index(seed, game_index)
    game = _default_starting_game(game_seed)
    baku.reset()

    safety_counter = 0
    while not game.game_over and safety_counter < _HALF_ROUND_SAFETY_LIMIT:
        safety_counter += 1
        dropper, checker = game.get_roles_for_half(game.current_half)
        turn_duration = game.get_turn_duration()
        if dropper.name.lower() == "hal":
            drop_time = _checked_action(
                agent,
                game,
                actor="hal",
                role="dropper",
                turn_duration=turn_duration,
            )
            check_time = _checked_action(
                baku,
                game,
                actor="baku",
                role="checker",
                turn_duration=turn_duration,
            )
        else:
            drop_time = _checked_action(
                baku,
                game,
                actor="baku",
                role="dropper",
                turn_duration=turn_duration,
            )
            check_time = _checked_action(
                agent,
                game,
                actor="hal",
                role="checker",
                turn_duration=turn_duration,
            )
        game.play_half_round(drop_time, check_time)

    return {
        "opponent": opponent,
        "seed": seed,
        "game_index": game_index,
        "game_seed": game_seed,
        "winner": game.winner.name if game.winner else None,
        "loser": game.loser.name if game.loser else None,
        "cause": _classify_termination(game),
        "half_rounds": len(game.history),
        "final_state": game.get_state_summary(),
        "history": [_record_to_json(record) for record in game.history],
    }


def _play_games(
    args,
    *,
    checkpoint: str,
    opponent: str,
    seed: int,
    games: int,
    start_game_index: int = 0,
) -> list[dict]:
    agent = _make_agent(args, checkpoint, seed)
    baku = _make_ladder_opponent(opponent, seed)
    rows = []
    for game_index in range(start_game_index + games):
        row = _play_one(
            args,
            agent=agent,
            baku=baku,
            opponent=opponent,
            seed=seed,
            game_index=game_index,
        )
        if game_index >= start_game_index:
            row["checkpoint"] = checkpoint
            rows.append(row)
    return rows


def _summary(games: list[dict]) -> dict:
    wins = sum(1 for game in games if game["winner"] == "Hal")
    losses = sum(1 for game in games if game["winner"] == "Baku")
    draws = len(games) - wins - losses
    causes: dict[str, int] = {}
    for game in games:
        causes[game["cause"]] = causes.get(game["cause"], 0) + 1
    return {
        "games": len(games),
        "wins": wins,
        "losses": losses,
        "draws": draws,
        "causes": causes,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--champion-checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--candidate-checkpoint", required=True)
    parser.add_argument("--opponent", required=True)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--policy-ensemble-size", type=int, default=1)
    parser.add_argument("--policy-uniform-mix", type=float, default=0.0)
    parser.add_argument(
        "--resolve-at-critical",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use SolverAgent critical-state subgame resolve during deployed search.",
    )
    parser.add_argument(
        "--resolve-horizon",
        type=int,
        default=3,
        help="Selective-solve horizon for --resolve-at-critical.",
    )
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--games", type=int, default=5)
    parser.add_argument(
        "--start-game-index",
        type=int,
        default=0,
        help=(
            "First ladder game index to include for each seed. Earlier games are "
            "still played to preserve the agent action RNG stream."
        ),
    )
    parser.add_argument("--out", required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    seeds = [int(item) for item in _split_csv(args.seeds)]
    champion_games = []
    candidate_games = []
    for seed in seeds:
        champion_games.extend(
            _play_games(
                args,
                checkpoint=args.champion_checkpoint,
                opponent=args.opponent,
                seed=seed,
                games=args.games,
                start_game_index=args.start_game_index,
            )
        )
        candidate_games.extend(
            _play_games(
                args,
                checkpoint=args.candidate_checkpoint,
                opponent=args.opponent,
                seed=seed,
                games=args.games,
                start_game_index=args.start_game_index,
            )
        )

    report = {
        "config": vars(args),
        "seeds": seeds,
        "summaries": {
            "champion": _summary(champion_games),
            "candidate": _summary(candidate_games),
        },
        "games": {
            "champion": champion_games,
            "candidate": candidate_games,
        },
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as fh:
        json.dump(_json_safe(report), fh, indent=2, allow_nan=False)

    print(f"Champion: {report['summaries']['champion']}")
    print(f"Candidate: {report['summaries']['candidate']}")
    print(f"Report: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
