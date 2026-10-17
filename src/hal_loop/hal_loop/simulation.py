"""Paired-seat series of a Hal provider against the histogram emulator.

Each base seed plays twice, once with the candidate in each seat, as
hal_lab's series harness does. A win scores 1, a loss 0, and a capped game 0.5.
The providers come from arena.policies.registry with the terminal's play options.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable
from pathlib import Path

from arena.match import play_match_game
from arena.policies.registry import add_play_arguments, make_provider

OPENING_START_CLOCK = 720
MAX_HALF_ROUNDS = 200


def provider_args(kind: str, dth_artifact_dir: Path, *, prior_path: Path | None = None, seed: int = 0) -> argparse.Namespace:
    """Parse the terminal's default play options for one provider kind."""
    parser = argparse.ArgumentParser()
    add_play_arguments(parser)
    argv = ["--dth-complete-tablebase", str(dth_artifact_dir)]
    if prior_path is not None:
        argv += ["--adaptive-prior-json", str(prior_path)]
    args = parser.parse_args(argv)
    args.hal_agent = kind
    args.seed = seed
    return args


def build_provider(kind: str, dth_artifact_dir: Path, *, prior_path: Path | None = None, seed: int = 0):
    """Build one Hal provider through the shared registry."""
    return make_provider(kind, provider_args(kind, dth_artifact_dir, prior_path=prior_path, seed=seed))


def paired_series(
    candidate,
    make_opponent: Callable[[], object],
    *,
    games: int,
    start_clock: int = OPENING_START_CLOCK,
    max_half_rounds: int = MAX_HALF_ROUNDS,
) -> dict:
    """Play `games` games in seat-swapped pairs and return the candidate's mean score."""
    opponent = make_opponent()
    wins = losses = capped = 0
    for seed in range(games // 2):
        for candidate_first in (True, False):
            one, two = (candidate, opponent) if candidate_first else (opponent, candidate)
            winner, _ = play_match_game(
                one, two, seed=seed, start_clock=start_clock,
                max_half_rounds=max_half_rounds, game_index=wins + losses + capped,
            )
            if winner is None:
                capped += 1
            elif winner == ("Hal" if candidate_first else "Baku"):
                wins += 1
            else:
                losses += 1
    played = wins + losses + capped
    return {
        "games": played,
        "wins": wins,
        "losses": losses,
        "capped": capped,
        "mean_score": (wins + 0.5 * capped) / played if played else 0.0,
    }
