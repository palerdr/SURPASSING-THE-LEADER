"""Paired-seat agent-versus-agent series with one predeclared SPRT.

The Hal seat drops first and the Baku seat holds the leap-window action
asymmetry, so ``run_paired_series`` plays each base seed twice, once in each
seating, and credits each result to an agent whichever seat it held.
``arena.match`` plays each game.

The strength gate is a predeclared one-sided SPRT on decisive games:
H0 win probability 0.5 against H1 0.65 at alpha = beta = 0.05. Games stopped
by the half-round cap count for neither hypothesis, and the report lists them.
``hal_lab.training.train_exploit_hal`` reads ``sprt_verdict`` from here too.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from arena.match import play_match_game

SPRT_P0 = 0.5
SPRT_P1 = 0.65
SPRT_ALPHA = 0.05
SPRT_BETA = 0.05


@dataclass(frozen=True)
class GameOutcome:
    seed: int
    first_seat_agent: str
    winner_agent: str | None
    half_rounds: int


def sprt_verdict(wins: int, losses: int) -> dict[str, float | str | int]:
    """One-sided SPRT for the candidate's decisive-game win probability."""

    decisive = wins + losses
    llr = wins * math.log(SPRT_P1 / SPRT_P0) + losses * math.log(
        (1.0 - SPRT_P1) / (1.0 - SPRT_P0)
    )
    upper = math.log((1.0 - SPRT_BETA) / SPRT_ALPHA)
    lower = math.log(SPRT_BETA / (1.0 - SPRT_ALPHA))
    if llr >= upper:
        decision = "accept-h1"
    elif llr <= lower:
        decision = "accept-h0"
    else:
        decision = "continue"
    return {
        "p0": SPRT_P0,
        "p1": SPRT_P1,
        "alpha": SPRT_ALPHA,
        "beta": SPRT_BETA,
        "decisive_games": decisive,
        "wins": wins,
        "losses": losses,
        "llr": llr,
        "upper_bound": upper,
        "lower_bound": lower,
        "decision": decision,
    }


def run_paired_series(
    candidate_name: str,
    opponent_name: str,
    make_candidate: Callable[[], object],
    make_opponent: Callable[[], object],
    *,
    base_seeds: int,
    seed_start: int = 0,
    start_clock: int,
    max_half_rounds: int,
    pure_dth: bool = False,
    stop_early: bool = True,
) -> dict[str, object]:
    """Play seat-swapped pairs until the SPRT decides or seeds run out."""

    candidate = make_candidate()
    opponent = make_opponent()
    outcomes: list[GameOutcome] = []
    wins = 0
    losses = 0
    stopped = 0
    per_seating = {
        "candidate_first_seat": {"wins": 0, "decisive": 0},
        "candidate_second_seat": {"wins": 0, "decisive": 0},
    }
    for seed in range(seed_start, seed_start + base_seeds):
        for candidate_first in (True, False):
            provider_one = candidate if candidate_first else opponent
            provider_two = opponent if candidate_first else candidate
            winner_seat, half_rounds = play_match_game(
                provider_one,
                provider_two,
                seed=seed,
                start_clock=start_clock,
                max_half_rounds=max_half_rounds,
                game_index=len(outcomes),
                pure_dth=pure_dth,
            )
            if winner_seat is None:
                winner_agent = None
                stopped += 1
            else:
                candidate_seat = "Hal" if candidate_first else "Baku"
                winner_agent = (
                    candidate_name if winner_seat == candidate_seat else opponent_name
                )
                seating = (
                    "candidate_first_seat"
                    if candidate_first
                    else "candidate_second_seat"
                )
                per_seating[seating]["decisive"] += 1
                if winner_agent == candidate_name:
                    wins += 1
                    per_seating[seating]["wins"] += 1
                else:
                    losses += 1
            outcomes.append(
                GameOutcome(
                    seed=seed,
                    first_seat_agent=(
                        candidate_name if candidate_first else opponent_name
                    ),
                    winner_agent=winner_agent,
                    half_rounds=half_rounds,
                )
            )
        if stop_early and sprt_verdict(wins, losses)["decision"] != "continue":
            break

    sprt = sprt_verdict(wins, losses)
    seating_rates = {
        name: (counts["wins"] / counts["decisive"] if counts["decisive"] else None)
        for name, counts in per_seating.items()
    }
    gates = {
        "sprt_accepts_candidate": sprt["decision"] == "accept-h1",
        "both_seatings_at_least_even": all(
            rate is not None and rate >= 0.5 for rate in seating_rates.values()
        ),
    }
    return {
        "schema_version": "arena-match-report-v1",
        "candidate": candidate_name,
        "opponent": opponent_name,
        "start_clock": start_clock,
        "max_half_rounds": max_half_rounds,
        "pure_dth": bool(pure_dth),
        "seed_start": seed_start,
        "games": [outcome.__dict__ for outcome in outcomes],
        "stopped_games": stopped,
        "per_seating": {
            name: {**counts, "win_rate": seating_rates[name]}
            for name, counts in per_seating.items()
        },
        "sprt": sprt,
        "gates": gates,
        "candidate_summaries": [
            summary()
            for summary in (getattr(candidate, "match_summary", None),)
            if callable(summary)
        ],
    }


def write_report(report: dict[str, object], output: str | Path) -> Path:
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return destination
