"""Exact public state keys for rigorous CFR/search.

No cylinder, clock, TTD, CPR, or death-count bucketing is allowed here. These
keys are for audits, transposition caches, and exact-vs-abstract comparisons.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.Game import Game


@dataclass(frozen=True)
class ExactPublicState:
    p1_name: str
    p1_physicality: float
    p1_cylinder: float
    p1_ttd: float
    p1_deaths: int
    p1_alive: bool
    p2_name: str
    p2_physicality: float
    p2_cylinder: float
    p2_ttd: float
    p2_deaths: int
    p2_alive: bool
    referee_cprs: int
    game_clock: float
    current_half: int
    round_num: int
    first_dropper_name: str
    game_over: bool
    winner_name: str | None
    loser_name: str | None


def exact_public_state(game: Game) -> ExactPublicState:
    first_dropper_name = game.first_dropper.name if game.first_dropper is not None else ""
    return ExactPublicState(
        p1_name=game.player1.name,
        p1_physicality=game.player1.physicality,
        p1_cylinder=game.player1.cylinder,
        p1_ttd=game.player1.ttd,
        p1_deaths=game.player1.deaths,
        p1_alive=game.player1.alive,
        p2_name=game.player2.name,
        p2_physicality=game.player2.physicality,
        p2_cylinder=game.player2.cylinder,
        p2_ttd=game.player2.ttd,
        p2_deaths=game.player2.deaths,
        p2_alive=game.player2.alive,
        referee_cprs=game.referee.cprs_performed,
        game_clock=game.game_clock,
        current_half=game.current_half,
        round_num=game.round_num,
        first_dropper_name=first_dropper_name,
        game_over=game.game_over,
        winner_name=game.winner.name if game.winner is not None else None,
        loser_name=game.loser.name if game.loser is not None else None,
    )

