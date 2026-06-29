"""Frozen-ladder match gate: ladder runner, Wilson CIs, gate report.

Plays a candidate Hal policy (any ``ChooseAction`` callable) against a
fixed ladder of scripted Baku opponents via ``training.tournament.play_match``
and aggregates per-opponent W/D/L, Wilson confidence intervals, and
simplified-GSPRT verdicts (see ``training.strength.sprt``).

Two facts about ``play_match`` this module compensates for (verified in
``training/tournament.py``):

  1. **Game length is already bounded.** The half-round loop carries
     ``_HALF_ROUND_SAFETY_LIMIT = 200``; a game that reaches it is
     abandoned, scored as a draw, and classified ``"unfinished"`` in
     ``cause_of_termination``. No extra wrapping is needed for runaway
     games.
  2. **It never calls ``Opponent.reset()``.** It only sees bare
     callables, so a stateful opponent would leak per-game state across
     the ``n_games`` of a match. :func:`reset_per_game` wraps an
     ``Opponent`` so ``reset()`` fires at every fresh-game boundary
     (the single call per game where ``game.history`` is empty).
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Callable

from environment.opponents.base import Opponent
from environment.opponents.factory import create_scripted_opponent
from training.tournament import ChooseAction, MatchResult, play_match

from .sprt import sprt_llr, sprt_verdict

__all__ = [
    "LadderEntry",
    "reset_per_game",
    "run_ladder",
    "run_ladder_entries",
    "wilson_interval",
    "gate_report",
]


# Mirrors training.tournament._HALF_ROUND_SAFETY_LIMIT (private there;
# duplicated rather than imported so this module never reaches into a
# private name — keep in sync if tournament.py ever changes it).
_ENGINE_HALF_ROUND_CAP = 200


def _opponent_seed(base_seed: int, name: str) -> int:
    digest = hashlib.sha256(f"{base_seed}|{name}".encode()).digest()
    return int.from_bytes(digest[:4], "little")


@dataclass(frozen=True)
class LadderEntry:
    """One rung of the frozen ladder: a label plus an opponent factory.

    The factory is invoked once per :func:`run_ladder_entries` call so
    every ladder run starts from a pristine opponent instance.
    """

    name: str
    opponent_factory: Callable[[], Opponent]


def reset_per_game(opponent: Opponent) -> ChooseAction:
    """Adapt an ``Opponent`` to ``ChooseAction`` with per-game resets.

    ``play_match`` consults each side exactly once per half-round, so
    within a default-start game there is exactly one call that observes
    an empty ``game.history`` — the first half-round of a fresh game.
    We fire ``opponent.reset()`` on that call. (Scenario starts with
    pre-populated history would defeat this heuristic; the ladder only
    uses canonical R1T1 starts.)
    """

    def choose(game, role: str, turn_duration: int) -> int:
        if not game.history:
            opponent.reset()
        return opponent.choose_action(game, role, turn_duration)

    return choose


def run_ladder_entries(
    hal_choose_action: ChooseAction,
    entries: list[LadderEntry],
    n_games: int,
    seed: int,
    max_half_rounds: int | None = None,
) -> dict[str, MatchResult]:
    """Play ``hal_choose_action`` against every ladder entry.

    ``max_half_rounds`` cannot be set tighter than the engine's built-in
    200-half-round safety cap without modifying ``tournament.py`` (which
    this module deliberately does not); passing a smaller value raises.
    ``None`` (or any value >= 200) defers to the engine cap, under which
    over-long games are scored as draws with cause ``"unfinished"``.
    """
    if max_half_rounds is not None and max_half_rounds < _ENGINE_HALF_ROUND_CAP:
        raise ValueError(
            f"max_half_rounds={max_half_rounds} is tighter than play_match's "
            f"built-in safety cap ({_ENGINE_HALF_ROUND_CAP}); the cap cannot "
            "be lowered without modifying training/tournament.py."
        )

    results: dict[str, MatchResult] = {}
    for entry in entries:
        opponent = entry.opponent_factory()
        results[entry.name] = play_match(
            hal_choose_action=hal_choose_action,
            baku_choose_action=reset_per_game(opponent),
            n_games=n_games,
            seed=seed,
        )
    return results


def run_ladder(
    hal_choose_action: ChooseAction,
    opponent_names: list[str],
    n_games: int,
    seed: int,
    max_half_rounds: int | None = None,
) -> dict[str, MatchResult]:
    """Run the ladder against factory-registered opponents by name."""
    entries = [
        LadderEntry(
            name=name,
            # Bind ``name`` at definition time (default-arg idiom).
            opponent_factory=lambda name=name: create_scripted_opponent(
                name,
                seed=_opponent_seed(seed, name),
            ),
        )
        for name in opponent_names
    ]
    return run_ladder_entries(
        hal_choose_action,
        entries,
        n_games=n_games,
        seed=seed,
        max_half_rounds=max_half_rounds,
    )


def wilson_interval(wins: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion ``wins / n``.

    Returns ``(lo, hi)``; ``(0.0, 1.0)`` when ``n == 0`` (no evidence).
    """
    if n <= 0:
        return (0.0, 1.0)
    if not (0 <= wins <= n):
        raise ValueError(f"wins must lie in [0, n]; got wins={wins}, n={n}")

    p_hat = wins / n
    z2 = z * z
    denom = 1.0 + z2 / n
    center = (p_hat + z2 / (2.0 * n)) / denom
    half = (z / denom) * math.sqrt(
        p_hat * (1.0 - p_hat) / n + z2 / (4.0 * n * n)
    )
    return (max(0.0, center - half), min(1.0, center + half))


def gate_report(
    results: dict[str, MatchResult],
    *,
    elo0: float = 0.0,
    elo1: float = 50.0,
    alpha: float = 0.05,
    beta: float = 0.05,
    z: float = 1.96,
) -> dict:
    """Aggregate ladder results into per-opponent stats + SPRT verdicts.

    All quantities are from the Hal (candidate) seat's perspective:
    ``wins`` = ``hal_wins``, ``losses`` = ``baku_wins``. The SPRT tests
    H1 "candidate is >= ``elo1`` stronger than this opponent" against
    H0 "<= ``elo0``" — with the defaults, a verdict of ``accept`` means
    "confidently stronger than 50%+elo1 score vs this rung".
    """
    per_opponent: dict[str, dict] = {}
    for name, result in results.items():
        wins = result.hal_wins
        losses = result.baku_wins
        draws = result.draws
        n = result.games_played
        win_rate = wins / n if n else 0.0
        score_rate = (wins + 0.5 * draws) / n if n else 0.0
        lo, hi = wilson_interval(wins, n, z=z)
        per_opponent[name] = {
            "games": n,
            "wins": wins,
            "draws": draws,
            "losses": losses,
            "win_rate": win_rate,
            "score_rate": score_rate,
            "wilson_lo": lo,
            "wilson_hi": hi,
            "llr": sprt_llr(wins, draws, losses, elo0=elo0, elo1=elo1),
            "sprt": sprt_verdict(
                wins, draws, losses,
                elo0=elo0, elo1=elo1, alpha=alpha, beta=beta,
            ),
            "avg_game_length_half_rounds": result.avg_game_length_half_rounds,
            "cause_of_termination": dict(result.cause_of_termination),
        }

    total_games = sum(r.games_played for r in results.values())
    total_wins = sum(r.hal_wins for r in results.values())
    total_draws = sum(r.draws for r in results.values())
    return {
        "elo0": elo0,
        "elo1": elo1,
        "alpha": alpha,
        "beta": beta,
        "opponents": per_opponent,
        "overall": {
            "games": total_games,
            "wins": total_wins,
            "draws": total_draws,
            "losses": total_games - total_wins - total_draws,
            "win_rate": total_wins / total_games if total_games else 0.0,
            "score_rate": (
                (total_wins + 0.5 * total_draws) / total_games
                if total_games
                else 0.0
            ),
        },
    }
