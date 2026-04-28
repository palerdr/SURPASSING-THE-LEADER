"""Foundational utility contract for rigorous CFR/search.

This module intentionally contains only terminal match utility. Route shaping,
handcrafted evaluation, and neural value estimates are experimental layers and
must not be imported by rigorous CFR modules.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.Game import Game


@dataclass(frozen=True)
class UtilityBreakdown:
    """Terminal-only utility summary from Hal's perspective."""

    value: float
    hal_win_probability: float
    baku_win_probability: float
    unresolved_probability: float


def terminal_value(game: Game, perspective_name: str = "Hal") -> float | None:
    """Return terminal utility, or None when the match is unresolved."""
    if not game.game_over:
        return None
    if game.winner is None:
        return 0.0
    return 1.0 if game.winner.name.lower() == perspective_name.lower() else -1.0


def terminal_breakdown(game: Game, perspective_name: str = "Hal") -> UtilityBreakdown:
    """Return a terminal-only breakdown for a single deterministic state."""
    value = terminal_value(game, perspective_name=perspective_name)
    if value is None:
        return UtilityBreakdown(
            value=0.0,
            hal_win_probability=0.0,
            baku_win_probability=0.0,
            unresolved_probability=1.0,
        )
    if value > 0:
        return UtilityBreakdown(value=value, hal_win_probability=1.0, baku_win_probability=0.0, unresolved_probability=0.0)
    if value < 0:
        return UtilityBreakdown(value=value, hal_win_probability=0.0, baku_win_probability=1.0, unresolved_probability=0.0)
    return UtilityBreakdown(value=value, hal_win_probability=0.0, baku_win_probability=0.0, unresolved_probability=0.0)

