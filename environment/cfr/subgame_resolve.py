"""Subgame re-solve at critical states (Phase 6, Stockfish-style runtime polish).

The blueprint (trained value net + MCTS) covers the public-state space at
finite resolution. For *critical decisions* — within K half-rounds of the
leap window, near-overflow cylinder, about to flip the Active LSR parity
— we want a fresh local solve at deeper horizon at runtime rather than
relying on the net's prediction. This is the **Pluribus / Libratus pattern**
(Brown & Sandholm 2017, 2018) adapted to perfect-info: at critical positions,
do extra search anchored to the blueprint at the boundary.

Pure rigorous-core module: no shaping imports, no bucketing, no curriculum
labels. Reads timing predicates from ``timing_features`` and runs a fresh
``selective_solve`` at the requested horizon.
"""

from __future__ import annotations

from src.Game import Game

from .exact import ExactSearchConfig
from .selective import SelectiveSearchResult, selective_solve
from .timing_features import (
    current_checker_fail_would_activate_lsr,
    is_active_lsr,
    rounds_until_leap_window,
)


__all__ = ["is_critical", "resolve_subgame"]


_NEAR_OVERFLOW_CYLINDER = 240.0
_LEAP_WINDOW_PROXIMITY = 2


def is_critical(game: Game) -> bool:
    """Heuristic gate for when subgame re-solving is worthwhile.

    Flags states near the leap window, near overflow, or about to flip
    LSR parity. The blueprint is most likely to misjudge these decisions
    because their consequences only become observable many half-rounds
    after the action.
    """
    if game.game_over:
        return False
    return any(
        (
            rounds_until_leap_window(game) <= _LEAP_WINDOW_PROXIMITY,
            current_checker_fail_would_activate_lsr(game),
            is_active_lsr(game) and game.current_half == 1,
            game.player1.cylinder >= _NEAR_OVERFLOW_CYLINDER,
            game.player2.cylinder >= _NEAR_OVERFLOW_CYLINDER,
        )
    )


def resolve_subgame(
    game: Game,
    horizon: int = 4,
    config: ExactSearchConfig | None = None,
) -> SelectiveSearchResult:
    """Fresh selective_solve at deeper horizon for a critical state.

    The blueprint is bypassed here: this returns the equilibrium value
    and strategies for the candidate matrix at depth ``horizon``, with
    no learned-net frontier. Used at runtime when ``is_critical`` flags
    the current state.
    """
    config = config or ExactSearchConfig()
    return selective_solve(game, horizon, config)
