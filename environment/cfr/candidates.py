"""Candidate exact-second generation for selective rigorous search.

A candidate set is a small subset of legal exact seconds that captures the
strategically meaningful actions for a state. Inclusion rules:

  - Always: legal critical seconds (1, 2, 58, 59, 60, 61).
  - Diagonal regime change: for each candidate drop d, also d-1, d, d+1
    in the checker's set, since the success/fail boundary lives at
    check == drop. (And the symmetric pass on the dropper side.)
  - Overflow boundary: ST = CYLINDER_MAX - checker.cylinder is the
    threshold ST that pushes a successful check into the terminal regime.
    Include the drop/check pairs straddling that ST.
  - Safe-check boundary: safe_st = CYLINDER_MAX - 1 - checker.cylinder is
    the largest ST that still leaves the checker safe; include the
    drop/check pairs straddling that ST.

No coarse buckets, no shaping. The output is two sorted tuples of distinct
legal seconds, ready to feed selective_search.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.Constants import CYLINDER_MAX
from src.Game import Game

from .exact_transition import ExactSearchConfig, legal_seconds_for_current_role


CRITICAL_SECONDS: tuple[int, ...] = (1, 2, 58, 59, 60, 61)


@dataclass(frozen=True)
class CandidateActions:
    drop_seconds: tuple[int, ...]
    check_seconds: tuple[int, ...]

    @property
    def joint_count(self) -> int:
        return len(self.drop_seconds) * len(self.check_seconds)


def overflow_st_threshold(checker_cylinder: float) -> int:
    """Smallest ST that drives the cylinder to CYLINDER_MAX or above."""
    return max(1, int(CYLINDER_MAX) - int(checker_cylinder))


def safe_st_budget(checker_cylinder: float) -> int:
    """Largest ST that leaves the cylinder strictly below CYLINDER_MAX."""
    return max(0, int(CYLINDER_MAX) - 1 - int(checker_cylinder))


def _legal_filter(seconds: set[int], legal: range) -> set[int]:
    return {s for s in seconds if s in legal}


def generate_candidates(game: Game, config: ExactSearchConfig | None = None) -> CandidateActions:
    """Return candidate exact seconds for the dropper and checker."""
    config = config or ExactSearchConfig()
    dropper, checker = game.get_roles_for_half(game.current_half)
    drop_legal = legal_seconds_for_current_role(game, dropper.name, "dropper", config)
    check_legal = legal_seconds_for_current_role(game, checker.name, "checker", config)

    drop_seconds: set[int] = _legal_filter(set(CRITICAL_SECONDS), drop_legal)
    check_seconds: set[int] = _legal_filter(set(CRITICAL_SECONDS), check_legal)

    overflow_st = overflow_st_threshold(checker.cylinder)
    safe_st = safe_st_budget(checker.cylinder)

    for d in tuple(drop_seconds):
        for c in (
            d - 1, d, d + 1,
            d + safe_st, d + safe_st + 1,
            d + overflow_st - 1, d + overflow_st,
        ):
            if c in check_legal:
                check_seconds.add(c)

    for c in tuple(check_seconds):
        for d in (
            c - 1, c, c + 1,
            c - safe_st - 1, c - safe_st,
            c - overflow_st, c - overflow_st + 1,
        ):
            if d in drop_legal:
                drop_seconds.add(d)

    return CandidateActions(
        drop_seconds=tuple(sorted(drop_seconds)),
        check_seconds=tuple(sorted(check_seconds)),
    )
