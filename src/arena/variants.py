"""Rule variants of the canonical STL referee that arena surfaces can select.

A variant changes one engine rule and nothing else. This module imports the
STL engine alone, so the headless referee in ``match.py`` and the play surfaces
can select a variant without loading a solver.
"""

from __future__ import annotations

from stl.engine.game import Game


class PureDTHGame(Game):
    """Shared canonical mechanics with pure DTH's permanent 60-action turn."""

    def get_turn_duration(self) -> int:
        return 60
