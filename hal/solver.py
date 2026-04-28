"""Compatibility exports for legacy Hal matrix solvers.

New code should import ``solve_minimax`` from ``environment.cfr.minimax`` and
``best_response`` from ``hal.action_model``.
"""

from environment.cfr.minimax import solve_minimax

from .action_model import best_response

__all__ = ["solve_minimax", "best_response"]

