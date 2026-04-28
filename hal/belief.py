"""Compatibility exports for Hal belief updates.

New code should import from ``hal.state``.
"""

from .state import EXPLOITATION_THRESHOLD, update_belief

__all__ = ["EXPLOITATION_THRESHOLD", "update_belief"]

