"""Compatibility exports for Hal memory updates.

New code should import from ``hal.state``.
"""

from .state import hal_can_check_leap, update_memory

__all__ = ["hal_can_check_leap", "update_memory"]

