"""Hal package public surface.

CanonicalHal is the playable legacy/search opponent. The package is organized
around a few explicit concerns:

- ``hal.state``: belief, memory, and Hal-owned state records.
- ``hal.action_model``: legacy bucketed action abstraction and matrix helpers.
- ``hal.search``: bucketed forward search for CanonicalHal.
- ``hal.value_net`` / ``hal.train`` / ``hal.self_play``: experimental value
  approximation and training.

Rigorous exact-second CFR/search lives in ``environment.cfr``.
"""

from .hal_opponent import CanonicalHal
from .state import BeliefState, Bucket, HalState, MemoryMode

__all__ = [
    "BeliefState",
    "Bucket",
    "CanonicalHal",
    "HalState",
    "MemoryMode",
]
