"""Algorithm-neutral policy interface for canonical live play."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Protocol, runtime_checkable


@dataclass(frozen=True, slots=True)
class CanonicalDecision:
    """Public canonical decision context passed to a policy provider.

    The numeric fields are literal engine seconds. ``native_state`` is an
    opaque escape hatch for a provider that natively understands the canonical
    engine (for example the STL MCTS agent); abstraction adapters ignore it.
    """

    role: str
    actor_name: str
    turn_duration: int
    legal_seconds: tuple[int, ...]
    checker_cylinder_seconds: float
    checker_ttd_seconds: float
    dropper_cylinder_seconds: float
    dropper_ttd_seconds: float
    native_state: object


@runtime_checkable
class CanonicalPolicyProvider(Protocol):
    """Provide an unnormalized literal-second policy for one decision."""

    def policy(self, decision: CanonicalDecision) -> Mapping[int, float]: ...
