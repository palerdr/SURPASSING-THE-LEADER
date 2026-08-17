"""Compatibility adapter for externally supplied legacy STL policy objects.

The repository no longer ships the ``stl.play`` stack, so Arena does not
advertise or construct this provider. Keeping the structural adapter importable
lets downstream experiments fail at their own explicit dependency boundary.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Protocol

import numpy as np

from arena.contracts import CanonicalDecision


class LegacySTLSolver(Protocol):
    def policy(self, native_state: object, role: str) -> tuple[object, object]: ...


@dataclass(slots=True)
class STLSolverPolicyProvider:
    """Wrap a full canonical SolverAgent without changing its algorithm."""

    solver: LegacySTLSolver

    def policy(self, decision: CanonicalDecision) -> Mapping[int, float]:
        seconds, probabilities = self.solver.policy(decision.native_state, decision.role)
        return {
            int(second): float(probability)
            for second, probability in zip(seconds, np.asarray(probabilities, dtype=np.float64))
        }
