"""Adapter exposing the existing STL MCTS/RL policy through the arena API."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np

from arena.contracts import CanonicalDecision
from stl.play.agent import SolverAgent


@dataclass(slots=True)
class STLSolverPolicyProvider:
    """Wrap a full canonical SolverAgent without changing its algorithm."""

    solver: SolverAgent

    def policy(self, decision: CanonicalDecision) -> Mapping[int, float]:
        seconds, probabilities = self.solver.policy(decision.native_state, decision.role)
        return {
            int(second): float(probability)
            for second, probability in zip(seconds, np.asarray(probabilities, dtype=np.float64))
        }
