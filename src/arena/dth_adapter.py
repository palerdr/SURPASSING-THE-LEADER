"""Complete DTH tablebase adapter for the canonical STL arena.

The four load/TTD coordinates and every ordinary action are identical between
pure DTH and canonical STL. The sole prospective rules difference is the
public leap window: Baku as Dropper may play 61 there, while a DTH policy still
supplies actions 1..60. Arena owns legality and sampling around that one extra
canonical action.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping

import numpy as np

from arena.contracts import CanonicalDecision
from dth.agent import CompleteDTHAgent, MoveDecision
from stl.engine.game import Game


class PureDTHGame(Game):
    """Shared canonical mechanics with pure DTH's permanent 60-action turn."""

    def get_turn_duration(self) -> int:
        return 60


def project_to_dth_state(decision: CanonicalDecision) -> tuple[int, int, int, int]:
    """Project the shared literal-second coordinates without approximation."""

    coordinates = (
        decision.checker_cylinder_seconds,
        decision.checker_ttd_seconds,
        decision.dropper_cylinder_seconds,
        decision.dropper_ttd_seconds,
    )
    if any(value != int(value) for value in coordinates):
        raise ValueError("canonical STL state is not on the literal-second DTH grid")
    return tuple(int(value) for value in coordinates)  # type: ignore[return-value]


@dataclass
class DTHCompletePolicyProvider:
    """Serve exact equilibrium policies from the completed tablebase."""

    artifact_dir: Path
    decisions: list[MoveDecision] = field(default_factory=list, repr=False)
    _agent: CompleteDTHAgent = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._agent = CompleteDTHAgent(self.artifact_dir)

    def close(self) -> None:
        """Match the provider lifecycle contract; memmaps need no explicit close."""

    def policy(self, decision: CanonicalDecision) -> Mapping[int, float]:
        move = self._agent.decide(project_to_dth_state(decision))
        self.decisions.append(move)
        row = move.drop_policy if decision.role == "dropper" else move.check_policy
        return {
            second: float(probability)
            for second, probability in enumerate(row, start=1)
            if probability > 0.0
        }

    def match_summary(self) -> str:
        if not self.decisions:
            return "dth complete: no moves played"
        latencies = np.asarray(
            [move.elapsed_seconds for move in self.decisions], dtype=np.float64
        )
        worst_gap = max(move.saddle_gap for move in self.decisions)
        return (
            f"dth complete: {len(self.decisions)} exact moves; "
            f"worst saddle gap {worst_gap:.3g}; "
            f"latency p95 {float(np.quantile(latencies, 0.95)):.3f}s "
            f"max {float(np.max(latencies)):.3f}s"
        )
