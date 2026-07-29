"""Bounded-resolve pure-DTH policy adapter for the canonical arena.

The projection is the identity at one-second resolution: the four canonical
load/TTD coordinates are field-for-field the role-relative pure-DTH state.

Documented approximations, per this package's adapter rules:

- Pure DTH has no leap second.  The Hal seat can never legally play 61, so
  the agent's own action space matches the engine exactly; an opponent's
  leap-window drop at 61 is unmodeled, the same leap-blind posture as the
  abstract adapter.
- The live engine still uses the pre-freeze revival surface (the pending
  migration is recorded in ``docs/REVIVAL_MODEL.md``).  Its eligibility
  zero-set matches pure DTH exactly, so reachability and terminal boundaries
  agree; only revival probabilities differ.
- Engine cylinder/TTD floats are floored and clamped to the integer DTH grid.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping

import numpy as np

from arena.contracts import CanonicalDecision
from dth.agent import BoundedResolveAgent, MoveDecision, NetworkLeafModel, ResolveBudget


def project_to_dth_state(decision: CanonicalDecision) -> tuple[int, int, int, int]:
    """Floor canonical seconds onto the integer role-relative DTH state."""

    def load(seconds: float) -> int:
        return min(299, max(0, int(seconds)))

    def ttd(seconds: float) -> int:
        return min(300, max(0, int(seconds)))

    return (
        load(decision.checker_cylinder_seconds),
        ttd(decision.checker_ttd_seconds),
        load(decision.dropper_cylinder_seconds),
        ttd(decision.dropper_ttd_seconds),
    )


@dataclass
class DTHResolvePolicyProvider:
    """Serve bounded-resolve mixed policies and keep per-move provenance."""

    tablebase_path: Path | None = None
    checkpoint_path: Path | None = None
    budget: ResolveBudget = field(default_factory=ResolveBudget)
    decisions: list[MoveDecision] = field(default_factory=list, repr=False)
    _agent: BoundedResolveAgent = field(init=False, repr=False)

    def __post_init__(self) -> None:
        network = (
            NetworkLeafModel(self.checkpoint_path)
            if self.checkpoint_path is not None
            else None
        )
        self._agent = BoundedResolveAgent(
            tablebase_path=self.tablebase_path,
            network=network,
            budget=self.budget,
        )
        self._agent.__enter__()

    def close(self) -> None:
        self._agent.__exit__(None, None, None)

    def policy(self, decision: CanonicalDecision) -> Mapping[int, float]:
        move = self._agent.decide(project_to_dth_state(decision))
        self.decisions.append(move)
        row = (
            move.drop_policy if decision.role == "dropper" else move.check_policy
        )
        return {
            second: float(probability)
            for second, probability in enumerate(row, start=1)
            if probability > 0.0
        }

    def match_summary(self) -> str:
        """G5 certified-play fraction and G6 latency, per docs/AGENT_GOAL.md."""

        if not self.decisions:
            return "dth: no moves played"
        certified = sum(
            1
            for move in self.decisions
            if move.provenance in ("complete-game-exact", "finite-horizon-exact")
        )
        latencies = np.asarray(
            [move.elapsed_seconds for move in self.decisions], dtype=np.float64
        )
        provenance_counts: dict[str, int] = {}
        for move in self.decisions:
            provenance_counts[move.provenance] = (
                provenance_counts.get(move.provenance, 0) + 1
            )
        breakdown = ", ".join(
            f"{name}={count}" for name, count in sorted(provenance_counts.items())
        )
        return (
            f"dth: {len(self.decisions)} moves, certified {certified}"
            f"/{len(self.decisions)}"
            f" ({breakdown}); latency p95 {float(np.quantile(latencies, 0.95)):.3f}s"
            f" max {float(np.max(latencies)):.3f}s"
        )
