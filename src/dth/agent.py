"""Exact play-time agent backed only by the completed DTH tablebase."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

from dth.complete_tablebase import CompleteTablebase
from dth.solver import NTState, validate_live_state


@dataclass(frozen=True)
class MoveDecision:
    """One exact equilibrium decision and its freshly checked certificate."""

    state: NTState
    value: float
    drop_policy: tuple[float, ...]
    check_policy: tuple[float, ...]
    saddle_gap: float
    elapsed_seconds: float


class CompleteDTHAgent:
    """Read-only facade over the canonical complete-game artifact.

    Canonical STL play reaches only TTD 0 or TTD at least 60, so every state
    projected from the arena lies in the tablebase's transition-closed domain.
    Missing, corrupt, or off-domain artifacts fail closed; play never silently
    substitutes a partial or learned answer.
    """

    def __init__(self, artifact_dir: str | Path) -> None:
        self.tablebase = CompleteTablebase(artifact_dir=artifact_dir)

    def decide(self, state: NTState) -> MoveDecision:
        started = time.monotonic()
        normalized = validate_live_state(state)
        certificate = self.tablebase.certificate(normalized)
        return MoveDecision(
            state=normalized,
            value=float(certificate["value"]),
            drop_policy=tuple(float(p) for p in certificate["drop_policy"]),
            check_policy=tuple(float(p) for p in certificate["check_policy"]),
            saddle_gap=float(certificate["saddle_gap"]),
            elapsed_seconds=time.monotonic() - started,
        )
