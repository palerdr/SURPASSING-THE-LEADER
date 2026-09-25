"""Exact play-time agent backed only by the completed DTH tablebase."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from dth.complete_tablebase import CompleteTablebase, _source_digest_inputs
from dth.solver import (
    NTState,
    SADDLE_GAP_TOLERANCE,
    continuation_class_values,
    reconstruct_transition_class_matrix,
    validate_live_state,
)

CERTIFIED_SADDLE_GAP_TOLERANCE = SADDLE_GAP_TOLERANCE


def runtime_source_files() -> tuple[str, ...]:
    """Return the files that a copy of this agent needs, by repository path.

    A serving bundle that copies these paths under one directory, in the
    repository layout, can import `CompleteDTHAgent` and open a complete
    artifact built by the C backend. The list holds the modules that the agent
    imports and each input of that artifact's `code_config_digest`, the root
    `uv.lock` included. The digest labels each input with its path from the
    common root, so a bundle must keep these paths unchanged.
    """

    package = Path(__file__).resolve().parent
    root = package.parent.parent
    files = [
        package / "__init__.py",
        package / "agent.py",
        *_source_digest_inputs(include_rust=False, include_fast=True),
    ]
    return tuple(path.relative_to(root).as_posix() for path in files)


@dataclass(frozen=True)
class MoveDecision:
    """One exact equilibrium decision and its freshly checked certificate."""

    state: NTState
    value: float
    drop_policy: tuple[float, ...]
    check_policy: tuple[float, ...]
    saddle_gap: float
    elapsed_seconds: float


@dataclass(frozen=True, slots=True)
class CertifiedStageGame:
    """One complete-tablebase Bellman stage game and its saddle certificate."""

    state: NTState
    value: float
    matrix: np.ndarray
    drop_policy: np.ndarray
    check_policy: np.ndarray
    saddle_gap: float


class CompleteDTHAgent:
    """Read-only facade over the canonical complete-game artifact.

    Canonical STL play reaches only TTD 0 or TTD at least 60, so every state
    projected from the arena lies in the tablebase's transition-closed domain.
    Missing, corrupt, or off-domain artifacts fail closed; play never silently
    substitutes a partial or learned answer.
    """

    def __init__(self, artifact_dir: str | Path, *, verify_hashes: bool = True) -> None:
        """Open the artifact. Hashing the arrays is the default and stays the
        default; `verify_hashes=False` is for a serving copy whose arrays were
        hashed when it was packaged, where rehashing gigabytes at every process
        start only delays the first answer. The manifest, schema, source digest,
        and array contracts are checked either way.
        """
        self.tablebase = CompleteTablebase(
            artifact_dir=artifact_dir, verify_hashes=verify_hashes
        )

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

    def stage_game(self, state: NTState) -> CertifiedStageGame:
        """Return the certified continuation-adjusted literal-action game.

        The complete artifact remains the sole continuation-value authority.
        The matrix and both policies are rebuilt and checked on demand; a
        corrupt, incompatible, or uncertified artifact therefore fails before
        any downstream controller can act on it.
        """

        normalized = validate_live_state(state)
        certificate = self.tablebase.certificate(normalized)
        successful, failed = continuation_class_values(
            normalized,
            lambda child: float(self.tablebase.lookup(child)["value"]),
        )
        matrix = reconstruct_transition_class_matrix(successful, failed)
        drop = np.asarray(certificate["drop_policy"], dtype=np.float64).copy()
        check = np.asarray(certificate["check_policy"], dtype=np.float64).copy()
        if matrix.shape != (60, 60) or not np.all(np.isfinite(matrix)):
            raise RuntimeError("complete DTH stage matrix is not finite 60x60")
        for label, policy in (("Dropper", drop), ("Checker", check)):
            if (
                policy.shape != (60,)
                or not np.all(np.isfinite(policy))
                or np.any(policy < 0.0)
                or abs(float(policy.sum()) - 1.0) > 1e-8
            ):
                raise RuntimeError(f"complete DTH {label} policy is malformed")
        lower = float(np.min(matrix.T @ drop))
        upper = float(np.max(matrix @ check))
        gap = max(0.0, upper - lower)
        if gap > SADDLE_GAP_TOLERANCE:
            raise RuntimeError(
                f"complete DTH stage saddle gap {gap:g} exceeds "
                f"{SADDLE_GAP_TOLERANCE:g}"
            )
        value = float(certificate["value"])
        if value < lower - SADDLE_GAP_TOLERANCE or value > upper + SADDLE_GAP_TOLERANCE:
            raise RuntimeError(
                "complete DTH stored value lies outside the certified saddle edges"
            )
        matrix.setflags(write=False)
        drop.setflags(write=False)
        check.setflags(write=False)
        return CertifiedStageGame(
            state=normalized,
            value=value,
            matrix=matrix,
            drop_policy=drop,
            check_policy=check,
            saddle_gap=gap,
        )
