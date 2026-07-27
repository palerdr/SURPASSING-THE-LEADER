"""Literal-second policy adapter for the exact abstract tablebases.

This is intentionally a policy transfer, not a claim that the tablebase solves
the full canonical STL game.  It projects canonical load/TTD downward to
completed buckets and lifts bucket action ``a`` to literal second
``bucket_seconds * a``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping

import numpy as np

from abstract.packed_tablebase import PackedTablebase
from abstract.rules import AbstractRuleset, ruleset_for_name
from abstract.state import AbstractState
from abstract.tablebase import load_tablebase
from arena.contracts import CanonicalDecision


def _completed_bucket(seconds: float, *, bucket_seconds: int, maximum: int) -> int:
    return min(maximum, max(0, int(seconds) // bucket_seconds))


def project_to_abstract_state(
    decision: CanonicalDecision,
    rules: AbstractRuleset | None = None,
) -> AbstractState:
    """Floor canonical seconds to the tablebase's role-relative bucket state."""

    rules = ruleset_for_name("bucket6_frozen95") if rules is None else rules
    bucket_seconds = rules.bucket_seconds
    maximum_load = rules.load_cap_units - 1
    maximum_ttd = rules.load_cap_units
    return AbstractState(
        checker_load=_completed_bucket(
            decision.checker_cylinder_seconds,
            bucket_seconds=bucket_seconds,
            maximum=maximum_load,
        ),
        checker_ttd=_completed_bucket(
            decision.checker_ttd_seconds,
            bucket_seconds=bucket_seconds,
            maximum=maximum_ttd,
        ),
        dropper_load=_completed_bucket(
            decision.dropper_cylinder_seconds,
            bucket_seconds=bucket_seconds,
            maximum=maximum_load,
        ),
        dropper_ttd=_completed_bucket(
            decision.dropper_ttd_seconds,
            bucket_seconds=bucket_seconds,
            maximum=maximum_ttd,
        ),
    )


def _state_key(state: AbstractState) -> tuple[int, int, int, int]:
    return (state.checker_load, state.checker_ttd, state.dropper_load, state.dropper_ttd)


@dataclass
class AbstractTablebasePolicyProvider:
    """Use an exact abstract policy as a sparse canonical literal-second policy."""

    tablebase_path: Path
    bucket_seconds: int = 10
    tablebase_manifest: Path | None = None
    _policies: dict[tuple[int, int, int, int], tuple[np.ndarray, np.ndarray]] = field(
        init=False,
        repr=False,
        default_factory=dict,
    )
    _packed: PackedTablebase | None = field(init=False, repr=False, default=None)
    _rules: AbstractRuleset = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.bucket_seconds not in (5, 10):
            raise ValueError("abstract bucket width must be 5 or 10 seconds")
        ruleset_id = (
            "bucket12_frozen95"
            if self.bucket_seconds == 5
            else "bucket6_frozen95"
        )
        self._rules = ruleset_for_name(ruleset_id)
        self.tablebase_path = Path(self.tablebase_path)

        if self.tablebase_path.is_dir() or self.tablebase_path.name == "tablebase.json":
            artifact_dir = (
                self.tablebase_path.parent
                if self.tablebase_path.name == "tablebase.json"
                else self.tablebase_path
            )
            self._packed = PackedTablebase(artifact_dir)
            metadata = self._packed.manifest["metadata"]
            self._validate_ruleset(str(metadata["ruleset_id"]))
            return

        manifest = self.tablebase_manifest or self.tablebase_path.with_suffix(".json")
        loaded = load_tablebase(self.tablebase_path, manifest)
        metadata = loaded["metadata"]
        self._validate_ruleset(str(metadata["ruleset_id"]))
        arrays = loaded["arrays"]
        self._policies = {
            tuple(int(value) for value in state): (drop, check)
            for state, drop, check in zip(arrays["states"], arrays["drop_policy"], arrays["check_policy"])
        }

    def _validate_ruleset(self, actual_ruleset_id: str) -> None:
        if actual_ruleset_id != self._rules.ruleset_id:
            raise ValueError(
                f"abstract tablebase ruleset {actual_ruleset_id!r} is incompatible with "
                f"adapter {self._rules.ruleset_id!r}"
            )

    def policy(self, decision: CanonicalDecision) -> Mapping[int, float]:
        projected = project_to_abstract_state(decision, self._rules)
        if self._packed is not None:
            packed_row = self._packed.lookup(projected)
            row = (
                packed_row["drop_policy"]
                if decision.role == "dropper"
                else packed_row["check_policy"]
            )
        else:
            key = _state_key(projected)
            rows = self._policies.get(key)
            if rows is None:
                raise LookupError(
                    f"projected canonical state {key} is not covered by the abstract tablebase"
                )
            row = rows[0] if decision.role == "dropper" else rows[1]
        return {
            self.bucket_seconds * (index + 1): float(probability)
            for index, probability in enumerate(row)
            if probability > 0.0
        }
