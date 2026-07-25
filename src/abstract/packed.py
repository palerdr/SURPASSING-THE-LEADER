"""Packed integer state encoding and transition parity surface.

The tablebase hot path operates only on integer indices.  ``AbstractState``
objects remain the readable Python rules authority and are constructed only at
API, test, lookup, and export boundaries.
"""

from __future__ import annotations

from dataclasses import dataclass

from abstract.rules import AbstractRuleset


@dataclass(frozen=True, slots=True)
class PackedStateCodec:
    """Mixed-radix bijection for the complete physical state domain."""

    load_cap_units: int

    def __post_init__(self) -> None:
        if self.load_cap_units <= 0:
            raise ValueError("load_cap_units must be positive")

    @property
    def ttd_size(self) -> int:
        return self.load_cap_units + 1

    @property
    def state_count(self) -> int:
        cap = self.load_cap_units
        return cap * (cap + 1) * cap * (cap + 1)

    @property
    def maximum_potential(self) -> int:
        return 4 * self.load_cap_units - 2

    def encode(
        self,
        checker_load: int,
        checker_ttd: int,
        dropper_load: int,
        dropper_ttd: int,
    ) -> int:
        cap = self.load_cap_units
        ttd_size = cap + 1
        fields = (checker_load, checker_ttd, dropper_load, dropper_ttd)
        if any(isinstance(value, bool) or not isinstance(value, int) for value in fields):
            raise TypeError("packed state fields must be integers")
        if not 0 <= checker_load < cap or not 0 <= dropper_load < cap:
            raise ValueError(f"load fields must be in 0..{cap - 1}")
        if not 0 <= checker_ttd <= cap or not 0 <= dropper_ttd <= cap:
            raise ValueError(f"TTD fields must be in 0..{cap}")
        return (((checker_load * ttd_size + checker_ttd) * cap + dropper_load) * ttd_size) + dropper_ttd

    def encode_unchecked(
        self,
        checker_load: int,
        checker_ttd: int,
        dropper_load: int,
        dropper_ttd: int,
    ) -> int:
        cap = self.load_cap_units
        ttd_size = cap + 1
        return (((checker_load * ttd_size + checker_ttd) * cap + dropper_load) * ttd_size) + dropper_ttd

    def decode(self, index: int) -> tuple[int, int, int, int]:
        if isinstance(index, bool) or not isinstance(index, int):
            raise TypeError("packed state index must be an integer")
        if not 0 <= index < self.state_count:
            raise ValueError(f"packed state index must be in 0..{self.state_count - 1}")
        return self.decode_unchecked(index)

    def decode_unchecked(self, index: int) -> tuple[int, int, int, int]:
        cap = self.load_cap_units
        ttd_size = cap + 1
        quotient, dropper_ttd = divmod(index, ttd_size)
        quotient, dropper_load = divmod(quotient, cap)
        checker_load, checker_ttd = divmod(quotient, ttd_size)
        return checker_load, checker_ttd, dropper_load, dropper_ttd

    def potential(self, index: int) -> int:
        return sum(self.decode(index))


@dataclass(frozen=True, slots=True)
class PackedBranch:
    """Compact transition result used by the Python/Rust parity boundary."""

    probability: float
    state_index: int | None
    terminal_value: float | None
    event: str
    squandered_units: int
    death_dose_units: int | None


def packed_branches(
    index: int,
    drop: int,
    check: int,
    rules: AbstractRuleset,
    *,
    codec: PackedStateCodec | None = None,
) -> tuple[PackedBranch, ...]:
    """Expand one packed joint action with the authoritative Python algebra."""

    codec = PackedStateCodec(rules.load_cap_units) if codec is None else codec
    checker_load, checker_ttd, dropper_load, dropper_ttd = codec.decode(index)
    if drop not in rules.action_values:
        raise ValueError(f"illegal drop action {drop}")
    if check not in rules.action_values:
        raise ValueError(f"illegal check action {check}")

    if check >= drop:
        squandered = check - drop + 1
        candidate_load = checker_load + squandered
        if candidate_load >= rules.load_cap_units:
            return (
                PackedBranch(
                    probability=1.0,
                    state_index=None,
                    terminal_value=1.0,
                    event="overflow_died",
                    squandered_units=squandered,
                    death_dose_units=rules.load_cap_units,
                ),
            )
        child = codec.encode_unchecked(
            dropper_load,
            dropper_ttd,
            candidate_load,
            checker_ttd,
        )
        return (
            PackedBranch(
                probability=1.0,
                state_index=child,
                terminal_value=None,
                event="check_success",
                squandered_units=squandered,
                death_dose_units=None,
            ),
        )

    dose = checker_load + rules.failed_check_penalty_units
    probability = rules.revival_probability(checker_ttd, dose)
    result: list[PackedBranch] = []
    if probability > 0.0:
        child = codec.encode_unchecked(
            dropper_load,
            dropper_ttd,
            0,
            checker_ttd + dose,
        )
        result.append(
            PackedBranch(
                probability=probability,
                state_index=child,
                terminal_value=None,
                event="check_failure_survived",
                squandered_units=0,
                death_dose_units=dose,
            )
        )
    if probability < 1.0:
        result.append(
            PackedBranch(
                probability=1.0 - probability,
                state_index=None,
                terminal_value=1.0,
                event="check_failure_died",
                squandered_units=0,
                death_dose_units=dose,
            )
        )
    return tuple(result)


def packed_live_successors(
    index: int,
    rules: AbstractRuleset,
    *,
    codec: PackedStateCodec | None = None,
) -> tuple[int, ...]:
    """Return the distinct live children without expanding an action square."""

    codec = PackedStateCodec(rules.load_cap_units) if codec is None else codec
    checker_load, checker_ttd, dropper_load, dropper_ttd = codec.decode_unchecked(index)
    cap = rules.load_cap_units
    children: list[int] = []

    # Successful checks depend only on inclusive squandered time.  Across the
    # contiguous 1..N action square every squandered value 1..N occurs.
    for squandered in rules.action_values:
        candidate_load = checker_load + squandered
        if candidate_load < cap:
            children.append(
                codec.encode_unchecked(
                    dropper_load,
                    dropper_ttd,
                    candidate_load,
                    checker_ttd,
                )
            )

    # Every failed-check cell has the same possible survival successor.
    dose = checker_load + rules.failed_check_penalty_units
    if dose < cap and checker_ttd + dose <= cap:
        children.append(
            codec.encode_unchecked(
                dropper_load,
                dropper_ttd,
                0,
                checker_ttd + dose,
            )
        )

    current_potential = checker_load + checker_ttd + dropper_load + dropper_ttd
    if any(sum(codec.decode_unchecked(child)) <= current_potential for child in children):
        raise RuntimeError("packed successor does not increase the acyclic potential")
    return tuple(children)
