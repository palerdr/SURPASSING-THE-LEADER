"""Immutable public states and chance branches for ToySTL."""

from __future__ import annotations

from dataclasses import dataclass, replace


@dataclass(frozen=True, slots=True)
class ToyState:
    """Public Markov state.

    The canonical ruleset uses ``hal_load``, ``baku_load``, and ``role_phase``.
    Additional zero-default fields remain part of the stable artifact record.
    """

    hal_load: int = 0
    baku_load: int = 0
    role_phase: int = 0  # 0: Hal drops, 1: Baku drops.
    hal_ttd: int = 0
    baku_ttd: int = 0
    cprs_performed: int = 0
    game_clock: int = 0
    half_in_round: int = 1
    round_num: int = 0

    def __post_init__(self) -> None:
        integer_fields = (
            self.hal_load,
            self.baku_load,
            self.role_phase,
            self.hal_ttd,
            self.baku_ttd,
            self.cprs_performed,
            self.game_clock,
            self.half_in_round,
            self.round_num,
        )
        if any(isinstance(value, bool) or not isinstance(value, int) for value in integer_fields):
            raise TypeError("ToyState fields must be integers")
        if self.hal_load < 0 or self.baku_load < 0:
            raise ValueError("loads must be nonnegative")
        if self.hal_ttd < 0 or self.baku_ttd < 0:
            raise ValueError("TTD must be nonnegative")
        if self.cprs_performed < 0 or self.game_clock < 0 or self.round_num < 0:
            raise ValueError("counts and clock must be nonnegative")
        if self.role_phase not in (0, 1):
            raise ValueError("role_phase must be 0 or 1")
        if self.half_in_round not in (1, 2):
            raise ValueError("half_in_round must be 1 or 2")

    @property
    def hal_is_dropper(self) -> bool:
        return self.role_phase == 0

    @property
    def checker_is_hal(self) -> bool:
        return not self.hal_is_dropper

    def with_updates(self, **updates: int) -> "ToyState":
        return replace(self, **updates)


@dataclass(frozen=True, slots=True)
class ToyBranch:
    """One deterministic successor or terminal outcome of a joint action."""

    probability: float
    state: ToyState | None
    terminal_value: float | None
    event: str
    survived: bool | None = None
    squandered_units: int = 0
    death_dose_units: int | None = None

    def __post_init__(self) -> None:
        if not 0.0 <= self.probability <= 1.0:
            raise ValueError("branch probability must be in [0, 1]")
        if self.state is None and self.terminal_value is None:
            raise ValueError("a branch must have a state or terminal value")
        if self.state is not None and self.terminal_value is not None:
            raise ValueError("a branch cannot be both stateful and terminal")
        if self.terminal_value is not None and self.terminal_value not in (-1.0, 0.0, 1.0):
            raise ValueError("terminal value must be -1, 0, or 1")

    @property
    def is_terminal(self) -> bool:
        return self.terminal_value is not None
