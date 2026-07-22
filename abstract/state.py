"""Role-relative public states and chance branches for the exact abstraction."""

from __future__ import annotations

from dataclasses import dataclass, replace


@dataclass(frozen=True, slots=True)
class AbstractState:
    """Public Markov state, always from the current roles' perspective.

    ``checker_*`` belongs to the player who is checking this half-round and
    ``dropper_*`` belongs to their opponent.  The next nonterminal state
    swaps these two role records, so no identity or round/phase field is
    needed.
    """

    checker_load: int = 0
    checker_ttd: int = 0
    dropper_load: int = 0
    dropper_ttd: int = 0

    def __post_init__(self) -> None:
        fields = (
            self.checker_load,
            self.checker_ttd,
            self.dropper_load,
            self.dropper_ttd,
        )
        if any(isinstance(value, bool) or not isinstance(value, int) for value in fields):
            raise TypeError("AbstractState fields must be integers")
        if any(value < 0 for value in fields):
            raise ValueError("load and TTD fields must be nonnegative")

    @property
    def potential(self) -> int:
        """Strictly increases on every nonterminal transition."""

        return self.checker_load + self.checker_ttd + self.dropper_load + self.dropper_ttd

    def with_updates(self, **updates: int) -> "AbstractState":
        return replace(self, **updates)


@dataclass(frozen=True, slots=True)
class AbstractBranch:
    """One deterministic successor or terminal outcome of a joint action."""

    probability: float
    state: AbstractState | None
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
