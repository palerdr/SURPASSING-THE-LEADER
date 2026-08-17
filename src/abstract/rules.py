"""Rules for the exact role-relative TTD bucket abstractions."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral
from typing import Iterator

from abstract.state import AbstractBranch, AbstractState


TIMING_CONVENTION_ID = "ordinal-buckets-inclusive-st-v1"
FROZEN_REVIVAL_MODEL = "linear_st_geometric_ttd_v1"
REVIVAL_BASELINE = 0.95
REVIVAL_TTD_DECAY_PER_DEATH_DOSE = 0.75


@dataclass(frozen=True, slots=True)
class AbstractRuleset:
    """One exact role-relative discretization of the canonical abstraction.

    Every numeric transition quantity is an ordinal time bucket.  The
    seconds mapping is metadata only; it never enters the solver algebra.
    """

    ruleset_id: str
    action_values: tuple[int, ...]
    bucket_seconds: int
    load_cap_units: int
    failed_check_penalty_units: int

    def __post_init__(self) -> None:
        if not isinstance(self.ruleset_id, str) or not self.ruleset_id:
            raise ValueError("ruleset_id must be non-empty")
        if any(
            isinstance(value, bool) or not isinstance(value, Integral)
            for value in (
                self.bucket_seconds,
                self.load_cap_units,
                self.failed_check_penalty_units,
            )
        ):
            raise ValueError("ruleset dimensions must be literal integers")
        if not isinstance(self.action_values, tuple):
            raise ValueError("action_values must be a non-empty sorted tuple")
        if any(
            isinstance(action, bool) or not isinstance(action, Integral)
            for action in self.action_values
        ):
            raise ValueError("action_values must contain literal integers")
        # ``numbers.Integral`` deliberately admits integer scalar types such
        # as ``numpy.int64`` at the public boundary.  Persisted metadata is
        # JSON, however, so canonicalize every accepted value immediately.
        object.__setattr__(
            self,
            "action_values",
            tuple(int(action) for action in self.action_values),
        )
        object.__setattr__(self, "bucket_seconds", int(self.bucket_seconds))
        object.__setattr__(self, "load_cap_units", int(self.load_cap_units))
        object.__setattr__(
            self,
            "failed_check_penalty_units",
            int(self.failed_check_penalty_units),
        )
        if not self.action_values or tuple(sorted(self.action_values)) != self.action_values:
            raise ValueError("action_values must be a non-empty sorted tuple")
        if len(set(self.action_values)) != len(self.action_values):
            raise ValueError("action_values must be unique")
        if any(action <= 0 for action in self.action_values):
            raise ValueError("actions must be positive")
        if self.bucket_seconds <= 0 or self.load_cap_units <= 0:
            raise ValueError("bucket_seconds and load_cap_units must be positive")
        if not 0 < self.failed_check_penalty_units < self.load_cap_units:
            raise ValueError("failed-check penalty must lie strictly inside the load cap")
        if self.action_values != tuple(range(1, self.failed_check_penalty_units + 1)):
            raise ValueError("action buckets must be the contiguous range 1..failed_check_penalty_units")

    @property
    def load_cap_seconds(self) -> int:
        return self.load_cap_units * self.bucket_seconds

    @property
    def action_size(self) -> int:
        return max(self.action_values)

    @property
    def schema_version(self) -> str:
        return f"abstract.state.{self.ruleset_id}.v1"

    @property
    def revival_model_metadata(self) -> dict[str, float | str]:
        """Return the complete, hot-artifact revival-model contract."""

        return {
            "kind": FROZEN_REVIVAL_MODEL,
            "baseline": REVIVAL_BASELINE,
            "st_shape": "linear_pre_failure_load",
            "ttd_decay_per_death_dose": REVIVAL_TTD_DECAY_PER_DEATH_DOSE,
        }

    def initial_state(self) -> AbstractState:
        return AbstractState()

    def legal_drop_actions(self, state: AbstractState) -> tuple[int, ...]:
        del state
        return self.action_values

    def legal_check_actions(self, state: AbstractState) -> tuple[int, ...]:
        del state
        return self.action_values

    def action_seconds(self, action: int) -> int:
        return action * self.bucket_seconds

    def state_fields(self, state: AbstractState) -> tuple[int, ...]:
        return (
            state.checker_load,
            state.checker_ttd,
            state.dropper_load,
            state.dropper_ttd,
        )

    def validate_state(self, state: AbstractState) -> AbstractState:
        """Reject states outside this ruleset's live physical domain."""

        if not isinstance(state, AbstractState):
            raise TypeError("state must be an AbstractState")
        checker_load, checker_ttd, dropper_load, dropper_ttd = self.state_fields(state)
        if not (0 <= checker_load < self.load_cap_units):
            raise ValueError("checker_load is outside the live ruleset domain")
        if not (0 <= dropper_load < self.load_cap_units):
            raise ValueError("dropper_load is outside the live ruleset domain")
        if not (0 <= checker_ttd <= self.load_cap_units):
            raise ValueError("checker_ttd is outside the live ruleset domain")
        if not (0 <= dropper_ttd <= self.load_cap_units):
            raise ValueError("dropper_ttd is outside the live ruleset domain")
        return state

    def validate_action(self, action: int, *, role: str) -> int:
        """Return a canonical ordinal action or fail closed on coercible values."""

        if isinstance(action, bool) or not isinstance(action, Integral):
            raise ValueError(f"{role} action must be a literal integer")
        normalized = int(action)
        if normalized not in self.action_values:
            raise ValueError(
                f"illegal {role} action {action!r}; legal={self.action_values}"
            )
        return normalized

    @property
    def state_field_names(self) -> tuple[str, ...]:
        return ("checker_load", "checker_ttd", "dropper_load", "dropper_ttd")

    @property
    def feature_names(self) -> tuple[str, ...]:
        return (
            "checker_load_normalized",
            "checker_ttd_normalized",
            "dropper_load_normalized",
            "dropper_ttd_normalized",
        )

    @property
    def physical_state_upper_bound(self) -> int:
        return self.load_cap_units * (self.load_cap_units + 1) * self.load_cap_units * (self.load_cap_units + 1)

    def enumerate_states(self) -> Iterator[AbstractState]:
        """Enumerate the full role-relative physical state domain."""

        for checker_load in range(self.load_cap_units):
            for checker_ttd in range(self.load_cap_units + 1):
                for dropper_load in range(self.load_cap_units):
                    for dropper_ttd in range(self.load_cap_units + 1):
                        yield AbstractState(checker_load, checker_ttd, dropper_load, dropper_ttd)

    def revival_probability(self, prior_ttd: int, dose_units: int) -> float:
        """Return the repository-wide frozen revival probability."""

        if dose_units >= self.load_cap_units or prior_ttd + dose_units > self.load_cap_units:
            return 0.0
        pre_failure_st = dose_units - self.failed_check_penalty_units
        survivable_st_span = self.load_cap_units - self.failed_check_penalty_units
        dose_factor = 1.0 - pre_failure_st / survivable_st_span
        ttd_factor = REVIVAL_TTD_DECAY_PER_DEATH_DOSE ** (
            prior_ttd / self.failed_check_penalty_units
        )
        return max(0.0, min(1.0, REVIVAL_BASELINE * dose_factor * ttd_factor))

    def expand_joint_action(self, state: AbstractState, drop: int, check: int) -> tuple[AbstractBranch, ...]:
        state = self.validate_state(state)
        drop = self.validate_action(drop, role="drop")
        check = self.validate_action(check, role="check")

        if check >= drop:
            squandered_units = check - drop + 1
            candidate_load = state.checker_load + squandered_units
            if candidate_load < self.load_cap_units:
                return (
                    AbstractBranch(
                        probability=1.0,
                        state=AbstractState(
                            checker_load=state.dropper_load,
                            checker_ttd=state.dropper_ttd,
                            dropper_load=candidate_load,
                            dropper_ttd=state.checker_ttd,
                        ),
                        terminal_value=None,
                        event="check_success",
                        survived=None,
                        squandered_units=squandered_units,
                    ),
                )
            return (
                AbstractBranch(
                    probability=1.0,
                    state=None,
                    terminal_value=1.0,
                    event="overflow_died",
                    survived=False,
                    squandered_units=squandered_units,
                    death_dose_units=self.load_cap_units,
                ),
            )

        dose_units = state.checker_load + self.failed_check_penalty_units
        probability = self.revival_probability(state.checker_ttd, dose_units)
        branches: list[AbstractBranch] = []
        if probability > 0.0:
            branches.append(
                AbstractBranch(
                    probability=probability,
                    state=AbstractState(
                        checker_load=state.dropper_load,
                        checker_ttd=state.dropper_ttd,
                        dropper_load=0,
                        dropper_ttd=state.checker_ttd + dose_units,
                    ),
                    terminal_value=None,
                    event="check_failure_survived",
                    survived=True,
                    squandered_units=0,
                    death_dose_units=dose_units,
                )
            )
        if probability < 1.0:
            branches.append(
                AbstractBranch(
                    probability=1.0 - probability,
                    state=None,
                    terminal_value=1.0,
                    event="check_failure_died",
                    survived=False,
                    squandered_units=0,
                    death_dose_units=dose_units,
                )
            )
        return tuple(branches)


def Bucket6Frozen95Rules() -> AbstractRuleset:
    """Ten-second discretization of the repository-wide frozen surface."""

    return AbstractRuleset(
        ruleset_id="bucket6_frozen95",
        action_values=tuple(range(1, 7)),
        bucket_seconds=10,
        load_cap_units=30,
        failed_check_penalty_units=6,
    )


def Bucket12Frozen95Rules() -> AbstractRuleset:
    """Five-second discretization of the repository-wide frozen surface."""

    return AbstractRuleset(
        ruleset_id="bucket12_frozen95",
        action_values=tuple(range(1, 13)),
        bucket_seconds=5,
        load_cap_units=60,
        failed_check_penalty_units=12,
    )


def ruleset_for_name(name: str) -> AbstractRuleset:
    factories = {
        "bucket6_frozen95": Bucket6Frozen95Rules,
        "bucket12_frozen95": Bucket12Frozen95Rules,
    }
    try:
        factory = factories[name]
    except KeyError as exc:
        expected = ", ".join(repr(value) for value in factories)
        raise ValueError(f"unknown abstract ruleset {name!r}; expected one of {expected}") from exc
    return factory()
