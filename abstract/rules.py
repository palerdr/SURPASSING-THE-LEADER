"""Rules for the exact role-relative TTD bucket abstractions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import numpy as np

from abstract.state import AbstractBranch, AbstractState


TIMING_CONVENTION_ID = "ordinal-buckets-inclusive-st-v1"
LEGACY_REVIVAL_MODEL = "dose_cubic_ttd_stretched_exponential_v1"
UNIFIED_REVIVAL_MODEL = "linear_st_ttd_effective_referee_v1"


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
    revival_model_kind: str = LEGACY_REVIVAL_MODEL
    revival_baseline: float = 0.95
    dose_curve_exponent: float = 3.0
    ttd_half_life_units: float = 12.0
    ttd_curve_exponent: float = 1.3
    referee_decay_per_death_dose: float = 1.0
    referee_floor: float = 1.0

    def __post_init__(self) -> None:
        if not self.ruleset_id:
            raise ValueError("ruleset_id must be non-empty")
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
        if not 0.0 < self.revival_baseline <= 1.0:
            raise ValueError("revival_baseline must be in (0, 1]")
        if self.revival_model_kind not in {LEGACY_REVIVAL_MODEL, UNIFIED_REVIVAL_MODEL}:
            raise ValueError(f"unknown revival_model_kind {self.revival_model_kind!r}")
        if self.dose_curve_exponent <= 0.0:
            raise ValueError("dose_curve_exponent must be positive")
        if self.ttd_half_life_units <= 0.0:
            raise ValueError("ttd_half_life_units must be positive")
        if self.ttd_curve_exponent <= 0.0:
            raise ValueError("ttd_curve_exponent must be positive")
        if not 0.0 < self.referee_decay_per_death_dose <= 1.0:
            raise ValueError("referee_decay_per_death_dose must be in (0, 1]")
        if not 0.0 < self.referee_floor <= 1.0:
            raise ValueError("referee_floor must be in (0, 1]")

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

        metadata: dict[str, float | str] = {
            "kind": self.revival_model_kind,
            "baseline": self.revival_baseline,
            "ttd_half_life_units": self.ttd_half_life_units,
            "ttd_curve_exponent": self.ttd_curve_exponent,
        }
        if self.revival_model_kind == LEGACY_REVIVAL_MODEL:
            metadata["dose_curve_exponent"] = self.dose_curve_exponent
        else:
            metadata["st_shape"] = "linear_pre_failure_load"
            metadata["referee_decay_per_death_dose"] = self.referee_decay_per_death_dose
            metadata["referee_floor"] = self.referee_floor
        return metadata

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
        """Return revival probability from pre-failure ST and accrued TTD.

        The current rules use a linear factor in pre-failure ST, a stretched
        TTD decay, and a TTD-derived effective referee burden.  Legacy rulesets
        retain the earlier cubic dose curve for artifact reproducibility.
        """

        if dose_units >= self.load_cap_units or prior_ttd + dose_units > self.load_cap_units:
            return 0.0
        if self.revival_model_kind == LEGACY_REVIVAL_MODEL:
            dose_factor = 1.0 - (dose_units / self.load_cap_units) ** self.dose_curve_exponent
            referee_factor = 1.0
        else:
            pre_failure_st = dose_units - self.failed_check_penalty_units
            survivable_st_span = self.load_cap_units - self.failed_check_penalty_units
            dose_factor = 1.0 - pre_failure_st / survivable_st_span
            effective_deaths = prior_ttd / self.failed_check_penalty_units
            referee_factor = max(
                self.referee_floor,
                self.referee_decay_per_death_dose**effective_deaths,
            )
        ttd_factor = 2.0 ** (-(prior_ttd / self.ttd_half_life_units) ** self.ttd_curve_exponent)
        return float(
            np.clip(
                self.revival_baseline * dose_factor * ttd_factor * referee_factor,
                0.0,
                1.0,
            )
        )

    def expand_joint_action(self, state: AbstractState, drop: int, check: int) -> tuple[AbstractBranch, ...]:
        legal_drop = self.legal_drop_actions(state)
        legal_check = self.legal_check_actions(state)
        if drop not in legal_drop:
            raise ValueError(f"illegal drop action {drop}; legal={legal_drop}")
        if check not in legal_check:
            raise ValueError(f"illegal check action {check}; legal={legal_check}")

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


def Bucket6TTDCurve95Rules() -> AbstractRuleset:
    return AbstractRuleset(
        ruleset_id="bucket6_ttd_curve95",
        action_values=tuple(range(1, 7)),
        bucket_seconds=10,
        load_cap_units=30,
        failed_check_penalty_units=6,
        revival_baseline=0.95,
        dose_curve_exponent=3.0,
        ttd_half_life_units=12.0,
        ttd_curve_exponent=1.3,
    )


def Bucket12TTDCurve95Rules() -> AbstractRuleset:
    """Five-second formulation with the same physical rules as bucket6.

    Loads and TTD are measured in five-second units, so every unit-valued
    parameter in :func:`Bucket6TTDCurve95Rules` is doubled while probabilities
    and boundary inequalities are unchanged.
    """

    return AbstractRuleset(
        ruleset_id="bucket12_ttd_curve95",
        action_values=tuple(range(1, 13)),
        bucket_seconds=5,
        load_cap_units=60,
        failed_check_penalty_units=12,
        revival_baseline=0.95,
        dose_curve_exponent=3.0,
        ttd_half_life_units=24.0,
        ttd_curve_exponent=1.3,
    )


def Bucket6Unified80Rules() -> AbstractRuleset:
    """Ten-second formulation of the unified two-variable revival model."""

    return AbstractRuleset(
        ruleset_id="bucket6_unified80",
        action_values=tuple(range(1, 7)),
        bucket_seconds=10,
        load_cap_units=30,
        failed_check_penalty_units=6,
        revival_model_kind=UNIFIED_REVIVAL_MODEL,
        revival_baseline=0.80,
        ttd_half_life_units=12.0,
        ttd_curve_exponent=1.3,
        referee_decay_per_death_dose=0.88,
        referee_floor=0.40,
    )


def Bucket12Unified80Rules() -> AbstractRuleset:
    """Five-second formulation physically equivalent to bucket6_unified80."""

    return AbstractRuleset(
        ruleset_id="bucket12_unified80",
        action_values=tuple(range(1, 13)),
        bucket_seconds=5,
        load_cap_units=60,
        failed_check_penalty_units=12,
        revival_model_kind=UNIFIED_REVIVAL_MODEL,
        revival_baseline=0.80,
        ttd_half_life_units=24.0,
        ttd_curve_exponent=1.3,
        referee_decay_per_death_dose=0.88,
        referee_floor=0.40,
    )


def ruleset_for_name(name: str) -> AbstractRuleset:
    factories = {
        "bucket6_unified80": Bucket6Unified80Rules,
        "bucket12_unified80": Bucket12Unified80Rules,
        "bucket6_ttd_curve95": Bucket6TTDCurve95Rules,
        "bucket12_ttd_curve95": Bucket12TTDCurve95Rules,
    }
    try:
        factory = factories[name]
    except KeyError as exc:
        expected = ", ".join(repr(value) for value in factories)
        raise ValueError(f"unknown abstract ruleset {name!r}; expected one of {expected}") from exc
    return factory()
