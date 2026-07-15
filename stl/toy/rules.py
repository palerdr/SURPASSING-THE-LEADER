"""Versioned standalone ToySTL rulesets.

The v0 ruleset deliberately has no dependency on the canonical STL engine.
Later rulesets add one source-game feature at a time while preserving the same
exact-search and MCTS interfaces.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Iterator

import numpy as np

from stl.toy.state import ToyBranch, ToyState


LS_WINDOW_START = 59 * 60
LS_WINDOW_END = 60 * 60
TURN_DURATION_NORMAL = 60
TURN_DURATION_LEAP = 61
FAILED_CHECK_PENALTY_SECONDS = 60
DEATH_PROCEDURE_OVERHEAD = 120
WITHIN_ROUND_OVERHEAD = 60


@dataclass(frozen=True, slots=True)
class ToyRuleset:
    """Rules and transition kernel consumed by exact search and MCTS."""

    ruleset_id: str
    action_values: tuple[int, ...]
    bucket_seconds: int
    load_cap_units: int
    max_half_rounds: int = 8
    revival_mode: str = "fixed"
    fixed_revival_probability: float = 0.5
    history_enabled: bool = False
    leap_enabled: bool = False
    initial_clock: int = 0
    failed_check_penalty_seconds: int | None = None
    hal_physicality: float = 1.0
    baku_physicality: float = 0.94
    cardiac_decay: float = 0.85
    referee_decay: float = 0.88
    referee_floor: float = 0.4

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
        if self.max_half_rounds <= 0:
            raise ValueError("max_half_rounds must be positive")
        if self.revival_mode not in {"fixed", "dose_curve"}:
            raise ValueError(f"unsupported revival_mode {self.revival_mode!r}")
        if not 0.0 <= self.fixed_revival_probability <= 1.0:
            raise ValueError("fixed revival probability must be in [0, 1]")
        if self.history_enabled and self.revival_mode != "dose_curve":
            raise ValueError("history-enabled rules require dose_curve revival")
        if self.leap_enabled and self.bucket_seconds != 1:
            raise ValueError("leap rules require literal-second actions")

    @property
    def load_cap_seconds(self) -> int:
        return self.load_cap_units * self.bucket_seconds

    @property
    def action_size(self) -> int:
        # The leap ruleset adds Baku-only action 61 dynamically.  Keeping the
        # full head width explicit makes policy arrays stable across states.
        return 61 if self.leap_enabled else max(self.action_values)

    @property
    def schema_version(self) -> str:
        return f"toy.state.{self.ruleset_id}.v1"

    def initial_state(self) -> ToyState:
        return ToyState(game_clock=self.initial_clock)

    def current_dropper_is_hal(self, state: ToyState) -> bool:
        return state.hal_is_dropper

    def is_leap_window(self, state: ToyState) -> bool:
        return self.leap_enabled and LS_WINDOW_START <= state.game_clock <= LS_WINDOW_END

    def turn_duration(self, state: ToyState) -> int:
        return TURN_DURATION_LEAP if self.is_leap_window(state) else TURN_DURATION_NORMAL

    def legal_drop_actions(self, state: ToyState) -> tuple[int, ...]:
        if self.is_leap_window(state) and not state.hal_is_dropper:
            return tuple(range(1, 62))
        return self.action_values

    def legal_check_actions(self, state: ToyState) -> tuple[int, ...]:
        if self.leap_enabled:
            return tuple(range(1, 61))
        return self.action_values

    def action_seconds(self, action: int) -> int:
        return action * self.bucket_seconds

    def state_fields(self, state: ToyState) -> tuple[int, ...]:
        """Return the active serialized state fields in stable order."""

        fields = [state.hal_load, state.baku_load, state.role_phase]
        if self.history_enabled:
            fields.extend((state.hal_ttd, state.baku_ttd, state.cprs_performed))
        if self.leap_enabled:
            fields.extend((state.game_clock, state.half_in_round, state.round_num))
        return tuple(fields)

    @property
    def state_field_names(self) -> tuple[str, ...]:
        names = ["hal_load", "baku_load", "role_phase"]
        if self.history_enabled:
            names.extend(("hal_ttd", "baku_ttd", "cprs_performed"))
        if self.leap_enabled:
            names.extend(("game_clock", "half_in_round", "round_num"))
        return tuple(names)

    @property
    def feature_names(self) -> tuple[str, ...]:
        names = (
            "hal_load_normalized",
            "baku_load_normalized",
            "role_phase",
            "remaining_horizon_normalized",
        )
        if self.history_enabled:
            names += (
                "hal_ttd_normalized",
                "baku_ttd_normalized",
                "cprs_performed_normalized",
                "hal_physicality",
                "baku_physicality",
            )
        if self.leap_enabled:
            names += ("game_clock_normalized", "half_in_round_two", "leap_window")
        return names

    def encode_state(self, state: ToyState, remaining_horizon: int) -> np.ndarray:
        """Encode active state fields plus explicit horizon context."""

        if not 0 <= remaining_horizon <= self.max_half_rounds:
            raise ValueError("remaining_horizon is outside the configured horizon")
        values = [
            state.hal_load / self.load_cap_units,
            state.baku_load / self.load_cap_units,
            float(state.role_phase),
            remaining_horizon / self.max_half_rounds,
        ]
        if self.history_enabled:
            values.extend(
                (
                    state.hal_ttd / self.load_cap_seconds,
                    state.baku_ttd / self.load_cap_seconds,
                    state.cprs_performed / 10.0,
                    self.hal_physicality,
                    self.baku_physicality,
                )
            )
        if self.leap_enabled:
            values.extend(
                (
                    state.game_clock / 3600.0,
                    float(state.half_in_round == 2),
                    float(self.is_leap_window(state)),
                )
            )
        return np.asarray(values, dtype=np.float32)

    def enumerate_states(self) -> Iterator[ToyState]:
        """Enumerate the complete v0 physical state domain.

        Larger rulesets deliberately do not claim exhaustive tablebase coverage
        through this helper; their expanded state spaces are audited through
        declared exact fixtures instead.
        """

        if self.ruleset_id != "bucket12_fixed50":
            raise NotImplementedError("exhaustive enumeration is defined for ToySTL-v0 only")
        # The serialized state schema is (hal_load, baku_load, role_phase),
        # so nested loops follow that tuple's lexicographic order.
        for hal_load in range(self.load_cap_units):
            for baku_load in range(self.load_cap_units):
                for phase in (0, 1):
                    yield ToyState(hal_load=hal_load, baku_load=baku_load, role_phase=phase)

    def survival_probability(self, state: ToyState, checker_is_hal: bool, dose_units: int) -> float:
        if self.revival_mode == "fixed":
            return float(self.fixed_revival_probability)

        dose_seconds = min(dose_units * self.bucket_seconds, self.load_cap_seconds)
        if dose_seconds >= self.load_cap_seconds:
            base = 0.0
        else:
            base = max(0.0, 1.0 - (dose_seconds / self.load_cap_seconds) ** 3)
        if not self.history_enabled:
            return float(np.clip(base, 0.0, 1.0))

        ttd = state.hal_ttd if checker_is_hal else state.baku_ttd
        physicality = self.hal_physicality if checker_is_hal else self.baku_physicality
        cardiac = self.cardiac_decay ** (ttd / 60.0)
        referee = max(self.referee_floor, self.referee_decay ** state.cprs_performed)
        return float(np.clip(base * cardiac * referee * physicality, 0.0, 1.0))

    def _advance_clock(self, state: ToyState, death_dose_units: int | None) -> ToyState:
        """Advance final-stage clock semantics; inert for earlier rulesets."""

        if not self.leap_enabled:
            return state

        game_clock = state.game_clock + self.turn_duration(state)
        if death_dose_units is not None:
            game_clock += death_dose_units * self.bucket_seconds + DEATH_PROCEDURE_OVERHEAD

        if state.half_in_round == 1:
            game_clock += WITHIN_ROUND_OVERHEAD
            return replace(
                state,
                game_clock=game_clock,
                half_in_round=2,
            )

        game_clock = self._snap_clock_to_next_minute(game_clock)
        return replace(
            state,
            game_clock=game_clock,
            half_in_round=1,
            round_num=state.round_num + 1,
        )

    @staticmethod
    def _snap_clock_to_next_minute(game_clock: int) -> int:
        gc = int(game_clock)
        if gc < 3600:
            snapped = ((gc // 60) + 1) * 60
            return 3601 if snapped == 3600 else snapped
        if gc <= 3600:
            return 3601
        elapsed = gc - 3601
        return 3601 + ((elapsed // 60) + 1) * 60

    def _advance_half(self, state: ToyState, *, death_dose_units: int | None) -> ToyState:
        next_phase = 1 - state.role_phase
        advanced = self._advance_clock(state, death_dose_units)
        return replace(advanced, role_phase=next_phase)

    def _checker_load(self, state: ToyState) -> tuple[bool, int]:
        checker_is_hal = state.checker_is_hal
        return checker_is_hal, state.hal_load if checker_is_hal else state.baku_load

    def _replace_checker_load(self, state: ToyState, checker_is_hal: bool, load: int) -> ToyState:
        return replace(state, hal_load=load if checker_is_hal else state.hal_load,
                       baku_load=load if not checker_is_hal else state.baku_load)

    def _death_branches(
        self,
        state: ToyState,
        *,
        checker_is_hal: bool,
        squandered_units: int,
        dose_units: int,
        event: str,
    ) -> tuple[ToyBranch, ...]:
        probability = self.survival_probability(state, checker_is_hal, dose_units)
        revived = self._replace_checker_load(state, checker_is_hal, 0)

        if self.history_enabled:
            revived = replace(
                revived,
                hal_ttd=state.hal_ttd + (dose_units * self.bucket_seconds if checker_is_hal else 0),
                baku_ttd=state.baku_ttd + (dose_units * self.bucket_seconds if not checker_is_hal else 0),
                cprs_performed=state.cprs_performed + 1,
            )
        revived = self._advance_half(revived, death_dose_units=dose_units)
        hal_wins = 1.0 if state.hal_is_dropper else -1.0

        branches: list[ToyBranch] = []
        if probability > 0.0:
            branches.append(
                ToyBranch(
                    probability=probability,
                    state=revived,
                    terminal_value=None,
                    event=f"{event}_survived",
                    survived=True,
                    squandered_units=squandered_units,
                    death_dose_units=dose_units,
                )
            )
        if probability < 1.0:
            branches.append(
                ToyBranch(
                    probability=1.0 - probability,
                    state=None,
                    terminal_value=hal_wins,
                    event=f"{event}_died",
                    survived=False,
                    squandered_units=squandered_units,
                    death_dose_units=dose_units,
                )
            )
        return tuple(branches)

    def expand_joint_action(self, state: ToyState, drop: int, check: int) -> tuple[ToyBranch, ...]:
        legal_drop = self.legal_drop_actions(state)
        legal_check = self.legal_check_actions(state)
        if drop not in legal_drop:
            raise ValueError(f"illegal drop action {drop}; legal={legal_drop}")
        if check not in legal_check:
            raise ValueError(f"illegal check action {check}; legal={legal_check}")

        checker_is_hal, checker_load = self._checker_load(state)
        if check >= drop:
            squandered_units = check - drop
            candidate_load = checker_load + squandered_units
            if candidate_load < self.load_cap_units:
                successor = self._replace_checker_load(state, checker_is_hal, candidate_load)
                successor = self._advance_half(successor, death_dose_units=None)
                return (
                    ToyBranch(
                        probability=1.0,
                        state=successor,
                        terminal_value=None,
                        event="check_success",
                        survived=None,
                        squandered_units=squandered_units,
                    ),
                )
            # Dose is the post-attempt load, clipped at the cap.  In v0 the
            # fixed revival rule makes the distinction strategically inert;
            # later dose curves rely on this exact value.
            dose_units = min(candidate_load, self.load_cap_units)
            return self._death_branches(
                state,
                checker_is_hal=checker_is_hal,
                squandered_units=squandered_units,
                dose_units=dose_units,
                event="overflow",
            )

        if self.failed_check_penalty_seconds is None:
            dose_units = self.load_cap_units
        else:
            penalty_units = max(1, self.failed_check_penalty_seconds // self.bucket_seconds)
            dose_units = min(checker_load + penalty_units, self.load_cap_units)
        return self._death_branches(
            state,
            checker_is_hal=checker_is_hal,
            squandered_units=0,
            dose_units=dose_units,
            event="check_failure",
        )


def Bucket12Fixed50Rules(*, max_half_rounds: int = 8) -> ToyRuleset:
    return ToyRuleset(
        ruleset_id="bucket12_fixed50",
        action_values=tuple(range(1, 13)),
        bucket_seconds=5,
        load_cap_units=60,
        max_half_rounds=max_half_rounds,
        revival_mode="fixed",
        fixed_revival_probability=0.5,
    )


def FullSecondFixed50Rules(*, max_half_rounds: int = 8) -> ToyRuleset:
    return ToyRuleset(
        ruleset_id="seconds60_fixed50",
        action_values=tuple(range(1, 61)),
        bucket_seconds=1,
        load_cap_units=300,
        max_half_rounds=max_half_rounds,
        revival_mode="fixed",
        fixed_revival_probability=0.5,
        failed_check_penalty_seconds=FAILED_CHECK_PENALTY_SECONDS,
    )


def FullSecondVariableRevivalRules(*, max_half_rounds: int = 8) -> ToyRuleset:
    return ToyRuleset(
        ruleset_id="seconds60_variable_revival",
        action_values=tuple(range(1, 61)),
        bucket_seconds=1,
        load_cap_units=300,
        max_half_rounds=max_half_rounds,
        revival_mode="dose_curve",
        fixed_revival_probability=0.5,
        failed_check_penalty_seconds=FAILED_CHECK_PENALTY_SECONDS,
    )


def FullSecondTTDCPRPhysicalityRules(*, max_half_rounds: int = 8) -> ToyRuleset:
    return ToyRuleset(
        ruleset_id="seconds60_ttd_cpr_physicality",
        action_values=tuple(range(1, 61)),
        bucket_seconds=1,
        load_cap_units=300,
        max_half_rounds=max_half_rounds,
        revival_mode="dose_curve",
        fixed_revival_probability=0.5,
        history_enabled=True,
        failed_check_penalty_seconds=FAILED_CHECK_PENALTY_SECONDS,
    )


def FullSecondLeapRules(*, max_half_rounds: int = 8) -> ToyRuleset:
    return ToyRuleset(
        ruleset_id="seconds60_leap",
        action_values=tuple(range(1, 61)),
        bucket_seconds=1,
        load_cap_units=300,
        max_half_rounds=max_half_rounds,
        revival_mode="dose_curve",
        fixed_revival_probability=0.5,
        history_enabled=True,
        leap_enabled=True,
        initial_clock=LS_WINDOW_START,
        failed_check_penalty_seconds=FAILED_CHECK_PENALTY_SECONDS,
    )


def ruleset_for_name(name: str, *, max_half_rounds: int = 8) -> ToyRuleset:
    factories = {
        "bucket12_fixed50": Bucket12Fixed50Rules,
        "seconds60_fixed50": FullSecondFixed50Rules,
        "seconds60_variable_revival": FullSecondVariableRevivalRules,
        "seconds60_ttd_cpr_physicality": FullSecondTTDCPRPhysicalityRules,
        "seconds60_leap": FullSecondLeapRules,
    }
    try:
        return factories[name](max_half_rounds=max_half_rounds)
    except KeyError as exc:
        raise ValueError(f"unknown ToySTL ruleset {name!r}") from exc
