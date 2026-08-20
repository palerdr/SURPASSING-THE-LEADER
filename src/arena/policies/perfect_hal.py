"""Unrestricted causal opponent exploitation for repeated pure-DTH play.

Perfect Hal is a policy label, not a claim that the DTH game has another
solution.  The completed DTH tablebase remains the continuation-value and
stage-matrix authority.  This provider uses only revealed public history to
forecast the opponent, then plays an unrestricted best response in the exact
continuation-adjusted matrix.

The forecaster is an online mixture of role-separated experts.  Long, short,
and flash recency models compete with first-order, self-response, state-regime,
delta, and periodic models under discounted prequential log loss.  This gives
the provider both persistent memory and rapid response to changed play without a
checkpoint or simulator-truth input.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from operator import index as integer_index
from pathlib import Path
from typing import Mapping

import numpy as np

from arena.contracts import (
    CanonicalDecision,
    CanonicalPolicyProvider,
    PublicGameOutcome,
    PublicHalfRound,
)
from arena.dth_adapter import project_to_dth_state
from dth.agent import (
    CERTIFIED_SADDLE_GAP_TOLERANCE,
    CertifiedStageGame,
    CompleteDTHAgent,
)

ACTION_COUNT = 60
ACTIONS = tuple(range(1, ACTION_COUNT + 1))
PERFECT_HAL_SCHEMA = "arena-perfect-hal-online-adaptive-v1"
PERFECT_HAL_DIAGNOSTICS_SCHEMA = "arena-perfect-hal-diagnostics-v1"

_BASE_EXPERT_NAMES = (
    "uniform",
    "global",
    "slow_recency",
    "fast_recency",
    "flash_recency",
    "repeat",
    "opponent_markov",
    "self_response",
    "action_delta",
    "state_regime",
)


def _finite(value: float, *, label: str) -> float:
    resolved = float(value)
    if not np.isfinite(resolved):
        raise ValueError(f"{label} must be finite")
    return resolved


def _normalize(raw: np.ndarray, *, label: str) -> np.ndarray:
    values = np.asarray(raw, dtype=np.float64)
    if (
        values.shape != (ACTION_COUNT,)
        or not np.all(np.isfinite(values))
        or np.any(values < 0.0)
        or float(values.sum()) <= 0.0
    ):
        raise ValueError(f"{label} must be finite nonnegative length 60 with mass")
    result = values / float(values.sum())
    result.setflags(write=False)
    return result


def _readonly(raw: np.ndarray) -> np.ndarray:
    result = np.asarray(raw, dtype=np.float64).copy()
    result.setflags(write=False)
    return result


def _point(action: int) -> np.ndarray:
    resolved = _literal_action(action, label="pure-DTH action")
    result = np.zeros(ACTION_COUNT, dtype=np.float64)
    result[resolved - 1] = 1.0
    return result


def _literal_action(raw: object, *, label: str) -> int:
    if isinstance(raw, (bool, np.bool_)):
        raise ValueError(f"{label} must be an integer second in 1..60")
    try:
        resolved = int(integer_index(raw))
    except TypeError as error:
        raise ValueError(f"{label} must be an integer second in 1..60") from error
    if not 1 <= resolved <= ACTION_COUNT:
        raise ValueError(f"{label} must be an integer second in 1..60")
    return resolved


def _band(value: int | float) -> int:
    resolved = _finite(float(value), label="public DTH state coordinate")
    if resolved < 0.0:
        raise ValueError("public DTH state coordinates must be nonnegative")
    return min(5, int(resolved // 60.0))


@dataclass(frozen=True, slots=True)
class PerfectHalConfig:
    """Online forecast and unrestricted response controls.

    The zero response temperature is intentional: the default policy puts all
    mass on the current forecast's exact best-response set and never blends in
    equilibrium for safety.  Positive temperatures remain available for
    experiments that need a softer response.
    """

    action_count: int = ACTION_COUNT
    prior_strength: float = 0.02
    conditional_prior_strength: float = 0.10
    recency_retentions: tuple[float, float, float] = (0.995, 0.82, 0.30)
    expert_learning_rate: float = 1.25
    expert_weight_retention: float = 0.92
    expert_weight_floor: float = 0.002
    log_probability_floor: float = 1e-9
    periodicities: tuple[int, ...] = (2, 3, 4, 5, 6, 8)
    response_temperature: float = 0.0
    best_response_tolerance: float = 1e-12

    def __post_init__(self) -> None:
        if self.action_count != ACTION_COUNT:
            raise ValueError("Perfect Hal v1 requires exactly 60 pure-DTH actions")
        if _finite(self.prior_strength, label="prior_strength") <= 0.0:
            raise ValueError("prior_strength must be positive")
        if (
            _finite(
                self.conditional_prior_strength,
                label="conditional_prior_strength",
            )
            <= 0.0
        ):
            raise ValueError("conditional_prior_strength must be positive")
        if len(self.recency_retentions) != 3 or not all(
            0.0 < _finite(value, label="recency retention") < 1.0
            for value in self.recency_retentions
        ):
            raise ValueError("recency_retentions must contain three values in (0, 1)")
        if not (
            self.recency_retentions[0]
            > self.recency_retentions[1]
            > self.recency_retentions[2]
        ):
            raise ValueError("recency_retentions must be ordered slow to flash")
        if _finite(self.expert_learning_rate, label="expert_learning_rate") <= 0.0:
            raise ValueError("expert_learning_rate must be positive")
        if not (
            0.0
            <= _finite(
                self.expert_weight_retention,
                label="expert_weight_retention",
            )
            <= 1.0
        ):
            raise ValueError("expert_weight_retention must lie in [0, 1]")
        if not self.periodicities or any(
            isinstance(period, bool) or not isinstance(period, int) or period < 2
            for period in self.periodicities
        ):
            raise ValueError("periodicities must be nonempty integers at least two")
        if len(set(self.periodicities)) != len(self.periodicities):
            raise ValueError("periodicities must be unique")
        expert_count = len(_BASE_EXPERT_NAMES) + len(self.periodicities)
        floor = _finite(self.expert_weight_floor, label="expert_weight_floor")
        if floor < 0.0 or floor * expert_count >= 1.0:
            raise ValueError("expert_weight_floor leaves no probability mass")
        probability_floor = _finite(
            self.log_probability_floor,
            label="log_probability_floor",
        )
        if not 0.0 < probability_floor < 1.0:
            raise ValueError("log_probability_floor must lie in (0, 1)")
        if _finite(self.response_temperature, label="response_temperature") < 0.0:
            raise ValueError("response_temperature must be nonnegative")
        if (
            _finite(
                self.best_response_tolerance,
                label="best_response_tolerance",
            )
            < 0.0
        ):
            raise ValueError("best_response_tolerance must be nonnegative")

    @property
    def expert_names(self) -> tuple[str, ...]:
        return _BASE_EXPERT_NAMES + tuple(
            f"period_{period}" for period in self.periodicities
        )


@dataclass(frozen=True, slots=True)
class PerfectHalContext:
    """Causal context fixed before the simultaneous action is revealed."""

    opponent_role: str
    observation_index: int
    game_index: int
    game_decision_index: int
    state_regime: tuple[int, int, int, int]
    previous_opponent_action: int | None
    previous_self_action: int | None


@dataclass(frozen=True, slots=True)
class PerfectHalForecast:
    """One immutable pre-decision opponent forecast and its expert evidence."""

    context: PerfectHalContext
    policy: np.ndarray
    expert_names: tuple[str, ...]
    expert_weights: np.ndarray
    expert_policies: np.ndarray
    entropy: float
    confidence: float


@dataclass(slots=True)
class _RoleState:
    observations: int
    global_counts: np.ndarray
    recency_counts: tuple[np.ndarray, np.ndarray, np.ndarray]
    transition_counts: np.ndarray
    response_counts: np.ndarray
    delta_counts: np.ndarray
    state_counts: dict[tuple[int, int, int, int], np.ndarray]
    phase_counts: dict[tuple[int, int], np.ndarray]
    log_weights: np.ndarray
    previous_opponent_action: int | None = None
    previous_self_action: int | None = None


def _new_role_state(config: PerfectHalConfig) -> _RoleState:
    recency_counts = (
        np.zeros(ACTION_COUNT, dtype=np.float64),
        np.zeros(ACTION_COUNT, dtype=np.float64),
        np.zeros(ACTION_COUNT, dtype=np.float64),
    )
    return _RoleState(
        observations=0,
        global_counts=np.zeros(ACTION_COUNT, dtype=np.float64),
        recency_counts=recency_counts,
        transition_counts=np.zeros((ACTION_COUNT, ACTION_COUNT), dtype=np.float64),
        response_counts=np.zeros((ACTION_COUNT, ACTION_COUNT), dtype=np.float64),
        delta_counts=np.zeros(2 * ACTION_COUNT - 1, dtype=np.float64),
        state_counts={},
        phase_counts={},
        log_weights=np.zeros(len(config.expert_names), dtype=np.float64),
    )


class PerfectHalOpponentModel:
    """Role-separated online expert mixture trained only by public reveals."""

    def __init__(self, config: PerfectHalConfig = PerfectHalConfig()) -> None:
        self.config = config
        self._roles = {
            "dropper": _new_role_state(config),
            "checker": _new_role_state(config),
        }

    def reset(self) -> None:
        """Forget the opponent and restore an independent session prior."""

        self._roles = {
            "dropper": _new_role_state(self.config),
            "checker": _new_role_state(self.config),
        }

    @property
    def total_observations(self) -> int:
        return sum(state.observations for state in self._roles.values())

    def observations(self, role: str) -> int:
        return self._state(role).observations

    def _state(self, role: str) -> _RoleState:
        if role not in self._roles:
            raise ValueError("opponent role must be 'dropper' or 'checker'")
        return self._roles[role]

    @staticmethod
    def state_regime(decision: CanonicalDecision) -> tuple[int, int, int, int]:
        """Coarsen the public load state without using native engine internals."""

        if decision.role == "dropper":
            own = (
                decision.dropper_cylinder_seconds,
                decision.dropper_ttd_seconds,
            )
            opponent = (
                decision.checker_cylinder_seconds,
                decision.checker_ttd_seconds,
            )
        elif decision.role == "checker":
            own = (
                decision.checker_cylinder_seconds,
                decision.checker_ttd_seconds,
            )
            opponent = (
                decision.dropper_cylinder_seconds,
                decision.dropper_ttd_seconds,
            )
        else:
            raise ValueError("decision role must be 'dropper' or 'checker'")
        return (_band(own[0]), _band(own[1]), _band(opponent[0]), _band(opponent[1]))

    def _predictive(
        self,
        counts: np.ndarray,
        prior: np.ndarray,
        *,
        strength: float,
    ) -> np.ndarray:
        values = np.asarray(counts, dtype=np.float64)
        if (
            values.shape != (ACTION_COUNT,)
            or not np.all(np.isfinite(values))
            or np.any(values < 0.0)
        ):
            raise RuntimeError("Perfect Hal count state is malformed")
        return _normalize(
            values + float(strength) * np.asarray(prior, dtype=np.float64),
            label="Perfect Hal predictive policy",
        )

    def _delta_policy(
        self,
        state: _RoleState,
        global_policy: np.ndarray,
    ) -> np.ndarray:
        previous = state.previous_opponent_action
        if previous is None or float(state.delta_counts.sum()) <= 0.0:
            return global_policy
        projected = np.zeros(ACTION_COUNT, dtype=np.float64)
        for index, count in enumerate(state.delta_counts):
            if count <= 0.0:
                continue
            delta = index - (ACTION_COUNT - 1)
            action = previous + delta
            if 1 <= action <= ACTION_COUNT:
                projected[action - 1] += count
        return self._predictive(
            projected,
            global_policy,
            strength=self.config.conditional_prior_strength,
        )

    def _expert_policies(
        self,
        state: _RoleState,
        context: PerfectHalContext,
    ) -> np.ndarray:
        uniform = np.full(ACTION_COUNT, 1.0 / ACTION_COUNT, dtype=np.float64)
        global_policy = self._predictive(
            state.global_counts,
            uniform,
            strength=self.config.prior_strength,
        )
        policies: list[np.ndarray] = [uniform, global_policy]
        policies.extend(
            self._predictive(
                counts,
                uniform,
                strength=self.config.prior_strength,
            )
            for counts in state.recency_counts
        )
        policies.append(
            _point(state.previous_opponent_action)
            if state.previous_opponent_action is not None
            else global_policy
        )
        policies.append(
            self._predictive(
                state.transition_counts[state.previous_opponent_action - 1],
                global_policy,
                strength=self.config.conditional_prior_strength,
            )
            if state.previous_opponent_action is not None
            else global_policy
        )
        policies.append(
            self._predictive(
                state.response_counts[state.previous_self_action - 1],
                global_policy,
                strength=self.config.conditional_prior_strength,
            )
            if state.previous_self_action is not None
            else global_policy
        )
        policies.append(self._delta_policy(state, global_policy))
        policies.append(
            self._predictive(
                state.state_counts.get(
                    context.state_regime,
                    np.zeros(ACTION_COUNT, dtype=np.float64),
                ),
                global_policy,
                strength=self.config.conditional_prior_strength,
            )
        )
        for period in self.config.periodicities:
            phase = context.observation_index % period
            policies.append(
                self._predictive(
                    state.phase_counts.get(
                        (period, phase),
                        np.zeros(ACTION_COUNT, dtype=np.float64),
                    ),
                    global_policy,
                    strength=self.config.conditional_prior_strength,
                )
            )
        matrix = np.stack(policies, axis=0)
        if matrix.shape != (len(self.config.expert_names), ACTION_COUNT):
            raise RuntimeError("Perfect Hal expert matrix shape drifted")
        return matrix

    def predict(
        self,
        opponent_role: str,
        *,
        state_regime: tuple[int, int, int, int],
        game_index: int,
        game_decision_index: int,
    ) -> PerfectHalForecast:
        """Forecast one opponent action before the simultaneous reveal exists."""

        state = self._state(opponent_role)
        if game_index < 0 or game_decision_index < 0:
            raise ValueError("game and decision indices must be nonnegative")
        if len(state_regime) != 4 or any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in state_regime
        ):
            raise ValueError("state_regime must contain four nonnegative integers")
        context = PerfectHalContext(
            opponent_role=opponent_role,
            observation_index=state.observations,
            game_index=int(game_index),
            game_decision_index=int(game_decision_index),
            state_regime=state_regime,
            previous_opponent_action=state.previous_opponent_action,
            previous_self_action=state.previous_self_action,
        )
        expert_policies = self._expert_policies(state, context)
        centered = state.log_weights - float(np.max(state.log_weights))
        weights = np.exp(np.clip(centered, -700.0, 0.0))
        weights /= float(weights.sum())
        floor = self.config.expert_weight_floor
        if floor > 0.0:
            weights = (1.0 - floor * weights.size) * weights + floor
        policy = _normalize(weights @ expert_policies, label="Perfect Hal forecast")
        clipped = np.clip(
            policy,
            self.config.log_probability_floor,
            1.0,
        )
        entropy = -float(np.sum(policy * np.log(clipped)))
        evidence = state.observations / (
            state.observations + self.config.prior_strength
        )
        confidence = evidence * max(0.0, 1.0 - entropy / np.log(ACTION_COUNT))
        expert_matrix = np.asarray(expert_policies, dtype=np.float64).copy()
        expert_matrix.setflags(write=False)
        return PerfectHalForecast(
            context=context,
            policy=policy,
            expert_names=self.config.expert_names,
            expert_weights=_readonly(weights),
            expert_policies=expert_matrix,
            entropy=entropy,
            confidence=float(np.clip(confidence, 0.0, 1.0)),
        )

    def observe(
        self,
        forecast: PerfectHalForecast,
        *,
        opponent_action: int,
        self_action: int,
    ) -> None:
        """Score the frozen forecast, then learn from one public reveal."""

        resolved_opponent_action = _literal_action(
            opponent_action,
            label="opponent action",
        )
        resolved_self_action = _literal_action(
            self_action,
            label="self action",
        )
        state = self._state(forecast.context.opponent_role)
        if forecast.context.observation_index != state.observations:
            raise RuntimeError(
                "Perfect Hal forecast token is stale or already observed"
            )
        if forecast.expert_names != self.config.expert_names or (
            forecast.expert_policies.shape
            != (len(self.config.expert_names), ACTION_COUNT)
        ):
            raise ValueError("Perfect Hal forecast token is incompatible")

        action_index = resolved_opponent_action - 1
        realized = np.clip(
            forecast.expert_policies[:, action_index],
            self.config.log_probability_floor,
            1.0,
        )
        state.log_weights *= self.config.expert_weight_retention
        state.log_weights += self.config.expert_learning_rate * np.log(realized)
        state.log_weights -= float(np.max(state.log_weights))

        state.global_counts[action_index] += 1.0
        for retention, counts in zip(
            self.config.recency_retentions,
            state.recency_counts,
            strict=True,
        ):
            counts *= retention
            counts[action_index] += 1.0
        previous_opponent = forecast.context.previous_opponent_action
        previous_self = forecast.context.previous_self_action
        if previous_opponent is not None:
            state.transition_counts[previous_opponent - 1, action_index] += 1.0
            delta_index = (
                resolved_opponent_action - previous_opponent + (ACTION_COUNT - 1)
            )
            state.delta_counts[delta_index] += 1.0
        if previous_self is not None:
            state.response_counts[previous_self - 1, action_index] += 1.0
        state.state_counts.setdefault(
            forecast.context.state_regime,
            np.zeros(ACTION_COUNT, dtype=np.float64),
        )[action_index] += 1.0
        for period in self.config.periodicities:
            phase = forecast.context.observation_index % period
            state.phase_counts.setdefault(
                (period, phase),
                np.zeros(ACTION_COUNT, dtype=np.float64),
            )[action_index] += 1.0
        state.previous_opponent_action = resolved_opponent_action
        state.previous_self_action = resolved_self_action
        state.observations += 1


def _validate_stage(stage: CertifiedStageGame) -> None:
    matrix = np.asarray(stage.matrix, dtype=np.float64)
    if matrix.shape != (ACTION_COUNT, ACTION_COUNT) or not np.all(np.isfinite(matrix)):
        raise ValueError("certified DTH stage matrix must be finite 60x60")
    for label, raw in (
        ("dropper", stage.drop_policy),
        ("checker", stage.check_policy),
    ):
        policy = np.asarray(raw, dtype=np.float64)
        if (
            policy.shape != (ACTION_COUNT,)
            or not np.all(np.isfinite(policy))
            or np.any(policy < 0.0)
            or abs(float(policy.sum()) - 1.0) > 1e-8
        ):
            raise ValueError(f"certified DTH {label} policy is malformed")
    if (
        not np.isfinite(stage.value)
        or not np.isfinite(stage.saddle_gap)
        or stage.saddle_gap < 0.0
        or stage.saddle_gap > CERTIFIED_SADDLE_GAP_TOLERANCE
    ):
        raise ValueError("certified DTH value or saddle gap is malformed")


def _best_response_policy(
    action_values: np.ndarray,
    config: PerfectHalConfig,
) -> np.ndarray:
    values = np.asarray(action_values, dtype=np.float64)
    if values.shape != (ACTION_COUNT,) or not np.all(np.isfinite(values)):
        raise ValueError("Perfect Hal action values must be finite length 60")
    best = float(np.max(values))
    if config.response_temperature == 0.0:
        winners = values >= best - config.best_response_tolerance
        policy = winners.astype(np.float64)
    else:
        logits = (values - best) / config.response_temperature
        policy = np.exp(np.clip(logits, -700.0, 0.0))
    return _normalize(policy, label="Perfect Hal best response")


@dataclass(frozen=True, slots=True)
class PerfectHalDecision:
    """Public-history and exact-matrix diagnostics for one decision."""

    role: str
    opponent_role: str
    policy: tuple[float, ...]
    opponent_policy: tuple[float, ...]
    action_values: tuple[float, ...]
    expert_weights: tuple[tuple[str, float], ...]
    evidence_count: int
    forecast_entropy: float
    forecast_confidence: float
    expected_payoff: float
    exact_policy_expected_payoff: float
    expected_exploit_gain: float
    best_response_actions: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class _PendingDecision:
    self_name: str
    self_role: str
    forecast: PerfectHalForecast


class PerfectHalPolicyProvider(CanonicalPolicyProvider):
    """Causal, session-persistent, unrestricted pure-DTH policy provider."""

    def __init__(
        self,
        artifact_dir: str | Path,
        config: PerfectHalConfig = PerfectHalConfig(),
        *,
        agent: CompleteDTHAgent | None = None,
        opponent_model: PerfectHalOpponentModel | None = None,
    ) -> None:
        self.artifact_dir = Path(artifact_dir)
        self.config = config
        self.agent = agent or CompleteDTHAgent(self.artifact_dir)
        self.opponent_model = opponent_model or PerfectHalOpponentModel(config)
        if self.opponent_model.config != config:
            raise ValueError("Perfect Hal provider and opponent-model configs differ")
        self.decisions: list[PerfectHalDecision] = []
        self._pending: _PendingDecision | None = None
        self._seen_reveals: set[tuple[int, int, int]] = set()
        self._current_actor_name: str | None = None
        self._game_started = False
        self._game_epoch = -1
        self._game_decision_index = 0
        self._last_outcome: PublicGameOutcome | None = None

    @property
    def last_decision(self) -> PerfectHalDecision | None:
        return self.decisions[-1] if self.decisions else None

    @property
    def has_session_memory(self) -> bool:
        return self.opponent_model.total_observations > 0

    def close(self) -> None:
        """Match the provider lifecycle; tablebase memory maps need no close."""

    def reset_session(self) -> None:
        """Forget the opponent and restore an independent repeated session."""

        if self._pending is not None:
            raise RuntimeError("Perfect Hal session reset with an unrevealed action")
        self.opponent_model.reset()
        self.decisions.clear()
        self._seen_reveals.clear()
        self._current_actor_name = None
        self._game_started = False
        self._game_epoch = -1
        self._game_decision_index = 0
        self._last_outcome = None

    def reset_game(self) -> None:
        """Start a fresh game while retaining the repeated-opponent model."""

        if self._pending is not None:
            raise RuntimeError("Perfect Hal game reset with an unrevealed action")
        if self._game_started:
            self._game_epoch += 1
        else:
            self._game_started = True
            self._game_epoch = 0
        self._current_actor_name = None
        self._game_decision_index = 0
        self._last_outcome = None

    @staticmethod
    def _validate_decision(decision: CanonicalDecision) -> None:
        if decision.role not in {"dropper", "checker"}:
            raise ValueError("Perfect Hal role must be dropper or checker")
        if (
            decision.turn_duration != ACTION_COUNT
            or tuple(decision.legal_seconds) != ACTIONS
        ):
            raise ValueError(
                "Perfect Hal supports pure DTH only: turn duration and legal "
                "actions must be exactly literal seconds 1..60"
            )

    def policy(self, decision: CanonicalDecision) -> Mapping[int, float]:
        """Return an unrestricted exact-matrix response to the causal forecast."""

        self._validate_decision(decision)
        if self._pending is not None:
            raise RuntimeError("Perfect Hal was asked to act twice before a reveal")
        if not self._game_started:
            self.reset_game()
        if self._current_actor_name is None:
            self._current_actor_name = decision.actor_name
        elif self._current_actor_name.casefold() != decision.actor_name.casefold():
            raise RuntimeError("Perfect Hal cannot switch player identity mid-game")

        stage = self.agent.stage_game(project_to_dth_state(decision))
        _validate_stage(stage)
        opponent_role = "checker" if decision.role == "dropper" else "dropper"
        forecast = self.opponent_model.predict(
            opponent_role,
            state_regime=self.opponent_model.state_regime(decision),
            game_index=self._game_epoch,
            game_decision_index=self._game_decision_index,
        )
        matrix = np.asarray(stage.matrix, dtype=np.float64)
        if decision.role == "dropper":
            action_values = matrix @ forecast.policy
            exact_policy = np.asarray(stage.drop_policy, dtype=np.float64)
        else:
            action_values = -(matrix.T @ forecast.policy)
            exact_policy = np.asarray(stage.check_policy, dtype=np.float64)
        policy = _best_response_policy(action_values, self.config)
        expected_payoff = float(policy @ action_values)
        exact_expected = float(exact_policy @ action_values)
        best = float(np.max(action_values))
        best_actions = tuple(
            index + 1
            for index, value in enumerate(action_values)
            if value >= best - self.config.best_response_tolerance
        )
        diagnostic = PerfectHalDecision(
            role=decision.role,
            opponent_role=opponent_role,
            policy=tuple(float(value) for value in policy),
            opponent_policy=tuple(float(value) for value in forecast.policy),
            action_values=tuple(float(value) for value in action_values),
            expert_weights=tuple(
                (name, float(weight))
                for name, weight in zip(
                    forecast.expert_names,
                    forecast.expert_weights,
                    strict=True,
                )
            ),
            evidence_count=forecast.context.observation_index,
            forecast_entropy=forecast.entropy,
            forecast_confidence=forecast.confidence,
            expected_payoff=expected_payoff,
            exact_policy_expected_payoff=exact_expected,
            expected_exploit_gain=expected_payoff - exact_expected,
            best_response_actions=best_actions,
        )
        self.decisions.append(diagnostic)
        self._pending = _PendingDecision(
            self_name=decision.actor_name,
            self_role=decision.role,
            forecast=forecast,
        )
        self._game_decision_index += 1
        return {
            action: float(policy[action - 1])
            for action in ACTIONS
            if policy[action - 1] > 0.0
        }

    def observe(self, record: PublicHalfRound) -> None:
        """Learn from one revealed result after the pending policy was fixed."""

        pending = self._pending
        if pending is None:
            raise RuntimeError("Perfect Hal received a reveal without a pending action")
        if record.game_index < 0 or record.half_round_index < 0:
            raise ValueError("public reveal indices must be nonnegative")
        key = (self._game_epoch, record.game_index, record.half_round_index)
        if key in self._seen_reveals:
            raise RuntimeError("public reveal was delivered more than once")
        self_name = pending.self_name.casefold()
        if record.dropper_name.casefold() == self_name:
            self_role = "dropper"
            self_action = _literal_action(record.drop_time, label="self action")
            opponent_action = _literal_action(
                record.check_time,
                label="opponent action",
            )
        elif record.checker_name.casefold() == self_name:
            self_role = "checker"
            self_action = _literal_action(record.check_time, label="self action")
            opponent_action = _literal_action(
                record.drop_time,
                label="opponent action",
            )
        else:
            raise ValueError("public reveal does not include the Perfect Hal player")
        if self_role != pending.self_role:
            raise RuntimeError("public reveal role disagrees with the pending decision")
        self.opponent_model.observe(
            pending.forecast,
            opponent_action=opponent_action,
            self_action=self_action,
        )
        self._seen_reveals.add(key)
        self._pending = None

    def end_game(self, outcome: PublicGameOutcome) -> None:
        """Record a terminal public result without erasing opponent memory."""

        if self._pending is not None:
            raise RuntimeError("Perfect Hal game ended with an unrevealed action")
        if outcome.game_index < 0 or outcome.half_rounds < 0:
            raise ValueError("public game outcome indices must be nonnegative")
        self._last_outcome = outcome

    def match_summary(self) -> str:
        if not self.decisions:
            return "Perfect Hal: no moves played"
        gains = np.asarray(
            [decision.expected_exploit_gain for decision in self.decisions],
            dtype=np.float64,
        )
        latest = self.decisions[-1]
        dominant = max(latest.expert_weights, key=lambda item: item[1])
        return (
            f"Perfect Hal: {len(self.decisions)} unrestricted pure-DTH moves; "
            f"{self.opponent_model.total_observations} public reveals; "
            f"mean forecasted gain over exact {float(gains.mean()):+.4f}; "
            f"latest dominant expert {dominant[0]} ({dominant[1]:.1%})"
        )

    def experiment_diagnostics(self) -> dict[str, object]:
        latest = self.last_decision
        return {
            "schema_version": PERFECT_HAL_DIAGNOSTICS_SCHEMA,
            "model_schema": PERFECT_HAL_SCHEMA,
            "pure_dth_only": True,
            "public_history_only": True,
            "unrestricted_best_response": True,
            "equilibrium_safety_blend": False,
            "config": asdict(self.config),
            "decision_count": len(self.decisions),
            "observations_by_opponent_role": {
                role: self.opponent_model.observations(role)
                for role in ("dropper", "checker")
            },
            "latest": None
            if latest is None
            else {
                "role": latest.role,
                "opponent_role": latest.opponent_role,
                "evidence_count": latest.evidence_count,
                "forecast_entropy": latest.forecast_entropy,
                "forecast_confidence": latest.forecast_confidence,
                "expected_exploit_gain": latest.expected_exploit_gain,
                "best_response_actions": list(latest.best_response_actions),
                "expert_weights": dict(latest.expert_weights),
            },
        }


def make_live_provider(
    *,
    artifact_dir: str | Path,
    config: PerfectHalConfig = PerfectHalConfig(),
) -> PerfectHalPolicyProvider:
    """Construct the checkpoint-free live provider over exact DTH artifacts."""

    return PerfectHalPolicyProvider(artifact_dir, config)


__all__ = [
    "ACTION_COUNT",
    "ACTIONS",
    "PERFECT_HAL_SCHEMA",
    "PerfectHalConfig",
    "PerfectHalContext",
    "PerfectHalForecast",
    "PerfectHalOpponentModel",
    "PerfectHalDecision",
    "PerfectHalPolicyProvider",
    "make_live_provider",
]
