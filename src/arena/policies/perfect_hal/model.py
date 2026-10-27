"""Perfect Hal's public-history opponent model.

Perfect Hal is a policy label, not a claim that the DTH game has another
solution. The completed DTH tablebase remains the continuation-value and
stage-matrix authority. The model here forecasts the opponent from revealed
public history alone; ``decision`` answers that forecast in the exact
continuation-adjusted matrix, and ``policy`` plays it.

The live forecaster uses Bayesian latent-model filtering and integrates
change-point uncertainty. We retain the v1 forecaster as the frozen PM source
and evaluation baseline. Both learn from public reveals.
"""

from __future__ import annotations

from dataclasses import dataclass
from operator import index as integer_index

import numpy as np

from arena.contracts import CanonicalDecision

ACTION_COUNT = 60
ACTIONS = tuple(range(1, ACTION_COUNT + 1))

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
