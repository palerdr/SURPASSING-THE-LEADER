"""Perfect Mode Hal: change-aware, risk-measured pure-DTH exploitation.

PM Hal is a meta-policy, not another game solver.  One completed DTH agent
supplies the continuation-adjusted matrix and equilibrium certificate.  PM Hal
combines four kinds of public-history evidence around that authority:

* Adaptive Hal's role-separated Dirichlet posterior;
* Perfect Hal's interpretable online pattern experts;
* a categorical Bayesian online change-point model and an outcome-conditioned
  human-behavior expert; and
* an optional trained Aggro Hal recurrent forecast and direct policy.

Forecast sources are combined by actual Fixed Share updates under causal
prequential log loss.  The fused forecast is offered to Adaptive Hal's
independently checked epsilon frontier.  Hard Perfect/fused responses and the
optional Aggro direct policy may compete only when their freshly recomputed
local worst-case loss fits the active mode and the remaining per-game budget.

The four modes are deliberately asymmetric.  A cold start or change-point
shock enters ``shield``; partial evidence enters ``probe``; stable evidence
enters ``press``; and long, agreeing evidence may enter ``dominate``.  Thus the
default is aggressive against a stable repeated opponent while retaining an
explicit retreat path when the opponent changes or deceives the model.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
import math
from operator import index as integer_index
from pathlib import Path
from typing import Literal, Mapping, Protocol, Sequence

import numpy as np
import torch
from torch import Tensor

from arena.contracts import (
    CanonicalDecision,
    CanonicalPolicyProvider,
    PublicGameOutcome,
    PublicHalfRound,
)
from arena.dth_adapter import project_to_dth_state
from arena.policies.adaptive import (
    CertifiedCandidateGenerator,
    DirichletPrior,
    RoleDirichletOpponent,
)
from arena.policies.aggro_hal import (
    AggroHalConfig,
    AggroHalNetwork,
    dth_compatibility,
    encode_public_observation,
    load_checkpoint,
)
from arena.policies.perfect_hal import (
    PerfectHalConfig,
    PerfectHalForecast,
    PerfectHalOpponentModel,
)
from dth.agent import (
    CERTIFIED_SADDLE_GAP_TOLERANCE,
    CertifiedStageGame,
    CompleteDTHAgent,
)

Role = Literal["dropper", "checker"]
PMMode = Literal["shield", "probe", "press", "dominate"]

ACTION_COUNT = 60
ACTIONS = tuple(range(1, ACTION_COUNT + 1))
ROLES: tuple[Role, Role] = ("dropper", "checker")
LOG_ACTION_COUNT = math.log(ACTION_COUNT)
UNIFORM_BRIER = (ACTION_COUNT - 1.0) / ACTION_COUNT
PM_HAL_SCHEMA = "arena-pm-hal-fixed-share-change-aware-v2"
PM_HAL_DIAGNOSTICS_SCHEMA = "arena-pm-hal-diagnostics-v2"
PM_HAL_CONFIG_FILE_SCHEMA_V2 = "arena-pm-hal-controller-config-v2"
PM_HAL_CONFIG_FILE_SCHEMA = "arena-pm-hal-controller-config-v3"
DEFAULT_PM_HAL_CONFIG = Path("src/arena/config/pm_hal_controller_v3.json")
_NUMERICAL_TOLERANCE = 1e-9


def _finite(value: float, *, label: str) -> float:
    resolved = float(value)
    if not np.isfinite(resolved):
        raise ValueError(f"{label} must be finite")
    return resolved


def _role(raw: str) -> Role:
    if raw not in ROLES:
        raise ValueError("role must be 'dropper' or 'checker'")
    return raw


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


def _distribution(raw: object, *, label: str) -> np.ndarray:
    values = np.asarray(raw, dtype=np.float64)
    if (
        values.shape != (ACTION_COUNT,)
        or not np.all(np.isfinite(values))
        or np.any(values < 0.0)
        or float(values.sum()) <= 0.0
    ):
        raise ValueError(f"{label} must be a finite nonnegative length-60 distribution")
    result = values / float(values.sum())
    result.setflags(write=False)
    return result


def _readonly(raw: object) -> np.ndarray:
    result = np.asarray(raw, dtype=np.float64).copy()
    result.setflags(write=False)
    return result


def _entropy(policy: np.ndarray, *, floor: float) -> float:
    return -float(np.sum(policy * np.log(np.clip(policy, float(floor), 1.0))))


def _fixed_share(
    prior: np.ndarray,
    score: np.ndarray,
    *,
    share: float,
) -> np.ndarray:
    """Apply one normalized exponential update followed by Fixed Share."""

    weights = np.asarray(prior, dtype=np.float64)
    scores = np.asarray(score, dtype=np.float64)
    if (
        weights.ndim != 1
        or scores.shape != weights.shape
        or not np.all(np.isfinite(weights))
        or not np.all(np.isfinite(scores))
        or np.any(weights <= 0.0)
    ):
        raise ValueError("Fixed Share requires positive weights and finite scores")
    centered = scores - float(np.max(scores))
    posterior = weights * np.exp(np.clip(centered, -700.0, 0.0))
    posterior /= float(posterior.sum())
    if posterior.size > 1 and share > 0.0:
        posterior = (1.0 - share) * posterior + share / (posterior.size - 1) * (
            1.0 - posterior
        )
    posterior /= float(posterior.sum())
    return posterior


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


def _best_response(
    action_values: np.ndarray,
    *,
    temperature: float,
    tolerance: float,
) -> np.ndarray:
    values = np.asarray(action_values, dtype=np.float64)
    if values.shape != (ACTION_COUNT,) or not np.all(np.isfinite(values)):
        raise ValueError("best-response values must be finite length 60")
    best = float(np.max(values))
    if temperature == 0.0:
        raw = (values >= best - tolerance).astype(np.float64)
    else:
        raw = np.exp(np.clip((values - best) / temperature, -700.0, 0.0))
    return _distribution(raw, label="best-response policy")


@dataclass(frozen=True, slots=True)
class PMHalConfig:
    """PM Hal evidence, mode, and locally measured risk controls.

    The confidence rule combines a data-biased-response evidence term with
    PM-specific change, disagreement, and prequential-skill guards.  Only the
    evidence term has the published DBR interpretation; the product is a
    declared controller heuristic.  Every chosen policy is still checked
    directly against the current exact matrix.
    """

    action_count: int = ACTION_COUNT
    adaptive_prior_strength: float = 0.25
    adaptive_decay: float = 0.70
    fixed_share: float = 0.04
    source_learning_rate: float = 1.0
    controller_learning_rate: float = 2.0
    controller_prior_scale: float = 0.002
    metric_retention: float = 0.92
    log_probability_floor: float = 1e-9
    bocpd_hazard: float = 0.08
    bocpd_prior_strength: float = 0.50
    bocpd_max_run_length: int = 256
    outcome_prior_strength: float = 0.25
    evidence_strength: float = 4.0
    maximum_confidence: float = 1.0
    minimum_role_observations: int = 1
    change_shield_threshold: float = 0.35
    press_confidence_threshold: float = 0.15
    dominate_confidence_threshold: float = 0.32
    probe_disagreement_threshold: float = 0.40
    dominate_disagreement_threshold: float = 0.18
    dominate_change_threshold: float = 0.10
    epsilon_grid: tuple[float, ...] = (
        0.0,
        0.01,
        0.02,
        0.05,
        0.10,
        0.20,
        0.50,
        1.0,
        2.0,
    )
    game_epsilon_budget: float = 12.0
    probe_epsilon_cap: float = 0.05
    press_epsilon_cap: float = 0.50
    dominate_epsilon_cap: float = 2.0
    posterior_samples: int = 256
    posterior_meta_concentration: float = 6.0
    minimum_improvement_support: float = 0.55
    improvement_tolerance: float = 1e-10
    response_temperature: float = 0.0
    best_response_tolerance: float = 1e-12
    perfect_config: PerfectHalConfig = field(default_factory=PerfectHalConfig)

    def __post_init__(self) -> None:
        if self.action_count != ACTION_COUNT:
            raise ValueError("PM Hal requires exactly 60 pure-DTH actions")
        positive = (
            ("adaptive_prior_strength", self.adaptive_prior_strength),
            ("source_learning_rate", self.source_learning_rate),
            ("controller_learning_rate", self.controller_learning_rate),
            ("log_probability_floor", self.log_probability_floor),
            ("bocpd_prior_strength", self.bocpd_prior_strength),
            ("outcome_prior_strength", self.outcome_prior_strength),
            ("evidence_strength", self.evidence_strength),
            ("posterior_meta_concentration", self.posterior_meta_concentration),
        )
        for label, value in positive:
            if _finite(value, label=label) <= 0.0:
                raise ValueError(f"{label} must be positive")
        unit_interval = (
            ("adaptive_decay", self.adaptive_decay, False),
            ("fixed_share", self.fixed_share, True),
            ("metric_retention", self.metric_retention, True),
            ("bocpd_hazard", self.bocpd_hazard, False),
            ("maximum_confidence", self.maximum_confidence, False),
            (
                "minimum_improvement_support",
                self.minimum_improvement_support,
                True,
            ),
        )
        for label, value, allow_zero in unit_interval:
            resolved = _finite(value, label=label)
            lower_ok = resolved >= 0.0 if allow_zero else resolved > 0.0
            if not lower_ok or resolved > 1.0:
                bracket = "[0, 1]" if allow_zero else "(0, 1]"
                raise ValueError(f"{label} must lie in {bracket}")
        thresholds = (
            self.change_shield_threshold,
            self.press_confidence_threshold,
            self.dominate_confidence_threshold,
            self.probe_disagreement_threshold,
            self.dominate_disagreement_threshold,
            self.dominate_change_threshold,
        )
        if any(
            not 0.0 <= _finite(value, label="mode threshold") <= 1.0
            for value in thresholds
        ):
            raise ValueError("PM mode thresholds must lie in [0, 1]")
        if self.dominate_confidence_threshold < self.press_confidence_threshold:
            raise ValueError("dominate confidence must not be below press confidence")
        if self.dominate_disagreement_threshold > self.probe_disagreement_threshold:
            raise ValueError("dominate disagreement must not exceed probe disagreement")
        if (
            isinstance(self.minimum_role_observations, bool)
            or self.minimum_role_observations < 0
        ):
            raise ValueError("minimum_role_observations must be nonnegative")
        if isinstance(self.bocpd_max_run_length, bool) or self.bocpd_max_run_length < 2:
            raise ValueError("bocpd_max_run_length must be at least two")
        if isinstance(self.posterior_samples, bool) or self.posterior_samples <= 0:
            raise ValueError("posterior_samples must be positive")
        epsilons = tuple(sorted(set(float(value) for value in self.epsilon_grid)))
        if (
            not epsilons
            or epsilons[0] != 0.0
            or any(not np.isfinite(value) or value < 0.0 for value in epsilons)
        ):
            raise ValueError("epsilon_grid must contain finite nonnegative 0.0")
        caps = (
            self.game_epsilon_budget,
            self.probe_epsilon_cap,
            self.press_epsilon_cap,
            self.dominate_epsilon_cap,
        )
        if any(_finite(value, label="epsilon cap") < 0.0 for value in caps):
            raise ValueError("epsilon budgets and caps must be nonnegative")
        if not (
            self.probe_epsilon_cap
            <= self.press_epsilon_cap
            <= self.dominate_epsilon_cap
            <= self.game_epsilon_budget
        ):
            raise ValueError(
                "epsilon caps must increase probe/press/dominate within the game budget"
            )
        if _finite(self.controller_prior_scale, label="controller_prior_scale") < 0.0:
            raise ValueError("controller_prior_scale must be nonnegative")
        if not 0.0 < self.log_probability_floor < 1.0:
            raise ValueError("log_probability_floor must lie in (0, 1)")
        if _finite(self.improvement_tolerance, label="improvement_tolerance") < 0.0:
            raise ValueError("improvement_tolerance must be nonnegative")
        if _finite(self.response_temperature, label="response_temperature") < 0.0:
            raise ValueError("response_temperature must be nonnegative")
        if _finite(self.best_response_tolerance, label="best_response_tolerance") < 0.0:
            raise ValueError("best_response_tolerance must be nonnegative")
        object.__setattr__(self, "epsilon_grid", epsilons)


@dataclass(frozen=True, slots=True)
class ChangePointForecast:
    """One role's categorical run-length mixture before the next reveal."""

    role: Role
    policy: np.ndarray
    change_probability: float
    expected_run_length: float
    hypotheses: int


@dataclass(slots=True)
class _ChangeRoleState:
    probabilities: np.ndarray
    counts: np.ndarray
    change_probability: float = 0.0
    observations: int = 0


class CategoricalChangePointModel:
    """Truncated Dirichlet-multinomial Bayesian online change detection."""

    def __init__(
        self,
        *,
        hazard: float = 0.08,
        prior_strength: float = 0.50,
        max_run_length: int = 256,
    ) -> None:
        if not 0.0 < _finite(hazard, label="hazard") <= 1.0:
            raise ValueError("hazard must lie in (0, 1]")
        if _finite(prior_strength, label="prior_strength") <= 0.0:
            raise ValueError("prior_strength must be positive")
        if isinstance(max_run_length, bool) or max_run_length < 2:
            raise ValueError("max_run_length must be at least two")
        self.hazard = float(hazard)
        self.prior_strength = float(prior_strength)
        self.max_run_length = int(max_run_length)
        self._roles = {role: self._new_state() for role in ROLES}

    @staticmethod
    def _new_state() -> _ChangeRoleState:
        return _ChangeRoleState(
            probabilities=np.ones(1, dtype=np.float64),
            counts=np.zeros((1, ACTION_COUNT), dtype=np.float64),
        )

    def reset(self) -> None:
        self._roles = {role: self._new_state() for role in ROLES}

    def _predictives(self, state: _ChangeRoleState) -> np.ndarray:
        alpha = self.prior_strength / ACTION_COUNT
        totals = state.counts.sum(axis=1, keepdims=True)
        return (state.counts + alpha) / (totals + self.prior_strength)

    def predict(self, role: str) -> ChangePointForecast:
        resolved = _role(role)
        state = self._roles[resolved]
        predictives = self._predictives(state)
        policy = _distribution(
            state.probabilities @ predictives,
            label="change-point predictive",
        )
        run_lengths = state.counts.sum(axis=1)
        return ChangePointForecast(
            role=resolved,
            policy=policy,
            change_probability=float(state.change_probability),
            expected_run_length=float(state.probabilities @ run_lengths),
            hypotheses=int(len(state.probabilities)),
        )

    def observe(self, role: str, action: int) -> None:
        resolved = _role(role)
        selected = _literal_action(action, label="change-point action")
        state = self._roles[resolved]
        index = selected - 1
        predictives = self._predictives(state)
        growth = (1.0 - self.hazard) * state.probabilities * predictives[:, index]
        prior_likelihood = 1.0 / ACTION_COUNT
        change = self.hazard * prior_likelihood
        masses = np.concatenate((np.asarray([change]), growth))
        point = np.zeros(ACTION_COUNT, dtype=np.float64)
        point[index] = 1.0
        counts = np.concatenate(
            (
                point.reshape(1, -1),
                state.counts + point.reshape(1, -1),
            ),
            axis=0,
        )
        limit = self.max_run_length + 1
        if len(masses) > limit:
            head_mass = masses[: limit - 1]
            tail_mass = float(masses[limit - 1 :].sum())
            tail_weights = masses[limit - 1 :]
            if tail_mass > 0.0:
                tail_counts = np.average(
                    counts[limit - 1 :], axis=0, weights=tail_weights
                )
            else:
                tail_counts = counts[-1]
            masses = np.concatenate((head_mass, np.asarray([tail_mass])))
            counts = np.concatenate(
                (counts[: limit - 1], tail_counts.reshape(1, -1)), axis=0
            )
        total = float(masses.sum())
        if not np.isfinite(total) or total <= 0.0:
            raise RuntimeError("change-point posterior lost all probability mass")
        probabilities = masses / total
        state.probabilities = probabilities
        state.counts = counts
        state.change_probability = float(probabilities[0])
        state.observations += 1

    def change_probability(self, role: str) -> float:
        return float(self._roles[_role(role)].change_probability)


@dataclass(slots=True)
class _OutcomeRoleState:
    counts: dict[str, np.ndarray]


class _OutcomeConditionedModel:
    """Sparse expert for the action that follows the last public outcome."""

    def __init__(self, *, prior_strength: float) -> None:
        self.prior_strength = float(prior_strength)
        self._roles = {role: _OutcomeRoleState(counts={}) for role in ROLES}
        self._previous_outcome: str | None = None

    def reset(self) -> None:
        self._roles = {role: _OutcomeRoleState(counts={}) for role in ROLES}
        self._previous_outcome = None

    def predict(self, role: str, prior: np.ndarray) -> np.ndarray:
        resolved = _role(role)
        state = self._roles[resolved]
        if self._previous_outcome is None:
            return _distribution(prior, label="outcome-expert prior")
        counts = state.counts.get(self._previous_outcome)
        if counts is None:
            return _distribution(prior, label="outcome-expert cold context")
        return _distribution(
            counts + self.prior_strength * np.asarray(prior, dtype=np.float64),
            label="outcome-conditioned predictive",
        )

    def observe(self, role: str, action: int, *, outcome: str) -> None:
        resolved = _role(role)
        selected = _literal_action(action, label="outcome-expert action")
        if not isinstance(outcome, str) or not outcome:
            raise ValueError("public outcome must be a nonempty string")
        state = self._roles[resolved]
        if self._previous_outcome is not None:
            counts = state.counts.setdefault(
                self._previous_outcome,
                np.zeros(ACTION_COUNT, dtype=np.float64),
            )
            counts[selected - 1] += 1.0
        self._previous_outcome = outcome


@dataclass(slots=True)
class _MetaRoleState:
    source_weights: np.ndarray
    controller_weights: np.ndarray
    observations: int = 0
    realized_nll_ema: float = LOG_ACTION_COUNT
    brier_ema: float = UNIFORM_BRIER


@dataclass(frozen=True, slots=True)
class PMHalForecast:
    """Immutable causal forecast token consumed exactly once after reveal."""

    opponent_role: Role
    observation_index: int
    component_names: tuple[str, ...]
    component_weights: np.ndarray
    component_policies: np.ndarray
    policy: np.ndarray
    perfect_forecast: PerfectHalForecast
    entropy: float
    disagreement: float
    change_probability: float
    expected_run_length: float
    effective_evidence: float
    prequential_skill: float
    confidence: float


class PMHalOpponentModel:
    """Role-separated fixed-share synthesis of PM Hal's forecast sources."""

    def __init__(
        self,
        config: PMHalConfig = PMHalConfig(),
        *,
        aggro_enabled: bool = False,
    ) -> None:
        self.config = config
        self.aggro_enabled = bool(aggro_enabled)
        prior = DirichletPrior.uniform(strength=config.adaptive_prior_strength)
        self.adaptive = RoleDirichletOpponent(
            prior,
            prior,
            decay=config.adaptive_decay,
        )
        self.perfect = PerfectHalOpponentModel(config.perfect_config)
        self.change = CategoricalChangePointModel(
            hazard=config.bocpd_hazard,
            prior_strength=config.bocpd_prior_strength,
            max_run_length=config.bocpd_max_run_length,
        )
        self.outcome = _OutcomeConditionedModel(
            prior_strength=config.outcome_prior_strength
        )
        self.source_names = (
            "uniform",
            "equilibrium",
            "adaptive",
            "perfect",
            "change_point",
            "outcome",
        ) + (("aggro",) if self.aggro_enabled else ())
        self.controller_names = (
            "exact",
            *(f"frontier_{epsilon:g}" for epsilon in config.epsilon_grid),
            "fused_hard",
            "perfect_hard",
        ) + (("aggro_direct",) if self.aggro_enabled else ())
        self._roles = {role: self._new_meta_state() for role in ROLES}

    def _new_meta_state(self) -> _MetaRoleState:
        return _MetaRoleState(
            source_weights=np.full(
                len(self.source_names),
                1.0 / len(self.source_names),
                dtype=np.float64,
            ),
            controller_weights=np.full(
                len(self.controller_names),
                1.0 / len(self.controller_names),
                dtype=np.float64,
            ),
        )

    def reset(self) -> None:
        self.adaptive = RoleDirichletOpponent(
            DirichletPrior.uniform(strength=self.config.adaptive_prior_strength),
            DirichletPrior.uniform(strength=self.config.adaptive_prior_strength),
            decay=self.config.adaptive_decay,
        )
        self.perfect.reset()
        self.change.reset()
        self.outcome.reset()
        self._roles = {role: self._new_meta_state() for role in ROLES}

    @property
    def total_observations(self) -> int:
        return sum(state.observations for state in self._roles.values())

    def observations(self, role: str) -> int:
        return self._roles[_role(role)].observations

    def controller_weight(self, role: str, name: str) -> float:
        resolved = _role(role)
        try:
            index = self.controller_names.index(name)
        except ValueError as error:
            raise ValueError(f"unknown PM controller {name!r}") from error
        return float(self._roles[resolved].controller_weights[index])

    def predict(
        self,
        opponent_role: str,
        *,
        decision: CanonicalDecision,
        equilibrium_policy: np.ndarray,
        game_index: int,
        game_decision_index: int,
        aggro_policy: np.ndarray | None = None,
    ) -> PMHalForecast:
        resolved = _role(opponent_role)
        state = self._roles[resolved]
        equilibrium = _distribution(
            equilibrium_policy, label="opponent equilibrium policy"
        )
        perfect = self.perfect.predict(
            resolved,
            state_regime=self.perfect.state_regime(decision),
            game_index=game_index,
            game_decision_index=game_decision_index,
        )
        adaptive = _distribution(
            self.adaptive.predictive(resolved), label="adaptive predictive"
        )
        change = self.change.predict(resolved)
        outcome = self.outcome.predict(resolved, adaptive)
        bank: dict[str, np.ndarray] = {
            "uniform": np.full(ACTION_COUNT, 1.0 / ACTION_COUNT),
            "equilibrium": equilibrium,
            "adaptive": adaptive,
            "perfect": perfect.policy,
            "change_point": change.policy,
            "outcome": outcome,
        }
        if self.aggro_enabled:
            if aggro_policy is None:
                raise ValueError("Aggro-enabled PM Hal requires a recurrent forecast")
            bank["aggro"] = _distribution(aggro_policy, label="Aggro opponent forecast")
        elif aggro_policy is not None:
            raise ValueError("Aggro forecast supplied to a checkpoint-free PM model")
        policies = np.stack([bank[name] for name in self.source_names], axis=0)
        weights = np.asarray(state.source_weights, dtype=np.float64)
        fused = _distribution(weights @ policies, label="PM fused forecast")
        component_entropies = np.asarray(
            [
                _entropy(policy, floor=self.config.log_probability_floor)
                for policy in policies
            ],
            dtype=np.float64,
        )
        entropy = _entropy(fused, floor=self.config.log_probability_floor)
        js = max(0.0, entropy - float(weights @ component_entropies))
        normalization = math.log(len(self.source_names))
        disagreement = float(
            np.clip(js / normalization if normalization > 0.0 else 0.0, 0.0, 1.0)
        )
        effective = max(
            0.0,
            self.adaptive.effective_concentration(resolved)
            - self.config.adaptive_prior_strength,
        )
        evidence_factor = effective / (self.config.evidence_strength + effective)
        prequential_skill = math.exp(
            -max(0.0, state.realized_nll_ema - LOG_ACTION_COUNT)
        )
        confidence = (
            self.config.maximum_confidence
            * evidence_factor
            * (1.0 - change.change_probability)
            * (1.0 - disagreement)
            * prequential_skill
        )
        matrix = np.asarray(policies, dtype=np.float64).copy()
        matrix.setflags(write=False)
        return PMHalForecast(
            opponent_role=resolved,
            observation_index=state.observations,
            component_names=self.source_names,
            component_weights=_readonly(weights),
            component_policies=matrix,
            policy=fused,
            perfect_forecast=perfect,
            entropy=entropy,
            disagreement=disagreement,
            change_probability=change.change_probability,
            expected_run_length=change.expected_run_length,
            effective_evidence=effective,
            prequential_skill=prequential_skill,
            confidence=float(np.clip(confidence, 0.0, 1.0)),
        )

    def observe(
        self,
        forecast: PMHalForecast,
        *,
        opponent_action: int,
        self_action: int,
        outcome: str,
    ) -> None:
        opponent = _literal_action(opponent_action, label="opponent action")
        own = _literal_action(self_action, label="self action")
        state = self._roles[forecast.opponent_role]
        if forecast.observation_index != state.observations:
            raise RuntimeError("PM forecast token is stale or already observed")
        if forecast.component_names != self.source_names or (
            forecast.component_policies.shape != (len(self.source_names), ACTION_COUNT)
        ):
            raise ValueError("PM forecast token is incompatible")
        index = opponent - 1
        likelihood = np.clip(
            forecast.component_policies[:, index],
            self.config.log_probability_floor,
            1.0,
        )
        source_score = self.config.source_learning_rate * np.log(likelihood)
        state.source_weights = _fixed_share(
            state.source_weights,
            source_score,
            share=self.config.fixed_share,
        )
        nll = -math.log(
            max(float(forecast.policy[index]), self.config.log_probability_floor)
        )
        target = np.zeros(ACTION_COUNT, dtype=np.float64)
        target[index] = 1.0
        brier = float(np.sum((forecast.policy - target) ** 2))
        retention = self.config.metric_retention
        state.realized_nll_ema = (
            retention * state.realized_nll_ema + (1.0 - retention) * nll
        )
        state.brier_ema = retention * state.brier_ema + (1.0 - retention) * brier
        self.perfect.observe(
            forecast.perfect_forecast,
            opponent_action=opponent,
            self_action=own,
        )
        self.adaptive.observe(forecast.opponent_role, opponent)
        self.change.observe(forecast.opponent_role, opponent)
        self.outcome.observe(forecast.opponent_role, opponent, outcome=outcome)
        state.observations += 1

    def update_controllers(
        self,
        role: str,
        *,
        candidate_names: Sequence[str],
        advantages: Sequence[float],
    ) -> None:
        resolved = _role(role)
        if len(candidate_names) != len(advantages):
            raise ValueError("candidate names and advantages must have equal length")
        scores = np.zeros(len(self.controller_names), dtype=np.float64)
        seen: set[str] = set()
        for name, advantage in zip(candidate_names, advantages, strict=True):
            if name in seen:
                raise ValueError("candidate controller names must be unique")
            seen.add(name)
            try:
                index = self.controller_names.index(name)
            except ValueError as error:
                raise ValueError(f"unknown PM controller {name!r}") from error
            scores[index] = self.config.controller_learning_rate * _finite(
                advantage, label="controller counterfactual advantage"
            )
        state = self._roles[resolved]
        state.controller_weights = _fixed_share(
            state.controller_weights,
            scores,
            share=self.config.fixed_share,
        )


class _PosteriorView:
    """Heuristic ensemble perturbations used as a controller support score.

    These draws are deliberately not represented as a calibrated Bayesian
    posterior.  One cached draw bank is shared by every candidate considered at
    a decision so Monte Carlo noise cannot favor a candidate merely because it
    received a different sample.
    """

    def __init__(
        self,
        model: PMHalOpponentModel,
        forecast: PMHalForecast,
    ) -> None:
        self.model = model
        self.forecast = forecast
        self._cached_samples: dict[tuple[Role, int], np.ndarray] = {}

    def predictive(self, role: str) -> np.ndarray:
        if _role(role) != self.forecast.opponent_role:
            raise ValueError("PM posterior view is valid for one opponent role")
        return np.asarray(self.forecast.policy, dtype=np.float64).copy()

    def sample(
        self,
        role: str,
        *,
        size: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        resolved = _role(role)
        if resolved != self.forecast.opponent_role:
            raise ValueError("PM posterior view is valid for one opponent role")
        if isinstance(size, bool) or size <= 0:
            raise ValueError("posterior sample size must be positive")
        key = (resolved, int(size))
        cached = self._cached_samples.get(key)
        if cached is not None:
            return cached.copy()
        adaptive_samples = self.model.adaptive.sample(resolved, size=int(size), rng=rng)
        concentration = (
            self.model.config.posterior_meta_concentration
            + self.forecast.effective_evidence
        )
        meta_alpha = np.maximum(
            self.forecast.component_weights * concentration,
            1e-3,
        )
        meta = rng.dirichlet(meta_alpha, size=int(size))
        policies = np.broadcast_to(
            self.forecast.component_policies,
            (int(size),) + self.forecast.component_policies.shape,
        ).copy()
        adaptive_index = self.forecast.component_names.index("adaptive")
        policies[:, adaptive_index, :] = adaptive_samples
        samples = np.einsum("sk,ska->sa", meta, policies)
        samples /= samples.sum(axis=1, keepdims=True)
        self._cached_samples[key] = samples.copy()
        return samples.copy()

    def observation_count(self, role: str) -> int:
        return self.model.observations(role)


@dataclass(frozen=True, slots=True)
class PMHalCandidate:
    """One exact, frontier, or direct policy considered by PM Hal."""

    name: str
    source: str
    policy: tuple[float, ...]
    actual_worst_case_loss: float
    budget_charge: float
    expected_improvement: float
    improvement_support: float
    valid: bool
    reason: str


@dataclass(frozen=True, slots=True)
class PMHalCandidateSummary:
    name: str
    source: str
    actual_worst_case_loss: float
    budget_charge: float
    expected_improvement: float
    improvement_support: float
    controller_weight: float
    admitted: bool
    reason: str


@dataclass(frozen=True, slots=True)
class PMHalDecision:
    """Read-only decision diagnostics for evaluation and human-facing audits."""

    state: tuple[int, int, int, int]
    game_index: int
    role: Role
    opponent_role: Role
    mode: PMMode
    policy: tuple[float, ...]
    opponent_policy: tuple[float, ...]
    component_weights: tuple[tuple[str, float], ...]
    component_policies: tuple[tuple[str, tuple[float, ...]], ...]
    evidence_count: int
    effective_evidence: float
    forecast_entropy: float
    forecast_disagreement: float
    change_probability: float
    expected_run_length: float
    prequential_skill: float
    confidence: float
    selected_candidate: str
    selected_source: str
    selected_actual_worst_case_loss: float
    selected_budget_charge: float
    game_epsilon_spent: float
    game_epsilon_remaining: float
    expected_improvement: float
    improvement_support: float
    aggro_enabled: bool
    aggro_direct_weight: float | None
    candidates: tuple[PMHalCandidateSummary, ...]


@dataclass(frozen=True, slots=True)
class PMHalObservation:
    """Reveal-time proper scores and full-information controller feedback."""

    role: Role
    opponent_role: Role
    opponent_action: int
    realized_nll: float
    brier_score: float
    selected_counterfactual_gain_over_exact: float
    one_step_candidate_regret: float
    post_reveal_change_probability: float


@dataclass(frozen=True, slots=True)
class _AggroInference:
    opponent_policy: np.ndarray
    direct_policy: np.ndarray
    direct_weight: float


@dataclass(frozen=True, slots=True)
class _PendingDecision:
    self_name: str
    self_role: Role
    forecast: PMHalForecast
    stage: CertifiedStageGame
    candidates: tuple[PMHalCandidate, ...]
    selected_index: int


class _StageAgent(Protocol):
    def stage_game(self, state: tuple[int, int, int, int]) -> CertifiedStageGame: ...


class PMHalPolicyProvider(CanonicalPolicyProvider):
    """Causal PM Hal provider for repeated pure-DTH sessions."""

    def __init__(
        self,
        artifact_dir: str | Path,
        config: PMHalConfig = PMHalConfig(),
        *,
        agent: CompleteDTHAgent | _StageAgent | None = None,
        opponent_model: PMHalOpponentModel | None = None,
        aggro_model: AggroHalNetwork | None = None,
        aggro_config: AggroHalConfig | None = None,
        device: str | torch.device = "cpu",
        seed: int | None = None,
    ) -> None:
        self.artifact_dir = Path(artifact_dir)
        self.config = config
        self.agent = agent or CompleteDTHAgent(self.artifact_dir)
        self.device = torch.device(device)
        self.aggro_model = aggro_model
        if aggro_model is None and aggro_config is not None:
            raise ValueError("aggro_config requires an Aggro model")
        if aggro_model is not None:
            resolved_aggro = (
                aggro_model.config if aggro_config is None else aggro_config
            )
            if aggro_model.config != resolved_aggro:
                raise ValueError("Aggro model and PM provider configurations differ")
            self.aggro_config = resolved_aggro
            self.aggro_model = aggro_model.to(self.device)
            self.aggro_model.eval()
        else:
            self.aggro_config = None
        self.opponent_model = opponent_model or PMHalOpponentModel(
            config, aggro_enabled=aggro_model is not None
        )
        if self.opponent_model.config != config:
            raise ValueError("PM provider and opponent-model configs differ")
        if self.opponent_model.aggro_enabled != (aggro_model is not None):
            raise ValueError("PM provider and opponent model disagree about Aggro")
        self._candidate_source = CertifiedCandidateGenerator(
            epsilon_grid=config.epsilon_grid,
            posterior_samples=config.posterior_samples,
            improvement_tolerance=config.improvement_tolerance,
        )
        self._initial_seed = seed
        self._rng = np.random.default_rng(self._initial_seed)
        self.decisions: list[PMHalDecision] = []
        self.observation_metrics: list[PMHalObservation] = []
        self._pending: _PendingDecision | None = None
        self._seen_reveals: set[tuple[int, int, int]] = set()
        self._current_actor_name: str | None = None
        self._game_started = False
        self._game_epoch = -1
        self._game_decision_index = 0
        self._epsilon_spent = 0.0
        self._last_outcome: PublicGameOutcome | None = None
        self._aggro_hidden: Tensor | None = None
        self._aggro_previous_reveal: PublicHalfRound | None = None
        self._aggro_previous_self_name: str | None = None
        self._aggro_new_game = True

    @classmethod
    def from_checkpoint(
        cls,
        *,
        artifact_dir: str | Path,
        checkpoint: str | Path,
        config: PMHalConfig = PMHalConfig(),
        aggro_config: AggroHalConfig | None = None,
        device: str | torch.device = "cpu",
        seed: int | None = None,
    ) -> "PMHalPolicyProvider":
        agent = CompleteDTHAgent(artifact_dir)
        model, _ = load_checkpoint(
            checkpoint,
            expected_config=aggro_config,
            dth_ruleset=dth_compatibility(agent),
            device=device,
        )
        return cls(
            artifact_dir,
            config,
            agent=agent,
            aggro_model=model,
            aggro_config=model.config,
            device=device,
            seed=seed,
        )

    @property
    def last_decision(self) -> PMHalDecision | None:
        return self.decisions[-1] if self.decisions else None

    @property
    def pending_stage_game(self) -> CertifiedStageGame | None:
        """Return the exact public-state stage used by the pending decision."""

        return None if self._pending is None else self._pending.stage

    @property
    def has_session_memory(self) -> bool:
        return self.opponent_model.total_observations > 0

    @property
    def epsilon_spent(self) -> float:
        return self._epsilon_spent

    def close(self) -> None:
        """Match the provider lifecycle; tablebase memory maps need no close."""

    def reset_session(self) -> None:
        if self._pending is not None:
            raise RuntimeError("PM Hal session reset with an unrevealed action")
        self.opponent_model.reset()
        self._rng = np.random.default_rng(self._initial_seed)
        self.decisions.clear()
        self.observation_metrics.clear()
        self._seen_reveals.clear()
        self._current_actor_name = None
        self._game_started = False
        self._game_epoch = -1
        self._game_decision_index = 0
        self._epsilon_spent = 0.0
        self._last_outcome = None
        self._aggro_hidden = None
        self._aggro_previous_reveal = None
        self._aggro_previous_self_name = None
        self._aggro_new_game = True

    def reset_game(self) -> None:
        if self._pending is not None:
            raise RuntimeError("PM Hal game reset with an unrevealed action")
        if self._game_started:
            self._game_epoch += 1
        else:
            self._game_started = True
            self._game_epoch = 0
        self._current_actor_name = None
        self._game_decision_index = 0
        self._epsilon_spent = 0.0
        self._last_outcome = None
        self._aggro_new_game = True

    @staticmethod
    def _validate_decision(decision: CanonicalDecision) -> None:
        if decision.role not in ROLES:
            raise ValueError("PM Hal role must be dropper or checker")
        if (
            decision.turn_duration != ACTION_COUNT
            or tuple(decision.legal_seconds) != ACTIONS
        ):
            raise ValueError(
                "PM Hal supports pure DTH only: turn duration and legal actions "
                "must be exactly literal seconds 1..60"
            )

    def _aggro_inference(
        self,
        decision: CanonicalDecision,
        stage: CertifiedStageGame,
    ) -> _AggroInference | None:
        model = self.aggro_model
        config = self.aggro_config
        if model is None or config is None:
            return None
        features = encode_public_observation(
            decision,
            stage,
            self._aggro_previous_reveal,
            previous_self_name=self._aggro_previous_self_name,
            new_game=self._aggro_new_game,
        )
        exact = stage.drop_policy if decision.role == "dropper" else stage.check_policy
        legal = np.ones(ACTION_COUNT, dtype=np.bool_)
        feature_array = np.array(features, dtype=np.float32, copy=True)
        stage_array = np.array(stage.matrix, dtype=np.float32, copy=True)
        exact_array = np.array(exact, dtype=np.float32, copy=True)
        with torch.inference_mode():
            output = model(
                torch.as_tensor(
                    feature_array, dtype=torch.float32, device=self.device
                ).view(1, 1, -1),
                torch.as_tensor(
                    stage_array,
                    dtype=torch.float32,
                    device=self.device,
                ).view(1, 1, ACTION_COUNT, ACTION_COUNT),
                torch.as_tensor(
                    exact_array,
                    dtype=torch.float32,
                    device=self.device,
                ).view(1, 1, ACTION_COUNT),
                torch.tensor(
                    [[decision.role == "dropper"]],
                    dtype=torch.bool,
                    device=self.device,
                ),
                torch.as_tensor(legal, dtype=torch.bool, device=self.device).view(
                    1, 1, ACTION_COUNT
                ),
                self._aggro_hidden,
            )
        self._aggro_hidden = output.hidden_state.detach()
        self._aggro_previous_reveal = None
        self._aggro_previous_self_name = None
        self._aggro_new_game = False
        opponent = _distribution(
            output.opponent_policy[0, 0].detach().cpu().numpy(),
            label="Aggro recurrent forecast",
        )
        direct = _distribution(
            output.direct_policy[0, 0].detach().cpu().numpy(),
            label="Aggro direct policy",
        )
        return _AggroInference(
            opponent_policy=opponent,
            direct_policy=direct,
            direct_weight=float(output.direct_weight[0, 0].item()),
        )

    @staticmethod
    def _oriented_values(
        stage: CertifiedStageGame,
        role: Role,
        opponent_policy: np.ndarray,
    ) -> np.ndarray:
        matrix = np.asarray(stage.matrix, dtype=np.float64)
        return (
            matrix @ opponent_policy
            if role == "dropper"
            else -(matrix.T @ opponent_policy)
        )

    @staticmethod
    def _actual_loss(
        stage: CertifiedStageGame,
        role: Role,
        policy: np.ndarray,
    ) -> float:
        matrix = np.asarray(stage.matrix, dtype=np.float64)
        drop = np.asarray(stage.drop_policy, dtype=np.float64)
        check = np.asarray(stage.check_policy, dtype=np.float64)
        lower = float(np.min(matrix.T @ drop))
        upper = float(np.max(matrix @ check))
        selected = (
            float(np.min(matrix.T @ policy))
            if role == "dropper"
            else float(np.max(matrix @ policy))
        )
        loss = lower - selected if role == "dropper" else selected - upper
        return max(0.0, float(loss))

    def _direct_candidate(
        self,
        *,
        name: str,
        source: str,
        policy: np.ndarray,
        stage: CertifiedStageGame,
        role: Role,
        posterior: _PosteriorView,
        exact_policy: np.ndarray,
    ) -> PMHalCandidate:
        resolved = _distribution(policy, label=f"{name} policy")
        samples = posterior.sample(
            "checker" if role == "dropper" else "dropper",
            size=self.config.posterior_samples,
            rng=self._rng,
        )
        matrix = np.asarray(stage.matrix, dtype=np.float64)
        if role == "dropper":
            improvements = samples @ (matrix.T @ (resolved - exact_policy))
        else:
            improvements = samples @ (matrix @ (exact_policy - resolved))
        loss = self._actual_loss(stage, role, resolved)
        return PMHalCandidate(
            name=name,
            source=source,
            policy=tuple(float(value) for value in resolved),
            actual_worst_case_loss=loss,
            budget_charge=loss,
            expected_improvement=float(np.mean(improvements)),
            improvement_support=float(
                np.mean(improvements > self.config.improvement_tolerance)
            ),
            valid=True,
            reason="independently-measured-direct-policy",
        )

    def _candidates(
        self,
        stage: CertifiedStageGame,
        role: Role,
        forecast: PMHalForecast,
        aggro: _AggroInference | None,
    ) -> tuple[PMHalCandidate, ...]:
        remaining = max(0.0, self.config.game_epsilon_budget - self._epsilon_spent)
        posterior = _PosteriorView(self.opponent_model, forecast)
        frontier = self._candidate_source.candidates(
            stage,
            role,
            posterior,
            remaining,
            self._rng,
        )
        result: list[PMHalCandidate] = []
        for index, candidate in enumerate(frontier):
            name = "exact" if index == 0 else f"frontier_{candidate.epsilon:g}"
            source = "exact" if index == 0 else "adaptive_frontier"
            result.append(
                PMHalCandidate(
                    name=name,
                    source=source,
                    policy=candidate.policy,
                    actual_worst_case_loss=candidate.actual_worst_case_loss,
                    budget_charge=0.0 if index == 0 else candidate.epsilon,
                    expected_improvement=candidate.expected_improvement,
                    improvement_support=candidate.improvement_probability,
                    valid=candidate.valid,
                    reason=candidate.reason,
                )
            )
        exact = np.asarray(
            stage.drop_policy if role == "dropper" else stage.check_policy,
            dtype=np.float64,
        )
        fused_values = self._oriented_values(stage, role, forecast.policy)
        fused_hard = _best_response(
            fused_values,
            temperature=self.config.response_temperature,
            tolerance=self.config.best_response_tolerance,
        )
        result.append(
            self._direct_candidate(
                name="fused_hard",
                source="fused_forecast",
                policy=fused_hard,
                stage=stage,
                role=role,
                posterior=posterior,
                exact_policy=exact,
            )
        )
        perfect_values = self._oriented_values(
            stage, role, forecast.perfect_forecast.policy
        )
        perfect_hard = _best_response(
            perfect_values,
            temperature=self.config.response_temperature,
            tolerance=self.config.best_response_tolerance,
        )
        result.append(
            self._direct_candidate(
                name="perfect_hard",
                source="perfect_forecast",
                policy=perfect_hard,
                stage=stage,
                role=role,
                posterior=posterior,
                exact_policy=exact,
            )
        )
        if aggro is not None:
            result.append(
                self._direct_candidate(
                    name="aggro_direct",
                    source="aggro_recurrent",
                    policy=aggro.direct_policy,
                    stage=stage,
                    role=role,
                    posterior=posterior,
                    exact_policy=exact,
                )
            )
        if (
            tuple(candidate.name for candidate in result)
            != self.opponent_model.controller_names
        ):
            raise RuntimeError("PM candidate bank drifted from controller weights")
        return tuple(result)

    def _mode(self, forecast: PMHalForecast) -> PMMode:
        if (
            forecast.observation_index < self.config.minimum_role_observations
            or forecast.change_probability >= self.config.change_shield_threshold
        ):
            return "shield"
        if (
            forecast.confidence >= self.config.dominate_confidence_threshold
            and forecast.disagreement <= self.config.dominate_disagreement_threshold
            and forecast.change_probability <= self.config.dominate_change_threshold
        ):
            return "dominate"
        if (
            forecast.confidence < self.config.press_confidence_threshold
            or forecast.disagreement >= self.config.probe_disagreement_threshold
        ):
            return "probe"
        return "press"

    def _risk_cap(self, mode: PMMode) -> float:
        remaining = max(0.0, self.config.game_epsilon_budget - self._epsilon_spent)
        if mode == "shield":
            return min(_NUMERICAL_TOLERANCE, remaining)
        if mode == "probe":
            return min(self.config.probe_epsilon_cap, remaining)
        if mode == "press":
            return min(self.config.press_epsilon_cap, remaining)
        return min(self.config.dominate_epsilon_cap, remaining)

    def _select(
        self,
        candidates: tuple[PMHalCandidate, ...],
        *,
        opponent_role: Role,
        mode: PMMode,
    ) -> tuple[int, tuple[PMHalCandidateSummary, ...]]:
        cap = self._risk_cap(mode)
        remaining = max(0.0, self.config.game_epsilon_budget - self._epsilon_spent)
        summaries: list[PMHalCandidateSummary] = []
        admitted: list[int] = []
        scores: list[float] = []
        for index, candidate in enumerate(candidates):
            weight = self.opponent_model.controller_weight(
                opponent_role, candidate.name
            )
            gain_supported = candidate.name == "exact" or (
                mode != "shield"
                and candidate.expected_improvement > self.config.improvement_tolerance
                and candidate.improvement_support
                >= self.config.minimum_improvement_support
            )
            within_risk = (
                candidate.actual_worst_case_loss <= cap + _NUMERICAL_TOLERANCE
                and candidate.budget_charge <= remaining + _NUMERICAL_TOLERANCE
            )
            allowed = candidate.valid and gain_supported and within_risk
            reason = candidate.reason
            if candidate.valid and not gain_supported:
                reason = (
                    "shield-exact-only"
                    if mode == "shield"
                    else "posterior-improvement-gate-failed"
                )
            elif candidate.valid and gain_supported and not within_risk:
                reason = "mode-or-budget-risk-cap"
            summaries.append(
                PMHalCandidateSummary(
                    name=candidate.name,
                    source=candidate.source,
                    actual_worst_case_loss=candidate.actual_worst_case_loss,
                    budget_charge=candidate.budget_charge,
                    expected_improvement=candidate.expected_improvement,
                    improvement_support=candidate.improvement_support,
                    controller_weight=weight,
                    admitted=allowed,
                    reason=reason,
                )
            )
            if allowed:
                admitted.append(index)
                scores.append(
                    candidate.expected_improvement
                    + self.config.controller_prior_scale
                    * math.log(max(weight, self.config.log_probability_floor))
                )
        if not admitted or 0 not in admitted:
            raise RuntimeError("PM Hal exact candidate must always remain admitted")
        selected = admitted[int(np.argmax(np.asarray(scores, dtype=np.float64)))]
        return selected, tuple(summaries)

    def policy(self, decision: CanonicalDecision) -> Mapping[int, float]:
        self._validate_decision(decision)
        if self._pending is not None:
            raise RuntimeError("PM Hal was asked to act twice before a reveal")
        if not self._game_started:
            self.reset_game()
        if self._current_actor_name is None:
            self._current_actor_name = decision.actor_name
        elif self._current_actor_name.casefold() != decision.actor_name.casefold():
            raise RuntimeError("PM Hal cannot switch player identity mid-game")
        role = _role(decision.role)
        opponent_role: Role = "checker" if role == "dropper" else "dropper"
        stage = self.agent.stage_game(project_to_dth_state(decision))
        _validate_stage(stage)
        aggro = self._aggro_inference(decision, stage)
        equilibrium_opponent = (
            np.asarray(stage.check_policy, dtype=np.float64)
            if role == "dropper"
            else np.asarray(stage.drop_policy, dtype=np.float64)
        )
        forecast = self.opponent_model.predict(
            opponent_role,
            decision=decision,
            equilibrium_policy=equilibrium_opponent,
            game_index=self._game_epoch,
            game_decision_index=self._game_decision_index,
            aggro_policy=None if aggro is None else aggro.opponent_policy,
        )
        candidates = self._candidates(stage, role, forecast, aggro)
        mode = self._mode(forecast)
        selected_index, summaries = self._select(
            candidates, opponent_role=opponent_role, mode=mode
        )
        selected = candidates[selected_index]
        self._epsilon_spent += selected.budget_charge
        if self._epsilon_spent > self.config.game_epsilon_budget + _NUMERICAL_TOLERANCE:
            raise RuntimeError("PM Hal exceeded its per-game epsilon budget")
        policy = _distribution(selected.policy, label="selected PM policy")
        diagnostic = PMHalDecision(
            state=stage.state,
            game_index=self._game_epoch,
            role=role,
            opponent_role=opponent_role,
            mode=mode,
            policy=tuple(float(value) for value in policy),
            opponent_policy=tuple(float(value) for value in forecast.policy),
            component_weights=tuple(
                (name, float(weight))
                for name, weight in zip(
                    forecast.component_names,
                    forecast.component_weights,
                    strict=True,
                )
            ),
            component_policies=tuple(
                (name, tuple(float(value) for value in component_policy))
                for name, component_policy in zip(
                    forecast.component_names,
                    forecast.component_policies,
                    strict=True,
                )
            ),
            evidence_count=forecast.observation_index,
            effective_evidence=forecast.effective_evidence,
            forecast_entropy=forecast.entropy,
            forecast_disagreement=forecast.disagreement,
            change_probability=forecast.change_probability,
            expected_run_length=forecast.expected_run_length,
            prequential_skill=forecast.prequential_skill,
            confidence=forecast.confidence,
            selected_candidate=selected.name,
            selected_source=selected.source,
            selected_actual_worst_case_loss=selected.actual_worst_case_loss,
            selected_budget_charge=selected.budget_charge,
            game_epsilon_spent=self._epsilon_spent,
            game_epsilon_remaining=max(
                0.0, self.config.game_epsilon_budget - self._epsilon_spent
            ),
            expected_improvement=selected.expected_improvement,
            improvement_support=selected.improvement_support,
            aggro_enabled=aggro is not None,
            aggro_direct_weight=None if aggro is None else aggro.direct_weight,
            candidates=summaries,
        )
        self.decisions.append(diagnostic)
        self._pending = _PendingDecision(
            self_name=decision.actor_name,
            self_role=role,
            forecast=forecast,
            stage=stage,
            candidates=candidates,
            selected_index=selected_index,
        )
        self._game_decision_index += 1
        return {
            action: float(policy[action - 1])
            for action in ACTIONS
            if policy[action - 1] > 0.0
        }

    def observe(self, record: PublicHalfRound) -> None:
        pending = self._pending
        if pending is None:
            raise RuntimeError("PM Hal received a reveal without a pending action")
        if record.game_index < 0 or record.half_round_index < 0:
            raise ValueError("public reveal indices must be nonnegative")
        key = (self._game_epoch, record.game_index, record.half_round_index)
        if key in self._seen_reveals:
            raise RuntimeError("public reveal was delivered more than once")
        self_name = pending.self_name.casefold()
        if record.dropper_name.casefold() == self_name:
            role: Role = "dropper"
            self_action = _literal_action(record.drop_time, label="self action")
            opponent_action = _literal_action(
                record.check_time, label="opponent action"
            )
        elif record.checker_name.casefold() == self_name:
            role = "checker"
            self_action = _literal_action(record.check_time, label="self action")
            opponent_action = _literal_action(record.drop_time, label="opponent action")
        else:
            raise ValueError("public reveal does not include the PM Hal player")
        if role != pending.self_role:
            raise RuntimeError("public reveal role disagrees with the PM decision")
        matrix = np.asarray(pending.stage.matrix, dtype=np.float64)
        opponent_index = opponent_action - 1
        rewards = []
        for candidate in pending.candidates:
            policy = np.asarray(candidate.policy, dtype=np.float64)
            reward = (
                float(policy @ matrix[:, opponent_index])
                if role == "dropper"
                else -float(matrix[opponent_index, :] @ policy)
            )
            rewards.append(reward)
        exact_reward = rewards[0]
        advantages = [reward - exact_reward for reward in rewards]
        self.opponent_model.update_controllers(
            pending.forecast.opponent_role,
            candidate_names=[candidate.name for candidate in pending.candidates],
            advantages=advantages,
        )
        realized_probability = max(
            float(pending.forecast.policy[opponent_index]),
            self.config.log_probability_floor,
        )
        target = np.zeros(ACTION_COUNT, dtype=np.float64)
        target[opponent_index] = 1.0
        selected_reward = rewards[pending.selected_index]
        self.opponent_model.observe(
            pending.forecast,
            opponent_action=opponent_action,
            self_action=self_action,
            outcome=record.outcome,
        )
        post_change = self.opponent_model.change.change_probability(
            pending.forecast.opponent_role
        )
        self.observation_metrics.append(
            PMHalObservation(
                role=role,
                opponent_role=pending.forecast.opponent_role,
                opponent_action=opponent_action,
                realized_nll=-math.log(realized_probability),
                brier_score=float(np.sum((pending.forecast.policy - target) ** 2)),
                selected_counterfactual_gain_over_exact=selected_reward - exact_reward,
                one_step_candidate_regret=max(rewards) - selected_reward,
                post_reveal_change_probability=post_change,
            )
        )
        if self.aggro_model is not None:
            if self._aggro_previous_reveal is not None:
                raise RuntimeError("Aggro reveal was not consumed before PM observe")
            self._aggro_previous_reveal = record
            self._aggro_previous_self_name = pending.self_name
        self._seen_reveals.add(key)
        self._pending = None

    def end_game(self, outcome: PublicGameOutcome) -> None:
        if self._pending is not None:
            raise RuntimeError("PM Hal game ended with an unrevealed action")
        if outcome.game_index < 0 or outcome.half_rounds < 0:
            raise ValueError("public game outcome indices must be nonnegative")
        self._last_outcome = outcome

    def match_summary(self) -> str:
        if not self.decisions:
            return "PM Hal: no moves played"
        modes = {mode: 0 for mode in ("shield", "probe", "press", "dominate")}
        for decision in self.decisions:
            modes[decision.mode] += 1
        mean_gain = (
            float(
                np.mean(
                    [
                        metric.selected_counterfactual_gain_over_exact
                        for metric in self.observation_metrics
                    ]
                )
            )
            if self.observation_metrics
            else 0.0
        )
        return (
            f"PM Hal: {len(self.decisions)} pure-DTH moves; "
            f"modes shield/probe/press/dominate="
            f"{modes['shield']}/{modes['probe']}/{modes['press']}/{modes['dominate']}; "
            f"current risk charge {self._epsilon_spent:.4f}/"
            f"{self.config.game_epsilon_budget:.4f}; "
            f"realized counterfactual gain over exact {mean_gain:+.4f}"
        )

    def experiment_diagnostics(self) -> dict[str, object]:
        latest = self.last_decision
        losses = np.asarray(
            [decision.selected_actual_worst_case_loss for decision in self.decisions],
            dtype=np.float64,
        )
        mode_counts = {
            mode: sum(decision.mode == mode for decision in self.decisions)
            for mode in ("shield", "probe", "press", "dominate")
        }
        return {
            "schema_version": PM_HAL_DIAGNOSTICS_SCHEMA,
            "model_schema": PM_HAL_SCHEMA,
            "pure_dth_only": True,
            "public_history_only": True,
            "aggro_enabled": self.aggro_model is not None,
            "fixed_share": True,
            "change_point_model": "categorical-dirichlet-multinomial-bocpd",
            "config": asdict(self.config),
            "decision_count": len(self.decisions),
            "observation_count": len(self.observation_metrics),
            "mode_counts": mode_counts,
            "max_selected_actual_worst_case_loss": (
                float(np.max(losses)) if len(losses) else None
            ),
            "observations_by_opponent_role": {
                role: self.opponent_model.observations(role) for role in ROLES
            },
            "latest": None
            if latest is None
            else {
                "mode": latest.mode,
                "confidence": latest.confidence,
                "change_probability": latest.change_probability,
                "forecast_disagreement": latest.forecast_disagreement,
                "selected_candidate": latest.selected_candidate,
                "selected_actual_worst_case_loss": latest.selected_actual_worst_case_loss,
                "game_epsilon_spent": latest.game_epsilon_spent,
                "component_weights": dict(latest.component_weights),
            },
        }


def make_live_provider(
    *,
    artifact_dir: str | Path,
    config: PMHalConfig = PMHalConfig(),
    aggro_checkpoint: str | Path | None = None,
    aggro_config: AggroHalConfig | None = None,
    device: str | torch.device = "cpu",
    seed: int | None = None,
) -> PMHalPolicyProvider:
    """Construct checkpoint-free PM Hal or add a compatible Aggro expert."""

    if aggro_checkpoint is None:
        return PMHalPolicyProvider(
            artifact_dir,
            config,
            device=device,
            seed=seed,
        )
    return PMHalPolicyProvider.from_checkpoint(
        artifact_dir=artifact_dir,
        checkpoint=aggro_checkpoint,
        config=config,
        aggro_config=aggro_config,
        device=device,
        seed=seed,
    )


def load_pm_hal_config(
    path: str | Path = DEFAULT_PM_HAL_CONFIG,
) -> PMHalConfig:
    """Load the fully enumerated, frozen PM Hal controller configuration."""

    source = Path(path)
    raw = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or raw.get("schema_version") not in {
        PM_HAL_CONFIG_FILE_SCHEMA_V2,
        PM_HAL_CONFIG_FILE_SCHEMA,
    }:
        raise ValueError("unsupported PM Hal controller config schema")
    values = raw.get("config")
    if not isinstance(values, dict):
        raise ValueError("PM Hal controller config must contain a config object")
    resolved = dict(values)
    if raw.get("schema_version") == PM_HAL_CONFIG_FILE_SCHEMA_V2:
        try:
            resolved["minimum_improvement_support"] = resolved.pop(
                "minimum_improvement_probability"
            )
        except KeyError as error:
            raise ValueError(
                "legacy PM Hal controller config lacks improvement probability"
            ) from error
    expected = set(asdict(PMHalConfig()))
    if set(resolved) != expected:
        raise ValueError("PM Hal controller config fields are incompatible")
    perfect = resolved.get("perfect_config")
    if not isinstance(perfect, dict):
        raise ValueError("PM Hal controller perfect_config must be an object")
    resolved["perfect_config"] = PerfectHalConfig(
        **{
            **perfect,
            "recency_retentions": tuple(perfect["recency_retentions"]),
            "periodicities": tuple(perfect["periodicities"]),
        }
    )
    resolved["epsilon_grid"] = tuple(resolved["epsilon_grid"])
    return PMHalConfig(**resolved)


__all__ = [
    "ACTION_COUNT",
    "ACTIONS",
    "DEFAULT_PM_HAL_CONFIG",
    "PM_HAL_CONFIG_FILE_SCHEMA",
    "PM_HAL_CONFIG_FILE_SCHEMA_V2",
    "PM_HAL_SCHEMA",
    "CategoricalChangePointModel",
    "ChangePointForecast",
    "PMHalCandidate",
    "PMHalConfig",
    "PMHalDecision",
    "PMHalForecast",
    "PMHalObservation",
    "PMHalOpponentModel",
    "PMHalPolicyProvider",
    "load_pm_hal_config",
    "make_live_provider",
]
