"""Opponent posteriors and certified one-step DTH policy candidates.

This module deliberately keeps population research out of live play.  A caller
supplies one learned Dirichlet prior per opponent role; the provider updates
those posteriors from revealed canonical half-rounds and uses the complete DTH
tablebase to decide whether a one-step exploit is safe enough to play.

The tablebase remains the authority.  Posterior beliefs only choose a point
inside a certified worst-case polytope, and every positive local epsilon is
charged to a per-game budget.  Continuation play is valued by the complete
tablebase, so the interpretation is exactly "exploit now, equilibrium later".

Fitting population priors, archetype mixtures, and comparing state-conditioned
models are offline experiments and intentionally are not hidden in this live
provider.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Literal, Mapping, Protocol, Sequence

import numpy as np
from scipy.optimize import linprog

from arena.contracts import CanonicalDecision, PublicGameOutcome, PublicHalfRound
from arena.dth_adapter import project_to_dth_state
from dth.agent import (
    CERTIFIED_SADDLE_GAP_TOLERANCE,
    CertifiedStageGame,
    CompleteDTHAgent,
)
from stl.engine.actions import legal_seconds
from stl.engine.game import Game, HalfRoundRecord

Role = Literal["dropper", "checker"]

ACTION_COUNT = 60
_ACTIONS = tuple(range(1, ACTION_COUNT + 1))
_ROLES: tuple[Role, Role] = ("dropper", "checker")
_LP_FEASIBILITY_TOLERANCE = 1e-9
DEFAULT_EPSILON_GRID = (0.0, 0.0025, 0.005, 0.01, 0.02, 0.05, 0.10)


def _role(raw: str) -> Role:
    if raw not in _ROLES:
        raise ValueError(f"role must be 'dropper' or 'checker', got {raw!r}")
    return raw


def _distribution(raw, *, name: str) -> np.ndarray:
    values = np.asarray(raw, dtype=np.float64)
    if values.shape != (ACTION_COUNT,):
        raise ValueError(f"{name} must contain exactly {ACTION_COUNT} actions")
    if not np.all(np.isfinite(values)) or np.any(values < 0.0):
        raise ValueError(f"{name} must be finite and nonnegative")
    total = float(values.sum())
    if total <= 0.0:
        raise ValueError(f"{name} must have positive mass")
    return values / total


@dataclass(frozen=True, slots=True)
class DirichletPrior:
    """Population action mean and its effective pseudo-observation count."""

    mean: tuple[float, ...]
    strength: float

    def __post_init__(self) -> None:
        mean = _distribution(self.mean, name="prior mean")
        if np.any(mean <= 0.0):
            raise ValueError("every prior-mean action must be strictly positive")
        if not np.isfinite(self.strength) or self.strength <= 0.0:
            raise ValueError("prior strength must be finite and positive")
        object.__setattr__(self, "mean", tuple(float(value) for value in mean))
        object.__setattr__(self, "strength", float(self.strength))

    @classmethod
    def uniform(cls, *, strength: float = 1.0) -> DirichletPrior:
        """Construct an explicit neutral fallback for tests or cold starts."""

        return cls((1.0 / ACTION_COUNT,) * ACTION_COUNT, strength)

    @property
    def alpha(self) -> np.ndarray:
        return self.strength * np.asarray(self.mean, dtype=np.float64)


@dataclass(slots=True)
class RoleDirichletOpponent:
    """Independent exact Dirichlet posteriors for Dropper and Checker play.

    ``decay`` discounts only accumulated evidence, never the learned prior.
    Setting it below one makes deliberate strategy switches eventually displace
    stale observations; one is the ordinary static conjugate model.
    """

    drop_prior: DirichletPrior
    check_prior: DirichletPrior
    decay: float = 1.0
    _evidence: dict[Role, np.ndarray] = field(init=False, repr=False)
    _observations: dict[Role, int] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if not np.isfinite(self.decay) or not 0.0 < self.decay <= 1.0:
            raise ValueError("decay must lie in (0, 1]")
        self.decay = float(self.decay)
        self._evidence = {
            "dropper": np.zeros(ACTION_COUNT, dtype=np.float64),
            "checker": np.zeros(ACTION_COUNT, dtype=np.float64),
        }
        self._observations = {"dropper": 0, "checker": 0}

    @classmethod
    def uniform(
        cls, *, strength: float = 1.0, decay: float = 1.0
    ) -> RoleDirichletOpponent:
        prior = DirichletPrior.uniform(strength=strength)
        return cls(prior, prior, decay=decay)

    def _prior(self, role: Role) -> DirichletPrior:
        return self.drop_prior if role == "dropper" else self.check_prior

    def observe(self, role: str, action: int) -> None:
        """Apply one exact conjugate update for a revealed literal second."""

        normalized_role = _role(role)
        if isinstance(action, bool) or not isinstance(action, (int, np.integer)):
            raise ValueError("observed action must be a literal integer in 1..60")
        normalized_action = int(action)
        if not 1 <= normalized_action <= ACTION_COUNT:
            raise ValueError(
                f"observed {normalized_role} action must be in 1..60, got {action!r}"
            )
        evidence = self._evidence[normalized_role]
        evidence *= self.decay
        evidence[normalized_action - 1] += 1.0
        self._observations[normalized_role] += 1

    def alpha(self, role: str) -> np.ndarray:
        normalized_role = _role(role)
        return self._prior(normalized_role).alpha + self._evidence[normalized_role]

    def predictive(self, role: str) -> np.ndarray:
        alpha = self.alpha(role)
        return alpha / float(alpha.sum())

    def sample(
        self, role: str, *, size: int, rng: np.random.Generator
    ) -> np.ndarray:
        if isinstance(size, bool) or not isinstance(size, (int, np.integer)) or size <= 0:
            raise ValueError("posterior sample size must be a positive integer")
        return rng.dirichlet(self.alpha(role), size=int(size))

    def observation_count(self, role: str) -> int:
        return self._observations[_role(role)]

    def effective_concentration(self, role: str) -> float:
        return float(self.alpha(role).sum())


class OpponentModel(Protocol):
    """Behavior required by the exploitation selector and live provider."""

    def observe(self, role: str, action: int) -> None: ...

    def predictive(self, role: str) -> np.ndarray: ...

    def sample(
        self, role: str, *, size: int, rng: np.random.Generator
    ) -> np.ndarray: ...

    def observation_count(self, role: str) -> int: ...


@dataclass(slots=True)
class RoleMixtureOpponent:
    """Mixture of role-specific Dirichlet archetypes with online membership."""

    mixture_weights: tuple[float, ...]
    drop_priors: tuple[DirichletPrior, ...]
    check_priors: tuple[DirichletPrior, ...]
    decay: float = 1.0
    _weights: np.ndarray = field(init=False, repr=False)
    _evidence: dict[Role, np.ndarray] = field(init=False, repr=False)
    _observations: dict[Role, int] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        component_count = len(self.mixture_weights)
        if component_count < 2:
            raise ValueError("a role mixture requires at least two archetypes")
        if len(self.drop_priors) != component_count or len(self.check_priors) != component_count:
            raise ValueError("mixture weights and role priors must have equal length")
        weights = np.asarray(self.mixture_weights, dtype=np.float64)
        if not np.all(np.isfinite(weights)) or np.any(weights <= 0.0):
            raise ValueError("mixture weights must be finite and positive")
        if not np.isfinite(self.decay) or not 0.0 < self.decay <= 1.0:
            raise ValueError("decay must lie in (0, 1]")
        self._weights = weights / float(weights.sum())
        self.mixture_weights = tuple(float(value) for value in self._weights)
        self.decay = float(self.decay)
        self._evidence = {
            "dropper": np.zeros(ACTION_COUNT, dtype=np.float64),
            "checker": np.zeros(ACTION_COUNT, dtype=np.float64),
        }
        self._observations = {"dropper": 0, "checker": 0}

    @property
    def posterior_weights(self) -> tuple[float, ...]:
        return tuple(float(value) for value in self._weights)

    def _priors(self, role: Role) -> tuple[DirichletPrior, ...]:
        return self.drop_priors if role == "dropper" else self.check_priors

    def _component_alpha(self, role: Role) -> np.ndarray:
        return np.stack(
            [prior.alpha + self._evidence[role] for prior in self._priors(role)]
        )

    def observe(self, role: str, action: int) -> None:
        normalized_role = _role(role)
        if isinstance(action, bool) or not isinstance(action, (int, np.integer)):
            raise ValueError("observed action must be a literal integer in 1..60")
        normalized_action = int(action)
        if not 1 <= normalized_action <= ACTION_COUNT:
            raise ValueError(
                f"observed {normalized_role} action must be in 1..60, got {action!r}"
            )
        self._evidence[normalized_role] *= self.decay
        self._weights = np.power(self._weights, self.decay)
        component_alpha = self._component_alpha(normalized_role)
        component_predictive = component_alpha / component_alpha.sum(
            axis=1, keepdims=True
        )
        self._weights *= component_predictive[:, normalized_action - 1]
        self._weights /= float(self._weights.sum())
        self._evidence[normalized_role][normalized_action - 1] += 1.0
        self._observations[normalized_role] += 1

    def predictive(self, role: str) -> np.ndarray:
        normalized_role = _role(role)
        component_alpha = self._component_alpha(normalized_role)
        component_predictive = component_alpha / component_alpha.sum(
            axis=1, keepdims=True
        )
        return self._weights @ component_predictive

    def sample(
        self, role: str, *, size: int, rng: np.random.Generator
    ) -> np.ndarray:
        normalized_role = _role(role)
        if isinstance(size, bool) or not isinstance(size, (int, np.integer)) or size <= 0:
            raise ValueError("posterior sample size must be a positive integer")
        components = rng.choice(len(self._weights), size=int(size), p=self._weights)
        alphas = self._component_alpha(normalized_role)
        draws = np.empty((int(size), ACTION_COUNT), dtype=np.float64)
        for component in range(len(self._weights)):
            selected = np.flatnonzero(components == component)
            if selected.size:
                draws[selected] = rng.dirichlet(alphas[component], size=selected.size)
        return draws

    def observation_count(self, role: str) -> int:
        return self._observations[_role(role)]

    def effective_concentration(self, role: str) -> float:
        normalized_role = _role(role)
        component_alpha = self._component_alpha(normalized_role)
        return float(self._weights @ component_alpha.sum(axis=1))


def load_opponent_model(
    prior_path: str | Path | None,
    *,
    default_strength: float = 1.0,
    decay: float = 1.0,
) -> OpponentModel:
    """Construct a fresh session posterior from a versioned population prior."""

    if prior_path is None:
        return RoleDirichletOpponent.uniform(
            strength=default_strength, decay=decay
        )
    source = Path(prior_path)
    payload = json.loads(source.read_text(encoding="utf-8"))

    def role_prior(raw: object, label: str) -> DirichletPrior:
        if not isinstance(raw, dict) or "mean" not in raw:
            raise ValueError(f"adaptive prior {source} is missing {label}.mean")
        return DirichletPrior(
            tuple(raw["mean"]),
            float(raw.get("strength", default_strength)),
        )

    schema = payload.get("schema_version")
    if schema == "adaptive-dth-role-prior-v1":
        return RoleDirichletOpponent(
            drop_prior=role_prior(payload.get("dropper"), "dropper"),
            check_prior=role_prior(payload.get("checker"), "checker"),
            decay=decay,
        )
    if schema == "adaptive-dth-role-mixture-prior-v1":
        components = payload.get("components")
        weights = payload.get("weights")
        if not isinstance(components, list) or not isinstance(weights, list):
            raise ValueError(
                f"adaptive mixture prior {source} requires weights and components"
            )
        if any(not isinstance(component, dict) for component in components):
            raise ValueError(
                f"adaptive mixture prior {source} components must be objects"
            )
        return RoleMixtureOpponent(
            mixture_weights=tuple(float(value) for value in weights),
            drop_priors=tuple(
                role_prior(component.get("dropper"), f"components[{index}].dropper")
                for index, component in enumerate(components)
            ),
            check_priors=tuple(
                role_prior(component.get("checker"), f"components[{index}].checker")
                for index, component in enumerate(components)
            ),
            decay=decay,
        )
    raise ValueError(f"adaptive prior {source} has unsupported schema {schema!r}")


@dataclass(frozen=True, slots=True)
class ExploitationConfig:
    """Predeclared posterior and match-safety gates.

    The safe default has no positive epsilon budget.  It can still exploit
    freedom inside the exact minimax polytope when posterior evidence clears
    the confidence gate.
    """

    epsilon_grid: tuple[float, ...] = (0.0,)
    match_epsilon_budget: float = 0.0
    confidence: float = 0.95
    posterior_samples: int = 512
    improvement_tolerance: float = 1e-10

    def __post_init__(self) -> None:
        epsilons = tuple(sorted(set(float(value) for value in self.epsilon_grid)))
        if not epsilons or epsilons[0] != 0.0:
            raise ValueError("epsilon_grid must include 0.0")
        if any(not np.isfinite(value) or value < 0.0 for value in epsilons):
            raise ValueError("epsilon_grid values must be finite and nonnegative")
        if (
            not np.isfinite(self.match_epsilon_budget)
            or self.match_epsilon_budget < 0.0
        ):
            raise ValueError("match_epsilon_budget must be finite and nonnegative")
        if not np.isfinite(self.confidence) or not 0.0 < self.confidence <= 1.0:
            raise ValueError("confidence must lie in (0, 1]")
        if (
            isinstance(self.posterior_samples, bool)
            or not isinstance(self.posterior_samples, (int, np.integer))
            or self.posterior_samples <= 0
        ):
            raise ValueError("posterior_samples must be a positive integer")
        if (
            not np.isfinite(self.improvement_tolerance)
            or self.improvement_tolerance < 0.0
        ):
            raise ValueError("improvement_tolerance must be finite and nonnegative")
        object.__setattr__(self, "epsilon_grid", epsilons)
        object.__setattr__(
            self, "match_epsilon_budget", float(self.match_epsilon_budget)
        )
        object.__setattr__(self, "confidence", float(self.confidence))
        object.__setattr__(self, "posterior_samples", int(self.posterior_samples))
        object.__setattr__(
            self, "improvement_tolerance", float(self.improvement_tolerance)
        )


@dataclass(frozen=True, slots=True)
class CertifiedPolicyCandidate:
    """One independently checked member of the fixed certified policy family."""

    policy: tuple[float, ...]
    epsilon: float
    posterior_expected_payoff: float
    expected_improvement: float
    improvement_probability: float
    baseline_worst_case_value: float
    selected_worst_case_value: float
    actual_worst_case_loss: float
    valid: bool
    reason: str


@dataclass(frozen=True, slots=True)
class ControllerObservation:
    """Public context shared by hand-written and learned policy controllers."""

    decision: CanonicalDecision
    stage_game: CertifiedStageGame
    opponent: OpponentModel
    remaining_epsilon: float
    game_epsilon_budget: float
    epsilon_spent: float
    exploit_decisions: int
    game_index: int
    half_round_index: int
    action_spaces_supported: bool
    leap_action_legal: bool


class CandidateSource(Protocol):
    def candidates(
        self,
        stage_game: CertifiedStageGame,
        role: str,
        opponent: OpponentModel,
        remaining_epsilon: float,
        rng: np.random.Generator,
    ) -> Sequence[CertifiedPolicyCandidate]: ...


class PolicyController(Protocol):
    def choose(
        self,
        observation: ControllerObservation,
        candidates: Sequence[CertifiedPolicyCandidate],
        valid_mask: np.ndarray,
    ) -> int: ...


def _invalid_candidate(
    equilibrium: np.ndarray,
    *,
    epsilon: float,
    posterior_expected_payoff: float,
    baseline_worst: float,
    reason: str,
) -> CertifiedPolicyCandidate:
    return CertifiedPolicyCandidate(
        policy=tuple(float(value) for value in equilibrium),
        epsilon=float(epsilon),
        posterior_expected_payoff=float(posterior_expected_payoff),
        expected_improvement=0.0,
        improvement_probability=0.0,
        baseline_worst_case_value=float(baseline_worst),
        selected_worst_case_value=float(baseline_worst),
        actual_worst_case_loss=0.0,
        valid=False,
        reason=reason,
    )


@dataclass(frozen=True, slots=True)
class CertifiedCandidateGenerator:
    """Solve and independently validate the fixed epsilon candidate family."""

    epsilon_grid: tuple[float, ...] = DEFAULT_EPSILON_GRID
    posterior_samples: int = 512
    improvement_tolerance: float = 1e-10

    def __post_init__(self) -> None:
        validated = ExploitationConfig(
            epsilon_grid=self.epsilon_grid,
            posterior_samples=self.posterior_samples,
            improvement_tolerance=self.improvement_tolerance,
        )
        object.__setattr__(self, "epsilon_grid", validated.epsilon_grid)
        object.__setattr__(self, "posterior_samples", validated.posterior_samples)
        object.__setattr__(
            self, "improvement_tolerance", validated.improvement_tolerance
        )

    def candidates(
        self,
        stage_game: CertifiedStageGame,
        role: str,
        opponent: OpponentModel,
        remaining_epsilon: float,
        rng: np.random.Generator,
    ) -> tuple[CertifiedPolicyCandidate, ...]:
        normalized_role = _role(role)
        matrix = np.asarray(stage_game.matrix, dtype=np.float64)
        if matrix.shape != (ACTION_COUNT, ACTION_COUNT) or not np.all(
            np.isfinite(matrix)
        ):
            raise ValueError("stage matrix must be a finite 60x60 array")
        drop = _distribution(
            stage_game.drop_policy, name="equilibrium Dropper policy"
        )
        check = _distribution(
            stage_game.check_policy, name="equilibrium Checker policy"
        )
        lower = float(np.min(matrix.T @ drop))
        upper = float(np.max(matrix @ check))
        gap = max(0.0, upper - lower)
        if gap > CERTIFIED_SADDLE_GAP_TOLERANCE:
            raise RuntimeError(
                f"fresh tablebase policy pair misses the "
                f"{CERTIFIED_SADDLE_GAP_TOLERANCE:g} saddle-gap gate: {gap:g}"
            )
        equilibrium = drop if normalized_role == "dropper" else check
        baseline_worst = lower if normalized_role == "dropper" else upper
        opponent_role: Role = (
            "checker" if normalized_role == "dropper" else "dropper"
        )
        try:
            opponent_mean = _distribution(
                opponent.predictive(opponent_role),
                name=f"opponent {opponent_role} predictive distribution",
            )
            samples = np.asarray(
                opponent.sample(
                    opponent_role,
                    size=self.posterior_samples,
                    rng=rng,
                ),
                dtype=np.float64,
            )
            if (
                samples.shape != (self.posterior_samples, ACTION_COUNT)
                or not np.all(np.isfinite(samples))
                or np.any(samples < 0.0)
                or not np.allclose(samples.sum(axis=1), 1.0, atol=1e-8)
            ):
                raise ValueError("posterior samples are malformed")
        except (TypeError, ValueError, FloatingPointError) as error:
            exact = CertifiedPolicyCandidate(
                policy=tuple(float(value) for value in equilibrium),
                epsilon=0.0,
                posterior_expected_payoff=0.0,
                expected_improvement=0.0,
                improvement_probability=0.0,
                baseline_worst_case_value=baseline_worst,
                selected_worst_case_value=baseline_worst,
                actual_worst_case_loss=0.0,
                valid=True,
                reason=f"exact-equilibrium-fallback: {error}",
            )
            return (exact,) + tuple(
                _invalid_candidate(
                    equilibrium,
                    epsilon=epsilon,
                    posterior_expected_payoff=0.0,
                    baseline_worst=baseline_worst,
                    reason="malformed-opponent-model",
                )
                for epsilon in self.epsilon_grid
            )

        baseline_expected = (
            float(equilibrium @ matrix @ opponent_mean)
            if normalized_role == "dropper"
            else -float(opponent_mean @ matrix @ equilibrium)
        )
        result: list[CertifiedPolicyCandidate] = [
            CertifiedPolicyCandidate(
                policy=tuple(float(value) for value in equilibrium),
                epsilon=0.0,
                posterior_expected_payoff=baseline_expected,
                expected_improvement=0.0,
                improvement_probability=0.0,
                baseline_worst_case_value=baseline_worst,
                selected_worst_case_value=baseline_worst,
                actual_worst_case_loss=0.0,
                valid=True,
                reason="exact-equilibrium",
            )
        ]
        for epsilon in self.epsilon_grid:
            if epsilon > remaining_epsilon + _LP_FEASIBILITY_TOLERANCE:
                result.append(
                    _invalid_candidate(
                        equilibrium,
                        epsilon=epsilon,
                        posterior_expected_payoff=baseline_expected,
                        baseline_worst=baseline_worst,
                        reason="epsilon-exceeds-remaining-budget",
                    )
                )
                continue
            solved = _constrained_policy(
                matrix,
                role=normalized_role,
                opponent_mean=opponent_mean,
                baseline_worst=baseline_worst,
                epsilon=epsilon,
            )
            if solved is None:
                result.append(
                    _invalid_candidate(
                        equilibrium,
                        epsilon=epsilon,
                        posterior_expected_payoff=baseline_expected,
                        baseline_worst=baseline_worst,
                        reason="lp-infeasible-or-uncertified",
                    )
                )
                continue
            policy, selected_worst = solved
            try:
                policy = _distribution(policy, name="candidate policy")
            except ValueError:
                result.append(
                    _invalid_candidate(
                        equilibrium,
                        epsilon=epsilon,
                        posterior_expected_payoff=baseline_expected,
                        baseline_worst=baseline_worst,
                        reason="malformed-lp-policy",
                    )
                )
                continue
            independently_selected_worst = (
                float(np.min(matrix.T @ policy))
                if normalized_role == "dropper"
                else float(np.max(matrix @ policy))
            )
            actual_loss = (
                baseline_worst - independently_selected_worst
                if normalized_role == "dropper"
                else independently_selected_worst - baseline_worst
            )
            if (
                not np.isfinite(independently_selected_worst)
                or abs(independently_selected_worst - selected_worst) > 1e-8
                or actual_loss > epsilon + _LP_FEASIBILITY_TOLERANCE
            ):
                result.append(
                    _invalid_candidate(
                        equilibrium,
                        epsilon=epsilon,
                        posterior_expected_payoff=baseline_expected,
                        baseline_worst=baseline_worst,
                        reason="independent-safety-check-failed",
                    )
                )
                continue
            if normalized_role == "dropper":
                improvement = samples @ (matrix.T @ (policy - equilibrium))
                expected_payoff = float(policy @ matrix @ opponent_mean)
            else:
                improvement = samples @ (matrix @ (equilibrium - policy))
                expected_payoff = -float(opponent_mean @ matrix @ policy)
            result.append(
                CertifiedPolicyCandidate(
                    policy=tuple(float(value) for value in policy),
                    epsilon=float(epsilon),
                    posterior_expected_payoff=expected_payoff,
                    expected_improvement=float(np.mean(improvement)),
                    improvement_probability=float(
                        np.mean(improvement > self.improvement_tolerance)
                    ),
                    baseline_worst_case_value=baseline_worst,
                    selected_worst_case_value=independently_selected_worst,
                    actual_worst_case_loss=max(0.0, float(actual_loss)),
                    valid=True,
                    reason="certified-opponent-directed-response",
                )
            )
        if len(result) != 1 + len(self.epsilon_grid):
            raise RuntimeError("candidate family changed width")
        return tuple(result)


@dataclass(frozen=True, slots=True)
class EvidenceGatedController:
    """The existing posterior-confidence rule over the shared candidate family."""

    confidence: float = 0.95
    improvement_tolerance: float = 1e-10

    def choose(
        self,
        observation: ControllerObservation,
        candidates: Sequence[CertifiedPolicyCandidate],
        valid_mask: np.ndarray,
    ) -> int:
        del observation
        if not len(candidates) or not bool(valid_mask[0]):
            raise RuntimeError("exact equilibrium candidate must remain valid")
        selected = 0
        for index, candidate in enumerate(candidates[1:], start=1):
            if (
                bool(valid_mask[index])
                and candidate.improvement_probability >= self.confidence
                and candidate.expected_improvement > self.improvement_tolerance
            ):
                selected = index
        return selected


@dataclass(frozen=True, slots=True)
class PolicySelection:
    """One selected policy and the evidence/safety facts behind it."""

    policy: tuple[float, ...]
    epsilon: float
    improvement_probability: float
    expected_improvement: float
    baseline_worst_case_value: float
    selected_worst_case_value: float
    exploited: bool
    reason: str


def _equilibrium_selection(
    policy: np.ndarray,
    *,
    role: Role,
    matrix: np.ndarray,
    reason: str,
) -> PolicySelection:
    worst = (
        float(np.min(matrix.T @ policy))
        if role == "dropper"
        else float(np.max(matrix @ policy))
    )
    return PolicySelection(
        policy=tuple(float(value) for value in policy),
        epsilon=0.0,
        improvement_probability=0.0,
        expected_improvement=0.0,
        baseline_worst_case_value=worst,
        selected_worst_case_value=worst,
        exploited=False,
        reason=reason,
    )


def _constrained_policy(
    matrix: np.ndarray,
    *,
    role: Role,
    opponent_mean: np.ndarray,
    baseline_worst: float,
    epsilon: float,
) -> tuple[np.ndarray, float] | None:
    if role == "dropper":
        objective = -(matrix @ opponent_mean)
        a_ub = -matrix.T
        b_ub = np.full(ACTION_COUNT, -(baseline_worst - epsilon))
    else:
        objective = opponent_mean @ matrix
        a_ub = matrix
        b_ub = np.full(ACTION_COUNT, baseline_worst + epsilon)
    solved = linprog(
        objective,
        A_ub=a_ub,
        b_ub=b_ub,
        A_eq=np.ones((1, ACTION_COUNT), dtype=np.float64),
        b_eq=np.ones(1, dtype=np.float64),
        bounds=(0.0, None),
        method="highs",
    )
    if not solved.success or solved.x is None:
        return None
    policy = np.maximum(np.asarray(solved.x, dtype=np.float64), 0.0)
    total = float(policy.sum())
    if not np.isfinite(total) or total <= 0.0:
        return None
    policy /= total
    selected_worst = (
        float(np.min(matrix.T @ policy))
        if role == "dropper"
        else float(np.max(matrix @ policy))
    )
    actual_loss = (
        baseline_worst - selected_worst
        if role == "dropper"
        else selected_worst - baseline_worst
    )
    if actual_loss > epsilon + _LP_FEASIBILITY_TOLERANCE:
        return None
    return policy, selected_worst


def select_evidence_gated_policy(
    matrix,
    *,
    role: str,
    equilibrium_drop,
    equilibrium_check,
    opponent: OpponentModel,
    config: ExploitationConfig,
    remaining_epsilon: float,
    rng: np.random.Generator,
) -> PolicySelection:
    """Select the most aggressive posterior-supported certified policy.

    The baseline worst-case value is the relevant edge of the freshly checked
    tablebase certificate: a lower bound for Dropper and an upper bound for
    Checker.  This makes epsilon zero feasible without spending unreported
    numerical slack around the tablebase value midpoint.
    """

    normalized_role = _role(role)
    stage = np.asarray(matrix, dtype=np.float64)
    if stage.shape != (ACTION_COUNT, ACTION_COUNT) or not np.all(
        np.isfinite(stage)
    ):
        raise ValueError("stage matrix must be a finite 60x60 array")
    drop = _distribution(equilibrium_drop, name="equilibrium Dropper policy")
    check = _distribution(equilibrium_check, name="equilibrium Checker policy")
    lower = float(np.min(stage.T @ drop))
    upper = float(np.max(stage @ check))
    gap = max(0.0, upper - lower)
    if gap > CERTIFIED_SADDLE_GAP_TOLERANCE:
        raise RuntimeError(
            f"fresh tablebase policy pair misses the "
            f"{CERTIFIED_SADDLE_GAP_TOLERANCE:g} "
            f"saddle-gap gate: {gap:g}"
        )
    equilibrium = drop if normalized_role == "dropper" else check
    baseline_worst = lower if normalized_role == "dropper" else upper
    opponent_role: Role = (
        "checker" if normalized_role == "dropper" else "dropper"
    )
    samples = opponent.sample(
        opponent_role, size=config.posterior_samples, rng=rng
    )
    opponent_mean = opponent.predictive(opponent_role)
    fallback = _equilibrium_selection(
        equilibrium,
        role=normalized_role,
        matrix=stage,
        reason="insufficient-posterior-evidence",
    )
    selected: PolicySelection | None = None
    for epsilon in config.epsilon_grid:
        if epsilon > remaining_epsilon + _LP_FEASIBILITY_TOLERANCE:
            continue
        candidate = _constrained_policy(
            stage,
            role=normalized_role,
            opponent_mean=opponent_mean,
            baseline_worst=baseline_worst,
            epsilon=epsilon,
        )
        if candidate is None:
            continue
        policy, selected_worst = candidate
        if normalized_role == "dropper":
            improvement = samples @ (stage.T @ (policy - equilibrium))
        else:
            improvement = samples @ (stage @ (equilibrium - policy))
        probability = float(
            np.mean(improvement > config.improvement_tolerance)
        )
        expected = float(np.mean(improvement))
        if probability < config.confidence or expected <= config.improvement_tolerance:
            continue
        selected = PolicySelection(
            policy=tuple(float(value) for value in policy),
            epsilon=float(epsilon),
            improvement_probability=probability,
            expected_improvement=expected,
            baseline_worst_case_value=baseline_worst,
            selected_worst_case_value=selected_worst,
            exploited=True,
            reason="posterior-confidence-gate-passed",
        )
    return selected or fallback


def reconstruct_stage_matrix(
    agent: CompleteDTHAgent, state: tuple[int, int, int, int]
) -> np.ndarray:
    """Compatibility wrapper around the public certified DTH facade."""

    return np.asarray(agent.stage_game(state).matrix, dtype=np.float64)


@dataclass(frozen=True, slots=True)
class AdaptiveDecision:
    match_index: int
    state: tuple[int, int, int, int]
    role: Role
    tablebase_value: float
    saddle_gap: float
    selection: PolicySelection


@dataclass
class AdaptiveDTHPolicyProvider:
    """Evidence-gated controller over the shared certified candidate family."""

    artifact_dir: Path
    opponent: OpponentModel
    config: ExploitationConfig = field(default_factory=ExploitationConfig)
    seed: int | None = None
    decisions: list[AdaptiveDecision] = field(default_factory=list, repr=False)
    agent: CompleteDTHAgent | None = field(default=None, repr=False)
    _agent: CompleteDTHAgent = field(init=False, repr=False)
    _rng: np.random.Generator = field(init=False, repr=False)
    _candidate_source: CertifiedCandidateGenerator = field(init=False, repr=False)
    _controller: EvidenceGatedController = field(init=False, repr=False)
    _game: Game | None = field(default=None, init=False, repr=False)
    _self_name: str | None = field(default=None, init=False, repr=False)
    _seen_records: int = field(default=0, init=False, repr=False)
    _spent_epsilon: float = field(default=0.0, init=False, repr=False)
    _match_index: int = field(default=0, init=False, repr=False)
    _half_round_index: int = field(default=0, init=False, repr=False)
    _exploit_decisions: int = field(default=0, init=False, repr=False)
    _unsupported_observations: int = field(default=0, init=False, repr=False)
    _explicit_lifecycle: bool = field(default=False, init=False, repr=False)
    _game_started: bool = field(default=False, init=False, repr=False)

    def __post_init__(self) -> None:
        self.artifact_dir = Path(self.artifact_dir)
        self._agent = self.agent or CompleteDTHAgent(self.artifact_dir)
        self._rng = np.random.default_rng(self.seed)
        self._candidate_source = CertifiedCandidateGenerator(
            epsilon_grid=self.config.epsilon_grid,
            posterior_samples=self.config.posterior_samples,
            improvement_tolerance=self.config.improvement_tolerance,
        )
        self._controller = EvidenceGatedController(
            confidence=self.config.confidence,
            improvement_tolerance=self.config.improvement_tolerance,
        )

    @property
    def spent_epsilon(self) -> float:
        return self._spent_epsilon

    def close(self) -> None:
        """Match the arena provider lifecycle; tablebase memmaps need no close."""

    def reset_game(self) -> None:
        """Reset per-game safety while retaining this opponent's posterior."""

        if not self._explicit_lifecycle and self._game is not None and self._self_name:
            self._consume_history(self._game, self._self_name)
        if self._game_started:
            self._match_index += 1
        else:
            self._game_started = True
        self._explicit_lifecycle = True
        self._game = None
        self._self_name = None
        self._seen_records = 0
        self._spent_epsilon = 0.0
        self._half_round_index = 0
        self._exploit_decisions = 0

    def reset_match(self) -> None:
        """Backward-compatible alias for the explicit per-game lifecycle."""

        self.reset_game()

    def observe(self, record: PublicHalfRound) -> None:
        """Update the role-separated posterior from one public reveal."""

        self._explicit_lifecycle = True
        if self._self_name is None:
            raise RuntimeError("adaptive provider observed a round before acting")
        self._observe_record(record, self._self_name)
        self._half_round_index = max(
            self._half_round_index, record.half_round_index + 1
        )

    def end_game(self, outcome: PublicGameOutcome) -> None:
        """Complete an explicit game; the posterior intentionally persists."""

        if outcome.game_index < 0 or outcome.half_rounds < 0:
            raise ValueError("public game outcome indices must be nonnegative")

    @staticmethod
    def _record_values(record: HalfRoundRecord | PublicHalfRound) -> tuple[str, str, int, int]:
        if isinstance(record, PublicHalfRound):
            return (
                record.dropper_name,
                record.checker_name,
                record.drop_time,
                record.check_time,
            )
        return record.dropper, record.checker, record.drop_time, record.check_time

    def _observe_record(
        self, record: HalfRoundRecord | PublicHalfRound, self_name: str
    ) -> None:
        dropper, checker, drop_time, check_time = self._record_values(record)
        if dropper.casefold() != self_name.casefold():
            if 1 <= drop_time <= ACTION_COUNT:
                self.opponent.observe("dropper", drop_time)
            else:
                self._unsupported_observations += 1
        if checker.casefold() != self_name.casefold():
            if 1 <= check_time <= ACTION_COUNT:
                self.opponent.observe("checker", check_time)
            else:
                self._unsupported_observations += 1

    def _consume_history(self, game: Game, self_name: str) -> None:
        if self._explicit_lifecycle:
            return
        if len(game.history) < self._seen_records:
            raise RuntimeError("canonical game history was rewound during live adaptation")
        for record in game.history[self._seen_records :]:
            self._observe_record(record, self_name)
        self._seen_records = len(game.history)

    def _sync_observations(self, decision: CanonicalDecision) -> None:
        game = decision.native_state
        if not isinstance(game, Game):
            if self._self_name is None:
                self._self_name = decision.actor_name
            return
        if self._game is not game:
            if not self._explicit_lifecycle:
                if self._game is not None and self._self_name is not None:
                    self._consume_history(self._game, self._self_name)
                    self._match_index += 1
                elif not self._game_started:
                    self._game_started = True
                self._spent_epsilon = 0.0
                self._half_round_index = 0
                self._exploit_decisions = 0
            self._game = game
            self._self_name = decision.actor_name
            self._seen_records = 0
        elif (
            self._self_name is not None
            and decision.actor_name.casefold() != self._self_name.casefold()
        ):
            raise RuntimeError("one adaptive provider cannot control both players")
        self._consume_history(game, decision.actor_name)

    @staticmethod
    def _action_spaces_supported(decision: CanonicalDecision) -> bool:
        if decision.legal_seconds != _ACTIONS:
            return False
        opponent_role: Role = "checker" if decision.role == "dropper" else "dropper"
        game = decision.native_state
        if isinstance(game, Game):
            dropper, checker = game.get_roles_for_half(game.current_half)
            opponent_actor = checker if opponent_role == "checker" else dropper
            actions = legal_seconds(
                opponent_actor.name,
                opponent_role,
                decision.turn_duration,
            )
            return actions == _ACTIONS
        return opponent_role == "checker" or decision.turn_duration == ACTION_COUNT

    @staticmethod
    def _leap_action_legal(decision: CanonicalDecision) -> bool:
        if 61 in decision.legal_seconds:
            return True
        game = decision.native_state
        if not isinstance(game, Game):
            return decision.turn_duration > ACTION_COUNT
        dropper, checker = game.get_roles_for_half(game.current_half)
        opponent_role = "checker" if decision.role == "dropper" else "dropper"
        opponent_actor = checker if opponent_role == "checker" else dropper
        return 61 in legal_seconds(
            opponent_actor.name, opponent_role, decision.turn_duration
        )

    @staticmethod
    def _policy_mapping(policy: tuple[float, ...]) -> Mapping[int, float]:
        return {
            action: probability
            for action, probability in enumerate(policy, start=1)
            if probability > 0.0
        }

    def _certified_stage(self, decision: CanonicalDecision) -> CertifiedStageGame:
        state = project_to_dth_state(decision)
        stage_method = getattr(self._agent, "stage_game", None)
        if callable(stage_method):
            return stage_method(state)
        # Compatibility for existing lightweight test doubles only.
        move = self._agent.decide(state)
        matrix = reconstruct_stage_matrix(self._agent, move.state)
        return CertifiedStageGame(
            state=move.state,
            value=move.value,
            matrix=matrix,
            drop_policy=np.asarray(move.drop_policy, dtype=np.float64),
            check_policy=np.asarray(move.check_policy, dtype=np.float64),
            saddle_gap=move.saddle_gap,
        )

    def _controller_observation(
        self,
        decision: CanonicalDecision,
        stage: CertifiedStageGame,
        *,
        remaining: float,
        supported: bool,
    ) -> ControllerObservation:
        return ControllerObservation(
            decision=decision,
            stage_game=stage,
            opponent=self.opponent,
            remaining_epsilon=remaining,
            game_epsilon_budget=self.config.match_epsilon_budget,
            epsilon_spent=self._spent_epsilon,
            exploit_decisions=self._exploit_decisions,
            game_index=self._match_index,
            half_round_index=self._half_round_index,
            action_spaces_supported=supported,
            leap_action_legal=self._leap_action_legal(decision),
        )

    def policy(self, decision: CanonicalDecision) -> Mapping[int, float]:
        self._sync_observations(decision)
        stage = self._certified_stage(decision)
        remaining = max(0.0, self.config.match_epsilon_budget - self._spent_epsilon)
        candidates = self._candidate_source.candidates(
            stage,
            decision.role,
            self.opponent,
            remaining,
            self._rng,
        )
        supported = self._action_spaces_supported(decision)
        valid_mask = np.asarray(
            [candidate.valid for candidate in candidates], dtype=np.bool_
        )
        if not supported:
            valid_mask[1:] = False
        if not bool(valid_mask[0]):
            raise RuntimeError("exact equilibrium candidate was masked")
        observation = self._controller_observation(
            decision, stage, remaining=remaining, supported=supported
        )
        selected_index = self._controller.choose(
            observation, candidates, valid_mask
        )
        if not 0 <= selected_index < len(candidates) or not valid_mask[selected_index]:
            raise RuntimeError("policy controller selected an invalid candidate")
        candidate = candidates[selected_index]
        reason = (
            "opponent-action-space-outside-dth"
            if not supported
            else candidate.reason
        )
        selection = PolicySelection(
            policy=candidate.policy,
            epsilon=candidate.epsilon,
            improvement_probability=candidate.improvement_probability,
            expected_improvement=candidate.expected_improvement,
            baseline_worst_case_value=candidate.baseline_worst_case_value,
            selected_worst_case_value=candidate.selected_worst_case_value,
            exploited=selected_index != 0,
            reason=reason,
        )
        self._spent_epsilon += selection.epsilon
        if selection.exploited:
            self._exploit_decisions += 1
        if self._spent_epsilon > (
            self.config.match_epsilon_budget + _LP_FEASIBILITY_TOLERANCE
        ):
            raise RuntimeError("adaptive policy exceeded its match epsilon budget")
        self.decisions.append(
            AdaptiveDecision(
                match_index=self._match_index,
                state=stage.state,
                role=_role(decision.role),
                tablebase_value=stage.value,
                saddle_gap=stage.saddle_gap,
                selection=selection,
            )
        )
        return self._policy_mapping(selection.policy)

    def match_summary(self) -> str:
        if not self._explicit_lifecycle and self._game is not None and self._self_name:
            self._consume_history(self._game, self._self_name)
        exploited = sum(decision.selection.exploited for decision in self.decisions)
        return (
            f"adaptive dth: {exploited}/{len(self.decisions)} moves exploited; "
            f"current epsilon {self._spent_epsilon:.6g}/"
            f"{self.config.match_epsilon_budget:.6g}; observations "
            f"drop={self.opponent.observation_count('dropper')} "
            f"check={self.opponent.observation_count('checker')}; "
            f"unsupported={self._unsupported_observations}"
        )

    def diagnostics(self) -> Mapping[str, object]:
        return self.experiment_diagnostics()

    def experiment_diagnostics(self) -> dict[str, object]:
        """Return bounded policy diagnostics after a public play session."""

        if not self._explicit_lifecycle and self._game is not None and self._self_name:
            self._consume_history(self._game, self._self_name)
        grouped: dict[int, list[AdaptiveDecision]] = {}
        for decision in self.decisions:
            grouped.setdefault(decision.match_index, []).append(decision)
        games = []
        for match_index, decisions in sorted(grouped.items()):
            reasons = Counter(decision.selection.reason for decision in decisions)
            games.append(
                {
                    "match_index": match_index,
                    "decisions": len(decisions),
                    "exploited": sum(
                        decision.selection.exploited for decision in decisions
                    ),
                    "epsilon_spent": sum(
                        decision.selection.epsilon for decision in decisions
                    ),
                    "max_saddle_gap": max(
                        decision.saddle_gap for decision in decisions
                    ),
                    "reasons": dict(sorted(reasons.items())),
                }
            )
        diagnostics: dict[str, object] = {
            "schema_version": "adaptive-dth-session-diagnostics-v1",
            "games": games,
            "unsupported_observations": self._unsupported_observations,
            "dropper_observations": self.opponent.observation_count("dropper"),
            "checker_observations": self.opponent.observation_count("checker"),
        }
        posterior_weights = getattr(self.opponent, "posterior_weights", None)
        if posterior_weights is not None:
            diagnostics["posterior_archetype_weights"] = list(posterior_weights)
        return diagnostics
