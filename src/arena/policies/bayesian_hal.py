"""Bayesian filtering for an unknown, changing pure-DTH opponent.

We integrate Dirichlet predictions over run lengths and change hazards, then
filter a latent predictor identity with a fixed transition prior. We condition
categorical predictors on public history. Context shrinkage uses a prefix-only
empirical Bayes prior. The response provider owns the exact game matrix.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Mapping

import numpy as np

from arena.policies.perfect_hal import (
    ACTION_COUNT, PerfectHalConfig, PerfectHalContext, PerfectHalForecast,
    PerfectHalOpponentModel, _literal_action, _normalize, _readonly,
)

MODEL_SCHEMA = "arena-perfect-hal-bayesian-v2"
ROLES = ("dropper", "checker")


@dataclass(frozen=True)
class BayesianHalConfig:
    concentration: float = 0.5
    context_strength: float = 1.0
    switch_probability: float = 0.02
    hazards: tuple[float, ...] = (0.005, 0.05, 0.2)
    max_run_lengths: int = 64

    def __post_init__(self):
        for value in (self.concentration, self.context_strength):
            if not np.isfinite(value) or value <= 0:
                raise ValueError("Bayesian concentrations must be finite and positive")
        if not np.isfinite(self.switch_probability) or not 0 <= self.switch_probability < 1:
            raise ValueError("switch probability must be in [0, 1)")
        if not self.hazards or any(not np.isfinite(h) or not 0 <= h < 1 for h in self.hazards):
            raise ValueError("hazards must be in [0, 1)")
        if isinstance(self.max_run_lengths, bool) or not isinstance(self.max_run_lengths, int) or self.max_run_lengths < 2:
            raise ValueError("max_run_lengths must be an integer at least two")


class RunLengthFilter:
    """Filter changes before each observation, with a finite run-length beam.

    The reset branch uses the prior predictive; growth uses each run's
    predictive. Thus the posterior change probability can respond to surprise.
    We report discarded posterior mass when pruning makes inference approximate.
    """

    def __init__(self, prior: np.ndarray, config: BayesianHalConfig):
        self.prior = prior.copy()
        self.config = config
        self.counts = np.zeros((1, ACTION_COUNT))
        self.weights = np.ones((len(config.hazards), 1))
        self.hazard_weights = np.full(len(config.hazards), 1 / len(config.hazards))
        self.last_change_probability = 0.0
        self.discarded_mass = 0.0

    def _predictives(self):
        alpha = self.counts + self.config.concentration * self.prior
        return alpha / alpha.sum(axis=1, keepdims=True)

    def predict(self):
        h = np.asarray(self.config.hazards)[:, None]
        by_hazard = (1 - h) * (self.weights @ self._predictives()) + h * self.prior
        return self.hazard_weights @ by_hazard

    def observe(self, action_index: int):
        h = np.asarray(self.config.hazards)
        growth = self.weights * (1 - h[:, None]) * self._predictives()[:, action_index]
        reset = h * self.prior[action_index]
        joint = np.column_stack((reset, growth))
        evidence = joint.sum(axis=1)
        joint /= evidence[:, None]
        self.hazard_weights *= evidence
        self.hazard_weights /= self.hazard_weights.sum()
        self.last_change_probability = float(self.hazard_weights @ joint[:, 0])
        counts = np.vstack((np.zeros(ACTION_COUNT), self.counts))
        counts[:, action_index] += 1
        if counts.shape[0] > self.config.max_run_lengths:
            marginal = self.hazard_weights @ joint
            keep = np.argsort(-marginal, kind="stable")[:self.config.max_run_lengths]
            self.discarded_mass += float(1 - marginal[keep].sum())
            joint = joint[:, keep]
            joint /= joint.sum(axis=1, keepdims=True)
            counts = counts[keep]
        self.weights, self.counts = joint, counts


@dataclass
class _Role:
    run: RunLengthFilter
    weights: np.ndarray
    counts: dict[tuple, np.ndarray] = field(default_factory=dict)
    observations: int = 0
    previous_opponent_action: int | None = None
    previous_self_action: int | None = None


class BayesianHalOpponentModel:
    """Predict before each reveal and apply Bayes' rule after that reveal."""

    state_regime = staticmethod(PerfectHalOpponentModel.state_regime)
    schema = MODEL_SCHEMA

    def __init__(
        self, config: PerfectHalConfig = PerfectHalConfig(), *,
        bayes: BayesianHalConfig = BayesianHalConfig(),
        role_priors: Mapping[str, object] | None = None,
    ):
        self.config, self.bayes = config, bayes
        self.expert_names = ("population", "global", "changepoint", "repeat", "markov", "response", "state") + tuple(
            f"period_{p}" for p in config.periodicities
        )
        if role_priors is not None and set(role_priors) != set(ROLES):
            raise ValueError("population prior requires both roles")
        self.priors = {
            role: _normalize(np.ones(ACTION_COUNT) if role_priors is None else role_priors[role], label="population prior")
            for role in ROLES
        }
        if any(np.any(prior <= 0) for prior in self.priors.values()):
            raise ValueError("population priors must have positive support on all 60 actions")
        self.reset()

    def reset(self):
        self._roles = {
            role: _Role(RunLengthFilter(self.priors[role], self.bayes), np.full(len(self.expert_names), 1 / len(self.expert_names)))
            for role in ROLES
        }

    @property
    def total_observations(self):
        return sum(s.observations for s in self._roles.values())

    def observations(self, role):
        return self._roles[role].observations

    def _keys(self, context):
        return (
            ("global",), ("markov", context.previous_opponent_action),
            ("response", context.previous_self_action), ("state", *context.state_regime),
            *((f"period_{p}", context.observation_index % p) for p in self.config.periodicities),
        )

    @staticmethod
    def _categorical(state, key, prior, strength):
        counts = state.counts.get(key)
        return prior if counts is None else (counts + strength * prior) / (counts.sum() + strength)

    def predict(self, opponent_role, *, state_regime, game_index, game_decision_index):
        if opponent_role not in ROLES:
            raise ValueError("opponent role must be dropper or checker")
        if len(state_regime) != 4 or any(isinstance(x, bool) or not isinstance(x, int) or not 0 <= x <= 5 for x in state_regime):
            raise ValueError("state_regime must contain four bands in 0..5")
        if game_index < 0 or game_decision_index < 0:
            raise ValueError("game and decision indices must be nonnegative")
        state = self._roles[opponent_role]
        context = PerfectHalContext(opponent_role, state.observations, game_index, game_decision_index,
                                    tuple(state_regime), state.previous_opponent_action, state.previous_self_action)
        prior = self.priors[opponent_role]
        keys = self._keys(context)
        global_policy = self._categorical(state, keys[0], prior, self.bayes.concentration)
        repeat = prior.copy()
        if state.previous_opponent_action is not None:
            repeat *= 0.05
            repeat[state.previous_opponent_action - 1] += 0.95
        policies = np.stack((prior, global_policy, state.run.predict(), repeat, *(
            self._categorical(state, key, global_policy, self.bayes.context_strength) for key in keys[1:]
        )))
        weights = (1 - self.bayes.switch_probability) * state.weights + self.bayes.switch_probability / len(state.weights)
        policy = _normalize(weights @ policies, label="Bayesian predictive")
        entropy = -float(policy @ np.log(policy))
        # This concentration summary has no calibrated confidence interpretation.
        concentration = state.observations / (state.observations + self.bayes.concentration) * (1 - entropy / np.log(ACTION_COUNT))
        return PerfectHalForecast(context, policy, self.expert_names, _readonly(weights), _readonly(policies),
                                  entropy, float(np.clip(concentration, 0, 1)))

    def observe(self, forecast, *, opponent_action, self_action):
        action = _literal_action(opponent_action, label="opponent action")
        own = _literal_action(self_action, label="self action")
        state = self._roles[forecast.context.opponent_role]
        if forecast.context.observation_index != state.observations:
            raise RuntimeError("Bayesian forecast token is stale or already observed")
        if forecast.expert_names != self.expert_names:
            raise ValueError("Bayesian forecast token is incompatible")
        log_weights = np.log(forecast.expert_weights) + np.log(forecast.expert_policies[:, action - 1])
        state.weights = np.exp(log_weights - np.logaddexp.reduce(log_weights))
        for key in self._keys(forecast.context):
            state.counts.setdefault(key, np.zeros(ACTION_COUNT))[action - 1] += 1
        state.run.observe(action - 1)
        state.previous_opponent_action, state.previous_self_action = action, own
        state.observations += 1

    def diagnostics(self):
        return {
            "schema": self.schema, "bayes_config": asdict(self.bayes),
            "roles": {role: {
                "observations": state.observations,
                "posterior_hazard_weights": state.run.hazard_weights.tolist(),
                "last_change_probability": state.run.last_change_probability,
                "sum_discarded_posterior_mass": state.run.discarded_mass,
            } for role, state in self._roles.items()},
        }
