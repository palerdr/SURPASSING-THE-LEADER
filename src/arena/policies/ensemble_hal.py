"""Mix Old and Bayesian Hal with full-information, reveal-time rewards."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from arena.dth_adapter import project_to_dth_state
from arena.policies.bayesian_hal import BayesianHalOpponentModel
from arena.policies.perfect_hal import (
    PerfectHalConfig, PerfectHalOpponentModel, PerfectHalPolicyProvider,
    _literal_action, _validate_stage,
)
from dth.agent import CompleteDTHAgent


@dataclass(frozen=True)
class EnsembleHalConfig:
    learning_rate: float = 2.0
    share: float = 0.04

    def __post_init__(self):
        if not np.isfinite(self.learning_rate) or self.learning_rate < 0:
            raise ValueError("ensemble learning rate must be finite and nonnegative")
        if not np.isfinite(self.share) or not 0 < self.share <= 1:
            raise ValueError("ensemble share must be in (0, 1]")


def update_weights(weights, rewards, config):
    """Apply exponential rewards in [0, 1], then share mass with both models."""
    logits = np.log(weights) + config.learning_rate * rewards
    posterior = np.exp(logits - np.max(logits))
    posterior /= posterior.sum()
    return (1 - config.share) * posterior + config.share / 2


class _StageView:
    """Let both providers consume the same certified stage for one decision."""
    stage = None

    def stage_game(self, state):
        if self.stage is None or tuple(state) != tuple(self.stage.state):
            raise RuntimeError("ensemble stage does not match this decision")
        return self.stage


class EnsembleHalPolicyProvider:
    def __init__(self, artifact_dir: str | Path,
                 config: PerfectHalConfig = PerfectHalConfig(), *,
                 ensemble: EnsembleHalConfig = EnsembleHalConfig(), agent=None):
        self.agent = agent if agent is not None else CompleteDTHAgent(artifact_dir)
        self.config = ensemble
        self._view = _StageView()
        self.providers = (
            PerfectHalPolicyProvider(artifact_dir, config, agent=self._view,
                                     opponent_model=PerfectHalOpponentModel(config)),
            PerfectHalPolicyProvider(artifact_dir, config, agent=self._view,
                                     opponent_model=BayesianHalOpponentModel(config)),
        )
        self.reset_session()

    @property
    def has_session_memory(self):
        return self.providers[0].has_session_memory

    def _require_reveal(self):
        if self._pending is not None:
            raise RuntimeError("ensemble has an unrevealed action")

    def reset_session(self):
        if hasattr(self, "_pending"):
            self._require_reveal()
        for provider in self.providers:
            provider.reset_session()
        self.weights = {role: np.full(2, 0.5) for role in ("dropper", "checker")}
        self._pending = None
        self._actor = None
        self.latest = None
        self.decisions = 0
        self.reveals = 0
        self.disagreement_sum = 0.0

    def reset_game(self):
        self._require_reveal()
        for provider in self.providers:
            provider.reset_game()
        self._actor = None

    def policy(self, decision):
        PerfectHalPolicyProvider._validate_decision(decision)
        self._require_reveal()
        if self._actor is not None and self._actor != decision.actor_name.casefold():
            raise RuntimeError("ensemble cannot switch player identity mid-game")
        stage = self.agent.stage_game(project_to_dth_state(decision))
        _validate_stage(stage)
        matrix = np.array(stage.matrix if decision.role == "dropper" else -stage.matrix.T, copy=True)
        if np.any(np.abs(matrix) > 1 + 1e-6):
            raise ValueError("ensemble matrix utilities must be in [-1, 1]")
        self._view.stage = stage
        try:
            proposals = [provider.policy(decision) for provider in self.providers]
        finally:
            self._view.stage = None
        policies = np.array([[proposal.get(a, 0.0) for a in range(1, 61)] for proposal in proposals])
        weights = self.weights[decision.role].copy()
        mixture = weights @ policies
        matrix.setflags(write=False)
        policies.setflags(write=False)
        self._pending = (decision.actor_name.casefold(), decision.role, policies, matrix, weights)
        self._actor = decision.actor_name.casefold()
        self.decisions += 1
        disagreement = float(np.abs(policies[0] - policies[1]).sum() / 2)
        self.disagreement_sum += disagreement
        self.latest = {"role": decision.role, "weights_before": weights.tolist(),
                       "policies": policies.tolist(), "policy": mixture.tolist(),
                       "disagreement": disagreement, "revealed_rewards": None}
        return {a + 1: float(p) for a, p in enumerate(mixture) if p > 0}

    def observe(self, record):
        if self._pending is None:
            raise RuntimeError("ensemble received a reveal without a pending decision")
        actor, role, policies, matrix, weights = self._pending
        if record.game_index < 0 or record.half_round_index < 0:
            raise ValueError("public reveal indices must be nonnegative")
        dropper, checker = record.dropper_name.casefold(), record.checker_name.casefold()
        if dropper == checker or actor != (dropper if role == "dropper" else checker):
            raise ValueError("public reveal role or player disagrees with pending decision")
        drop = _literal_action(record.drop_time, label="drop action")
        check = _literal_action(record.check_time, label="check action")
        action = check if role == "dropper" else drop
        utilities = policies @ matrix[:, action - 1]
        rewards = np.clip((utilities + 1) / 2, 0, 1)
        updated = update_weights(weights, rewards, self.config)
        # Each child validates the same reveal and learns the same public history.
        for provider in self.providers:
            provider.observe(record)
        self.weights[role] = updated
        self.latest.update(revealed_rewards=rewards.tolist(), weights_after=updated.tolist())
        self.reveals += 1
        self._pending = None

    def end_game(self, outcome):
        self._require_reveal()
        for provider in self.providers:
            provider.end_game(outcome)

    def close(self):
        for provider in self.providers:
            provider.close()

    def match_summary(self):
        return f"Ensemble Hal: {self.reveals} reveals; Old/Bayesian weights {self.weights}"

    def experiment_diagnostics(self):
        return {"schema_version": "arena-perfect-hal-policy-ensemble-v1",
                "config": asdict(self.config), "experts": ["old", "bayesian"],
                "weights": {role: weights.tolist() for role, weights in self.weights.items()},
                "decisions": self.decisions, "reveals": self.reveals,
                "mean_policy_disagreement": self.disagreement_sum / max(1, self.decisions),
                "latest": self.latest, "pure_dth_only": True, "public_history_only": True}
