"""The causal, session-persistent pure-DTH policy provider of Perfect Hal."""

from __future__ import annotations

from dataclasses import asdict, dataclass
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
from arena.policies.perfect_hal.decision import (
    PerfectHalDecision,
    _best_response_policy,
    _validate_stage,
)
from arena.policies.perfect_hal.model import (
    ACTION_COUNT,
    ACTIONS,
    PerfectHalConfig,
    PerfectHalForecast,
    PerfectHalOpponentModel,
    _literal_action,
)
from dth.agent import CompleteDTHAgent

PERFECT_HAL_DIAGNOSTICS_SCHEMA = "arena-perfect-hal-diagnostics-v1"


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
        if opponent_model is None:
            from arena.policies.bayesian_hal import BayesianHalOpponentModel

            opponent_model = BayesianHalOpponentModel(config)
        self.opponent_model = opponent_model
        if self.opponent_model.config != config:
            raise ValueError("Perfect Hal provider and opponent-model configs differ")
        self.decisions: list[PerfectHalDecision] = []
        self._pending: _PendingDecision | None = None
        self._seen_reveals: set[tuple[int, int, int]] = set()
        self._current_actor_name: str | None = None
        self._game_started = False
        self._game_epoch = -1
        self._game_decision_index = 0

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
            "model_schema": getattr(self.opponent_model, "schema", "arena-perfect-hal-online-adaptive-v1"),
            "opponent_model": self.opponent_model.diagnostics()
            if hasattr(self.opponent_model, "diagnostics") else None,
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
