"""Hal-derived opponents for pure-DTH candidate studies.

``hal_model_opponent`` plays the frozen translated Hal against a candidate.
``FullKnowledgeCounter`` runs a private copy of the candidate on the same
public history, reads the candidate's exact mixed policy at each decision,
and plays the best response to it in the exact stage matrix. It measures
what a one-step best response with complete knowledge of the candidate takes.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Callable, Mapping

import numpy as np

from arena.contracts import (
    CanonicalDecision,
    PublicGameOutcome,
    PublicHalfRound,
    end_provider_game,
    observe_provider,
    reset_provider_game,
)
from arena.dth_adapter import project_to_dth_state
from arena.policies.exploit_continuation import hal_view
from arena.policies.perfect_hal import PerfectHalPolicyProvider
from arena.policies.translated_hal import TranslatedHalOpponentModel
from arena.translated_hal_adapter import selected_config
from hal_lab.harness.telescope import policy_vector

ACTION_COUNT = 60
ACTIONS = tuple(range(1, ACTION_COUNT + 1))
_OTHER_SEAT = {"hal": "Baku", "baku": "Hal"}
_OTHER_ROLE = {"dropper": "checker", "checker": "dropper"}


def hal_model_opponent(artifact_dir, agent) -> PerfectHalPolicyProvider:
    """Return the selected translated Hal as a pure-DTH opponent."""

    config = selected_config()
    return PerfectHalPolicyProvider(
        artifact_dir,
        config,
        agent=agent,
        opponent_model=TranslatedHalOpponentModel(config),
    )


class FullKnowledgeCounter:
    """Best-respond to the exact mixed policy of a private candidate copy.

    ``candidate_factory`` builds a fresh candidate. The copy receives the
    candidate's decision and every public reveal, so a candidate whose policy
    depends on public history alone returns the same policy as the real one.
    ``candidate_policies`` keeps the copy's normalized policy at each decision.
    """

    def __init__(self, candidate_factory: Callable[[], object], agent) -> None:
        self.copy = candidate_factory()
        self.agent = agent
        self.candidate_policies: list[np.ndarray] = []
        self.tolerance = 1e-12

    @staticmethod
    def candidate_decision(decision: CanonicalDecision) -> CanonicalDecision:
        """Build the decision that the candidate faces in the same half-round."""

        if decision.role not in _OTHER_ROLE:
            raise ValueError("counter role must be dropper or checker")
        if (
            decision.turn_duration != ACTION_COUNT
            or tuple(decision.legal_seconds) != ACTIONS
        ):
            raise ValueError("FullKnowledgeCounter supports pure DTH only")
        other_name = _OTHER_SEAT.get(decision.actor_name.casefold())
        if other_name is None:
            raise ValueError(f"unknown seat name {decision.actor_name!r}")
        return replace(
            decision,
            role=_OTHER_ROLE[decision.role],
            actor_name=other_name,
            legal_seconds=ACTIONS,
            native_state=None,
        )

    def policy(self, decision: CanonicalDecision) -> Mapping[int, float]:
        own = self.candidate_decision(decision)
        pi = policy_vector(self.copy.policy(own), own)
        self.candidate_policies.append(pi)
        stage = self.agent.stage_game(project_to_dth_state(decision))
        matrix, _ = hal_view(stage.matrix, stage.value, own.role)
        values = pi @ np.asarray(matrix, dtype=np.float64)
        best = values <= float(np.min(values)) + self.tolerance
        weight = 1.0 / float(np.count_nonzero(best))
        return {action: weight for action in ACTIONS if best[action - 1]}

    def observe(self, record: PublicHalfRound) -> None:
        observe_provider(self.copy, record)

    def reset_game(self) -> None:
        reset_provider_game(self.copy)

    def end_game(self, outcome: PublicGameOutcome) -> None:
        end_provider_game(self.copy, outcome)

    def reset_session(self) -> None:
        hook = getattr(self.copy, "reset_session", None)
        if callable(hook):
            hook()
        self.candidate_policies.clear()


__all__ = ["FullKnowledgeCounter", "hal_model_opponent"]
