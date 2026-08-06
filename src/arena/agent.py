"""Canonical policy sampler shared by exact and approximate providers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

import numpy as np

from arena.contracts import (
    CanonicalDecision,
    CanonicalPolicyProvider,
    PublicDecisionState,
    PublicPlayerState,
)
from stl.engine.actions import legal_seconds
from stl.engine.game import Game


def decision_from_game(game: Game, *, role: str, turn_duration: int) -> CanonicalDecision:
    if role not in {"dropper", "checker"}:
        raise ValueError(f"role must be 'dropper' or 'checker', got {role!r}")
    dropper, checker = game.get_roles_for_half(game.current_half)
    actor = dropper if role == "dropper" else checker
    return CanonicalDecision(
        role=role,
        actor_name=actor.name,
        turn_duration=turn_duration,
        legal_seconds=legal_seconds(actor.name, role, turn_duration),
        checker_cylinder_seconds=checker.cylinder,
        checker_ttd_seconds=checker.ttd,
        dropper_cylinder_seconds=dropper.cylinder,
        dropper_ttd_seconds=dropper.ttd,
        native_state=game,
    )


def public_state_from_game(game: Game, *, turn_duration: int) -> PublicDecisionState:
    """Snapshot only the public state shared before simultaneous actions."""

    return PublicDecisionState(
        game_clock_seconds=float(game.game_clock),
        round_index=int(game.round_num),
        half_index=int(game.current_half),
        turn_duration=int(turn_duration),
        players=tuple(
            PublicPlayerState(
                name=player.name,
                cylinder_seconds=float(player.cylinder),
                ttd_seconds=float(player.ttd),
            )
            for player in (game.player1, game.player2)
        ),
    )


def normalize_legal_policy(
    raw_policy: Mapping[int, float],
    legal: tuple[int, ...],
) -> tuple[np.ndarray, np.ndarray]:
    """Validate then normalize a provider policy on canonical legal seconds."""

    if not raw_policy:
        raise ValueError("policy provider returned an empty policy")
    legal_set = set(legal)
    actions: list[int] = []
    weights: list[float] = []
    for action, weight in raw_policy.items():
        if isinstance(action, bool) or int(action) != action:
            raise ValueError(f"policy action must be an integer second, got {action!r}")
        if not np.isfinite(weight) or weight < 0.0:
            raise ValueError(f"policy weight for action {action!r} must be finite and nonnegative")
        if int(action) in legal_set and weight > 0.0:
            actions.append(int(action))
            weights.append(float(weight))
    if not actions:
        raise ValueError(f"policy provider assigned no positive mass to legal actions {legal}")
    probabilities = np.asarray(weights, dtype=np.float64)
    probabilities /= probabilities.sum()
    return np.asarray(actions, dtype=np.int64), probabilities


@dataclass(slots=True)
class PolicyDrivenAgent:
    """Algorithm-agnostic actor: provider policy in, valid action out."""

    provider: CanonicalPolicyProvider
    player_name: str = "Hal"
    seed: int | None = None
    _rng: np.random.Generator = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._rng = np.random.default_rng(self.seed)

    def choose_action(self, game: Game, role: str, turn_duration: int) -> int:
        decision = decision_from_game(game, role=role, turn_duration=turn_duration)
        if decision.actor_name.lower() != self.player_name.lower():
            raise ValueError(
                f"PolicyDrivenAgent({self.player_name}) asked to act for {decision.actor_name}"
            )
        actions, probabilities = normalize_legal_policy(
            self.provider.policy(decision), decision.legal_seconds
        )
        return int(self._rng.choice(actions, p=probabilities))
