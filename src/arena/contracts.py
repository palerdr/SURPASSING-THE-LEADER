"""Algorithm-neutral policy interface for canonical live play."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Protocol, runtime_checkable


@dataclass(frozen=True, slots=True)
class CanonicalDecision:
    """Public canonical decision context passed to a policy provider.

    The numeric fields are literal engine seconds. ``native_state`` is an
    opaque escape hatch for a provider that natively understands the canonical
    engine (for example the STL MCTS agent); abstraction adapters ignore it.
    """

    role: str
    actor_name: str
    turn_duration: int
    legal_seconds: tuple[int, ...]
    checker_cylinder_seconds: float
    checker_ttd_seconds: float
    dropper_cylinder_seconds: float
    dropper_ttd_seconds: float
    native_state: object


@dataclass(frozen=True, slots=True)
class PublicPlayerState:
    """One player's public load state before a simultaneous decision."""

    name: str
    cylinder_seconds: float
    ttd_seconds: float


@dataclass(frozen=True, slots=True)
class PublicDecisionState:
    """Public state captured before either half-round action is requested."""

    game_clock_seconds: float
    round_index: int
    half_index: int
    turn_duration: int
    players: tuple[PublicPlayerState, PublicPlayerState]


@dataclass(frozen=True, slots=True)
class PublicHalfRound:
    """Revealed public result delivered once to every session provider."""

    game_index: int
    half_round_index: int
    pre_decision_state: PublicDecisionState
    dropper_name: str
    checker_name: str
    drop_time: int
    check_time: int
    outcome: str
    game_over: bool
    winner_name: str | None


@dataclass(frozen=True, slots=True)
class PublicGameOutcome:
    """End-of-game notification, including capped games without a winner."""

    game_index: int
    winner_name: str | None
    half_rounds: int


@runtime_checkable
class CanonicalPolicyProvider(Protocol):
    """Provide an unnormalized literal-second policy for one decision."""

    def policy(self, decision: CanonicalDecision) -> Mapping[int, float]: ...


def reset_provider_game(provider: object) -> None:
    """Start a fresh per-game lifecycle when a provider implements the hook."""

    hook = getattr(provider, "reset_game", None)
    if callable(hook):
        hook()


def observe_provider(provider: object, record: PublicHalfRound) -> None:
    """Deliver one revealed public half-round to an interested provider."""

    hook = getattr(provider, "observe", None)
    if callable(hook):
        hook(record)


def end_provider_game(provider: object, outcome: PublicGameOutcome) -> None:
    """Finish one game without requiring hooks on exact/stateless providers."""

    hook = getattr(provider, "end_game", None)
    if callable(hook):
        hook(outcome)
