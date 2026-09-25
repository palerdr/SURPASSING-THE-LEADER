"""Shared fakes for the tests of the provider contract and the play session.

Tests in more than one project import these fakes. No runtime module imports
this one.

- ``StageAgent`` stands in for ``CompleteDTHAgent.stage_game`` with a fixed
  certified stage, and ``certified_stage`` builds that stage.
- ``pure_policy`` puts all mass on one of the 60 actions, and ``drop_stage``
  builds the certified stage that the Exploit Hal tests share.
- ``make_decision`` builds one canonical decision, and ``make_reveal`` builds
  one public half-round reveal. ``make_forecast`` asks an opponent model for
  one forecast.
- ``RecordingHal`` is a Hal stand-in that records each consultation, and
  ``make_session`` builds a ``PlaySession`` around it.
"""

from __future__ import annotations

import random
from typing import TYPE_CHECKING

import numpy as np

from arena.contracts import (
    CanonicalDecision,
    PublicDecisionState,
    PublicHalfRound,
    PublicPlayerState,
)
from arena.policies.perfect_hal import ACTION_COUNT
from arena.session import PlaySession
from dth.agent import CertifiedStageGame
from stl.engine.game import (
    OPENING_START_CLOCK,
    PHYSICALITY_BAKU,
    PHYSICALITY_HAL,
    Game,
    Player,
    Referee,
)

if TYPE_CHECKING:
    from arena.policies.perfect_hal import PerfectHalOpponentModel


def certified_stage(matrix: np.ndarray | None = None) -> CertifiedStageGame:
    resolved = (
        np.eye(ACTION_COUNT, dtype=np.float64)
        if matrix is None
        else np.asarray(matrix, dtype=np.float64)
    )
    uniform = np.full(ACTION_COUNT, 1.0 / ACTION_COUNT, dtype=np.float64)
    return CertifiedStageGame(
        state=(0, 60, 0, 60),
        value=1.0 / ACTION_COUNT,
        matrix=resolved,
        drop_policy=uniform.copy(),
        check_policy=uniform.copy(),
        saddle_gap=0.0,
    )


class StageAgent:
    def __init__(self, matrix: np.ndarray | None = None) -> None:
        self.matrix = matrix
        self.states: list[tuple[int, int, int, int]] = []

    def stage_game(self, state) -> CertifiedStageGame:
        normalized = tuple(int(value) for value in state)
        self.states.append(normalized)
        stage = certified_stage(self.matrix)
        return CertifiedStageGame(
            state=normalized,
            value=stage.value,
            matrix=stage.matrix,
            drop_policy=stage.drop_policy,
            check_policy=stage.check_policy,
            saddle_gap=stage.saddle_gap,
        )


def pure_policy(index: int) -> np.ndarray:
    policy = np.zeros(60, dtype=np.float64)
    policy[index] = 1.0
    return policy


def drop_stage() -> CertifiedStageGame:
    matrix = np.zeros((60, 60), dtype=np.float64)
    matrix[0, 0] = 1.0
    matrix[0, 1:] = -1.0
    return CertifiedStageGame(
        state=(0, 0, 0, 0),
        value=0.0,
        matrix=matrix,
        drop_policy=pure_policy(1),
        check_policy=pure_policy(1),
        saddle_gap=0.0,
    )


def make_decision(
    *,
    role: str = "dropper",
    actor_name: str = "Hal",
    legal_seconds: tuple[int, ...] = tuple(range(1, 61)),
    checker_cylinder: float = 12.0,
    checker_ttd: float = 60.0,
    dropper_cylinder: float = 24.0,
    dropper_ttd: float = 120.0,
) -> CanonicalDecision:
    return CanonicalDecision(
        role=role,
        actor_name=actor_name,
        turn_duration=60,
        legal_seconds=legal_seconds,
        checker_cylinder_seconds=checker_cylinder,
        checker_ttd_seconds=checker_ttd,
        dropper_cylinder_seconds=dropper_cylinder,
        dropper_ttd_seconds=dropper_ttd,
        native_state=object(),
    )


def make_reveal(
    *,
    self_role: str = "dropper",
    self_name: str = "Hal",
    opponent_name: str = "Baku",
    self_action: int = 3,
    opponent_action: int = 7,
    game_index: int = 0,
    half_round_index: int = 0,
    game_over: bool = False,
) -> PublicHalfRound:
    if self_role == "dropper":
        dropper_name, checker_name = self_name, opponent_name
        drop_time, check_time = self_action, opponent_action
    elif self_role == "checker":
        dropper_name, checker_name = opponent_name, self_name
        drop_time, check_time = opponent_action, self_action
    else:
        raise ValueError("self_role must be dropper or checker")
    return PublicHalfRound(
        game_index=game_index,
        half_round_index=half_round_index,
        pre_decision_state=PublicDecisionState(
            game_clock_seconds=720.0,
            round_index=1,
            half_index=1,
            turn_duration=60,
            players=(
                PublicPlayerState(self_name, 24.0, 120.0),
                PublicPlayerState(opponent_name, 12.0, 60.0),
            ),
        ),
        dropper_name=dropper_name,
        checker_name=checker_name,
        drop_time=drop_time,
        check_time=check_time,
        outcome="check_success",
        game_over=game_over,
        winner_name=self_name if game_over else None,
    )


def make_forecast(
    model: PerfectHalOpponentModel,
    role: str,
    *,
    game_index: int = 0,
    decision_index: int = 0,
):
    return model.predict(
        role,
        state_regime=(0, 1, 0, 2),
        game_index=game_index,
        game_decision_index=decision_index,
    )


class RecordingHal:
    """Hal stand-in that records exactly when it is consulted."""

    def __init__(self, second: int = 30) -> None:
        self.second = second
        self.calls: list[str] = []
        self.provider = object()

    def choose_action(self, game, role, turn_duration):
        del game
        self.calls.append(role)
        return min(self.second, turn_duration)


def make_session(
    *,
    hal_agent=None,
    seed: int | None = 41,
    start_clock: int = OPENING_START_CLOCK,
    max_half_rounds: int | None = None,
    human_display_name: str = "Baku",
) -> PlaySession:
    hal = Player(name="Hal", physicality=PHYSICALITY_HAL)
    human = Player(name="Baku", physicality=PHYSICALITY_BAKU)
    game = Game(
        player1=hal,
        player2=human,
        referee=Referee(),
        rng=random.Random(seed),
    )
    game.game_clock = start_clock
    return PlaySession(
        game=game,
        hal_agent=hal_agent if hal_agent is not None else RecordingHal(),
        hal=hal,
        human=human,
        human_display_name=human_display_name,
        game_seed=seed,
        start_clock=start_clock,
        max_half_rounds=max_half_rounds,
    )
