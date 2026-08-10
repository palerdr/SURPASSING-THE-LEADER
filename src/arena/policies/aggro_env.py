"""Whole-session pure-DTH environment for direct-action policy learning.

The learning policy chooses a literal second in ``1..60``.  Canonical STL
``Game`` and ``Referee`` objects resolve every half-round, while the complete
DTH agent supplies the exact continuation-adjusted stage matrix and equilibrium
policy used as training context.  One environment episode is a repeated-
opponent session containing several canonical games.

This module is intentionally independent of Gymnasium.  ``reset``/``step`` are
small enough for a custom recurrent trainer, and ``rollout`` is a convenience
for deterministic data generation and tests.
"""

from __future__ import annotations

import random
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from arena.agent import (
    decision_from_game,
    normalize_legal_policy,
    public_state_from_game,
)
from arena.contracts import (
    CanonicalDecision,
    CanonicalPolicyProvider,
    PublicGameOutcome,
    PublicHalfRound,
    end_provider_game,
    observe_provider,
    reset_provider_game,
)
from arena.dth_adapter import PureDTHGame, project_to_dth_state
from arena.policies.aggro_hal import encode_public_observation
from dth.agent import CompleteDTHAgent
from stl.engine.actions import validate_action
from stl.engine.game import PHYSICALITY_BAKU, PHYSICALITY_HAL, Player, Referee


PURE_DTH_ACTIONS = tuple(range(1, 61))
ACTION_COUNT = len(PURE_DTH_ACTIONS)


def _readonly_array(
    values: object, *, dtype: np.dtype = np.dtype(np.float64)
) -> np.ndarray:
    result = np.asarray(values, dtype=dtype).copy()
    result.setflags(write=False)
    return result


def _validate_distribution(values: object, *, label: str) -> np.ndarray:
    distribution = np.asarray(values, dtype=np.float64)
    if (
        distribution.shape != (ACTION_COUNT,)
        or not np.all(np.isfinite(distribution))
        or np.any(distribution < 0.0)
        or float(distribution.sum()) <= 0.0
    ):
        raise ValueError(f"{label} must be a finite positive 60-action distribution")
    return distribution / float(distribution.sum())


def _mapping_distribution(values: Mapping[int, float], *, label: str) -> np.ndarray:
    distribution = np.zeros(ACTION_COUNT, dtype=np.float64)
    for raw_action, raw_weight in values.items():
        if isinstance(raw_action, bool) or int(raw_action) != raw_action:
            raise ValueError(f"{label} action must be an integer second")
        action = int(raw_action)
        weight = float(raw_weight)
        if not np.isfinite(weight) or weight < 0.0:
            raise ValueError(f"{label} weights must be finite and nonnegative")
        if 1 <= action <= ACTION_COUNT:
            distribution[action - 1] += weight
    return _validate_distribution(distribution, label=label)


@dataclass(frozen=True, slots=True)
class AggroDecision:
    """Public decision context and exact pure-DTH training information."""

    game_index: int
    half_round_index: int
    session_decision_index: int
    learner_seat: str
    opponent_seat: str
    role: str
    opponent_role: str
    canonical_decision: CanonicalDecision
    observation: np.ndarray
    dth_state: tuple[int, int, int, int]
    stage_matrix: np.ndarray
    role_oriented_matrix: np.ndarray
    exact_policy: np.ndarray
    exact_value: float
    saddle_gap: float


@dataclass(frozen=True, slots=True)
class AggroTrainingRecord:
    """One direct action, its public reveal, and exact training context."""

    game_index: int
    half_round_index: int
    session_decision_index: int
    learner_seat: str
    role: str
    reveal: PublicHalfRound
    selected_action: int
    opponent_action: int
    terminal_game_reward: float
    game_terminated: bool
    game_truncated: bool
    game_boundary: bool
    session_done: bool
    opponent_true_distribution: np.ndarray | None


@dataclass(frozen=True, slots=True)
class AggroStepResult:
    """Result returned by ``AggroSessionEnv.step``."""

    record: AggroTrainingRecord
    next_decision: AggroDecision | None

    @property
    def session_done(self) -> bool:
        return self.record.session_done


class AggroSessionEnv:
    """Run several canonical games as one direct-action learning episode.

    The same opponent provider instance is retained for the complete session.
    Its standard ``reset_game`` hook is called once before each game, every
    public reveal is delivered once through ``observe``, and ``end_game`` is
    called once at each terminal or half-round-capped boundary.
    """

    def __init__(
        self,
        opponent_provider: CanonicalPolicyProvider,
        exact_agent: CompleteDTHAgent,
        *,
        games_per_session: int,
        seed: int,
        start_clocks: Sequence[int] = (720,),
        max_half_rounds: int = 24,
        learner_starts_in_hal_seat: bool = True,
    ) -> None:
        if isinstance(games_per_session, bool) or games_per_session <= 0:
            raise ValueError("games_per_session must be positive")
        if isinstance(max_half_rounds, bool) or max_half_rounds <= 0:
            raise ValueError("max_half_rounds must be positive")
        clocks = tuple(int(value) for value in start_clocks)
        if not clocks or any(value < 0 for value in clocks):
            raise ValueError("start_clocks must contain nonnegative canonical clocks")
        if isinstance(seed, bool) or seed < 0:
            raise ValueError("seed must be a nonnegative integer")
        self.opponent_provider = opponent_provider
        self.exact_agent = exact_agent
        self.games_per_session = int(games_per_session)
        self.seed = int(seed)
        self.start_clocks = clocks
        self.max_half_rounds = int(max_half_rounds)
        self.learner_starts_in_hal_seat = bool(learner_starts_in_hal_seat)

        self._session_seed = self.seed
        self._game_index = 0
        self._session_decision_index = 0
        self._half_rounds = 0
        self._game: PureDTHGame | None = None
        self._game_active = False
        self._session_done = True
        self._pending: AggroDecision | None = None
        self._pending_opponent_truth: np.ndarray | None = None
        self._previous_reveal: PublicHalfRound | None = None
        self._previous_reveal_self_name: str | None = None
        self._opponent_rng: np.random.Generator | None = None

    @property
    def session_done(self) -> bool:
        return self._session_done

    @property
    def current_decision(self) -> AggroDecision | None:
        return self._pending

    def _learner_seat_for_game(self, game_index: int) -> str:
        starts_hal = self.learner_starts_in_hal_seat
        learner_is_hal = starts_hal if game_index % 2 == 0 else not starts_hal
        return "Hal" if learner_is_hal else "Baku"

    def _start_game(self) -> None:
        game_seed = self._session_seed + self._game_index
        self._game = PureDTHGame(
            player1=Player(name="Hal", physicality=PHYSICALITY_HAL),
            player2=Player(name="Baku", physicality=PHYSICALITY_BAKU),
            referee=Referee(),
            rng=random.Random(game_seed),
        )
        self._game.game_clock = self.start_clocks[
            self._game_index % len(self.start_clocks)
        ]
        self._opponent_rng = np.random.default_rng(game_seed * 2 + 2)
        self._half_rounds = 0
        self._game_active = True
        reset_provider_game(self.opponent_provider)

    def _finish_game(self) -> None:
        if not self._game_active or self._game is None:
            return
        winner_name = (
            self._game.winner.name
            if self._game.game_over and self._game.winner is not None
            else None
        )
        end_provider_game(
            self.opponent_provider,
            PublicGameOutcome(
                game_index=self._game_index,
                winner_name=winner_name,
                half_rounds=self._half_rounds,
            ),
        )
        self._game_active = False

    def _opponent_truth(self, decision: CanonicalDecision) -> np.ndarray | None:
        method = getattr(self.opponent_provider, "true_distribution", None)
        if not callable(method):
            return None
        raw = method(decision)
        if isinstance(raw, Mapping):
            truth = _mapping_distribution(raw, label="opponent truth")
        else:
            truth = _validate_distribution(raw, label="opponent truth")
        return _readonly_array(truth)

    def _prepare_decision(self) -> AggroDecision:
        if self._game is None or not self._game_active:
            raise RuntimeError("Aggro session has no active game")
        learner_seat = self._learner_seat_for_game(self._game_index)
        opponent_seat = "Baku" if learner_seat == "Hal" else "Hal"
        dropper, _ = self._game.get_roles_for_half(self._game.current_half)
        role = "dropper" if dropper.name == learner_seat else "checker"
        opponent_role = "checker" if role == "dropper" else "dropper"
        turn_duration = self._game.get_turn_duration()
        decision = decision_from_game(
            self._game, role=role, turn_duration=turn_duration
        )
        opponent_decision = decision_from_game(
            self._game, role=opponent_role, turn_duration=turn_duration
        )
        dth_state = project_to_dth_state(decision)
        stage = self.exact_agent.stage_game(dth_state)
        matrix = np.asarray(stage.matrix, dtype=np.float64)
        if matrix.shape != (ACTION_COUNT, ACTION_COUNT) or not np.all(
            np.isfinite(matrix)
        ):
            raise RuntimeError("exact DTH stage matrix must be finite 60x60")
        if role == "dropper":
            role_matrix = matrix
            exact_policy = np.asarray(stage.drop_policy, dtype=np.float64)
            exact_value = float(stage.value)
        else:
            role_matrix = -matrix.T
            exact_policy = np.asarray(stage.check_policy, dtype=np.float64)
            exact_value = -float(stage.value)
        exact_policy = _validate_distribution(exact_policy, label="exact DTH policy")
        observation = encode_public_observation(
            decision,
            stage,
            self._previous_reveal,
            previous_self_name=self._previous_reveal_self_name,
            new_game=self._half_rounds == 0,
        )
        prepared = AggroDecision(
            game_index=self._game_index,
            half_round_index=self._half_rounds,
            session_decision_index=self._session_decision_index,
            learner_seat=learner_seat,
            opponent_seat=opponent_seat,
            role=role,
            opponent_role=opponent_role,
            canonical_decision=decision,
            observation=_readonly_array(observation, dtype=np.dtype(np.float32)),
            dth_state=dth_state,
            stage_matrix=_readonly_array(matrix),
            role_oriented_matrix=_readonly_array(role_matrix),
            exact_policy=_readonly_array(exact_policy),
            exact_value=exact_value,
            saddle_gap=float(stage.saddle_gap),
        )
        self._pending = prepared
        self._pending_opponent_truth = self._opponent_truth(opponent_decision)
        return prepared

    def reset(self, *, seed: int | None = None) -> AggroDecision:
        """Start a new repeated-opponent session and return its first decision."""

        self.close()
        reset_session = getattr(self.opponent_provider, "reset_session", None)
        if callable(reset_session):
            reset_session()
        if seed is not None and (isinstance(seed, bool) or seed < 0):
            raise ValueError("reset seed must be a nonnegative integer")
        self._session_seed = self.seed if seed is None else int(seed)
        self._game_index = 0
        self._session_decision_index = 0
        self._previous_reveal = None
        self._previous_reveal_self_name = None
        self._session_done = False
        self._pending = None
        self._pending_opponent_truth = None
        self._start_game()
        return self._prepare_decision()

    @staticmethod
    def _validate_learning_action(action: int | np.integer) -> int:
        if isinstance(action, bool) or not isinstance(action, (int, np.integer)):
            raise ValueError("learning action must be an integer literal second")
        normalized = int(action)
        if normalized not in PURE_DTH_ACTIONS:
            raise ValueError("pure-DTH learning action must lie in 1..60")
        return normalized

    def _sample_opponent_action(self, decision: CanonicalDecision) -> int:
        if self._opponent_rng is None:
            raise RuntimeError("Aggro session opponent RNG is unavailable")
        raw_policy = self.opponent_provider.policy(decision)
        actions, probabilities = normalize_legal_policy(raw_policy, PURE_DTH_ACTIONS)
        return int(self._opponent_rng.choice(actions, p=probabilities))

    def step(self, action: int | np.integer) -> AggroStepResult:
        """Resolve one direct action and return its complete training record."""

        selected_action = self._validate_learning_action(action)
        prepared = self._pending
        opponent_truth = self._pending_opponent_truth
        game = self._game
        if (
            prepared is None
            or game is None
            or not self._game_active
            or self._session_done
        ):
            raise RuntimeError(
                "Aggro session step requested without an active decision"
            )
        turn_duration = game.get_turn_duration()
        public_state = public_state_from_game(game, turn_duration=turn_duration)
        opponent_decision = decision_from_game(
            game, role=prepared.opponent_role, turn_duration=turn_duration
        )
        opponent_action = self._sample_opponent_action(opponent_decision)

        if prepared.role == "dropper":
            drop, check = selected_action, opponent_action
        else:
            drop, check = opponent_action, selected_action
        dropper, checker = game.get_roles_for_half(game.current_half)
        validate_action(
            drop, actor=dropper.name, role="dropper", turn_duration=turn_duration
        )
        validate_action(
            check, actor=checker.name, role="checker", turn_duration=turn_duration
        )
        result = game.play_half_round(drop, check)
        public_record = PublicHalfRound(
            game_index=self._game_index,
            half_round_index=self._half_rounds,
            pre_decision_state=public_state,
            dropper_name=result.dropper,
            checker_name=result.checker,
            drop_time=int(result.drop_time),
            check_time=int(result.check_time),
            outcome=result.result.value,
            game_over=bool(game.game_over),
            winner_name=game.winner.name if game.winner is not None else None,
        )
        observe_provider(self.opponent_provider, public_record)
        self._previous_reveal = public_record
        self._previous_reveal_self_name = prepared.learner_seat
        self._half_rounds += 1
        self._session_decision_index += 1

        game_terminated = bool(game.game_over)
        game_truncated = (
            not game_terminated and self._half_rounds >= self.max_half_rounds
        )
        game_boundary = game_terminated or game_truncated
        terminal_reward = 0.0
        if game_terminated:
            winner = game.winner.name if game.winner is not None else None
            terminal_reward = 1.0 if winner == prepared.learner_seat else -1.0

        final_game = game_boundary and self._game_index + 1 >= self.games_per_session
        record = AggroTrainingRecord(
            game_index=prepared.game_index,
            half_round_index=prepared.half_round_index,
            session_decision_index=prepared.session_decision_index,
            learner_seat=prepared.learner_seat,
            role=prepared.role,
            reveal=public_record,
            selected_action=selected_action,
            opponent_action=opponent_action,
            terminal_game_reward=terminal_reward,
            game_terminated=game_terminated,
            game_truncated=game_truncated,
            game_boundary=game_boundary,
            session_done=final_game,
            opponent_true_distribution=opponent_truth,
        )

        self._pending = None
        self._pending_opponent_truth = None
        next_decision: AggroDecision | None
        if game_boundary:
            self._finish_game()
            if final_game:
                self._session_done = True
                next_decision = None
            else:
                self._game_index += 1
                self._start_game()
                next_decision = self._prepare_decision()
        else:
            next_decision = self._prepare_decision()
        return AggroStepResult(record=record, next_decision=next_decision)

    def rollout(
        self,
        policy: Callable[[AggroDecision], int | np.integer],
        *,
        seed: int | None = None,
    ) -> tuple[AggroTrainingRecord, ...]:
        """Run one complete session with a direct-action callback."""

        decision = self.reset(seed=seed)
        records: list[AggroTrainingRecord] = []
        while True:
            result = self.step(policy(decision))
            records.append(result.record)
            if result.session_done:
                return tuple(records)
            if result.next_decision is None:
                raise RuntimeError(
                    "active Aggro session failed to prepare its next decision"
                )
            decision = result.next_decision

    def close(self) -> None:
        """End an unfinished game once; the environment remains reusable."""

        self._pending = None
        self._pending_opponent_truth = None
        self._finish_game()
        self._session_done = True
