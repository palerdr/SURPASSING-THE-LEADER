from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pytest

from arena.contracts import CanonicalDecision, PublicGameOutcome, PublicHalfRound
from arena.policies.aggro_env import (
    AggroDecision,
    AggroSessionEnv,
)
from arena.policies.aggro_hal import OBSERVATION_DIM, OBSERVATION_FEATURES
from arena.policies.opponent_league import ReactiveDTHOpponent, STATE_THRESHOLD
from dth.agent import CertifiedStageGame


@dataclass
class _ExactStageStub:
    matrix: np.ndarray
    calls: list[tuple[int, int, int, int]] = field(default_factory=list)

    def stage_game(self, state: tuple[int, int, int, int]) -> CertifiedStageGame:
        self.calls.append(state)
        drop = np.zeros(60, dtype=np.float64)
        check = np.zeros(60, dtype=np.float64)
        drop[2] = 1.0
        check[3] = 1.0
        return CertifiedStageGame(
            state=state,
            value=0.25,
            matrix=self.matrix,
            drop_policy=drop,
            check_policy=check,
            saddle_gap=1e-8,
        )


@dataclass
class _RecordingOpponent:
    session_resets: int = 0
    resets: int = 0
    reveals: list[PublicHalfRound] = field(default_factory=list)
    outcomes: list[PublicGameOutcome] = field(default_factory=list)
    truth_decisions: list[CanonicalDecision] = field(default_factory=list)

    def reset_session(self) -> None:
        self.session_resets += 1

    def reset_game(self) -> None:
        self.resets += 1

    def policy(self, decision: CanonicalDecision) -> dict[int, float]:
        return {1: 1.0} if decision.role == "checker" else {60: 1.0}

    def true_distribution(self, decision: CanonicalDecision) -> np.ndarray:
        self.truth_decisions.append(decision)
        result = np.zeros(60, dtype=np.float64)
        result[0 if decision.role == "checker" else 59] = 1.0
        return result

    def observe(self, record: PublicHalfRound) -> None:
        self.reveals.append(record)

    def end_game(self, outcome: PublicGameOutcome) -> None:
        self.outcomes.append(outcome)


def _matrix() -> np.ndarray:
    return np.arange(60 * 60, dtype=np.float64).reshape(60, 60) / 3600.0


def test_whole_session_alternates_seats_and_preserves_provider_lifecycle() -> None:
    matrix = _matrix()
    exact = _ExactStageStub(matrix)
    opponent = _RecordingOpponent()
    env = AggroSessionEnv(
        opponent,
        exact,  # type: ignore[arg-type]
        games_per_session=2,
        seed=2,
        start_clocks=(720, 840),
        max_half_rounds=1,
    )

    first = env.reset()
    feature = {name: index for index, name in enumerate(OBSERVATION_FEATURES)}
    assert first.learner_seat == "Hal"
    assert first.role == "dropper"
    assert first.observation.shape == (OBSERVATION_DIM,)
    assert first.observation[feature["previous_reveal_present"]] == 0.0
    assert first.observation[feature["current_new_game"]] == 1.0
    assert not hasattr(first, "opponent_true_distribution")
    np.testing.assert_allclose(first.stage_matrix, matrix)
    np.testing.assert_allclose(first.role_oriented_matrix, matrix)
    assert first.exact_policy[2] == 1.0
    assert first.exact_value == 0.25
    assert opponent.truth_decisions[0].role == "checker"
    assert opponent.truth_decisions[0].actor_name == "Baku"
    assert opponent.truth_decisions[0].legal_seconds == tuple(range(1, 61))
    assert opponent.truth_decisions[0].turn_duration == 60
    assert opponent.session_resets == 1
    assert opponent.resets == 1

    first_step = env.step(60)
    first_record = first_step.record
    assert first_record.game_boundary
    assert first_record.game_terminated
    assert not first_record.game_truncated
    assert first_record.terminal_game_reward == 1.0
    assert first_record.selected_action == 60
    assert first_record.opponent_action == 1
    assert first_record.reveal.dropper_name == "Hal"
    assert first_record.opponent_true_distribution is not None
    assert first_record.opponent_true_distribution[0] == 1.0
    assert len(opponent.reveals) == 1
    assert len(opponent.outcomes) == 1

    second = first_step.next_decision
    assert second is not None
    assert second.game_index == 1
    assert second.learner_seat == "Baku"
    assert second.role == "checker"
    assert second.observation[feature["previous_reveal_present"]] == 1.0
    assert second.observation[feature["previous_dropper_is_self"]] == 1.0
    assert second.observation[feature["previous_checker_is_self"]] == 0.0
    assert second.observation[feature["current_new_game"]] == 1.0
    np.testing.assert_allclose(second.stage_matrix, matrix)
    np.testing.assert_allclose(second.role_oriented_matrix, -matrix.T)
    assert second.exact_policy[3] == 1.0
    assert second.exact_value == -0.25
    assert not hasattr(second, "opponent_true_distribution")
    assert opponent.truth_decisions[1].role == "dropper"
    assert opponent.truth_decisions[1].actor_name == "Hal"
    assert opponent.truth_decisions[1].checker_cylinder_seconds == (
        second.canonical_decision.checker_cylinder_seconds
    )
    assert opponent.resets == 2

    second_step = env.step(1)
    assert second_step.session_done
    assert second_step.next_decision is None
    assert second_step.record.game_boundary
    assert second_step.record.game_terminated or second_step.record.game_truncated
    assert second_step.record.opponent_true_distribution is not None
    assert second_step.record.opponent_true_distribution[59] == 1.0
    assert len(opponent.reveals) == 2
    assert len(opponent.outcomes) == 2
    assert [item.game_index for item in opponent.outcomes] == [0, 1]
    assert opponent.resets == 2
    assert len(exact.calls) == 2

    env.close()
    assert len(opponent.outcomes) == 2


def test_rollout_exposes_live_observations_and_direct_action_records() -> None:
    exact = _ExactStageStub(_matrix())
    opponent = _RecordingOpponent()
    env = AggroSessionEnv(
        opponent,
        exact,  # type: ignore[arg-type]
        games_per_session=3,
        seed=100,
        max_half_rounds=1,
    )
    visited: list[tuple[int, str, tuple[int, ...]]] = []

    def choose(decision: AggroDecision) -> int:
        visited.append(
            (decision.game_index, decision.learner_seat, decision.observation.shape)
        )
        assert not hasattr(decision, "opponent_true_distribution")
        return 30

    records = env.rollout(choose, seed=100)

    assert len(records) == 3
    assert [item.learner_seat for item in records] == ["Hal", "Baku", "Hal"]
    assert [item.game_boundary for item in records] == [True, True, True]
    assert [item.session_done for item in records] == [False, False, True]
    assert all(shape == (OBSERVATION_DIM,) for _, _, shape in visited)
    assert all(isinstance(item.reveal, PublicHalfRound) for item in records)
    assert all(item.opponent_true_distribution is not None for item in records)
    assert len(opponent.reveals) == len(records)
    assert len(opponent.outcomes) == 3
    assert opponent.resets == 3


def test_direct_action_domain_and_close_are_fail_closed_and_idempotent() -> None:
    exact = _ExactStageStub(_matrix())
    opponent = _RecordingOpponent()
    env = AggroSessionEnv(
        opponent,
        exact,  # type: ignore[arg-type]
        games_per_session=1,
        seed=4,
        max_half_rounds=2,
    )
    env.reset()

    with pytest.raises(ValueError, match="1..60"):
        env.step(0)
    with pytest.raises(ValueError, match="integer literal second"):
        env.step(True)
    assert len(opponent.reveals) == 0

    env.close()
    env.close()
    assert len(opponent.outcomes) == 1
    assert opponent.outcomes[0].winner_name is None
    assert opponent.outcomes[0].half_rounds == 0


def test_pure_dth_environment_never_exposes_stl_leap_action() -> None:
    exact = _ExactStageStub(_matrix())
    opponent = _RecordingOpponent()
    env = AggroSessionEnv(
        opponent,
        exact,  # type: ignore[arg-type]
        games_per_session=1,
        seed=7,
        start_clocks=(3540,),
        max_half_rounds=1,
    )

    decision = env.reset()

    assert decision.canonical_decision.turn_duration == 60
    assert decision.canonical_decision.legal_seconds == tuple(range(1, 61))
    result = env.step(60)
    assert result.record.selected_action == 60
    assert result.session_done


def test_state_conditioned_league_truth_integrates_with_full_decision() -> None:
    opponent = ReactiveDTHOpponent(STATE_THRESHOLD, seed=1701)
    env = AggroSessionEnv(
        opponent,
        _ExactStageStub(_matrix()),  # type: ignore[arg-type]
        games_per_session=1,
        seed=8,
        start_clocks=(720,),
        max_half_rounds=1,
    )

    decision = env.reset()
    expected = opponent.true_distribution(decision_from_opponent_view(decision))

    assert not hasattr(decision, "opponent_true_distribution")
    result = env.step(30)
    assert result.record.opponent_true_distribution is not None
    np.testing.assert_allclose(result.record.opponent_true_distribution, expected)
    assert result.session_done
    assert opponent.observation_count == 1


def test_reset_starts_a_fresh_opponent_session_when_environment_is_reused() -> None:
    opponent = _RecordingOpponent()
    env = AggroSessionEnv(
        opponent,
        _ExactStageStub(_matrix()),  # type: ignore[arg-type]
        games_per_session=1,
        seed=9,
        max_half_rounds=1,
    )

    env.reset()
    env.step(30)
    env.reset(seed=10)

    assert opponent.session_resets == 2
    assert opponent.resets == 2
    env.close()


def decision_from_opponent_view(decision: AggroDecision) -> CanonicalDecision:
    """Build the exact state-conditioned decision seen by the opponent."""

    own = decision.canonical_decision
    return CanonicalDecision(
        role=decision.opponent_role,
        actor_name=decision.opponent_seat,
        turn_duration=own.turn_duration,
        legal_seconds=tuple(range(1, 61)),
        checker_cylinder_seconds=own.checker_cylinder_seconds,
        checker_ttd_seconds=own.checker_ttd_seconds,
        dropper_cylinder_seconds=own.dropper_cylinder_seconds,
        dropper_ttd_seconds=own.dropper_ttd_seconds,
        native_state=own.native_state,
    )
