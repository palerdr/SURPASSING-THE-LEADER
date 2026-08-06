from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from arena.adaptive_dth import (
    AdaptiveDecision,
    AdaptiveDTHPolicyProvider,
    DirichletPrior,
    ExploitationConfig,
    PolicySelection,
    RoleDirichletOpponent,
    RoleMixtureOpponent,
    select_evidence_gated_policy,
)
from arena.agent import decision_from_game
from arena.contracts import PublicDecisionState, PublicHalfRound, PublicPlayerState
from dth.agent import MoveDecision
from stl.engine.game import Game, Player, Referee


def _pure(index: int) -> np.ndarray:
    policy = np.zeros(60, dtype=np.float64)
    policy[index] = 1.0
    return policy


def _dropper_opportunity() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    matrix = np.zeros((60, 60), dtype=np.float64)
    matrix[0, 0] = 1.0
    matrix[0, 1:] = -1.0
    return matrix, _pure(1), _pure(1)


def _checker_opportunity() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    matrix = np.zeros((60, 60), dtype=np.float64)
    matrix[0, 0] = -1.0
    return matrix, _pure(1), _pure(1)


def test_role_posteriors_update_independently_and_exactly() -> None:
    model = RoleDirichletOpponent.uniform(strength=60.0)

    model.observe("dropper", 7)

    assert model.alpha("dropper")[6] == pytest.approx(2.0)
    assert model.alpha("dropper").sum() == pytest.approx(61.0)
    assert model.predictive("dropper")[6] == pytest.approx(2.0 / 61.0)
    assert model.alpha("checker") == pytest.approx(np.ones(60))
    assert model.observation_count("dropper") == 1
    assert model.observation_count("checker") == 0


def test_discounting_forgets_evidence_without_discounting_the_prior() -> None:
    model = RoleDirichletOpponent.uniform(strength=60.0, decay=0.5)
    model.observe("checker", 1)
    model.observe("checker", 2)

    expected = np.ones(60)
    expected[0] += 0.5
    expected[1] += 1.0
    assert model.alpha("checker") == pytest.approx(expected)


def test_role_mixture_updates_membership_and_samples_posterior() -> None:
    early = np.full(60, 0.1 / 59.0)
    early[0] = 0.9
    late = np.full(60, 0.1 / 59.0)
    late[-1] = 0.9
    early_prior = DirichletPrior(tuple(early), strength=4.0)
    late_prior = DirichletPrior(tuple(late), strength=4.0)
    model = RoleMixtureOpponent(
        mixture_weights=(0.5, 0.5),
        drop_priors=(early_prior, late_prior),
        check_priors=(early_prior, late_prior),
        decay=0.9,
    )

    before = model.posterior_weights
    model.observe("dropper", 1)
    draws = model.sample("dropper", size=64, rng=np.random.default_rng(91))

    assert before == pytest.approx((0.5, 0.5))
    assert model.posterior_weights[0] > model.posterior_weights[1]
    assert model.predictive("dropper")[0] > model.predictive("dropper")[-1]
    assert draws.shape == (64, 60)
    assert draws.sum(axis=1) == pytest.approx(np.ones(64))


def test_little_evidence_keeps_the_equilibrium_policy() -> None:
    matrix, drop, check = _dropper_opportunity()
    model = RoleDirichletOpponent.uniform(strength=60.0)

    selected = select_evidence_gated_policy(
        matrix,
        role="dropper",
        equilibrium_drop=drop,
        equilibrium_check=check,
        opponent=model,
        config=ExploitationConfig(
            epsilon_grid=(0.0, 0.1),
            match_epsilon_budget=0.1,
            posterior_samples=256,
        ),
        remaining_epsilon=0.1,
        rng=np.random.default_rng(3),
    )

    assert selected.exploited is False
    assert selected.policy == pytest.approx(drop)
    assert selected.epsilon == 0.0


def test_strong_checker_bias_selects_most_aggressive_safe_drop_policy() -> None:
    matrix, drop, check = _dropper_opportunity()
    model = RoleDirichletOpponent.uniform(strength=1.0)
    for _ in range(30):
        model.observe("checker", 1)

    selected = select_evidence_gated_policy(
        matrix,
        role="dropper",
        equilibrium_drop=drop,
        equilibrium_check=check,
        opponent=model,
        config=ExploitationConfig(
            epsilon_grid=(0.0, 0.02, 0.1),
            match_epsilon_budget=0.1,
            confidence=0.95,
            posterior_samples=512,
        ),
        remaining_epsilon=0.1,
        rng=np.random.default_rng(4),
    )

    policy = np.asarray(selected.policy)
    assert selected.exploited is True
    assert selected.epsilon == pytest.approx(0.1)
    assert selected.improvement_probability >= 0.95
    assert selected.selected_worst_case_value >= -0.1 - 1e-9
    assert policy[0] == pytest.approx(0.1, abs=1e-8)
    assert policy.sum() == pytest.approx(1.0)


def test_remaining_match_budget_caps_local_epsilon() -> None:
    matrix, drop, check = _dropper_opportunity()
    model = RoleDirichletOpponent.uniform(strength=1.0)
    for _ in range(30):
        model.observe("checker", 1)

    selected = select_evidence_gated_policy(
        matrix,
        role="dropper",
        equilibrium_drop=drop,
        equilibrium_check=check,
        opponent=model,
        config=ExploitationConfig(
            epsilon_grid=(0.0, 0.02, 0.1),
            match_epsilon_budget=0.1,
            posterior_samples=256,
        ),
        remaining_epsilon=0.02,
        rng=np.random.default_rng(5),
    )

    assert selected.exploited is True
    assert selected.epsilon == pytest.approx(0.02)
    assert selected.selected_worst_case_value >= -0.02 - 1e-9


def test_checker_uses_the_opponent_dropper_posterior() -> None:
    matrix, drop, check = _checker_opportunity()
    model = RoleDirichletOpponent.uniform(strength=1.0)
    for _ in range(20):
        model.observe("dropper", 1)

    selected = select_evidence_gated_policy(
        matrix,
        role="checker",
        equilibrium_drop=drop,
        equilibrium_check=check,
        opponent=model,
        config=ExploitationConfig(posterior_samples=256),
        remaining_epsilon=0.0,
        rng=np.random.default_rng(6),
    )

    assert selected.exploited is True
    assert selected.epsilon == 0.0
    assert selected.policy[0] == pytest.approx(1.0)
    assert selected.selected_worst_case_value <= selected.baseline_worst_case_value


class _FakeAgent:
    def __init__(self, artifact_dir: Path) -> None:
        self.artifact_dir = artifact_dir

    def decide(self, state) -> MoveDecision:
        return MoveDecision(
            state=state,
            value=0.0,
            drop_policy=tuple(_pure(1)),
            check_policy=tuple(_pure(1)),
            saddle_gap=0.0,
            elapsed_seconds=0.0,
        )


def test_provider_learns_revealed_opponent_actions_once(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import arena.adaptive_dth as adaptive

    monkeypatch.setattr(adaptive, "CompleteDTHAgent", _FakeAgent)
    monkeypatch.setattr(
        adaptive,
        "reconstruct_stage_matrix",
        lambda agent, state: np.zeros((60, 60), dtype=np.float64),
    )
    provider = AdaptiveDTHPolicyProvider(
        tmp_path,
        RoleDirichletOpponent.uniform(strength=60.0),
        seed=7,
    )
    game = Game(Player("Hal"), Player("Baku"), Referee())

    provider.policy(
        decision_from_game(game, role="dropper", turn_duration=60)
    )
    game.play_half_round(2, 7)
    provider.policy(
        decision_from_game(game, role="checker", turn_duration=60)
    )
    provider.policy(
        decision_from_game(game, role="checker", turn_duration=60)
    )

    assert provider.opponent.observation_count("dropper") == 0
    assert provider.opponent.observation_count("checker") == 1
    assert provider.opponent.alpha("checker")[6] == pytest.approx(2.0)


def test_provider_retreats_when_opponent_can_play_leap_action_61(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import arena.adaptive_dth as adaptive

    matrix, _, _ = _checker_opportunity()
    monkeypatch.setattr(adaptive, "CompleteDTHAgent", _FakeAgent)
    monkeypatch.setattr(
        adaptive, "reconstruct_stage_matrix", lambda agent, state: matrix
    )
    model = RoleDirichletOpponent.uniform(strength=1.0)
    for _ in range(20):
        model.observe("dropper", 1)
    provider = AdaptiveDTHPolicyProvider(tmp_path, model, seed=8)
    game = Game(Player("Hal"), Player("Baku"), Referee())
    game.current_half = 2
    game.game_clock = 3540

    policy = provider.policy(
        decision_from_game(game, role="checker", turn_duration=61)
    )

    assert policy == {2: 1.0}
    assert provider.decisions[-1].selection.exploited is False
    assert (
        provider.decisions[-1].selection.reason
        == "opponent-action-space-outside-dth"
    )


def test_provider_returns_to_equilibrium_after_budget_is_exhausted(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import arena.adaptive_dth as adaptive

    matrix, _, _ = _dropper_opportunity()
    monkeypatch.setattr(adaptive, "CompleteDTHAgent", _FakeAgent)
    monkeypatch.setattr(
        adaptive, "reconstruct_stage_matrix", lambda agent, state: matrix
    )
    model = RoleDirichletOpponent.uniform(strength=1.0)
    for _ in range(30):
        model.observe("checker", 1)
    provider = AdaptiveDTHPolicyProvider(
        tmp_path,
        model,
        config=ExploitationConfig(
            epsilon_grid=(0.0, 0.1),
            match_epsilon_budget=0.1,
            posterior_samples=256,
        ),
        seed=9,
    )
    first_game = Game(Player("Hal"), Player("Baku"), Referee())
    decision = decision_from_game(first_game, role="dropper", turn_duration=60)

    first = provider.policy(decision)
    second = provider.policy(decision)

    assert first[1] == pytest.approx(0.1)
    assert second == {2: 1.0}
    assert provider.spent_epsilon == pytest.approx(0.1)
    assert provider.decisions[-1].selection.exploited is False

    second_game = Game(Player("Hal"), Player("Baku"), Referee())
    provider.policy(
        decision_from_game(second_game, role="dropper", turn_duration=60)
    )

    assert provider.decisions[-1].selection.exploited is True
    assert provider.spent_epsilon == pytest.approx(0.1)


def test_experiment_diagnostics_groups_decisions_by_match(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import arena.adaptive_dth as adaptive

    monkeypatch.setattr(adaptive, "CompleteDTHAgent", _FakeAgent)
    provider = AdaptiveDTHPolicyProvider(
        artifact_dir=tmp_path,
        opponent=RoleDirichletOpponent.uniform(),
    )
    provider.decisions.extend(
        [
            AdaptiveDecision(
                match_index=0,
                state=(0, 0, 0, 0),
                role="dropper",
                tablebase_value=0.0,
                saddle_gap=1e-8,
                selection=PolicySelection(
                    policy=(1 / 60,) * 60,
                    epsilon=0.0,
                    improvement_probability=0.0,
                    expected_improvement=0.0,
                    baseline_worst_case_value=0.0,
                    selected_worst_case_value=0.0,
                    exploited=False,
                    reason="insufficient-posterior-evidence",
                ),
            ),
            AdaptiveDecision(
                match_index=1,
                state=(0, 0, 0, 0),
                role="checker",
                tablebase_value=0.0,
                saddle_gap=2e-8,
                selection=PolicySelection(
                    policy=(1 / 60,) * 60,
                    epsilon=0.01,
                    improvement_probability=1.0,
                    expected_improvement=0.1,
                    baseline_worst_case_value=0.0,
                    selected_worst_case_value=0.01,
                    exploited=True,
                    reason="posterior-confidence-gate-passed",
                ),
            ),
        ]
    )

    diagnostics = provider.experiment_diagnostics()
    assert diagnostics["schema_version"] == "adaptive-dth-session-diagnostics-v1"
    assert diagnostics["games"][0]["decisions"] == 1
    assert diagnostics["games"][1]["exploited"] == 1
    assert diagnostics["games"][1]["epsilon_spent"] == pytest.approx(0.01)


def test_prior_requires_positive_support_for_every_dth_action() -> None:
    mean = np.full(60, 1.0 / 59.0)
    mean[-1] = 0.0
    with pytest.raises(ValueError, match="strictly positive"):
        DirichletPrior(tuple(mean), strength=10.0)


def test_public_action_61_is_counted_without_corrupting_posterior(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import arena.adaptive_dth as adaptive

    monkeypatch.setattr(adaptive, "CompleteDTHAgent", _FakeAgent)
    monkeypatch.setattr(
        adaptive,
        "reconstruct_stage_matrix",
        lambda agent, state: np.zeros((60, 60), dtype=np.float64),
    )
    provider = AdaptiveDTHPolicyProvider(
        tmp_path, RoleDirichletOpponent.uniform(), seed=10
    )
    game = Game(Player("Hal"), Player("Baku"), Referee())
    game.current_half = 2
    provider.reset_game()
    provider.policy(decision_from_game(game, role="checker", turn_duration=60))
    before = provider.opponent.predictive("dropper").copy()
    provider.observe(
        PublicHalfRound(
            game_index=0,
            half_round_index=0,
            pre_decision_state=PublicDecisionState(
                game_clock_seconds=3540.0,
                round_index=0,
                half_index=2,
                turn_duration=61,
                players=(
                    PublicPlayerState("Hal", 0.0, 0.0),
                    PublicPlayerState("Baku", 0.0, 0.0),
                ),
            ),
            dropper_name="Baku",
            checker_name="Hal",
            drop_time=61,
            check_time=60,
            outcome="check_fail_survived",
            game_over=False,
            winner_name=None,
        )
    )

    assert provider.opponent.observation_count("dropper") == 0
    assert provider.opponent.predictive("dropper") == pytest.approx(before)
    assert provider.experiment_diagnostics()["unsupported_observations"] == 1
