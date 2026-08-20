from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from arena.cli import build_parser, command_match, command_play
from arena.contracts import (
    CanonicalDecision,
    PublicDecisionState,
    PublicGameOutcome,
    PublicHalfRound,
    PublicPlayerState,
)
from arena.policies.aggro_hal import AggroHalConfig, AggroHalNetwork
from arena.policies.train_aggro_hal import load_training_config
from arena.policies.pm_hal import (
    ACTION_COUNT,
    CategoricalChangePointModel,
    PMHalConfig,
    PMHalOpponentModel,
    PMHalPolicyProvider,
    _OutcomeConditionedModel,
    _PosteriorView,
    load_pm_hal_config,
)
from dth.agent import CertifiedStageGame


def _stage(matrix: np.ndarray | None = None) -> CertifiedStageGame:
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


class _StageAgent:
    def __init__(self, matrix: np.ndarray | None = None) -> None:
        self.matrix = matrix
        self.states: list[tuple[int, int, int, int]] = []

    def stage_game(self, state) -> CertifiedStageGame:
        normalized = tuple(int(value) for value in state)
        self.states.append(normalized)
        stage = _stage(self.matrix)
        return CertifiedStageGame(
            state=normalized,
            value=stage.value,
            matrix=stage.matrix,
            drop_policy=stage.drop_policy,
            check_policy=stage.check_policy,
            saddle_gap=stage.saddle_gap,
        )


def _decision(
    *,
    role: str = "dropper",
    actor_name: str = "Hal",
    legal_seconds: tuple[int, ...] = tuple(range(1, 61)),
) -> CanonicalDecision:
    return CanonicalDecision(
        role=role,
        actor_name=actor_name,
        turn_duration=60,
        legal_seconds=legal_seconds,
        checker_cylinder_seconds=12.0,
        checker_ttd_seconds=60.0,
        dropper_cylinder_seconds=24.0,
        dropper_ttd_seconds=120.0,
        native_state=object(),
    )


def _reveal(
    *,
    self_role: str = "dropper",
    self_name: str = "Hal",
    opponent_name: str = "Baku",
    self_action: int = 3,
    opponent_action: int = 7,
    game_index: int = 0,
    half_round_index: int = 0,
    outcome: str = "check_success",
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
        outcome=outcome,
        game_over=False,
        winner_name=None,
    )


def _predict(
    model: PMHalOpponentModel,
    *,
    role: str = "checker",
    decision_index: int = 0,
):
    return model.predict(
        role,
        decision=_decision(),
        equilibrium_policy=np.full(ACTION_COUNT, 1.0 / ACTION_COUNT),
        game_index=0,
        game_decision_index=decision_index,
    )


def _aggressive_test_config(**overrides) -> PMHalConfig:
    values = {
        "minimum_role_observations": 1,
        "press_confidence_threshold": 0.05,
        "dominate_confidence_threshold": 0.20,
        "dominate_disagreement_threshold": 0.5,
        "probe_disagreement_threshold": 0.8,
        "minimum_improvement_support": 0.0,
        "posterior_samples": 64,
    }
    values.update(overrides)
    return PMHalConfig(**values)


def test_config_freezes_pure_dth_and_ordered_aggression_caps() -> None:
    config = PMHalConfig()
    assert config.action_count == 60
    assert config.probe_epsilon_cap < config.press_epsilon_cap
    assert config.press_epsilon_cap < config.dominate_epsilon_cap
    assert config.dominate_epsilon_cap < config.game_epsilon_budget

    with pytest.raises(ValueError, match="60"):
        PMHalConfig(action_count=61)
    with pytest.raises(ValueError, match="increase"):
        PMHalConfig(probe_epsilon_cap=0.60)
    with pytest.raises(ValueError, match="dominate confidence"):
        PMHalConfig(
            press_confidence_threshold=0.8,
            dominate_confidence_threshold=0.7,
        )


def test_tracked_pm_aggro_component_config_targets_the_current_artifact() -> None:
    model, trainer = load_training_config(
        "src/arena/config/pm_hal_aggro_component_v1.yaml"
    )
    assert model.hidden_size == 128
    assert trainer.dth_artifact == "src/dth/artifacts/complete_full_v1"
    assert trainer.warmstart_updates == 16
    assert trainer.ppo_updates == 4


def test_tracked_overbearing_controller_matches_code_defaults() -> None:
    tracked = load_pm_hal_config()
    assert tracked == PMHalConfig()
    assert tracked.minimum_role_observations == 1
    assert tracked.press_confidence_threshold == pytest.approx(0.15)
    assert tracked.game_epsilon_budget == pytest.approx(12.0)
    assert tracked.dominate_epsilon_cap == pytest.approx(2.0)
    assert load_pm_hal_config("src/arena/config/pm_hal_controller_v2.json") == tracked


def test_outcome_expert_uses_immediately_previous_public_outcome_across_roles() -> None:
    model = _OutcomeConditionedModel(prior_strength=0.25)
    prior = np.full(ACTION_COUNT, 1.0 / ACTION_COUNT)
    model.observe("checker", 1, outcome="A")
    model.observe("checker", 9, outcome="B")
    model.observe("checker", 9, outcome="B")
    assert model.predict("checker", prior)[8] > 0.80

    model.observe("dropper", 4, outcome="C")
    prediction = model.predict("checker", prior)
    np.testing.assert_allclose(prediction, prior, atol=1e-12)


def test_candidate_ensemble_uses_one_common_perturbation_bank() -> None:
    model = PMHalOpponentModel(PMHalConfig(posterior_samples=16))
    forecast = _predict(model)
    view = _PosteriorView(model, forecast)
    rng = np.random.default_rng(17)
    first = view.sample("checker", size=16, rng=rng)
    second = view.sample("checker", size=16, rng=rng)
    np.testing.assert_array_equal(first, second)


def test_categorical_change_point_spikes_on_a_surprising_switch() -> None:
    model = CategoricalChangePointModel(
        hazard=0.05,
        prior_strength=0.25,
        max_run_length=64,
    )
    for _ in range(20):
        model.observe("checker", 7)
    stable = model.predict("checker")
    assert stable.policy[6] > 0.95
    assert stable.change_probability < 0.10

    model.observe("checker", 49)
    switched = model.predict("checker")
    assert switched.change_probability > 0.35
    assert switched.expected_run_length < stable.expected_run_length


def test_fixed_share_forecast_learns_one_role_without_cross_role_leakage() -> None:
    model = PMHalOpponentModel(PMHalConfig(posterior_samples=32))
    untouched = _predict(model, role="dropper")
    for index in range(10):
        forecast = _predict(model, decision_index=index)
        model.observe(
            forecast,
            opponent_action=7,
            self_action=3,
            outcome="check_success",
        )

    learned = _predict(model, decision_index=10)
    still_untouched = _predict(model, role="dropper", decision_index=10)
    assert learned.policy[6] > 0.80
    assert learned.confidence > 0.25
    assert still_untouched.policy == pytest.approx(untouched.policy)
    assert model.observations("checker") == 10
    assert model.observations("dropper") == 0


def test_fixed_share_keeps_recovery_mass_for_every_source() -> None:
    model = PMHalOpponentModel(PMHalConfig(posterior_samples=32))
    for index in range(30):
        forecast = _predict(model, decision_index=index)
        model.observe(
            forecast,
            opponent_action=7,
            self_action=3,
            outcome="check_success",
        )
    forecast = _predict(model, decision_index=30)
    assert np.all(forecast.component_weights > 0.0)
    assert float(forecast.component_weights.min()) > 0.001


def test_forecast_token_is_causal_and_single_use() -> None:
    model = PMHalOpponentModel(PMHalConfig(posterior_samples=32))
    forecast = _predict(model)
    model.observe(
        forecast,
        opponent_action=7,
        self_action=3,
        outcome="check_success",
    )
    with pytest.raises(RuntimeError, match="stale or already observed"):
        model.observe(
            forecast,
            opponent_action=7,
            self_action=3,
            outcome="check_success",
        )


def test_provider_shields_at_cold_start_then_attacks_a_stable_pattern(
    tmp_path: Path,
) -> None:
    config = _aggressive_test_config()
    provider = PMHalPolicyProvider(
        tmp_path,
        config,
        agent=_StageAgent(),
        seed=11,
    )
    first = provider.policy(_decision())
    assert provider.last_decision is not None
    assert provider.pending_stage_game is not None
    assert provider.last_decision.mode == "shield"
    assert provider.last_decision.selected_candidate == "exact"
    assert first == pytest.approx(
        {action: 1.0 / ACTION_COUNT for action in range(1, ACTION_COUNT + 1)}
    )
    provider.observe(_reveal(opponent_action=7, half_round_index=0))
    assert provider.pending_stage_game is None

    second = provider.policy(_decision())
    assert provider.last_decision is not None
    assert provider.last_decision.mode in {"press", "dominate"}
    assert second[7] > first[7]
    assert provider.last_decision.expected_improvement > 0.0
    assert (
        provider.last_decision.selected_actual_worst_case_loss
        <= provider.config.dominate_epsilon_cap + 1e-9
    )


def test_checker_orientation_attacks_with_the_same_measured_risk_gate(
    tmp_path: Path,
) -> None:
    config = _aggressive_test_config()
    provider = PMHalPolicyProvider(
        tmp_path,
        config,
        agent=_StageAgent(),
        seed=13,
    )
    first = provider.policy(_decision(role="checker"))
    provider.observe(
        _reveal(self_role="checker", opponent_action=7, half_round_index=0)
    )

    second = provider.policy(_decision(role="checker"))
    decision = provider.last_decision
    assert decision is not None
    assert decision.mode in {"press", "dominate"}
    assert second.get(7, 0.0) < first[7]
    assert decision.expected_improvement > 0.0
    assert decision.selected_actual_worst_case_loss <= (
        provider.config.dominate_epsilon_cap + 1e-9
    )


def test_switch_shock_forces_the_next_decision_back_to_shield(tmp_path: Path) -> None:
    provider = PMHalPolicyProvider(
        tmp_path,
        _aggressive_test_config(),
        agent=_StageAgent(),
        seed=19,
    )
    for index in range(15):
        provider.policy(_decision())
        provider.observe(_reveal(opponent_action=7, half_round_index=index))
    provider.policy(_decision())
    provider.observe(_reveal(opponent_action=49, half_round_index=15))

    provider.policy(_decision())
    assert provider.last_decision is not None
    assert provider.last_decision.change_probability > 0.35
    assert provider.last_decision.mode == "shield"
    assert provider.last_decision.selected_candidate == "exact"
    assert provider.last_decision.selected_actual_worst_case_loss <= 1e-9


def test_budget_charge_never_exceeds_the_frozen_per_game_cap(tmp_path: Path) -> None:
    config = _aggressive_test_config(
        game_epsilon_budget=0.05,
        press_epsilon_cap=0.05,
        dominate_epsilon_cap=0.05,
    )
    provider = PMHalPolicyProvider(
        tmp_path,
        config,
        agent=_StageAgent(),
        seed=23,
    )
    for index in range(20):
        provider.policy(_decision())
        assert provider.epsilon_spent <= config.game_epsilon_budget + 1e-9
        provider.observe(_reveal(opponent_action=7, half_round_index=index))
    assert provider.epsilon_spent <= 0.05 + 1e-9
    assert all(
        decision.game_epsilon_spent <= 0.05 + 1e-9 for decision in provider.decisions
    )


def test_reveal_publishes_proper_scores_and_counterfactual_feedback(
    tmp_path: Path,
) -> None:
    provider = PMHalPolicyProvider(
        tmp_path,
        _aggressive_test_config(),
        agent=_StageAgent(),
        seed=29,
    )
    provider.policy(_decision())
    provider.observe(_reveal(opponent_action=7))
    sample = provider.observation_metrics[-1]
    assert sample.realized_nll == pytest.approx(np.log(ACTION_COUNT))
    assert sample.brier_score == pytest.approx((ACTION_COUNT - 1) / ACTION_COUNT)
    assert sample.one_step_candidate_regret >= 0.0
    assert 0.0 <= sample.post_reveal_change_probability <= 1.0


def test_optional_aggro_network_is_one_fixed_share_source(tmp_path: Path) -> None:
    torch.manual_seed(31)
    aggro_config = AggroHalConfig(hidden_size=8, head_hidden_size=8)
    network = AggroHalNetwork(aggro_config)
    provider = PMHalPolicyProvider(
        tmp_path,
        _aggressive_test_config(posterior_samples=16),
        agent=_StageAgent(),
        aggro_model=network,
        aggro_config=aggro_config,
        seed=31,
    )
    provider.policy(_decision())
    decision = provider.last_decision
    assert decision is not None
    assert decision.aggro_enabled
    assert "aggro" in dict(decision.component_weights)
    assert any(candidate.name == "aggro_direct" for candidate in decision.candidates)
    provider.observe(_reveal(opponent_action=7))
    provider.reset_game()
    provider.policy(_decision())
    assert provider.has_session_memory


def test_provider_fails_closed_outside_pure_dth_and_enforces_lifecycle(
    tmp_path: Path,
) -> None:
    provider = PMHalPolicyProvider(
        tmp_path,
        PMHalConfig(posterior_samples=16),
        agent=_StageAgent(),
    )
    with pytest.raises(ValueError, match="pure DTH"):
        provider.policy(_decision(legal_seconds=tuple(range(1, 62))))

    provider.policy(_decision())
    with pytest.raises(RuntimeError, match="twice before a reveal"):
        provider.policy(_decision())
    with pytest.raises(RuntimeError, match="unrevealed"):
        provider.reset_game()
    provider.observe(_reveal())
    provider.end_game(PublicGameOutcome(game_index=0, winner_name="Hal", half_rounds=1))
    provider.reset_game()
    assert provider.epsilon_spent == 0.0


def test_session_reset_restores_monte_carlo_stream(tmp_path: Path) -> None:
    provider = PMHalPolicyProvider(
        tmp_path,
        _aggressive_test_config(),
        agent=_StageAgent(),
        seed=43,
    )

    def run_two_decisions() -> tuple[object, object]:
        provider.policy(_decision())
        provider.observe(_reveal(opponent_action=7, half_round_index=0))
        provider.policy(_decision())
        decision = provider.last_decision
        assert decision is not None
        return decision.selected_candidate, decision.candidates

    first = run_two_decisions()
    provider.observe(_reveal(opponent_action=7, half_round_index=1))
    provider.reset_session()
    second = run_two_decisions()
    assert second == first


def test_diagnostics_are_json_serializable_and_scope_the_claim(tmp_path: Path) -> None:
    provider = PMHalPolicyProvider(
        tmp_path,
        PMHalConfig(posterior_samples=16),
        agent=_StageAgent(),
    )
    provider.policy(_decision())
    diagnostics = provider.experiment_diagnostics()
    assert diagnostics["pure_dth_only"] is True
    assert diagnostics["public_history_only"] is True
    assert diagnostics["fixed_share"] is True
    assert diagnostics["aggro_enabled"] is False
    json.dumps(diagnostics)


def test_match_cli_exposes_pm_hal_only_with_explicit_pure_dth() -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
            "match",
            "--candidate",
            "pm-hal",
            "--opponent",
            "dth",
            "--output",
            "unused.json",
        ]
    )
    with pytest.raises(ValueError, match="pure-DTH"):
        command_match(args)

    pure = parser.parse_args(
        [
            "match",
            "--candidate",
            "pm-hal",
            "--opponent",
            "dth",
            "--pure-dth",
            "--output",
            "unused.json",
        ]
    )
    assert pure.candidate == "pm-hal"
    assert pure.pure_dth is True
    assert pure.pm_hal_aggro_checkpoint is None
    assert pure.pm_hal_config.endswith("pm_hal_controller_v3.json")
    assert pure.pm_hal_game_epsilon_budget is None


def test_human_play_requires_pm_hal_pure_dth_surface() -> None:
    args = build_parser().parse_args(["play", "--hal-agent", "pm-hal", "--skip-rules"])
    with pytest.raises(ValueError, match="pure-DTH"):
        command_play(args)

    pure = build_parser().parse_args(
        ["play", "--hal-agent", "pm-hal", "--pure-dth", "--skip-rules"]
    )
    assert pure.pure_dth is True
