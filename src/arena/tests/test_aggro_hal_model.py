from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from arena.contracts import (
    CanonicalDecision,
    PublicDecisionState,
    PublicGameOutcome,
    PublicHalfRound,
    PublicPlayerState,
)
from arena.policies.aggro_hal import (
    ACTION_COUNT,
    OBSERVATION_DIM,
    OBSERVATION_FEATURES,
    AggroHalConfig,
    AggroHalNetwork,
    AggroHalPolicyProvider,
    dth_compatibility,
    load_checkpoint,
    save_checkpoint,
)
from dth.agent import CertifiedStageGame

CPU = torch.device("cpu")


def _stage() -> CertifiedStageGame:
    row = np.linspace(-0.8, 0.9, ACTION_COUNT, dtype=np.float64)
    column = np.linspace(0.6, -0.7, ACTION_COUNT, dtype=np.float64)
    matrix = row[:, None] + 0.5 * column[None, :]
    uniform = np.full(ACTION_COUNT, 1.0 / ACTION_COUNT, dtype=np.float64)
    return CertifiedStageGame(
        state=(0, 60, 0, 60),
        value=0.05,
        matrix=matrix,
        drop_policy=uniform.copy(),
        check_policy=uniform.copy(),
        saddle_gap=0.0,
    )


def _metadata() -> dict[str, object]:
    return {
        "solver_schema_hash": "solver-hash",
        "table_digest": "table-digest",
        "class_encoding": "test-encoding",
        "profile_count": 123,
        "class_count": 45,
        "max_support": 8,
        "saddle_gap_tolerance": 1e-6,
        "canonical_table": True,
        "ladder": "pure-dth",
        "code_config_digest": "config-digest",
    }


class _StageAgent:
    def __init__(self) -> None:
        self.tablebase = SimpleNamespace(metadata=_metadata())
        self.states: list[tuple[int, int, int, int]] = []

    def stage_game(self, state) -> CertifiedStageGame:
        self.states.append(tuple(state))
        stage = _stage()
        return CertifiedStageGame(
            state=tuple(state),
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
    legal=tuple(range(1, 61)),
) -> CanonicalDecision:
    return CanonicalDecision(
        role=role,
        actor_name=actor_name,
        turn_duration=60,
        legal_seconds=tuple(legal),
        checker_cylinder_seconds=12.0,
        checker_ttd_seconds=60.0,
        dropper_cylinder_seconds=24.0,
        dropper_ttd_seconds=120.0,
        native_state=object(),
    )


def _reveal(*, game_over: bool = False) -> PublicHalfRound:
    return PublicHalfRound(
        game_index=0,
        half_round_index=0,
        pre_decision_state=PublicDecisionState(
            game_clock_seconds=720.0,
            round_index=1,
            half_index=1,
            turn_duration=60,
            players=(
                PublicPlayerState("Hal", 24.0, 120.0),
                PublicPlayerState("Baku", 12.0, 60.0),
            ),
        ),
        dropper_name="Hal",
        checker_name="Baku",
        drop_time=3,
        check_time=7,
        outcome="check_success",
        game_over=game_over,
        winner_name="Hal" if game_over else None,
    )


def _model_inputs(batch: int, time: int):
    generator = torch.Generator(device=CPU).manual_seed(81)
    features = torch.randn(
        batch, time, OBSERVATION_DIM, generator=generator, device=CPU
    )
    matrices = torch.randn(
        batch, time, ACTION_COUNT, ACTION_COUNT, generator=generator, device=CPU
    )
    exact = torch.softmax(
        torch.randn(batch, time, ACTION_COUNT, generator=generator, device=CPU), dim=-1
    )
    roles = torch.tensor(
        [[(step + item) % 2 == 0 for step in range(time)] for item in range(batch)],
        dtype=torch.bool,
        device=CPU,
    )
    legal = torch.ones(batch, time, ACTION_COUNT, dtype=torch.bool, device=CPU)
    return features, matrices, exact, roles, legal


def test_network_uses_role_oriented_analytic_values_and_returns_legal_mixture() -> None:
    torch.manual_seed(3)
    config = AggroHalConfig(
        hidden_size=16, head_hidden_size=8, tactical_logit_scale=2.5
    )
    model = AggroHalNetwork(config).to(CPU).eval()
    features, matrices, exact, roles, legal = _model_inputs(2, 1)
    roles[:, 0] = torch.tensor([True, False], device=CPU)
    legal[:, :, 5:] = False

    output = model(features, matrices, exact, roles, legal)

    q = output.opponent_policy
    expected_drop = torch.matmul(matrices[0, 0], q[0, 0].unsqueeze(-1)).squeeze(-1)
    expected_check = -torch.matmul(matrices[1, 0].T, q[1, 0].unsqueeze(-1)).squeeze(-1)
    assert torch.allclose(output.analytic_action_values[0, 0], expected_drop)
    assert torch.allclose(output.analytic_action_values[1, 0], expected_check)
    assert torch.all(output.policy[:, :, 5:] == 0.0)
    assert torch.allclose(output.policy.sum(dim=-1), torch.ones(2, 1, device=CPU))

    masked_exact = exact * legal
    masked_exact = masked_exact / masked_exact.sum(dim=-1, keepdim=True)
    gate = output.direct_weight.unsqueeze(-1)
    expected_mixture = (1.0 - gate) * masked_exact + gate * output.direct_policy
    assert torch.allclose(output.policy, expected_mixture, atol=1e-6)


def test_two_layer_gru_matches_stepwise_sequence_execution_on_cpu() -> None:
    torch.manual_seed(5)
    config = AggroHalConfig(hidden_size=12, head_hidden_size=7, gru_dropout=0.0)
    model = AggroHalNetwork(config).to(CPU).eval()
    assert model.gru.num_layers == 2
    inputs = _model_inputs(1, 4)

    whole = model(*inputs)
    hidden = None
    step_policies = []
    step_opponents = []
    for index in range(4):
        step = model(
            inputs[0][:, index : index + 1],
            inputs[1][:, index : index + 1],
            inputs[2][:, index : index + 1],
            inputs[3][:, index : index + 1],
            inputs[4][:, index : index + 1],
            hidden,
        )
        hidden = step.hidden_state
        step_policies.append(step.policy)
        step_opponents.append(step.opponent_policy)

    assert torch.allclose(whole.policy, torch.cat(step_policies, dim=1), atol=1e-6)
    assert torch.allclose(
        whole.opponent_policy, torch.cat(step_opponents, dim=1), atol=1e-6
    )
    assert torch.allclose(whole.hidden_state, hidden, atol=1e-6)


def test_v1_rejects_recurrent_dropout_that_would_break_ppo_ratios() -> None:
    with pytest.raises(ValueError, match="gru_dropout=0"):
        AggroHalConfig(gru_dropout=0.1)


class _CapturingNetwork(AggroHalNetwork):
    def __init__(self, config: AggroHalConfig) -> None:
        super().__init__(config)
        self.captured_features: list[torch.Tensor] = []

    def forward(self, features, *args, **kwargs):
        self.captured_features.append(features.detach().cpu().clone())
        return super().forward(features, *args, **kwargs)


def test_provider_consumes_only_observed_reveal_and_preserves_memory_across_seat_change(
    tmp_path: Path,
) -> None:
    torch.manual_seed(7)
    config = AggroHalConfig(hidden_size=10, head_hidden_size=6)
    model = _CapturingNetwork(config).to(CPU)
    agent = _StageAgent()
    provider = AggroHalPolicyProvider(
        tmp_path,
        model,
        config,
        agent=agent,
        device=CPU,
    )
    feature_index = {name: index for index, name in enumerate(OBSERVATION_FEATURES)}

    first = provider.policy(_decision())
    assert set(first) <= set(range(1, 61))
    assert sum(first.values()) == pytest.approx(1.0)
    assert (
        model.captured_features[0][0, 0, feature_index["previous_reveal_present"]]
        == 0.0
    )
    assert agent.states == [(12, 60, 24, 120)]
    hidden_after_first = provider._hidden_state.detach().clone()

    provider.observe(_reveal(game_over=True))
    assert len(model.captured_features) == 1
    provider.end_game(PublicGameOutcome(game_index=0, winner_name="Hal", half_rounds=1))
    provider.reset_game()
    assert torch.equal(provider._hidden_state, hidden_after_first)

    second = provider.policy(_decision(role="checker", actor_name="Baku"))
    encoded = model.captured_features[1][0, 0]
    assert set(second) <= set(range(1, 61))
    assert encoded[feature_index["previous_reveal_present"]] == 1.0
    assert encoded[feature_index["previous_drop_action_3"]] == 1.0
    assert encoded[feature_index["previous_check_action_7"]] == 1.0
    assert encoded[feature_index["previous_dropper_is_self"]] == 1.0
    assert encoded[feature_index["previous_checker_is_self"]] == 0.0
    assert encoded[feature_index["previous_self_cylinder_over_300"]] == pytest.approx(
        24.0 / 300.0
    )
    assert encoded[
        feature_index["previous_opponent_cylinder_over_300"]
    ] == pytest.approx(12.0 / 300.0)
    assert encoded[feature_index["current_new_game"]] == 1.0
    assert provider.has_session_memory

    provider.reset_session()
    assert not provider.has_session_memory
    assert provider.last_decision is None


def test_live_provider_is_cpu_fail_safe_even_if_cuda_is_reported_available(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    config = AggroHalConfig(hidden_size=8, head_hidden_size=5)
    model = AggroHalNetwork(config)
    provider = AggroHalPolicyProvider(
        tmp_path,
        model,
        config,
        agent=_StageAgent(),
    )

    provider.policy(_decision())

    assert provider.device.type == "cpu"
    assert all(
        parameter.device.type == "cpu" for parameter in provider.model.parameters()
    )
    assert provider._hidden_state is not None
    assert provider._hidden_state.device.type == "cpu"


def test_live_provider_fails_closed_outside_pure_dth(tmp_path: Path) -> None:
    config = AggroHalConfig(hidden_size=8, head_hidden_size=5)
    agent = _StageAgent()
    provider = AggroHalPolicyProvider(
        tmp_path,
        AggroHalNetwork(config),
        config,
        agent=agent,
    )

    with pytest.raises(ValueError, match="pure DTH only"):
        provider.policy(_decision(legal=tuple(range(1, 62))))
    assert agent.states == []


def test_fast_adaptation_concentrates_on_a_revealed_predictable_action(
    tmp_path: Path,
) -> None:
    class _DiagonalAgent(_StageAgent):
        def stage_game(self, state) -> CertifiedStageGame:
            self.states.append(tuple(state))
            uniform = np.full(ACTION_COUNT, 1.0 / ACTION_COUNT, dtype=np.float64)
            return CertifiedStageGame(
                state=tuple(state),
                value=0.0,
                matrix=np.eye(ACTION_COUNT, dtype=np.float64),
                drop_policy=uniform.copy(),
                check_policy=uniform.copy(),
                saddle_gap=0.0,
            )

    torch.manual_seed(13)
    config = AggroHalConfig(
        hidden_size=8, head_hidden_size=5, tactical_logit_scale=12.0
    )
    model = AggroHalNetwork(config).to(CPU)
    for parameter in model.parameters():
        torch.nn.init.zeros_(parameter)
    provider = AggroHalPolicyProvider(
        tmp_path,
        model,
        config,
        agent=_DiagonalAgent(),
        device=CPU,
        fast_adaptation=True,
    )

    before = provider.policy(_decision(role="dropper"))
    provider.observe(_reveal())
    after = provider.policy(_decision(role="dropper"))

    assert provider.last_decision is not None
    assert provider.last_decision.fast_adaptation_weight > 0.5
    assert after[7] > before[7]
    assert sum(after.values()) == pytest.approx(1.0)

    provider.reset_session()
    provider.policy(_decision(role="dropper"))
    assert provider.last_decision is not None
    assert provider.last_decision.fast_adaptation_weight == 0.0


def test_checkpoint_round_trip_is_strictly_bound_to_config_and_dth_artifact(
    tmp_path: Path,
) -> None:
    torch.manual_seed(9)
    config = AggroHalConfig(hidden_size=14, head_hidden_size=9)
    model = AggroHalNetwork(config).to(CPU)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    compatibility = dth_compatibility(_StageAgent())
    path = save_checkpoint(
        tmp_path / "aggro.pt",
        model=model,
        config=config,
        dth_ruleset=compatibility,
        optimizer=optimizer,
        training_state={"updates": 4},
    )

    restored, payload = load_checkpoint(
        path,
        dth_ruleset=compatibility,
        device=CPU,
    )
    assert payload["training_state"] == {"updates": 4}
    assert restored.config == config
    assert next(restored.parameters()).device.type == "cpu"
    for expected, actual in zip(model.parameters(), restored.parameters(), strict=True):
        assert torch.equal(expected, actual)

    changed_config = AggroHalConfig(
        hidden_size=14,
        head_hidden_size=9,
        tactical_logit_scale=config.tactical_logit_scale + 1.0,
    )
    with pytest.raises(ValueError, match="incompatible"):
        load_checkpoint(
            path,
            expected_config=changed_config,
            dth_ruleset=compatibility,
            device=CPU,
        )
    changed_artifact = {**compatibility, "table_digest": "different"}
    with pytest.raises(ValueError, match="incompatible"):
        load_checkpoint(
            path,
            expected_config=config,
            dth_ruleset=changed_artifact,
            device=CPU,
        )


def test_dth_compatibility_rejects_noncanonical_or_incomplete_metadata() -> None:
    agent = _StageAgent()
    agent.tablebase.metadata["canonical_table"] = False
    with pytest.raises(ValueError, match="canonical"):
        dth_compatibility(agent)

    agent.tablebase.metadata = _metadata()
    del agent.tablebase.metadata["table_digest"]
    with pytest.raises(ValueError, match="missing"):
        dth_compatibility(agent)
