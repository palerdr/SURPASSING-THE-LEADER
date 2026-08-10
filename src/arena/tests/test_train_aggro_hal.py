from __future__ import annotations

from dataclasses import asdict, replace

import numpy as np
import pytest
import torch

import arena.policies.train_aggro_hal as train_aggro_hal
from arena.policies.aggro_hal import (
    ACTION_COUNT,
    OBSERVATION_DIM,
    AggroHalConfig,
    AggroHalNetwork,
)
from arena.policies.train_aggro_hal import (
    SESSION_COLLECTOR_BINDING_SCHEMA,
    AggroTrainerConfig,
    TrainingSequence,
    _default_session_collector_binding,
    _validate_resume_session_collector,
    _validate_resume_trainer_config,
    collect_teacher_session,
    generalized_advantages,
    pad_sequences,
    warmstart_step,
)
from arena.policies.opponent_league import EARLY, LATE
from dth.agent import CertifiedStageGame

CPU = torch.device("cpu")


class _ExactStageStub:
    def stage_game(self, state: tuple[int, int, int, int]) -> CertifiedStageGame:
        matrix = np.zeros((ACTION_COUNT, ACTION_COUNT), dtype=np.float64)
        drop = np.zeros(ACTION_COUNT, dtype=np.float64)
        check = np.zeros(ACTION_COUNT, dtype=np.float64)
        drop[2] = 1.0
        check[3] = 1.0
        return CertifiedStageGame(
            state=state,
            value=0.0,
            matrix=matrix,
            drop_policy=drop,
            check_policy=check,
            saddle_gap=0.0,
        )


def _single_step_sequence() -> TrainingSequence:
    features = np.zeros((1, OBSERVATION_DIM), dtype=np.float32)
    matrices = np.zeros((1, ACTION_COUNT, ACTION_COUNT), dtype=np.float32)
    matrices[:, 11, 5] = 1.0
    exact = np.full((1, ACTION_COUNT), 1.0 / ACTION_COUNT, dtype=np.float32)
    opponent = np.zeros((1, ACTION_COUNT), dtype=np.float32)
    opponent[:, 5] = 1.0
    return TrainingSequence(
        features=features,
        stage_matrices=matrices,
        exact_policies=exact,
        roles_are_dropper=np.ones(1, dtype=np.bool_),
        legal_masks=np.ones((1, ACTION_COUNT), dtype=np.bool_),
        opponent_targets=opponent,
        opponent_actions=np.asarray([5], dtype=np.int64),
        learner_actions=np.asarray([11], dtype=np.int64),
        rewards=np.asarray([1.0], dtype=np.float32),
        returns=np.asarray([1.0], dtype=np.float32),
    )


class _InjectedCollector:
    def __init__(self, *, mode: str) -> None:
        self.mode = mode
        self.calls: list[tuple[int, bool]] = []

    def checkpoint_binding(self) -> dict[str, object]:
        return {
            "schema_version": SESSION_COLLECTOR_BINDING_SCHEMA,
            "identity": "test.memory-necessity-curriculum-v1",
            "config": {"mode": self.mode},
        }

    def collect_update(self, **kwargs: object) -> list[TrainingSequence]:
        self.calls.append((int(kwargs["update_index"]), bool(kwargs["on_policy"])))
        return [_single_step_sequence()]


def test_gae_carries_credit_across_game_boundaries_inside_one_session() -> None:
    rewards = np.asarray([1.0, 0.0, -1.0], dtype=np.float32)
    values = np.zeros(3, dtype=np.float32)

    advantages, returns = generalized_advantages(
        rewards, values, gamma=1.0, gae_lambda=1.0
    )

    np.testing.assert_allclose(advantages, [0.0, -1.0, -1.0])
    np.testing.assert_allclose(returns, advantages)


def test_one_cpu_warmstart_step_updates_the_recurrent_model() -> None:
    torch.manual_seed(9)
    length = 3
    features = np.zeros((length, OBSERVATION_DIM), dtype=np.float32)
    matrices = np.zeros((length, ACTION_COUNT, ACTION_COUNT), dtype=np.float32)
    matrices[:, 11, 5] = 1.0
    exact = np.full((length, ACTION_COUNT), 1.0 / ACTION_COUNT, dtype=np.float32)
    opponent = np.zeros((length, ACTION_COUNT), dtype=np.float32)
    opponent[:, 5] = 1.0
    sequence = TrainingSequence(
        features=features,
        stage_matrices=matrices,
        exact_policies=exact,
        roles_are_dropper=np.ones(length, dtype=np.bool_),
        legal_masks=np.ones((length, ACTION_COUNT), dtype=np.bool_),
        opponent_targets=opponent,
        opponent_actions=np.full(length, 5, dtype=np.int64),
        learner_actions=np.full(length, 11, dtype=np.int64),
        rewards=np.asarray([0.0, 0.0, 1.0], dtype=np.float32),
        returns=np.asarray([1.0, 1.0, 1.0], dtype=np.float32),
    )
    model = AggroHalNetwork(
        AggroHalConfig(hidden_size=12, head_hidden_size=8, tactical_logit_scale=4.0)
    ).to(CPU)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    trainer = AggroTrainerConfig(
        warmstart_updates=1,
        ppo_updates=0,
        sessions_per_update=1,
        games_per_session=1,
        max_half_rounds=1,
        cpu_threads=1,
    )
    before = next(model.parameters()).detach().clone()

    metrics = warmstart_step(
        model,
        optimizer,
        pad_sequences([sequence], device=CPU),
        trainer,
    )

    assert all(np.isfinite(value) for value in metrics.values())
    assert metrics["total_loss"] > 0.0
    assert "tactical" in metrics
    assert "direct_tactical" in metrics
    assert "gate" not in metrics
    assert "teacher" not in metrics
    assert not torch.equal(before, next(model.parameters()).detach())


def test_objective_weights_keep_prefix_steps_as_context_only() -> None:
    base = _single_step_sequence()
    length = 3
    sequence = TrainingSequence(
        features=np.repeat(base.features, length, axis=0),
        stage_matrices=np.repeat(base.stage_matrices, length, axis=0),
        exact_policies=np.repeat(base.exact_policies, length, axis=0),
        roles_are_dropper=np.repeat(base.roles_are_dropper, length, axis=0),
        legal_masks=np.repeat(base.legal_masks, length, axis=0),
        opponent_targets=np.repeat(base.opponent_targets, length, axis=0),
        opponent_actions=np.repeat(base.opponent_actions, length, axis=0),
        learner_actions=np.repeat(base.learner_actions, length, axis=0),
        rewards=np.zeros(length, dtype=np.float32),
        returns=np.zeros(length, dtype=np.float32),
        objective_weights=np.asarray([0.0, 0.0, 1.0], dtype=np.float32),
    )

    weighted = pad_sequences([sequence], device=CPU)
    legacy = pad_sequences([base], device=CPU)

    torch.testing.assert_close(
        weighted.objective_weights,
        torch.tensor([[0.0, 0.0, 1.0]], dtype=torch.float32),
    )
    torch.testing.assert_close(
        legacy.objective_weights,
        torch.ones((1, 1), dtype=torch.float32),
    )


def test_trainer_keeps_high_clocks_on_the_pure_dth_action_contract() -> None:
    trainer = AggroTrainerConfig(start_clocks=(3540,))
    assert trainer.start_clocks == (3540,)


def test_warmstart_behavior_action_does_not_depend_on_privileged_truth() -> None:
    trainer = AggroTrainerConfig(
        games_per_session=1,
        max_half_rounds=1,
        sessions_per_update=1,
        warmstart_updates=1,
        ppo_updates=0,
        cpu_threads=1,
    )
    common = {
        "exact_agent": _ExactStageStub(),
        "trainer": trainer,
        "session_seed": 1234,
        "learner_starts_in_hal_seat": True,
    }

    early = collect_teacher_session(family=EARLY, opponent_seed=1301, **common)
    late = collect_teacher_session(family=LATE, opponent_seed=1401, **common)

    np.testing.assert_array_equal(early.learner_actions, late.learner_actions)
    assert not np.allclose(early.opponent_targets, late.opponent_targets)


def test_resume_allows_only_targets_and_runtime_fields_to_change() -> None:
    stored_config = AggroTrainerConfig(
        warmstart_updates=2,
        ppo_updates=1,
        sessions_per_update=1,
        games_per_session=1,
        max_half_rounds=1,
        cpu_threads=1,
    )
    stored = asdict(stored_config)
    allowed = replace(
        stored_config,
        warmstart_updates=4,
        ppo_updates=3,
        cpu_threads=2,
    )

    _validate_resume_trainer_config(stored, allowed)

    with pytest.raises(ValueError, match="gamma"):
        _validate_resume_trainer_config(stored, replace(allowed, gamma=0.9))
    with pytest.raises(ValueError, match="learning_rate"):
        _validate_resume_trainer_config(
            stored, replace(allowed, learning_rate=stored_config.learning_rate * 2.0)
        )


def test_resume_binds_session_collector_identity_and_config() -> None:
    default_binding = _default_session_collector_binding()
    _validate_resume_session_collector(None, default_binding)

    current = _InjectedCollector(mode="paired-hidden-modes").checkpoint_binding()
    _validate_resume_session_collector(current, current)

    with pytest.raises(ValueError, match="session collector is incompatible"):
        _validate_resume_session_collector(
            _InjectedCollector(mode="different-modes").checkpoint_binding(),
            current,
        )
    with pytest.raises(ValueError, match="session collector is incompatible"):
        _validate_resume_session_collector(None, current)


def test_train_accepts_injected_collector_for_warmstart_only(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path = tmp_path / "aggro-warmstart-only.yaml"
    config_path.write_text(
        """\
schema_version: arena-aggro-hal-training-config-v2
model:
  hidden_size: 12
  gru_layers: 2
  head_hidden_size: 8
  tactical_logit_scale: 4.0
training:
  seed: 17
  device: cpu
  cpu_threads: 1
  dth_artifact: unused-by-test
  games_per_session: 1
  max_half_rounds: 1
  start_clocks: [720]
  sessions_per_update: 1
  warmstart_updates: 1
  ppo_updates: 0
  ppo_epochs: 1
  snapshot_interval: 0
""",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        train_aggro_hal,
        "dth_compatibility",
        lambda _agent: {"schema_version": "test-dth-compatibility"},
    )
    collector = _InjectedCollector(mode="paired-hidden-modes")

    report = train_aggro_hal.train(
        config_path,
        tmp_path / "run",
        exact_agent_override=_ExactStageStub(),
        session_collector=collector,
    )

    assert collector.calls == [(0, False)]
    assert report["completed_warmstart_updates"] == 1
    assert report["completed_ppo_updates"] == 0
    assert report["session_collector"] == collector.checkpoint_binding()
    assert report["experiment"] is None
    payload = torch.load(
        tmp_path / "run" / "checkpoint.pt",
        map_location=CPU,
        weights_only=False,
    )
    assert (
        payload["training_state"]["session_collector"] == collector.checkpoint_binding()
    )
    assert payload["training_state"]["experiment"] is None

    initialized_collector = _InjectedCollector(mode="new-curriculum")
    initialized_report = train_aggro_hal.train(
        config_path,
        tmp_path / "initialized-run",
        initial_checkpoint=tmp_path / "run" / "checkpoint.pt",
        exact_agent_override=_ExactStageStub(),
        session_collector=initialized_collector,
    )

    assert initialized_collector.calls == [(0, False)]
    assert initialized_report["completed_warmstart_updates"] == 1
    assert initialized_report["completed_ppo_updates"] == 0
    assert initialized_report["initial_checkpoint"]["path"].endswith("checkpoint.pt")
    assert len(initialized_report["initial_checkpoint"]["sha256"]) == 64


def test_train_rejects_initial_checkpoint_with_resume() -> None:
    with pytest.raises(ValueError, match="mutually exclusive"):
        train_aggro_hal.train(
            "unused.yaml",
            "unused-output",
            resume="resume.pt",
            initial_checkpoint="initial.pt",
        )
