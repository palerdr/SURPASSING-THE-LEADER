from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
import torch

from arena.policies.aggro_hal import (
    OBSERVATION_FEATURES,
    AggroHalConfig,
    AggroHalNetwork,
)
from arena.policies.aggro_memory_curriculum import (
    MEMORY_CURRICULUM_SCHEMA,
    TRAIN_MEMORY_CURRICULUM,
    VALIDATION_MEMORY_CURRICULUM,
    MemoryCurriculumCase,
    assert_split_disjointness,
    audit_exact_best_responses,
    build_memory_curriculum_case,
    build_memory_curriculum_role_pair,
    memory_curriculum_config_payload,
    memory_curriculum_config_sha256,
    memory_curriculum_generator_contract,
    memory_curriculum_generator_contract_sha256,
    mode_target_distributions,
    select_curriculum_parameters,
)
from dth.agent import CertifiedStageGame


def _crossing_matrix() -> np.ndarray:
    targets = mode_target_distributions()
    truth_a = np.asarray(targets["a"], dtype=np.float64)
    truth_b = np.asarray(targets["b"], dtype=np.float64)
    matrix = np.zeros((60, 60), dtype=np.float64)
    matrix[0] = truth_a
    matrix[1] = truth_b
    matrix[np.flatnonzero(truth_a), 2] = -1.0
    matrix[np.flatnonzero(truth_b), 3] = -1.0
    return matrix


class _ExactAgent:
    @staticmethod
    def stage_game(state: tuple[int, int, int, int]) -> CertifiedStageGame:
        uniform = np.full(60, 1.0 / 60.0, dtype=np.float64)
        return CertifiedStageGame(
            state=tuple(state),
            value=0.0,
            matrix=_crossing_matrix(),
            drop_policy=uniform.copy(),
            check_policy=uniform.copy(),
            saddle_gap=0.0,
        )


@pytest.mark.parametrize("role", ["dropper", "checker"])
def test_current_target_is_byte_identical_but_history_identifies_mode(
    role: str,
) -> None:
    case = build_memory_curriculum_case(
        _ExactAgent(),
        split="train",
        example_seed=7,
        role=role,  # type: ignore[arg-type]
    )

    assert case.mode_a.target.bitwise_equal(case.mode_b.target)
    assert case.target_sha256 == case.mode_a.target.sha256()
    assert case.mode_b.target.sha256() == case.target_sha256
    assert set(case.mode_a.cue_actions).isdisjoint(case.mode_b.cue_actions)
    assert not np.array_equal(case.mode_a.target_truth, case.mode_b.target_truth)

    action_features = np.asarray(
        [
            name.startswith("previous_drop_action_")
            or name.startswith("previous_check_action_")
            for name in OBSERVATION_FEATURES
        ]
    )
    differences = [
        left.features != right.features
        for left, right in zip(case.mode_a.tokens, case.mode_b.tokens, strict=True)
    ]
    assert any(np.any(difference) for difference in differences)
    assert all(not np.any(difference & ~action_features) for difference in differences)
    assert all(
        left.bitwise_equal(right)
        for left, right in zip(
            case.mode_a.tokens[-case.parameters.cover_games :],
            case.mode_b.tokens[-case.parameters.cover_games :],
            strict=True,
        )
    )


def test_exact_targets_conflict_in_every_role_and_mode() -> None:
    audit = audit_exact_best_responses(_crossing_matrix())

    assert audit.dropper_actions == (1, 2)
    assert audit.checker_actions == (3, 4)
    assert all(gap > 0.5 for gap in audit.dropper_gaps + audit.checker_gaps)
    assert audit.action("dropper", "a") != audit.action("dropper", "b")
    assert audit.action("checker", "a") != audit.action("checker", "b")

    dropper, checker = build_memory_curriculum_role_pair(
        _ExactAgent(), split="train", example_seed=11
    )
    assert dropper.role == "dropper"
    assert checker.role == "checker"
    assert dropper.best_responses == checker.best_responses == audit


def test_generator_and_tensor_batch_are_deterministic_and_network_ready() -> None:
    first = build_memory_curriculum_case(
        _ExactAgent(), split="train", example_seed=19, role="dropper"
    )
    second = build_memory_curriculum_case(
        _ExactAgent(), split="train", example_seed=19, role="dropper"
    )

    assert first.parameters == second.parameters
    assert first.target_sha256 == second.target_sha256
    assert all(
        left.bitwise_equal(right)
        for left, right in zip(first.mode_a.tokens, second.mode_a.tokens, strict=True)
    )

    batch = first.to_batch()
    repeat = second.to_batch()
    assert all(
        torch.equal(batch.network_inputs()[name], repeat.network_inputs()[name])
        for name in batch.network_inputs()
    )
    assert batch.features.shape[0] == 2
    assert batch.target_mask[:, -1].tolist() == [True, True]
    assert int(batch.target_mask[:, :-1].sum()) == 0
    assert batch.best_response_actions.tolist() == [0, 1]

    model = AggroHalNetwork(
        AggroHalConfig(hidden_size=16, head_hidden_size=8, tactical_logit_scale=1.0)
    )
    output = model(**batch.network_inputs())
    assert output.policy.shape == (2, first.target_index + 1, 60)
    assert output.opponent_policy.shape == output.policy.shape
    assert output.hidden_state.shape == (2, 2, 16)


def test_train_and_validation_seed_and_parameter_namespaces_are_disjoint() -> None:
    assert_split_disjointness()
    assert TRAIN_MEMORY_CURRICULUM.cover_games == (2, 4, 6)
    assert VALIDATION_MEMORY_CURRICULUM.cover_games == (8,)
    assert set(TRAIN_MEMORY_CURRICULUM.example_seeds).isdisjoint(
        VALIDATION_MEMORY_CURRICULUM.example_seeds
    )
    for name, train_values in TRAIN_MEMORY_CURRICULUM.parameter_support.items():
        assert train_values.isdisjoint(
            VALIDATION_MEMORY_CURRICULUM.parameter_support[name]
        )

    train = select_curriculum_parameters("train", example_seed=0, role="dropper")
    validation = select_curriculum_parameters(
        "validation", example_seed=10_000, role="dropper"
    )
    assert train.session_seed != validation.session_seed
    assert set(train.cue_actions_a + train.cue_actions_b).isdisjoint(
        validation.cue_actions_a + validation.cue_actions_b
    )
    with pytest.raises(ValueError, match="not registered"):
        select_curriculum_parameters("train", example_seed=10_000, role="dropper")

    payload = memory_curriculum_config_payload("validation")
    assert payload == memory_curriculum_config_payload("validation")
    assert payload["loss_contract"] == {
        "prefix_consumes_recurrence": True,
        "supervision": "target_only",
        "target_input_identical_across_modes": True,
        "required_roles": ["dropper", "checker"],
    }
    contract = memory_curriculum_generator_contract()
    assert contract["environment"]["max_half_rounds"] == 1
    assert payload["generator_contract"] == contract
    assert payload["generator_contract_sha256"] == (
        memory_curriculum_generator_contract_sha256()
    )
    digest = memory_curriculum_config_sha256("validation")
    assert digest == memory_curriculum_config_sha256(VALIDATION_MEMORY_CURRICULUM)
    assert len(digest) == 64


def test_case_schema_and_target_digest_fail_closed() -> None:
    case = build_memory_curriculum_case(
        _ExactAgent(), split="train", example_seed=3, role="checker"
    )
    assert isinstance(case, MemoryCurriculumCase)
    assert case.schema_version == MEMORY_CURRICULUM_SCHEMA
    with pytest.raises(ValueError, match="unsupported"):
        replace(case, schema_version="arena-aggro-hal-memory-curriculum-v0")
    with pytest.raises(ValueError, match="digest"):
        replace(case, target_sha256="0" * 64)
