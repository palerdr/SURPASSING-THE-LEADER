from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace

import numpy as np
import pytest
import torch

from arena.policies.aggro_hal import (
    OBSERVATION_FEATURES,
    AggroHalConfig,
    AggroHalNetwork,
)
from arena.policies.evaluate_aggro_hal_memory import (
    LatentTwinCase,
    adaptation_conclusion,
    build_latent_twin_case,
    cluster_bootstrap_interval,
    evaluate_latent_twin_case,
    latent_twin_protocol,
    symmetric_crossover_metrics,
)
from dth.agent import CertifiedStageGame


def _crossing_matrix() -> np.ndarray:
    protocol = latent_twin_protocol(twin_seeds=(0,), cover_games=(1,))
    truth_a = _protocol_distribution(protocol, "a")
    truth_b = _protocol_distribution(protocol, "b")
    matrix = np.zeros((60, 60), dtype=np.float64)
    matrix[0] = truth_a
    matrix[1] = truth_b
    matrix[np.flatnonzero(truth_a), 2] = -1.0
    matrix[np.flatnonzero(truth_b), 3] = -1.0
    return matrix


class _ExactAgent:
    @staticmethod
    def stage_game(state) -> CertifiedStageGame:
        uniform = np.full(60, 1.0 / 60.0, dtype=np.float64)
        return CertifiedStageGame(
            state=tuple(state),
            value=0.0,
            matrix=_crossing_matrix(),
            drop_policy=uniform.copy(),
            check_policy=uniform.copy(),
            saddle_gap=0.0,
        )


@pytest.fixture(scope="module")
def twin_case() -> LatentTwinCase:
    return build_latent_twin_case(
        _ExactAgent(),
        role="dropper",
        cover_games=3,
        twin_seed=7,
    )


def _protocol_distribution(protocol: Mapping[str, object], mode: str) -> np.ndarray:
    raw_modes = protocol["mode_targets"]
    assert isinstance(raw_modes, Mapping)
    raw = raw_modes[mode]
    assert isinstance(raw, Mapping)
    result = np.zeros(60, dtype=np.float64)
    for action, probability in raw.items():
        result[int(action) - 1] = float(probability)
    return result


def test_latent_twin_target_is_frozen_after_a_common_suffix(
    twin_case: LatentTwinCase,
) -> None:
    assert twin_case.target_sha256 == twin_case.target.sha256()
    assert twin_case.target_state == (0, 0, 0, 0)
    assert len(twin_case.prefix_a) == len(twin_case.prefix_b)
    assert not twin_case.prefix_a[-twin_case.cover_games].bitwise_equal(
        twin_case.prefix_b[-twin_case.cover_games]
    )

    common_suffix = twin_case.cover_games - 1
    assert common_suffix > 0
    assert all(
        left.bitwise_equal(right)
        for left, right in zip(
            twin_case.prefix_a[-common_suffix:],
            twin_case.prefix_b[-common_suffix:],
            strict=True,
        )
    )

    feature = {name: index for index, name in enumerate(OBSERVATION_FEATURES)}
    target = twin_case.target.features
    assert target[feature["current_new_game"]] == 1.0
    assert target[feature["previous_reveal_present"]] == 1.0
    assert (
        target[feature["previous_drop_action_30"]]
        + target[feature["previous_check_action_30"]]
        == 1.0
    )

    another_seed = build_latent_twin_case(
        _ExactAgent(),
        role="dropper",
        cover_games=twin_case.cover_games,
        twin_seed=8,
    )
    assert twin_case.target.bitwise_equal(another_seed.target)
    assert twin_case.target_sha256 == another_seed.target_sha256

    different_dtype = replace(
        twin_case.target,
        features=twin_case.target.features.astype(np.float64),
    )
    assert not twin_case.target.bitwise_equal(different_dtype)
    positive_zeros = np.flatnonzero(
        (twin_case.target.features == 0.0) & ~np.signbit(twin_case.target.features)
    )
    assert positive_zeros.size > 0
    signed_zero_features = twin_case.target.features.copy()
    signed_zero_features[positive_zeros[0]] = np.float32(-0.0)
    signed_zero = replace(twin_case.target, features=signed_zero_features)
    assert not twin_case.target.bitwise_equal(signed_zero)


def test_protocol_targets_require_opposite_responses_in_both_roles() -> None:
    protocol = latent_twin_protocol(twin_seeds=(3, 4), cover_games=(1, 3))
    truth_a = _protocol_distribution(protocol, "a")
    truth_b = _protocol_distribution(protocol, "b")
    assert truth_a.sum() == pytest.approx(1.0)
    assert truth_b.sum() == pytest.approx(1.0)
    assert not np.array_equal(truth_a, truth_b)

    # Embed two independent crossing games in one 60x60 matrix. Rows 1 and 2
    # distinguish the two Checker targets; columns 3 and 4 distinguish the
    # same distributions when they are Dropper targets.
    matrix = _crossing_matrix()

    drop_a, drop_b = matrix @ truth_a, matrix @ truth_b
    check_a, check_b = -matrix.T @ truth_a, -matrix.T @ truth_b
    assert (int(np.argmax(drop_a)), int(np.argmax(drop_b))) == (0, 1)
    assert (int(np.argmax(check_a)), int(np.argmax(check_b))) == (2, 3)
    assert drop_a[0] - drop_a[1] > 0.5
    assert drop_b[1] - drop_b[0] > 0.5
    assert check_a[2] - check_a[3] > 0.5
    assert check_b[3] - check_b[2] > 0.5


def test_symmetric_crossover_metrics_preserve_direction_and_mode_signs() -> None:
    metrics = symmetric_crossover_metrics(
        payoff_a_correct=5.0,
        payoff_a_swapped=1.0,
        payoff_a_zero=2.0,
        payoff_b_correct=7.0,
        payoff_b_swapped=3.0,
        payoff_b_zero=4.0,
        nll_a_correct=0.2,
        nll_a_swapped=0.8,
        nll_a_zero=0.5,
        nll_b_correct=0.4,
        nll_b_swapped=1.0,
        nll_b_zero=0.7,
    )

    assert metrics == pytest.approx(
        {
            "payoff_correct_minus_swapped": 4.0,
            "payoff_correct_minus_zero": 3.0,
            "nll_swapped_minus_correct": 0.6,
            "nll_zero_minus_correct": 0.3,
            "mode_a_nll_swapped_minus_correct": 0.6,
            "mode_b_nll_swapped_minus_correct": 0.6,
            "mode_a_payoff_correct_minus_swapped": 4.0,
            "mode_b_payoff_correct_minus_swapped": 4.0,
        }
    )


def test_cluster_bootstrap_is_seed_clustered_order_invariant_and_deterministic() -> (
    None
):
    values = {10: 1.0, 2: -1.0, 7: 3.0}
    reverse_order = dict(reversed(tuple(values.items())))

    first = cluster_bootstrap_interval(values, replicates=2_000, seed=91)
    second = cluster_bootstrap_interval(values, replicates=2_000, seed=91)
    reordered = cluster_bootstrap_interval(reverse_order, replicates=2_000, seed=91)

    assert first == second == reordered
    assert cluster_bootstrap_interval(
        {1: 2.5, 9: 2.5}, replicates=100, seed=4
    ) == pytest.approx((2.5, 2.5))


def test_conclusion_rejects_a_pooled_gain_with_wrong_role_signs() -> None:
    def metric(mean: float, low: float, high: float) -> dict[str, object]:
        return {"mean": mean, "bootstrap_95": [low, high]}

    primary = {
        "metrics": {
            "payoff_correct_minus_swapped": metric(1.0, 0.8, 1.2),
            "mode_a_payoff_correct_minus_swapped": metric(1.0, 0.8, 1.2),
            "mode_b_payoff_correct_minus_swapped": metric(1.0, 0.8, 1.2),
            "nll_swapped_minus_correct": metric(0.0, -0.001, 0.001),
            "normalized_payoff_crossover": metric(0.005, 0.004, 0.006),
            "policy_total_variation": metric(0.001, 0.0009, 0.0011),
            "forecast_total_variation": metric(0.001, 0.0009, 0.0011),
        }
    }

    def role_summary(sign: float, *, outside_rope: bool = False) -> dict[str, object]:
        normalized = 0.03 if outside_rope else 0.005
        return {
            "metrics": {
                "mode_a_payoff_correct_minus_swapped": metric(
                    sign, sign - 0.1, sign + 0.1
                ),
                "mode_b_payoff_correct_minus_swapped": metric(
                    sign, sign - 0.1, sign + 0.1
                ),
                "mode_a_nll_swapped_minus_correct": metric(0.0, -0.001, 0.001),
                "mode_b_nll_swapped_minus_correct": metric(0.0, -0.001, 0.001),
                "mode_a_normalized_payoff_crossover": metric(
                    normalized, normalized - 0.001, normalized + 0.001
                ),
                "mode_b_normalized_payoff_crossover": metric(
                    normalized, normalized - 0.001, normalized + 0.001
                ),
                "policy_total_variation": metric(0.001, 0.0009, 0.0011),
            }
        }

    conclusion = adaptation_conclusion(
        {
            "by_cover": {"8": primary},
            "by_cover_and_role": {
                "8": {
                    "dropper": role_summary(1.0),
                    "checker": role_summary(-1.0, outside_rope=True),
                }
            },
        },
        primary_cover=8,
        thresholds={
            "normalized_payoff_crossover": 0.02,
            "opponent_nll_nats": 0.01,
            "policy_total_variation": 0.01,
        },
    )

    assert conclusion["pooled_directional_action_effect_detected"] is True
    assert conclusion["directional_action_effect_detected"] is False
    assert conclusion["causal_recurrent_action_use_supported"] is False
    assert conclusion["practical_no_adaptation_equivalence_supported"] is False


def test_small_recurrent_network_produces_finite_probe_metrics(
    twin_case: LatentTwinCase,
) -> None:
    torch.manual_seed(23)
    model = AggroHalNetwork(AggroHalConfig(hidden_size=8, head_hidden_size=6)).cpu()

    result = evaluate_latent_twin_case(model, twin_case)

    assert result["target_identity_passed"] is True
    contrasts = result["contrasts"]
    assert isinstance(contrasts, Mapping)
    assert all(np.isfinite(float(value)) for value in contrasts.values())
    assert 0.0 <= float(contrasts["policy_total_variation"]) <= 1.0
    assert 0.0 <= float(contrasts["forecast_total_variation"]) <= 1.0
    assert float(contrasts["pre_target_hidden_l2"]) >= 0.0
    assert float(contrasts["post_target_hidden_l2"]) >= 0.0

    modes = result["modes"]
    assert isinstance(modes, Mapping)
    for mode in ("a", "b"):
        arms = modes[mode]
        assert isinstance(arms, Mapping)
        for condition in ("correct", "swapped", "zero"):
            score = arms[condition]
            assert isinstance(score, Mapping)
            for key in (
                "expected_payoff",
                "oracle_payoff",
                "oracle_regret",
                "opponent_expected_nll",
                "best_response_mass",
                "top_action_probability",
                "direct_weight",
            ):
                assert np.isfinite(float(score[key]))
            assert 1 <= int(score["top_action"]) <= 60
            assert 0.0 <= float(score["top_action_probability"]) <= 1.0
            assert 0.0 <= float(score["direct_weight"]) <= 1.0


def test_probe_rejects_a_non_cpu_model_without_querying_cuda(
    twin_case: LatentTwinCase,
) -> None:
    model = AggroHalNetwork(AggroHalConfig(hidden_size=4, head_hidden_size=4)).to(
        "meta"
    )

    with pytest.raises(ValueError, match="CPU-only"):
        evaluate_latent_twin_case(model, twin_case)
