from __future__ import annotations

from copy import deepcopy

import numpy as np
import pytest
import torch

from arena.policies.aggro_hal import AggroHalConfig, AggroHalNetwork
from arena.policies.aggro_memory_curriculum import (
    VALIDATION_MEMORY_CURRICULUM,
    build_memory_curriculum_case,
    memory_curriculum_config_sha256,
    mode_target_distributions,
)
from arena.policies.evaluate_aggro_hal_adaptive import (
    PROMOTION_THRESHOLDS,
    adaptive_memory_protocol,
    adaptive_promotion_decision,
    cluster_bootstrap_interval,
    evaluate_adaptive_memory_case,
    summarize_adaptive_memory_rows,
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
def test_case_uses_identical_target_for_all_memory_interventions(role: str) -> None:
    torch.manual_seed(71)
    model = AggroHalNetwork(
        AggroHalConfig(hidden_size=8, head_hidden_size=6, tactical_logit_scale=1.0)
    ).cpu()
    case = build_memory_curriculum_case(
        _ExactAgent(),
        split="validation",
        example_seed=10_000,
        role=role,  # type: ignore[arg-type]
    )

    rows = evaluate_adaptive_memory_case(model, case)

    assert [row["mode"] for row in rows] == ["a", "b"]
    assert all(row["target_integrity_passed"] is True for row in rows)
    assert all(row["cover_games"] == 8 for row in rows)
    assert all(row["history_free_matches_zero_target"] is True for row in rows)
    assert all(
        float(row["history_free_zero_target_max_abs_difference"]) <= 1e-7
        for row in rows
    )
    for row in rows:
        scores = row["scores"]
        contrasts = row["contrasts"]
        assert isinstance(scores, dict)
        assert isinstance(contrasts, dict)
        assert set(scores) == {
            "correct",
            "swapped",
            "zero_target",
            "history_free",
        }
        assert set(contrasts) == {
            "correct_vs_swapped",
            "correct_vs_zero_target",
            "correct_vs_history_free",
        }
        assert scores["zero_target"] == pytest.approx(scores["history_free"])
        for comparison in contrasts.values():
            assert np.isfinite(float(comparison["normalized_payoff_gain"]))
            assert np.isfinite(float(comparison["nll_gain_nats"]))


def test_example_seed_cluster_bootstrap_is_deterministic_and_order_invariant() -> None:
    values = {10_002: 0.08, 10_000: 0.02, 10_001: 0.05}
    reversed_values = dict(reversed(tuple(values.items())))

    first = cluster_bootstrap_interval(values, replicates=2_000, seed=93)
    second = cluster_bootstrap_interval(values, replicates=2_000, seed=93)
    reordered = cluster_bootstrap_interval(reversed_values, replicates=2_000, seed=93)

    assert first == second == reordered
    assert cluster_bootstrap_interval(
        {7: 0.04, 9: 0.04}, replicates=100, seed=1
    ) == pytest.approx((0.04, 0.04))


def _promotion_rows(
    *,
    payoff: float = 0.04,
    nll: float = 0.03,
    cover_games: int = 8,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for role in ("dropper", "checker"):
        for mode in ("a", "b"):
            for example_seed in (10_000, 10_001):
                comparisons = {
                    f"correct_vs_{baseline}": {
                        "normalized_payoff_gain": payoff,
                        "nll_gain_nats": nll,
                    }
                    for baseline in ("swapped", "zero_target", "history_free")
                }
                rows.append(
                    {
                        "split": "validation",
                        "example_seed": example_seed,
                        "role": role,
                        "mode": mode,
                        "cover_games": cover_games,
                        "target_integrity_passed": True,
                        "history_free_matches_zero_target": True,
                        "contrasts": comparisons,
                    }
                )
    return rows


def _summary(rows: list[dict[str, object]]) -> dict[str, object]:
    return summarize_adaptive_memory_rows(
        rows,
        expected_example_seeds=(10_000, 10_001),
        bootstrap_replicates=500,
        bootstrap_seed=37,
    )


def test_promotion_requires_every_role_mode_cell_without_pooled_rescue() -> None:
    passing_rows = _promotion_rows()
    passing = adaptive_promotion_decision(_summary(passing_rows), split="validation")
    assert passing["passed"] is True
    assert passing["decision"] == "unlock_ppo"
    assert passing["pooled_rescue_allowed"] is False

    failing_rows = deepcopy(passing_rows)
    for row in failing_rows:
        if row["role"] == "checker" and row["mode"] == "b":
            row["contrasts"]["correct_vs_swapped"][  # type: ignore[index]
                "normalized_payoff_gain"
            ] = -0.25
    failing = adaptive_promotion_decision(_summary(failing_rows), split="validation")

    assert failing["passed"] is False
    assert failing["decision"] == "hold_warmstart"
    assert failing["cells"]["checker/b"]["passed"] is False  # type: ignore[index]
    assert any(
        str(check).startswith("checker/b:correct_vs_swapped_payoff")
        for check in failing["failed_checks"]
    )


def test_promotion_thresholds_are_strict_and_cover_and_integrity_fail_closed() -> None:
    threshold_rows = _promotion_rows(
        payoff=PROMOTION_THRESHOLDS.normalized_payoff_gain,
        nll=PROMOTION_THRESHOLDS.nll_gain_nats,
    )
    threshold = adaptive_promotion_decision(
        _summary(threshold_rows), split="validation"
    )
    assert threshold["passed"] is False

    cover_rows = _promotion_rows(cover_games=7)
    cover = adaptive_promotion_decision(_summary(cover_rows), split="validation")
    assert cover["passed"] is False
    assert all(
        cell["checks"]["minimum_cover_delay"] is False
        for cell in cover["cells"].values()
    )

    integrity_rows = _promotion_rows()
    integrity_rows[0]["target_integrity_passed"] = False
    integrity = adaptive_promotion_decision(
        _summary(integrity_rows), split="validation"
    )
    assert integrity["passed"] is False
    assert any("target_integrity" in check for check in integrity["failed_checks"])

    wrong_split = adaptive_promotion_decision(
        _summary(_promotion_rows()), split="train"
    )
    assert wrong_split["passed"] is False
    assert "split:validation_required" in wrong_split["failed_checks"]


def test_validation_protocol_freezes_manifest_thresholds_and_no_pooling() -> None:
    protocol = adaptive_memory_protocol(
        "validation", bootstrap_replicates=123, bootstrap_seed=456
    )

    assert protocol["split"] == "validation"
    assert protocol["example_seeds"] == list(VALIDATION_MEMORY_CURRICULUM.example_seeds)
    assert protocol["curriculum_config_sha256"] == memory_curriculum_config_sha256(
        "validation"
    )
    assert protocol["bootstrap"] == {
        "unit": "example_seed",
        "replicates": 123,
        "seed": 456,
        "interval": "two-sided percentile 95%",
    }
    promotion = protocol["promotion"]
    assert promotion["pooled_rescue_allowed"] is False
    assert promotion["strict_inequality"] is True
    assert promotion["thresholds"] == {
        "normalized_payoff_gain": 0.02,
        "nll_gain_nats": 0.01,
        "minimum_cover_games": 8,
    }
