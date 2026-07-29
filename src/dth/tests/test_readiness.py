from __future__ import annotations

import pytest
from omegaconf import OmegaConf

from dth.readiness import compare_ladders, depth_gate, orientation_gate


ROOTS = (
    ((239, 0, 0, 240), 5),
    ((0, 240, 239, 0), 5),
    ((239, 0, 0, 240), 4),
    ((238, 0, 1, 240), 4),
)


def _report(gaps: tuple[float, ...], errors: tuple[float, ...]) -> dict:
    records = []
    for budget in (0, 256, 1024, 4096):
        for seed in (0, 1, 2):
            for (state, horizon), gap, error in zip(ROOTS, gaps, errors, strict=True):
                records.append(
                    {
                        "budget": budget,
                        "seed": seed,
                        "evaluator": "network",
                        "state": list(state),
                        "horizon": horizon,
                        "saddle_gap": gap,
                        "value_error": error,
                        "mcts_value": 0.25,
                    }
                )
    return {"records": records}


def _depth_report(
    depth: int,
    gaps: tuple[float, ...],
    *,
    checkpoint: str = "src/dth/checkpoints/depth_baseline_v1/best.pt",
    with_fallbacks: bool = True,
) -> dict:
    records = []
    for (state, horizon), gap in zip(ROOTS, gaps, strict=True):
        record = {
            "budget": 0,
            "seed": 0,
            "evaluator": "network",
            "state": list(state),
            "horizon": horizon,
            "saddle_gap": gap,
            "value_error": gap / 2.0,
            "mcts_value": 0.25,
        }
        if with_fallbacks:
            record["lp_fallbacks"] = 0
        records.append(record)
    return {
        "checkpoint": checkpoint,
        "config": {"mcts": {"max_depth": depth}},
        "records": records,
    }


def test_depth_gate_accepts_a_strictly_improving_resolve_ladder() -> None:
    reports = [
        _depth_report(1, (0.20, 0.10, 0.15, 0.25)),
        _depth_report(2, (0.10, 0.05, 0.08, 0.20)),
        _depth_report(3, (0.05, 0.02, 0.04, 0.10)),
    ]

    verdict = depth_gate(reports)

    assert verdict["depths"] == [1, 2, 3]
    assert all(verdict["gates"].values())
    assert verdict["recommendation"] == "depth-effective"


def test_depth_gate_tail_metric_ignores_zero_inflation() -> None:
    # Depth one leaves half the pack at exactly zero gap; depth two spreads
    # tiny gaps onto those roots while improving every material one.  The
    # zero-inflated median regresses, the gated tail mean must not.
    reports = [
        _depth_report(1, (0.28, 0.22, 0.0, 0.0)),
        _depth_report(2, (0.21, 0.14, 0.09, 0.08)),
    ]

    verdict = depth_gate(reports)

    assert verdict["per_depth"][0]["median_gap"] < verdict["per_depth"][1]["median_gap"]
    assert verdict["gates"]["cvar_gap_strictly_decreasing"]
    assert verdict["recommendation"] == "depth-effective"


def test_depth_gate_rejects_a_flat_ladder() -> None:
    reports = [
        _depth_report(1, (0.20, 0.10, 0.15, 0.25)),
        _depth_report(2, (0.20, 0.10, 0.15, 0.25)),
    ]

    verdict = depth_gate(reports)

    assert not verdict["gates"]["cvar_gap_strictly_decreasing"]
    assert verdict["recommendation"] == "depth-ineffective"


def test_depth_gate_fails_closed_without_fallback_counts() -> None:
    reports = [
        _depth_report(1, (0.20, 0.10, 0.15, 0.25)),
        _depth_report(2, (0.10, 0.05, 0.08, 0.20), with_fallbacks=False),
    ]

    with pytest.raises(ValueError, match="uncounted fallbacks"):
        depth_gate(reports)


def test_depth_gate_rejects_mixed_checkpoints_and_unordered_depths() -> None:
    with pytest.raises(ValueError, match="one checkpoint"):
        depth_gate(
            [
                _depth_report(1, (0.2, 0.1, 0.15, 0.25)),
                _depth_report(2, (0.1, 0.05, 0.08, 0.2), checkpoint="other.pt"),
            ]
        )
    with pytest.raises(ValueError, match="strictly increasing"):
        depth_gate(
            [
                _depth_report(2, (0.1, 0.05, 0.08, 0.2)),
                _depth_report(1, (0.2, 0.1, 0.15, 0.25)),
            ]
        )


def _orientation_report(gap_by_root: dict[tuple[tuple[int, ...], int], float]) -> dict:
    records = []
    for (state, horizon), gap in gap_by_root.items():
        records.append(
            {
                "budget": 0,
                "seed": 0,
                "evaluator": "network",
                "state": list(state),
                "horizon": horizon,
                "saddle_gap": gap,
                "value_error": gap / 2.0,
                "mcts_value": 0.25,
                "lp_fallbacks": 0,
            }
        )
    return {
        "checkpoint": "src/dth/checkpoints/depth_baseline_v1/best.pt",
        "config": {"mcts": {"max_depth": 1}},
        "records": records,
    }


def test_orientation_gate_accepts_balanced_and_near_exact_pairs() -> None:
    report = _orientation_report(
        {
            ((179, 60, 59, 180), 4): 0.10,
            ((59, 180, 179, 60), 4): 0.12,
            ((219, 0, 19, 240), 3): 1e-9,
            ((19, 240, 219, 0), 3): 5e-9,
        }
    )

    verdict = orientation_gate(report)

    assert len(verdict["pairs"]) == 2
    assert verdict["recommendation"] == "orientation-consistent"


def test_orientation_gate_flags_a_five_fold_mirror_imbalance() -> None:
    report = _orientation_report(
        {
            ((179, 60, 59, 180), 4): 0.2734,
            ((59, 180, 179, 60), 4): 0.0547,
        }
    )

    verdict = orientation_gate(report)

    assert not verdict["gates"]["orientation_pairs_consistent"]
    assert verdict["recommendation"] == "orientation-inconsistent"


def test_compare_ladders_accepts_a_seed_stable_generalizing_candidate() -> None:
    baseline = _report((0.20, 0.10, 0.15, 0.25), (0.10,) * 4)
    candidate = _report((0.16, 0.08, 0.12, 0.20), (0.08,) * 4)

    comparison = compare_ladders(baseline, candidate)

    assert comparison["primary_budget"] == 4096
    assert all(comparison["gates"].values())


def test_compare_ladders_rejects_different_root_sets() -> None:
    baseline = _report((0.20, 0.10, 0.15, 0.25), (0.10,) * 4)
    candidate = _report((0.16, 0.08, 0.12, 0.20), (0.08,) * 4)
    candidate["records"][0]["state"] = [1, 2, 3, 4]

    with pytest.raises(ValueError, match="roots differ"):
        compare_ladders(baseline, candidate)


def test_v2_development_roots_are_mirrored_and_frozen_disjoint() -> None:
    configured = set()
    for path, horizon in (
        ("src/dth/config/readiness_development_h3_v2.yaml", 3),
        ("src/dth/config/readiness_development_h4a_v2.yaml", 4),
        ("src/dth/config/readiness_development_h4_v2.yaml", 4),
    ):
        config = OmegaConf.load(path)
        configured.update((tuple(state), horizon) for state in config.root_states)
    frozen = {
        ((238, 0, 1, 240), 4),
        ((1, 240, 238, 0), 4),
        ((179, 60, 59, 180), 4),
        ((59, 180, 179, 60), 4),
        ((219, 0, 19, 240), 3),
        ((19, 240, 219, 0), 3),
        ((119, 120, 179, 60), 3),
        ((179, 60, 119, 120), 3),
    }

    assert len(configured) == 24
    assert configured.isdisjoint(frozen)
    assert all(
        ((state[2], state[3], state[0], state[1]), horizon) in configured
        for state, horizon in configured
    )


def test_value_only_selection_roots_are_disjoint_from_training_and_frozen_roots() -> None:
    config = OmegaConf.load("src/dth/config/train_value_only_bellman_composition_v2.yaml")
    frozen_config = OmegaConf.load("src/dth/config/mcts_readiness_mixed_v7_v1.yaml")
    selected = {
        (tuple(root.state), int(root.horizon))
        for root in config.decision_selection.roots
    }
    guarded = {
        (tuple(root.state), int(root.horizon))
        for root in config.decision_selection.guard_roots
    }
    trained = {
        (tuple(root.state), int(root.horizon)) for root in config.decision_loss.roots
    }
    frozen = {
        (tuple(root.state), int(root.horizon)) for root in frozen_config.roots
    }

    assert selected == guarded
    assert len(selected) == 12
    assert selected.isdisjoint(trained)
    assert selected.isdisjoint(frozen)
