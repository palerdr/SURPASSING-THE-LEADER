from __future__ import annotations

import pytest
from omegaconf import OmegaConf

from dth.readiness import compare_ladders


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
        ("dth/config/readiness_development_h3_v2.yaml", 3),
        ("dth/config/readiness_development_h4a_v2.yaml", 4),
        ("dth/config/readiness_development_h4_v2.yaml", 4),
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
