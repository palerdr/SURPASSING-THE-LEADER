from pathlib import Path

import numpy as np
import pytest

from pure.generate_dataset import (
    TARGET_SCHEMA,
    failure_margin_class,
    generate_exact_targets,
    generate_strategic_targets,
    live_successors,
    reachable_layers,
    sample_strategic_roots,
)


def test_opening_has_all_transition_distinct_live_successors() -> None:
    children = live_successors((0, 0, 0, 0))
    assert len(children) == 61
    assert (0, 0, 60, 0) in children
    assert (0, 0, 0, 60) in children


def test_reachable_layers_reject_invalid_configuration() -> None:
    with pytest.raises(ValueError, match="horizon"):
        reachable_layers([(0, 0, 0, 0)], 0)
    with pytest.raises(ValueError, match="root state"):
        reachable_layers([(0, 0, 0)], 1)
    with pytest.raises(ValueError, match="TTD"):
        reachable_layers([(0, 301, 0, 0)], 1)


def test_generate_horizon_one_artifact(tmp_path: Path) -> None:
    output = generate_exact_targets(
        output=tmp_path / "targets.npz",
        horizon=1,
        root_states=[(239, 0, 0, 0), (240, 0, 0, 0)],
        progress_every=0,
    )

    with np.load(output) as artifact:
        np.testing.assert_array_equal(
            artifact["states"],
            np.asarray([(239, 0, 0, 0), (240, 0, 0, 0)], dtype=np.int16),
        )
        np.testing.assert_array_equal(
            artifact["horizons"], np.ones(2, dtype=np.uint8)
        )
        assert artifact["drop_policies"].shape == (2, 60)
        assert artifact["check_policies"].shape == (2, 60)
        assert str(artifact["schema_version"]) == TARGET_SCHEMA


def test_failure_margin_classes_pin_strict_total_boundary() -> None:
    assert failure_margin_class((240, 0)) == "dose_fatal"
    assert failure_margin_class((0, 240)) == "exact_300"
    assert failure_margin_class((0, 241)) == "ttd_fatal"
    assert failure_margin_class((239, 0)) == "near_1_5"


def test_strategic_sampling_is_deterministic_and_keeps_forced_roots() -> None:
    kwargs = {
        "count": 8,
        "st_values": [0, 180, 239, 240],
        "ttd_values": [0, 60, 240],
        "forced_roots": [(0, 0, 0, 0), (0, 240, 0, 0)],
        "seed": 4,
    }
    first = sample_strategic_roots(**kwargs)
    second = sample_strategic_roots(**kwargs)
    assert first == second
    assert first[:2] == ((0, 0, 0, 0), (0, 240, 0, 0))
    assert len(set(first)) == 8


def test_generate_strategic_artifact_emits_only_requested_roots(tmp_path: Path) -> None:
    output = generate_strategic_targets(
        output=tmp_path / "strategic.npz",
        target_sets=[{"horizon": 1, "count": 3}],
        st_values=[0, 239, 240],
        ttd_values=[0, 240],
        forced_roots=[(0, 0, 0, 0)],
        seed=4,
        progress_every=0,
    )
    with np.load(output) as artifact:
        assert artifact["states"].shape == (3, 4)
        np.testing.assert_array_equal(
            artifact["horizons"], np.ones(3, dtype=np.uint8)
        )
        assert str(artifact["dataset_version"]) == "strategic_exact_v1"
        assert str(artifact["emission"]) == "roots_only"
