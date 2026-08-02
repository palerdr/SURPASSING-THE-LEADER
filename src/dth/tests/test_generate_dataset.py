from pathlib import Path
import json

import numpy as np
import pytest

from dth.solver import CHECKER_ACTIONS, DROPPER_ACTIONS
from dth.generate_dataset import (
    TARGET_SCHEMA,
    boundary_tablebase_identities,
    failure_margin_class,
    generate_boundary_tablebase,
    generate_exact_targets,
    generate_paired_orientation_targets,
    generate_strategic_targets,
    live_successors,
    merge_exact_target_artifacts,
    mirror_state,
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


def test_paired_orientation_targets_hold_out_only_exact_mirrors(
    tmp_path: Path,
) -> None:
    train, holdout = generate_paired_orientation_targets(
        train_output=tmp_path / "train.npz",
        holdout_output=tmp_path / "holdout.npz",
        pairs=[
            {"state": [240, 0, 0, 0], "horizons": [1, 2]},
            {"state": [200, 41, 60, 180], "horizons": [1]},
        ],
        train_orientation="primary",
        progress_every=0,
        dataset_version="paired-test",
    )

    with np.load(train) as train_artifact, np.load(holdout) as holdout_artifact:
        train_rows = {
            (tuple(int(value) for value in state), int(horizon))
            for state, horizon in zip(
                train_artifact["states"], train_artifact["horizons"], strict=True
            )
        }
        holdout_rows = {
            (tuple(int(value) for value in state), int(horizon))
            for state, horizon in zip(
                holdout_artifact["states"],
                holdout_artifact["horizons"],
                strict=True,
            )
        }

        assert str(train_artifact["emission"]) == "paired_mirror_roots"
        assert str(holdout_artifact["paired_role"]) == "heldout_mirror"

    assert train_rows.isdisjoint(holdout_rows)
    assert {
        (mirror_state(state), horizon) for state, horizon in train_rows
    } == holdout_rows


def test_paired_orientation_targets_reject_self_mirrors(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="self-mirrored"):
        generate_paired_orientation_targets(
            train_output=tmp_path / "train.npz",
            holdout_output=tmp_path / "holdout.npz",
            pairs=[{"state": [0, 0, 0, 0], "horizons": [1]}],
            train_orientation="primary",
            progress_every=0,
        )


def test_boundary_tablebase_materializes_paired_full_closure(tmp_path: Path) -> None:
    pairs = [{"state": [240, 0, 0, 0], "horizons": [1, 2]}]
    output = tmp_path / "tablebase.npz"
    report_output = tmp_path / "tablebase_report.json"

    roots, expected = boundary_tablebase_identities(pairs)
    generated = generate_boundary_tablebase(
        output=output,
        report_output=report_output,
        pairs=pairs,
        progress_every=0,
        dataset_version="tablebase-test",
    )

    with np.load(generated) as artifact:
        rows = {
            (tuple(int(value) for value in state), int(horizon))
            for state, horizon in zip(
                artifact["states"], artifact["horizons"], strict=True
            )
        }
        assert rows == expected
        assert str(artifact["dataset_version"]) == "tablebase-test"
        assert str(artifact["emission"]) == "boundary_tablebase_closure"
        assert str(artifact["schema_version"]) == TARGET_SCHEMA
        assert set(artifact["root_orientations"].tolist()) == {"primary", "mirror"}
        assert artifact["root_states"].shape == (len(roots), 4)

    report = json.loads(report_output.read_text(encoding="utf-8"))
    assert report["artifact"]["rows"] == len(expected)
    assert report["artifact"]["bytes"] == output.stat().st_size
    assert report["roots"]["primary_rows"] == 2
    assert report["roots"]["mirror_rows"] == 2
    assert report["coverage_by_remaining_horizon"] == {
        str(horizon): sum(1 for _, current in expected if current == horizon)
        for horizon in (1, 2)
    }


def test_merge_exact_artifacts_deduplicates_state_horizon_rows(tmp_path: Path) -> None:
    first = generate_exact_targets(
        output=tmp_path / "first.npz",
        horizon=1,
        root_states=[(239, 0, 0, 0)],
        progress_every=0,
    )
    second = generate_exact_targets(
        output=tmp_path / "second.npz",
        horizon=1,
        root_states=[(239, 0, 0, 0), (240, 0, 0, 0)],
        progress_every=0,
    )
    output = merge_exact_target_artifacts(
        [first, second],
        tmp_path / "merged.npz",
        dataset_version="merged_test",
    )

    with np.load(output) as artifact:
        assert artifact["states"].shape == (2, 4)
        assert str(artifact["dataset_version"]) == "merged_test"
        assert str(artifact["emission"]) == "merged_reachable"
        assert artifact["value_semantics"].tolist() == [0, 0]


def test_merge_keeps_play_semantics_and_lets_exact_rows_win_collisions(
    tmp_path: Path,
) -> None:
    exact = generate_exact_targets(
        output=tmp_path / "exact.npz",
        horizon=1,
        root_states=[(239, 0, 0, 0)],
        progress_every=0,
    )
    with np.load(exact, allow_pickle=False) as artifact:
        exact_value = float(artifact["values"][0])
        drop_policy = artifact["drop_policies"][0]
        check_policy = artifact["check_policies"][0]
    play = tmp_path / "play.npz"
    np.savez_compressed(
        play,
        states=np.asarray([[239, 0, 0, 0], [240, 0, 0, 0]], dtype=np.int16),
        horizons=np.ones(2, dtype=np.uint8),
        values=np.asarray([0.5, 0.5], dtype=np.float32),
        drop_policies=np.asarray([drop_policy, drop_policy], dtype=np.float32),
        check_policies=np.asarray([check_policy, check_policy], dtype=np.float32),
        saddle_gaps=np.zeros(2, dtype=np.float32),
        drop_actions=np.asarray(DROPPER_ACTIONS, dtype=np.int16),
        check_actions=np.asarray(CHECKER_ACTIONS, dtype=np.int16),
        dataset_version=np.asarray("play-test"),
        emission=np.asarray("resolve_labeled"),
        schema_version=np.asarray(TARGET_SCHEMA),
    )

    output = merge_exact_target_artifacts(
        [play, exact],
        tmp_path / "merged.npz",
        dataset_version="merged_mixed_test",
    )

    with np.load(output) as artifact:
        rows = {
            tuple(int(value) for value in state): (
                float(row_value),
                int(semantics),
            )
            for state, row_value, semantics in zip(
                artifact["states"],
                artifact["values"],
                artifact["value_semantics"],
                strict=True,
            )
        }
    assert rows[(239, 0, 0, 0)] == (pytest.approx(exact_value), 0)
    assert rows[(240, 0, 0, 0)] == (pytest.approx(0.5), 1)


def test_resolve_labeled_emission_writes_play_coverage_rows(tmp_path):
    from dataclasses import dataclass

    import numpy as np

    from dth.research_agent import BoundedResolveAgent, ResolveBudget
    from dth.generate_dataset import generate_resolve_labeled_targets

    @dataclass(frozen=True)
    class _StubConfig:
        transition_class_head: bool = False

    class _ZeroNetwork:
        config = _StubConfig()

        def values(self, states, horizon):
            del horizon
            return np.zeros(len(states), dtype=np.float64)

    agent = BoundedResolveAgent(
        network=_ZeroNetwork(),
        budget=ResolveBudget(deadline_seconds=30.0, max_depth=2, leaf_horizon=4),
    )
    output = tmp_path / "resolve_labeled.npz"
    generate_resolve_labeled_targets(
        output=output,
        report_output=tmp_path / "resolve_labeled.json",
        games=1,
        max_half_rounds=3,
        seed=4,
        label_depth=2,
        label_deadline_seconds=30.0,
        max_resolves=4,
        leaf_horizon=4,
        agent=agent,
    )

    with np.load(output, allow_pickle=False) as artifact:
        assert str(np.asarray(artifact["emission"]).item()) == "resolve_labeled"
        assert len(artifact["states"]) > 1
        assert set(artifact["horizons"].tolist()) == {4}
        assert set(artifact["value_semantics"].tolist()) == {1}
        drop = artifact["drop_policies"]
        assert np.allclose(drop.sum(axis=1), 1.0, atol=1e-5)
