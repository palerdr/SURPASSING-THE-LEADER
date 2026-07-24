from __future__ import annotations

import json

import numpy as np
import pytest

from abstract.exact import solve_exact
from abstract.packed import PackedStateCodec, packed_branches, packed_live_successors
from abstract.packed_tablebase import (
    PACKED_TABLEBASE_SCHEMA,
    PackedTablebase,
    PackedTablebaseBuilder,
)
from abstract.rules import AbstractRuleset, Bucket12TTDCurve95Rules
from abstract.state import AbstractState


def _tiny_rules() -> AbstractRuleset:
    return AbstractRuleset(
        ruleset_id="test_bucket2",
        action_values=(1, 2),
        bucket_seconds=5,
        load_cap_units=4,
        failed_check_penalty_units=2,
        ttd_half_life_units=2.0,
    )


def test_five_second_formulation_doubles_physical_unit_counts() -> None:
    rules = Bucket12TTDCurve95Rules()
    assert rules.action_values == tuple(range(1, 13))
    assert tuple(rules.action_seconds(action) for action in rules.action_values) == tuple(
        range(5, 61, 5)
    )
    assert rules.load_cap_units == 60
    assert rules.load_cap_seconds == 300
    assert rules.failed_check_penalty_units == 12
    assert rules.ttd_half_life_units == 24.0
    assert rules.physical_state_upper_bound == 60 * 61 * 60 * 61


@pytest.mark.parametrize(
    ("fields", "drop", "check"),
    [
        ((0, 0, 0, 0), 12, 1),
        ((0, 0, 0, 0), 1, 1),
        ((31, 4, 9, 7), 5, 11),
        ((58, 0, 3, 2), 1, 3),
        ((48, 0, 8, 5), 12, 1),
    ],
)
def test_packed_transition_contract_matches_python_rules(
    fields: tuple[int, int, int, int],
    drop: int,
    check: int,
) -> None:
    rules = Bucket12TTDCurve95Rules()
    codec = PackedStateCodec(rules.load_cap_units)
    state = AbstractState(*fields)
    packed = packed_branches(codec.encode(*fields), drop, check, rules, codec=codec)
    authoritative = rules.expand_joint_action(state, drop, check)
    assert len(packed) == len(authoritative)
    for actual, expected in zip(packed, authoritative):
        assert actual.probability == pytest.approx(expected.probability)
        assert actual.terminal_value == expected.terminal_value
        assert actual.event == expected.event
        assert actual.squandered_units == expected.squandered_units
        assert actual.death_dose_units == expected.death_dose_units
        if expected.state is None:
            assert actual.state_index is None
        else:
            assert actual.state_index == codec.encode(*rules.state_fields(expected.state))


def test_packed_codec_is_bijective_and_successors_increase_potential() -> None:
    rules = Bucket12TTDCurve95Rules()
    codec = PackedStateCodec(rules.load_cap_units)
    samples = (
        (0, 0, 0, 0),
        (59, 60, 59, 60),
        (17, 23, 41, 9),
    )
    for fields in samples:
        index = codec.encode(*fields)
        assert codec.decode(index) == fields
        for child in packed_live_successors(index, rules, codec=codec):
            assert codec.potential(child) > sum(fields)


def test_packed_build_resumes_and_derives_ids_only_at_lookup(tmp_path) -> None:
    rules = _tiny_rules()
    output = tmp_path / "packed"

    first = PackedTablebaseBuilder(rules, output, checkpoint_states=2)
    assert not first.enumerate_reachable(stop_after_dequeues=2)

    resumed = PackedTablebaseBuilder(rules, output, checkpoint_states=2)
    assert resumed.enumerate_reachable()
    assert resumed.prepare_storage()
    assert not resumed.solve(stop_after_chunks=1)

    completed = PackedTablebaseBuilder(rules, output, checkpoint_states=2)
    assert completed.solve()
    manifest = json.loads((output / "tablebase.json").read_text(encoding="utf-8"))
    assert manifest["schema_version"] == PACKED_TABLEBASE_SCHEMA
    assert "state_id" not in manifest["arrays"]
    assert manifest["metadata"]["state_ids"] == "derived_on_lookup_or_export_sha256"
    assert (
        manifest["metadata"]["matrix_solver"]["pure_saddle_states"]
        + manifest["metadata"]["matrix_solver"]["mixed_lp_states"]
        == manifest["metadata"]["reachable_state_count"]
    )
    assert manifest["metadata"]["persisted_policy_max_saddle_gap"] <= 2e-7

    tablebase = PackedTablebase(output)
    root = tablebase.lookup(AbstractState())
    expected = solve_exact(AbstractState(), rules, include_transitions=False)
    assert root["value"] == pytest.approx(expected.value_for_dropper)
    assert np.asarray(root["drop_policy"]).sum() == pytest.approx(1.0)
    assert np.asarray(root["check_policy"]).sum() == pytest.approx(1.0)
    assert len(root["state_id"]) == 64


def test_float32_policy_validation_accumulates_in_float64(tmp_path) -> None:
    rules = _tiny_rules()
    output = tmp_path / "validation"
    builder = PackedTablebaseBuilder(
        rules,
        output,
        checkpoint_states=2,
        backend="python",
    )
    builder.run()
    tablebase = PackedTablebase(output)
    policy = np.asarray(tablebase.arrays["drop_policy"])
    assert np.allclose(
        policy.sum(axis=1, dtype=np.float64),
        1.0,
        atol=2e-7,
        rtol=0.0,
    )
