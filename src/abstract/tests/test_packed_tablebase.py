from __future__ import annotations

import copy
import json
from types import SimpleNamespace

import numpy as np
import pytest

import abstract.packed_tablebase as packed_tablebase_module
import abstract.tablebase as tablebase_module
from abstract.exact import solve_exact
from abstract.packed import PackedStateCodec, packed_branches, packed_live_successors
from abstract.packed_tablebase import (
    PACKED_TABLEBASE_SCHEMA,
    PackedTablebase,
    PackedTablebaseBuilder,
)
from abstract.rules import AbstractRuleset, Bucket12Frozen95Rules
from abstract.state import AbstractState
from abstract.tablebase import build_tablebase, load_tablebase, write_tablebase


def _tiny_rules() -> AbstractRuleset:
    return AbstractRuleset(
        ruleset_id="test_bucket2",
        action_values=(1, 2),
        bucket_seconds=5,
        load_cap_units=4,
        failed_check_penalty_units=2,
    )


def test_five_second_formulation_doubles_physical_unit_counts() -> None:
    rules = Bucket12Frozen95Rules()
    assert rules.action_values == tuple(range(1, 13))
    assert tuple(rules.action_seconds(action) for action in rules.action_values) == tuple(
        range(5, 61, 5)
    )
    assert rules.load_cap_units == 60
    assert rules.load_cap_seconds == 300
    assert rules.failed_check_penalty_units == 12
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
    rules = Bucket12Frozen95Rules()
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
    rules = Bucket12Frozen95Rules()
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


def test_packed_checkpoint_rejects_changed_implementation_source(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    rules = _tiny_rules()
    output = tmp_path / "source-bound"
    builder = PackedTablebaseBuilder(
        rules, output, checkpoint_states=2, backend="python"
    )
    assert not builder.enumerate_reachable(stop_after_dequeues=2)
    original = packed_tablebase_module._implementation_digest
    monkeypatch.setattr(
        packed_tablebase_module,
        "_implementation_digest",
        lambda *, include_rust: "0" * 64,
    )
    assert original(include_rust=False) != "0" * 64
    with pytest.raises(ValueError, match="configuration"):
        PackedTablebaseBuilder(rules, output, checkpoint_states=2, backend="python")


def test_packed_rust_backend_rejects_a_stale_loaded_binary(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    stale = SimpleNamespace(
        PARITY_CONTRACT_VERSION="abstract-packed-parity-v3",
        SOURCE_BUNDLE_DIGEST_ALGORITHM="sha256-framed-source-bundle-v1",
        SOURCE_BUNDLE_DIGEST="0" * 64,
    )
    monkeypatch.setattr(
        packed_tablebase_module.importlib,
        "import_module",
        lambda name: stale,
    )
    with pytest.raises(RuntimeError, match="current Rust source bundle"):
        PackedTablebaseBuilder(_tiny_rules(), tmp_path / "stale", backend="auto")


def test_packed_manifest_contract_fields_fail_closed(tmp_path) -> None:
    rules = _tiny_rules()
    output = tmp_path / "manifest-contract"
    PackedTablebaseBuilder(
        rules, output, checkpoint_states=2, backend="python"
    ).run()
    manifest_path = output / "tablebase.json"
    original = json.loads(manifest_path.read_text(encoding="utf-8"))
    cases = (
        (
            lambda manifest: manifest["metadata"].__setitem__(
                "packed_build_config_digest", "0" * 64
            ),
            "configuration",
        ),
        (
            lambda manifest: manifest["metadata"].__setitem__(
                "code_config_digest", "0" * 64
            ),
            "code/configuration",
        ),
        (
            lambda manifest: manifest["metadata"]["matrix_solver"].__setitem__(
                "policy_saddle_gap", 1e-3
            ),
            "matrix contract",
        ),
        (
            lambda manifest: manifest["metadata"].__setitem__(
                "revival_model", {"kind": "other"}
            ),
            "frozen revival",
        ),
        (lambda manifest: manifest["arrays"].pop("state_index"), "array set"),
        (
            lambda manifest: manifest["arrays"]["value"].__setitem__(
                "shape", [999]
            ),
            "array contract",
        ),
        (lambda manifest: manifest.__setitem__("extra", None), "manifest key set"),
        (lambda manifest: manifest["metadata"].__setitem__("extra", None), "metadata key set"),
        (lambda manifest: manifest["metadata"]["matrix_solver"].__setitem__("extra", None), "matrix-solver"),
        (lambda manifest: manifest["arrays"]["value"].__setitem__("extra", None), "array contract"),
        (lambda manifest: manifest["metadata"].__setitem__("execution_backends", ["python", "rust"]), "execution provenance"),
        (lambda manifest: manifest["metadata"].__setitem__("checkpoint_states", 0), "checkpoint size"),
    )
    for mutate, message in cases:
        altered = copy.deepcopy(original)
        mutate(altered)
        manifest_path.write_text(json.dumps(altered), encoding="utf-8")
        with pytest.raises(ValueError, match=message):
            PackedTablebase(output)
    manifest_path.write_text(json.dumps(original), encoding="utf-8")
    PackedTablebase(output)


@pytest.mark.parametrize("damage", ["manifest", "array"])
def test_completed_packed_builder_reverifies_artifact(tmp_path, damage) -> None:
    rules = _tiny_rules()
    output = tmp_path / damage
    PackedTablebaseBuilder(
        rules, output, checkpoint_states=2, backend="python"
    ).run()
    if damage == "manifest":
        (output / "tablebase.json").unlink()
    else:
        value_path = output / "value.npy"
        payload = bytearray(value_path.read_bytes())
        payload[-1] ^= 0xFF
        value_path.write_bytes(payload)
    resumed = PackedTablebaseBuilder(
        rules, output, checkpoint_states=2, backend="python"
    )
    with pytest.raises(RuntimeError, match="completed packed"):
        resumed.solve()


@pytest.mark.parametrize(
    "state",
    [
        (0.0, 0, 0, 0),
        (False, 0, 0, 0),
        AbstractState(checker_load=4),
    ],
)
def test_packed_lookup_rejects_coercible_or_out_of_domain_states(tmp_path, state) -> None:
    output = tmp_path / "lookup-validation"
    PackedTablebaseBuilder(
        _tiny_rules(), output, checkpoint_states=2, backend="python"
    ).run()
    with pytest.raises((TypeError, ValueError)):
        PackedTablebase(output).lookup(state)


def test_npz_tablebase_reader_rederives_manifest_contract(tmp_path) -> None:
    rules = _tiny_rules()
    npz_path, manifest_path, original = write_tablebase(
        build_tablebase(rules), tmp_path / "npz"
    )
    load_tablebase(npz_path, manifest_path)
    cases = (
        (
            lambda manifest: manifest["metadata"].__setitem__(
                "build_config_digest", "0" * 64
            ),
            "configuration",
        ),
        (
            lambda manifest: manifest["metadata"].__setitem__(
                "code_config_digest", "0" * 64
            ),
            "code/configuration",
        ),
        (
            lambda manifest: manifest["metadata"].__setitem__(
                "timing_convention_id", "exclusive"
            ),
            "timing_convention_id",
        ),
        (lambda manifest: manifest["arrays"].pop("states"), "array sets"),
        (lambda manifest: manifest.__setitem__("extra", None), "manifest key set"),
        (lambda manifest: manifest["metadata"].__setitem__("extra", None), "metadata key set"),
    )
    for mutate, message in cases:
        altered = copy.deepcopy(original)
        mutate(altered)
        manifest_path.write_text(json.dumps(altered), encoding="utf-8")
        with pytest.raises(ValueError, match=message):
            load_tablebase(npz_path, manifest_path)


def test_classic_tablebase_captures_fingerprints_before_solving(monkeypatch) -> None:
    events = []
    original_solve = tablebase_module.solve_all_reachable

    def digest_json_before_solve(payload):
        events.append("build-digest")
        return "1" * 64

    def digest_files_before_solve(paths, *, config):
        events.append("code-digest")
        return "2" * 64

    def solve_after_fingerprints(rules, *, root):
        assert events == ["build-digest", "code-digest"]
        events.append("solve")
        return original_solve(rules, root=root)

    monkeypatch.setattr(tablebase_module, "digest_json", digest_json_before_solve)
    monkeypatch.setattr(tablebase_module, "digest_files", digest_files_before_solve)
    monkeypatch.setattr(tablebase_module, "solve_all_reachable", solve_after_fingerprints)
    tablebase_module.build_tablebase(_tiny_rules())
    assert events == ["build-digest", "code-digest", "solve"]
