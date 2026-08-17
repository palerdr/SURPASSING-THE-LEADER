from __future__ import annotations

import importlib
import json
import os

import numpy as np
import pytest

from abstract.packed import PackedStateCodec, packed_live_successors
from abstract.packed_tablebase import (
    PackedTablebase,
    PackedTablebaseBuilder,
    _rust_source_bundle_digest,
)
from abstract.rules import (
    AbstractRuleset,
    Bucket12Frozen95Rules,
)
from abstract.state import AbstractState


def _load_rust_extension():
    try:
        return importlib.import_module("abstract_solver_rs")
    except ModuleNotFoundError as error:
        if error.name != "abstract_solver_rs":
            raise
        if os.environ.get("STL_REQUIRE_RUST_PARITY") == "1":
            raise RuntimeError(
                "abstract_solver_rs is required by STL_REQUIRE_RUST_PARITY=1; "
                "build it before running parity tests"
            ) from error
        pytest.skip("abstract_solver_rs is not installed", allow_module_level=True)


rust = _load_rust_extension()


def _tiny_rules() -> AbstractRuleset:
    return AbstractRuleset(
        ruleset_id="test_bucket2_rust_parity",
        action_values=(1, 2),
        bucket_seconds=5,
        load_cap_units=4,
        failed_check_penalty_units=2,
    )


def test_rust_contract_version_and_successor_parity() -> None:
    assert rust.PARITY_CONTRACT_VERSION == "abstract-packed-parity-v3"
    assert rust.SOURCE_BUNDLE_DIGEST_ALGORITHM == "sha256-framed-source-bundle-v1"
    assert rust.SOURCE_BUNDLE_DIGEST == _rust_source_bundle_digest()
    for rules in (
        _tiny_rules(),
        Bucket12Frozen95Rules(),
    ):
        codec = PackedStateCodec(rules.load_cap_units)
        indices = range(codec.state_count) if codec.state_count < 1_000 else (
            codec.encode(0, 0, 0, 0),
            codec.encode(59, 60, 59, 60),
            codec.encode(17, 23, 41, 9),
            codec.encode(48, 0, 8, 5),
        )
        for index in indices:
            expected = packed_live_successors(index, rules, codec=codec)
            actual = rust.live_successors_rs(
                index,
                rules.load_cap_units,
                rules.action_size,
                rules.failed_check_penalty_units,
            )
            assert tuple(actual) == expected


def test_rust_public_boundary_rejects_invalid_domains_and_shapes() -> None:
    rules = _tiny_rules()
    codec = PackedStateCodec(rules.load_cap_units)
    with pytest.raises(ValueError, match="physical state domain"):
        rust.live_successors_rs(
            codec.state_count,
            rules.load_cap_units,
            rules.action_size,
            rules.failed_check_penalty_units,
        )
    with pytest.raises(ValueError, match="action_size must equal"):
        rust.live_successors_rs(0, rules.load_cap_units, 1, 2)
    with pytest.raises(ValueError, match="queue length"):
        rust.expand_reachability_chunk_rs(
            np.zeros(1, dtype=np.uint32),
            np.zeros(1, dtype=np.uint8),
            0,
            0,
            1,
            rules.load_cap_units,
            rules.action_size,
            rules.failed_check_penalty_units,
        )
    with pytest.raises(ValueError, match="ordinal_by_index length"):
        rust.backup_chunk_rs(
            np.array([0], dtype=np.uint32),
            np.zeros(1, dtype=np.uint32),
            np.zeros(1, dtype=np.float64),
            np.zeros(1, dtype=np.float64),
            np.zeros(1, dtype=np.float64),
            rules.load_cap_units,
            rules.action_size,
            rules.failed_check_penalty_units,
        )


def test_rust_and_python_builds_have_exact_closure_and_numeric_parity(
    tmp_path,
) -> None:
    rules = _tiny_rules()
    python_dir = tmp_path / "python"
    rust_dir = tmp_path / "rust"
    assert PackedTablebaseBuilder(
        rules,
        python_dir,
        checkpoint_states=3,
        backend="python",
    ).run()
    assert PackedTablebaseBuilder(
        rules,
        rust_dir,
        checkpoint_states=3,
        backend="rust",
    ).run()

    python_manifest = json.loads((python_dir / "tablebase.json").read_text())
    rust_manifest = json.loads((rust_dir / "tablebase.json").read_text())
    assert (
        python_manifest["metadata"]["reachable_state_count"]
        == rust_manifest["metadata"]["reachable_state_count"]
    )
    assert (
        python_manifest["metadata"]["potential_counts"]
        == rust_manifest["metadata"]["potential_counts"]
    )
    assert (
        python_manifest["metadata"]["matrix_solver"]["pure_saddle_states"]
        == rust_manifest["metadata"]["matrix_solver"]["pure_saddle_states"]
    )
    assert (
        python_manifest["metadata"]["matrix_solver"]["mixed_lp_states"]
        == rust_manifest["metadata"]["matrix_solver"]["mixed_lp_states"]
    )

    python_tablebase = PackedTablebase(python_dir)
    rust_tablebase = PackedTablebase(rust_dir)
    for name in python_tablebase.arrays:
        left = np.asarray(python_tablebase.arrays[name])
        right = np.asarray(rust_tablebase.arrays[name])
        if np.issubdtype(left.dtype, np.floating):
            assert np.allclose(left, right, atol=2e-10, rtol=0.0), name
        else:
            assert np.array_equal(left, right), name
    assert (
        python_tablebase.lookup(AbstractState())["state_id"]
        == rust_tablebase.lookup(AbstractState())["state_id"]
    )
