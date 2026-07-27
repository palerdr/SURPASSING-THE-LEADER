from __future__ import annotations

import json

import numpy as np
import pytest

from abstract.packed import PackedStateCodec, packed_live_successors
from abstract.packed_tablebase import PackedTablebase, PackedTablebaseBuilder
from abstract.rules import (
    FROZEN_REVIVAL_MODEL,
    UNIFIED_REVIVAL_MODEL,
    AbstractRuleset,
    Bucket12Frozen95Rules,
    Bucket12Unified80Rules,
)
from abstract.state import AbstractState


rust = pytest.importorskip("abstract_solver_rs")


def _tiny_rules() -> AbstractRuleset:
    return AbstractRuleset(
        ruleset_id="test_bucket2_rust_parity",
        action_values=(1, 2),
        bucket_seconds=5,
        load_cap_units=4,
        failed_check_penalty_units=2,
        ttd_half_life_units=2.0,
    )


def _tiny_unified_rules() -> AbstractRuleset:
    return AbstractRuleset(
        ruleset_id="test_bucket2_unified_rust_parity",
        action_values=(1, 2),
        bucket_seconds=75,
        load_cap_units=4,
        failed_check_penalty_units=2,
        revival_model_kind=UNIFIED_REVIVAL_MODEL,
        revival_baseline=0.8,
        ttd_half_life_units=2.0,
        referee_decay_per_death_dose=0.88,
        referee_floor=0.4,
    )


def _tiny_frozen_rules() -> AbstractRuleset:
    return AbstractRuleset(
        ruleset_id="test_bucket2_frozen_rust_parity",
        action_values=(1, 2),
        bucket_seconds=5,
        load_cap_units=4,
        failed_check_penalty_units=2,
        revival_model_kind=FROZEN_REVIVAL_MODEL,
        revival_baseline=0.95,
        ttd_decay_per_death_dose=0.75,
    )


def test_rust_contract_version_and_successor_parity() -> None:
    assert rust.PARITY_CONTRACT_VERSION == "abstract-packed-parity-v2"
    for rules in (
        _tiny_rules(),
        _tiny_unified_rules(),
        _tiny_frozen_rules(),
        Bucket12Unified80Rules(),
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


@pytest.mark.parametrize(
    "rules_factory",
    [_tiny_unified_rules, _tiny_frozen_rules],
    ids=["unified", "frozen"],
)
def test_rust_and_python_builds_have_exact_closure_and_numeric_parity(
    tmp_path,
    rules_factory,
) -> None:
    rules = rules_factory()
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
