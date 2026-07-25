from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from stl.learning.certificates import (
    ExactPolicyCertificate,
    certificates_by_state,
    load_certificate_shard,
    save_certificate_shard,
)
from stl.learning.replay import ShardRole


def _digest(character: str) -> str:
    return character * 64


def _certificate(index: int = 0) -> ExactPolicyCertificate:
    return ExactPolicyCertificate(
        state_hash=_digest(f"{index:x}"),
        search_config_digest=_digest("a"),
        horizon=2,
        drop_actions=(1, 61),
        check_actions=(1, 60),
        payoff_for_hal=np.asarray([[0.0, 1.0], [-1.0, 0.25]]),
        dropper_strategy=np.asarray([0.25, 0.75]),
        checker_strategy=np.asarray([0.6, 0.4]),
        value_for_hal=0.125,
        unresolved_probability=0.25,
    )


def test_certificate_shard_round_trips_without_pickle(tmp_path: Path):
    path = tmp_path / "certificates.npz"
    originals = [_certificate(1), _certificate(2)]

    manifest = save_certificate_shard(
        originals,
        path,
        shard_role=ShardRole.DEVELOPMENT,
        generation_provenance={"fixture": True},
    )
    loaded = load_certificate_shard(path, expected_role=ShardRole.DEVELOPMENT)

    assert manifest["record_count"] == 2
    assert manifest["shard_role"] == ShardRole.DEVELOPMENT.value
    assert set(certificates_by_state(loaded)) == {item.state_hash for item in originals}
    for original, copy in zip(originals, loaded):
        assert copy.state_hash == original.state_hash
        assert copy.search_config_digest == original.search_config_digest
        assert copy.horizon == original.horizon
        assert copy.drop_actions == original.drop_actions
        assert copy.check_actions == original.check_actions
        assert copy.value_for_hal == pytest.approx(original.value_for_hal)
        np.testing.assert_array_equal(copy.payoff_for_hal, original.payoff_for_hal)
        np.testing.assert_array_equal(copy.dropper_strategy, original.dropper_strategy)
        np.testing.assert_array_equal(copy.checker_strategy, original.checker_strategy)

    with np.load(path, allow_pickle=False) as payload:
        assert all(not payload[name].dtype.hasobject for name in payload.files)


def test_certificate_role_and_payload_hash_are_enforced(tmp_path: Path):
    path = tmp_path / "certificates.npz"
    save_certificate_shard([_certificate(3)], path, shard_role=ShardRole.EXTERNAL_RULER)

    with pytest.raises(ValueError, match="required role"):
        load_certificate_shard(path, expected_role=ShardRole.REPLAY)

    payload = bytearray(path.read_bytes())
    payload[len(payload) // 2] ^= 0x01
    path.write_bytes(payload)
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        load_certificate_shard(path)


def test_certificate_rejects_invalid_equilibrium_distribution():
    with pytest.raises(ValueError, match="sum to one"):
        replace(_certificate(4), dropper_strategy=np.asarray([0.2, 0.2]))
