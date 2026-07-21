from pathlib import Path

import numpy as np
import pytest
import torch

from dth.migrate_boundary_checkpoint import (
    MIGRATION_METHOD,
    migrate_boundary_checkpoint,
    migrate_state_dict,
)
from dth.network import DTHNetworkConfig, DTHPolicyValueNet


def _audit_artifact(path: Path) -> None:
    np.savez(
        path,
        states=np.asarray(
            [[239, 0, 0, 240], [241, 0, 241, 0], [200, 41, 200, 41]],
            dtype=np.int16,
        ),
        horizons=np.asarray([5, 4, 3], dtype=np.uint8),
    )


def test_migration_copies_only_new_zero_columns_and_preserves_predictions(tmp_path):
    torch.manual_seed(17)
    source_config = DTHNetworkConfig(hidden_width=4, hidden_layers=2)
    source_model = DTHPolicyValueNet(source_config).eval()
    artifact = tmp_path / "reference.npz"
    _audit_artifact(artifact)
    source_path = tmp_path / "source.pt"
    destination = tmp_path / "migrated.pt"
    source_payload = {
        "state_dict": source_model.state_dict(),
        "model_config": {
            key: value
            for key, value in source_config.to_dict().items()
            if key != "feature_lift"
        },
        "epoch": 19,
    }
    torch.save(source_payload, source_path)

    migrated = migrate_boundary_checkpoint(
        source_path,
        destination,
        audit_artifacts=[artifact],
    )
    loaded = torch.load(destination, map_location="cpu", weights_only=False)

    assert migrated["model_config"]["feature_lift"] == "boundary_v1"
    assert migrated["boundary_lift_migration"]["method"] == MIGRATION_METHOD
    assert migrated["boundary_lift_migration"]["prediction_audit"]["passed"]
    assert loaded["epoch"] == 19
    assert torch.equal(
        loaded["state_dict"]["trunk.0.weight"][:, :5],
        source_payload["state_dict"]["trunk.0.weight"],
    )
    assert torch.count_nonzero(loaded["state_dict"]["trunk.0.weight"][:, 5:]) == 0
    for name, tensor in source_payload["state_dict"].items():
        if name != "trunk.0.weight":
            assert torch.equal(loaded["state_dict"][name], tensor)

    migrated_model = DTHPolicyValueNet(
        DTHNetworkConfig(**loaded["model_config"])
    ).eval()
    migrated_model.load_state_dict(loaded["state_dict"], strict=True)
    features = torch.tensor(
        [[239 / 300, 0, 0, 240 / 300, 5 / 3]], dtype=torch.float32
    )
    with torch.no_grad():
        for source_output, migrated_output in zip(
            source_model(features), migrated_model(features), strict=True
        ):
            torch.testing.assert_close(source_output, migrated_output, atol=1e-5, rtol=0)


def test_migration_rejects_non_identity_source():
    config = DTHNetworkConfig(feature_lift="boundary_v1")
    model = DTHPolicyValueNet(config)
    with pytest.raises(ValueError, match="identity feature lift"):
        migrate_state_dict(model.state_dict(), config)
