from __future__ import annotations

from types import SimpleNamespace

import stl.learning.train as train_module
from stl.commands.gen0_corpus import write_gen0_shard
from stl.learning.replay import ShardRole, load_replay_manifest, load_replay_shard
from stl.learning.train import TrainConfig, load_checkpoint, train


def test_gen0_smoke_writes_reconstructable_train_and_external_ruler(tmp_path):
    train_path = tmp_path / "train.npz"
    ruler_path = tmp_path / "ruler.npz"

    train_summary = write_gen0_shard(
        train_path,
        holdout=False,
        workers=1,
        smoke=True,
    )
    ruler_summary = write_gen0_shard(
        ruler_path,
        holdout=True,
        workers=1,
        smoke=True,
    )

    assert train_summary["records"] > 0
    assert ruler_summary["records"] > 0
    assert load_replay_shard(train_path, for_training=True)
    assert load_replay_shard(ruler_path)
    assert load_replay_manifest(train_path)["shard_role"] == ShardRole.REPLAY.value
    assert (
        load_replay_manifest(ruler_path)["shard_role"]
        == ShardRole.EXTERNAL_RULER.value
    )

    result = train(
        train_path,
        tmp_path / "checkpoint",
        TrainConfig(epochs=1, batch_size=8, val_fraction=0.2, seed=0),
    )
    assert load_checkpoint(result.checkpoint_path) is not None


def test_train_module_exposes_hydra_dispatchable_main(monkeypatch, tmp_path):
    captured = {}

    def fake_train(targets, out, config):
        captured.update(targets=targets, out=out, config=config)
        return SimpleNamespace(
            best_val_mse=0.0,
            best_epoch=0,
            checkpoint_path=str(tmp_path / "best.pt"),
            best_per_source_mse={},
        )

    monkeypatch.setattr(train_module, "train", fake_train)
    monkeypatch.setattr(
        "sys.argv",
        [
            "stl.learning.train",
            "--targets",
            "train-v2.npz",
            "--out",
            "checkpoint-v2",
            "--epochs",
            "1",
            "--batch-size",
            "8",
            "--no-allow-legacy-targets",
        ],
    )

    assert train_module.main() == 0
    assert captured["targets"] == "train-v2.npz"
    assert captured["out"] == "checkpoint-v2"
    assert captured["config"].epochs == 1
    assert captured["config"].batch_size == 8
    assert captured["config"].allow_legacy_targets is False
