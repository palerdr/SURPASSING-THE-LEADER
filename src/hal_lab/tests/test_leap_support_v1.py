"""Protocol tests of the leap support study: the key draw, the label decoding,
and a smoke run of both experiments on fabricated exports."""

from __future__ import annotations

import json

import numpy as np
import pytest

from hal_lab.experiments.leap_support_v1 import run as study


def test_training_keys_are_deterministic_and_never_hidden():
    keys = study.training_keys()
    assert keys == study.training_keys()
    assert len(keys) == 2 * study.PROTOCOL["training_keys_per_half"]
    assert len(set(keys)) == len(keys)
    assert not set(keys) & set(study.PROTOCOL["hidden_keys"])
    assert sum(k.startswith("H1") for k in keys) == study.PROTOCOL["training_keys_per_half"]
    assert set(keys) <= set(study.PROTOCOL["candidate_keys"])


def test_candidate_keys_are_the_residue_keys():
    # The artifact checkpoint counts at least a million residue classes in
    # exactly these 18 H keys; every other H key holds fewer or none.
    candidates = study.PROTOCOL["candidate_keys"]
    assert len(candidates) == 18 and len(set(candidates)) == 18
    assert set(study.PROTOCOL["hidden_keys"]) <= set(candidates)
    assert all(k.split("_")[0] in {"H1", "H2"} and 12 <= int(k.split("_")[1]) <= 59 for k in candidates)
    for half in ("H1", "H2"):
        remaining = [k for k in candidates if k.startswith(half) and k not in study.PROTOCOL["hidden_keys"]]
        assert len(remaining) >= study.PROTOCOL["training_keys_per_half"]


def _fabricate(path, rows, seed, residue_fraction=0.5):
    """We write one export whose support bits are a fixed function of the payoffs."""
    rng = np.random.default_rng(seed)
    success = np.sort(rng.uniform(-1, 1, size=(rows, 60)), axis=1)
    failure = rng.uniform(-1, 1, size=rows)
    residue = rng.random(rows) < residue_fraction
    # Residue rows drop the actions whose success payoff lies below the failure payoff.
    p_mask = np.zeros(rows, np.uint64)
    for a in range(60):
        keep = ~residue | (success[:, a] >= failure)
        p_mask |= keep.astype(np.uint64) << np.uint64(a)
    np.savez(path, checker=np.arange(rows, dtype=np.int32), dropper=np.arange(rows, dtype=np.int32),
             success=success, failure=failure, value=failure, kind=np.where(residue, 255, 1).astype(np.uint8),
             p_mask=p_mask, q_mask=p_mask, residue=residue, window=np.bool_(False))
    return p_mask


def test_load_rows_decodes_the_masks(tmp_path):
    path = tmp_path / "H1_17.npz"
    p_mask = _fabricate(path, 32, 1)
    x, y, residue = study.load_rows(path)
    assert x.shape == (32, 120) and y.shape == (32, 120) and residue.shape == (32,)
    assert np.allclose(x[:, 61:], np.diff(x[:, :60], axis=1), atol=1e-6)
    for row in range(32):
        bits = [(int(p_mask[row]) >> a) & 1 for a in range(60)]
        assert y[row, :60].tolist() == bits and y[row, 60:].tolist() == bits


def test_smoke_study_runs_both_experiments(tmp_path):
    pytest.importorskip("torch")
    protocol = dict(study.PROTOCOL, rows_per_key=200, steps=60, validation_every=30, batch_size=64,
                    hidden_layers=[16])
    export = tmp_path / "export"; export.mkdir()
    for index, key in enumerate(study.training_keys(protocol) + protocol["hidden_keys"]):
        _fabricate(export / f"{key}.npz", 200, index)
    report = study.study(export, tmp_path / "out", protocol, log=lambda line: None)
    written = json.loads((tmp_path / "out" / "report.json").read_text())
    assert written["training_keys"] == report["training_keys"] == study.training_keys(protocol)
    assert set(report["nets"]) == {"A", "B"}
    assert report["nets"]["B"]["parameters"] == report["nets"]["A"]["parameters"] == 120 * 16 + 16 + 16 * 120 + 120
    for net in report["nets"].values():
        assert set(net["hidden"]) == set(protocol["hidden_keys"])
        for scores in net["hidden"].values():
            assert 0.0 <= scores["exact_match"] <= 1.0 and sum(scores["wrong_bits_histogram_0_to_7"]) <= scores["rows"]
        assert (tmp_path / "out" / "net_A.npz").exists() and (tmp_path / "out" / "net_B.npz").exists()
    bound = report["sha256"]
    assert all("/" in key for key in bound) and len(bound) == 10 + 2 + 2
    frozen = study.record(tmp_path / "out", tmp_path / "results.json")
    assert list(frozen["report_sha256"]) == [study.repo_path(tmp_path / "out" / "report.json")]
    assert json.loads((tmp_path / "results.json").read_text()) == frozen
    assert report["experiment2"]["verdict"] in {"easy stages hurt", "easy stages help", "no measurable effect at this budget"}
    assert set(report["gate_results"]) == {"hidden_key_exact_match", "parameter_cap", "training_hours_cap"}


def test_study_refuses_a_missing_export(tmp_path):
    pytest.importorskip("torch")
    with pytest.raises(FileNotFoundError):
        study.study(tmp_path, tmp_path / "out", study.PROTOCOL, log=lambda line: None)
