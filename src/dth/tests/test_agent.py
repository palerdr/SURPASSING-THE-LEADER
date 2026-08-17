"""Play-time DTH agent uses only the completed exact artifact."""

from __future__ import annotations

import json
import pytest
import numpy as np
from pathlib import Path

import dth.agent as agent_module
from dth.agent import CompleteDTHAgent, MoveDecision
from dth.complete_tablebase import COMPLETE_TABLEBASE_SCHEMA


class _Complete:
    def __init__(self, artifact_dir) -> None:
        self.artifact_dir = artifact_dir

    def certificate(self, state):
        assert state == (0, 0, 0, 0)
        return {
            "value": 0.125,
            "drop_policy": [1.0] + [0.0] * 59,
            "check_policy": [0.0] * 59 + [1.0],
            "saddle_gap": 4e-7,
        }

    def lookup(self, state):
        del state
        return {"value": 0.0}


def test_complete_agent_returns_exact_certificate(monkeypatch) -> None:
    monkeypatch.setattr(agent_module, "CompleteTablebase", _Complete)
    decision = CompleteDTHAgent("complete").decide((0, 0, 0, 0))
    assert decision.value == pytest.approx(0.125)
    assert decision.drop_policy[0] == 1.0
    assert decision.check_policy[-1] == 1.0
    assert decision.saddle_gap == pytest.approx(4e-7)
    assert decision.elapsed_seconds >= 0.0


def test_decision_is_a_frozen_record() -> None:
    decision = MoveDecision(
        state=(0, 0, 0, 0),
        value=0.0,
        drop_policy=(),
        check_policy=(),
        saddle_gap=0.0,
        elapsed_seconds=0.0,
    )
    with pytest.raises(AttributeError):
        decision.value = 1.0  # type: ignore[misc]


def test_complete_agent_rejects_off_domain_state(monkeypatch) -> None:
    monkeypatch.setattr(agent_module, "CompleteTablebase", _Complete)
    with pytest.raises(ValueError):
        CompleteDTHAgent("complete").decide((0, 0, 0, 301))


def test_stage_game_reconstructs_and_rechecks_the_full_certificate(
    monkeypatch,
) -> None:
    class _StageComplete(_Complete):
        def certificate(self, state):
            result = super().certificate(state)
            result["value"] = 0.0
            return result

    monkeypatch.setattr(agent_module, "CompleteTablebase", _StageComplete)
    monkeypatch.setattr(
        agent_module,
        "continuation_class_values",
        lambda state, lookup: ((0.0,) * 60, 0.0),
    )
    stage = CompleteDTHAgent("complete").stage_game((0, 0, 0, 0))

    assert stage.state == (0, 0, 0, 0)
    assert stage.value == pytest.approx(0.0)
    assert stage.matrix.shape == (60, 60)
    assert np.all(np.isfinite(stage.matrix))
    assert stage.saddle_gap == 0.0
    assert stage.drop_policy.sum() == pytest.approx(1.0)
    assert stage.check_policy.sum() == pytest.approx(1.0)
    assert stage.matrix.flags.writeable is False


def test_stage_game_fails_closed_when_saddle_gap_is_too_large(monkeypatch) -> None:
    monkeypatch.setattr(agent_module, "CompleteTablebase", _Complete)
    matrix = np.zeros((60, 60), dtype=np.float64)
    matrix[0, :] = -1.0
    matrix[:, -1] = 1.0
    monkeypatch.setattr(
        agent_module,
        "reconstruct_transition_class_matrix",
        lambda successful, failed: matrix,
    )
    monkeypatch.setattr(
        agent_module,
        "continuation_class_values",
        lambda state, lookup: ((0.0,) * 60, 0.0),
    )
    with pytest.raises(RuntimeError, match="saddle gap"):
        CompleteDTHAgent("complete").stage_game((0, 0, 0, 0))


def test_real_complete_stage_facade_when_canonical_artifact_is_available() -> None:
    artifact = Path("src/dth/artifacts/complete_full_v1")
    manifest_path = artifact / "tablebase.json"
    if not manifest_path.is_file():
        pytest.skip("canonical complete tablebase artifact is not available")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != COMPLETE_TABLEBASE_SCHEMA:
        pytest.skip("canonical complete tablebase artifact requires regeneration")
    agent = CompleteDTHAgent(artifact)
    decision = agent.decide((0, 0, 0, 0))
    stage = agent.stage_game((0, 0, 0, 0))

    assert stage.value == pytest.approx(decision.value, abs=1e-12)
    assert stage.drop_policy == pytest.approx(decision.drop_policy)
    assert stage.check_policy == pytest.approx(decision.check_policy)
    assert stage.matrix.shape == (60, 60)
    lower = float(np.min(stage.matrix.T @ stage.drop_policy))
    upper = float(np.max(stage.matrix @ stage.check_policy))
    assert upper - lower <= 1e-6
