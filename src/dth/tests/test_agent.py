"""Play-time DTH agent uses only the completed exact artifact."""

from __future__ import annotations

import pytest

import dth.agent as agent_module
from dth.agent import CompleteDTHAgent, MoveDecision


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
