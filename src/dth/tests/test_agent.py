"""Contract tests for the bounded-resolve agent (docs/AGENT_GOAL.md M3)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from dth.agent import BoundedResolveAgent, MoveDecision, ResolveBudget
from dth.exact_agent import ExactDTHAgent
from dth.solver import solve
from dth.tablebase import CertifiedTablebase


@dataclass(frozen=True)
class _StubConfig:
    transition_class_head: bool = False


class _ZeroNetwork:
    """Deterministic scalar stub whose leaves equal the horizon-zero cutoff."""

    config = _StubConfig()

    def values(self, states, horizon):
        del horizon
        return np.zeros(len(states), dtype=np.float64)


class _ClassStub:
    config = _StubConfig(transition_class_head=True)

    def __init__(self) -> None:
        self.class_calls = 0
        self.scalar_calls = 0

    def values(self, states, horizon):
        del horizon
        self.scalar_calls += 1
        return np.zeros(len(states), dtype=np.float64)

    def class_matrix_values(self, states, horizon):
        del horizon
        self.class_calls += 1
        return np.zeros(len(states), dtype=np.float64)


def test_zero_leaf_resolve_matches_the_finite_horizon_solver() -> None:
    state = (180, 60, 180, 60)
    agent = BoundedResolveAgent(
        network=_ZeroNetwork(),
        budget=ResolveBudget(deadline_seconds=30.0, max_depth=2),
    )

    with agent:
        decision = agent.decide(state)

    assert decision.provenance == "approximate"
    assert decision.resolve_depth == 2
    # Zero-valued leaves are exactly the horizon-zero live cutoff, so the
    # depth-2 resolve must reproduce the exact finite-horizon value.
    assert decision.value == pytest.approx(solve(state, 2).value, abs=1e-9)


def test_decisions_are_deterministic_for_a_fixed_state() -> None:
    agent = BoundedResolveAgent(
        network=_ZeroNetwork(),
        budget=ResolveBudget(deadline_seconds=30.0, max_depth=2),
    )
    with agent:
        first = agent.decide((119, 120, 179, 60))
        second = agent.decide((119, 120, 179, 60))

    assert first.value == second.value
    assert first.drop_policy == second.drop_policy
    assert first.check_policy == second.check_policy
    assert first.provenance == second.provenance


def test_terminal_dominated_state_resolves_exactly_with_no_leaves() -> None:
    agent = BoundedResolveAgent(
        network=_ZeroNetwork(),
        budget=ResolveBudget(deadline_seconds=30.0, max_depth=1),
    )
    with agent:
        decision = agent.decide((299, 300, 299, 300))

    assert decision.value == pytest.approx(1.0)
    assert decision.exact_leaf_fraction == 1.0


def test_band_leaves_take_priority_over_the_network(tmp_path) -> None:
    database = tmp_path / "band.sqlite"
    with CertifiedTablebase(database) as tablebase:
        # Offline deposit workflow in miniature: one root, one manifest, and
        # the quotient classes (60,1) and (1,1)..(1,59) become durable.
        ExactDTHAgent(tablebase).evaluate(
            (240, 0, 299, 0),
            deadline_seconds=30.0,
            max_new_solutions=200,
            census_max_states=1_000,
        )

    agent = BoundedResolveAgent(
        tablebase_path=database,
        network=_ZeroNetwork(),
        budget=ResolveBudget(deadline_seconds=30.0, max_depth=1),
    )
    with agent:
        decision = agent.decide((239, 0, 299, 0))

    # Fifty-nine successful-lag children land in solved band classes; the
    # (1, 60) class and the revival child still need the network.
    assert decision.provenance == "approximate"
    assert decision.exact_leaf_fraction == pytest.approx(59.0 / 61.0)


def test_band_state_returns_a_complete_game_certificate(tmp_path) -> None:
    database = tmp_path / "band.sqlite"
    with CertifiedTablebase(database) as tablebase:
        ExactDTHAgent(tablebase).evaluate(
            (299, 0, 299, 0),
            deadline_seconds=10.0,
            max_new_solutions=10,
            census_max_states=100,
        )

    agent = BoundedResolveAgent(
        tablebase_path=database,
        network=_ZeroNetwork(),
        budget=ResolveBudget(deadline_seconds=30.0),
    )
    with agent:
        decision = agent.decide((299, 17, 299, 4))

    assert decision.provenance == "complete-game-exact"
    assert decision.value == pytest.approx(1.0)
    assert decision.saddle_gap <= 1e-6


def test_missing_network_falls_back_to_a_finite_certificate() -> None:
    agent = BoundedResolveAgent(
        budget=ResolveBudget(deadline_seconds=5.0, finite_fallback_horizon=1)
    )
    with agent:
        decision = agent.decide((0, 0, 0, 0))

    assert decision.provenance == "finite-horizon-exact"
    assert decision.horizon == 1
    assert decision.saddle_gap <= 1e-6


def test_deadline_caps_the_resolve_depth_but_keeps_an_answer() -> None:
    agent = BoundedResolveAgent(
        network=_ZeroNetwork(),
        budget=ResolveBudget(deadline_seconds=0.3, max_depth=3),
    )
    with agent:
        decision = agent.decide((0, 240, 0, 240))

    assert decision.provenance == "approximate"
    assert decision.resolve_depth is not None and decision.resolve_depth < 3
    assert decision.elapsed_seconds < 5.0


def test_class_head_stub_is_used_for_small_frontiers() -> None:
    stub = _ClassStub()
    agent = BoundedResolveAgent(
        network=stub,
        budget=ResolveBudget(
            deadline_seconds=30.0,
            max_depth=1,
            class_matrix_leaf_limit=1_000,
        ),
    )
    with agent:
        agent.decide((180, 60, 180, 60))

    assert stub.class_calls == 1
    assert stub.scalar_calls == 0


def test_decision_is_a_frozen_record() -> None:
    decision = MoveDecision(
        state=(0, 0, 0, 0),
        value=0.0,
        drop_policy=(),
        check_policy=(),
        provenance="approximate",
        scope_detail="test",
        horizon=None,
        saddle_gap=0.0,
        resolve_depth=1,
        exact_leaf_fraction=None,
        elapsed_seconds=0.0,
    )
    with pytest.raises(AttributeError):
        decision.value = 1.0  # type: ignore[misc]


def test_approximate_matrix_solver_reports_gap_instead_of_certifying(
    monkeypatch,
) -> None:
    import dth.agent as agent_module

    matrix = np.zeros((60, 60))
    matrix[0, 0] = 1.0

    def reject(_matrix):
        raise RuntimeError("LP saddle gap too large: 3e-06")

    monkeypatch.setattr(agent_module, "solve_certified_matrix", reject)
    value, drop, check, gap = agent_module.solve_approximate_matrix(matrix)

    assert np.isfinite(value)
    assert drop.shape == (60,) and check.shape == (60,)
    assert 0.0 <= gap < 0.01


def test_resolve_labels_harvests_interiors_and_excludes_leaves() -> None:
    state = (180, 60, 180, 60)
    agent = BoundedResolveAgent(
        network=_ZeroNetwork(),
        budget=ResolveBudget(deadline_seconds=30.0, max_depth=2),
    )
    with agent:
        labels = agent.resolve_labels(state, depth=2, deadline_seconds=30.0)

    assert state in labels
    value, drop, check, gap = labels[state]
    # Zero-valued leaves reproduce the horizon-two exact value at the root.
    assert value == pytest.approx(solve(state, 2).value, abs=1e-9)
    assert drop.shape == (60,) and check.shape == (60,)
    assert gap <= 1e-6
    # Interiors only: every harvested state was solved with children below it,
    # and one depth-two resolve harvests the root plus its level-one children.
    assert 1 < len(labels) <= 62


def test_resolve_labels_skips_certified_band_states(tmp_path) -> None:
    database = tmp_path / "band.sqlite"
    with CertifiedTablebase(database) as tablebase:
        ExactDTHAgent(tablebase).evaluate(
            (299, 0, 299, 0),
            deadline_seconds=10.0,
            max_new_solutions=10,
            census_max_states=100,
        )

    agent = BoundedResolveAgent(
        tablebase_path=database,
        network=_ZeroNetwork(),
        budget=ResolveBudget(deadline_seconds=30.0, max_depth=2),
    )
    with agent:
        labels = agent.resolve_labels(
            (299, 12, 299, 30), depth=2, deadline_seconds=30.0
        )

    assert labels == {}


def test_network_leaf_model_answers_from_the_play_head_when_present(
    tmp_path,
) -> None:
    import torch

    from dth.agent import NetworkLeafModel
    from dth.network import DTHNetworkConfig, DTHPolicyValueNet

    def checkpoint_with(config: DTHNetworkConfig, path) -> None:
        model = DTHPolicyValueNet(config)
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.zero_()
            model.value_head.bias.fill_(-0.3)
            if model.play_value_head is not None:
                model.play_value_head.bias.fill_(0.7)
        torch.save(
            {"state_dict": model.state_dict(), "model_config": config.to_dict()},
            path,
        )

    playless_path = tmp_path / "playless.pt"
    checkpoint_with(
        DTHNetworkConfig(hidden_width=4, hidden_layers=1), playless_path
    )
    play_path = tmp_path / "play.pt"
    checkpoint_with(
        DTHNetworkConfig(hidden_width=4, hidden_layers=1, play_value_head=True),
        play_path,
    )

    states = [(0, 0, 0, 0)]
    playless_values = NetworkLeafModel(playless_path).values(states, 4)
    play_values = NetworkLeafModel(play_path).values(states, 4)

    assert playless_values[0] == pytest.approx(float(np.tanh(-0.3)))
    assert play_values[0] == pytest.approx(float(np.tanh(0.7)))
