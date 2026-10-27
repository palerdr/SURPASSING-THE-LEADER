from __future__ import annotations

import json
from pathlib import Path

import pytest

from arena.contracts import PublicGameOutcome
from arena.policies.perfect_hal.model import ACTION_COUNT
from arena.policies.perfect_hal.policy import PerfectHalPolicyProvider
from arena.testing import (
    StageAgent as _StageAgent,
    make_decision as _decision,
    make_reveal as _reveal,
)


def test_provider_places_all_mass_on_the_forecast_best_response(tmp_path: Path) -> None:
    agent = _StageAgent()
    provider = PerfectHalPolicyProvider(tmp_path, agent=agent)

    first = provider.policy(_decision(role="dropper"))
    assert first == pytest.approx(
        {action: 1.0 / ACTION_COUNT for action in range(1, ACTION_COUNT + 1)}
    )
    provider.observe(_reveal(opponent_action=7, self_action=3))

    second = provider.policy(_decision(role="dropper"))
    assert second == {7: pytest.approx(1.0)}
    assert provider.last_decision is not None
    assert provider.last_decision.opponent_policy[6] > 0.5
    assert provider.last_decision.expected_exploit_gain > 0.0
    assert provider.last_decision.best_response_actions == (7,)
    assert agent.states == [(12, 60, 24, 120), (12, 60, 24, 120)]


def test_provider_preserves_session_memory_across_games_and_seat_names(
    tmp_path: Path,
) -> None:
    provider = PerfectHalPolicyProvider(tmp_path, agent=_StageAgent())
    provider.reset_game()
    provider.policy(_decision(role="dropper", actor_name="Hal"))
    provider.observe(
        _reveal(
            self_role="dropper",
            self_name="Hal",
            opponent_name="Baku",
            opponent_action=13,
            game_over=True,
        )
    )
    provider.end_game(PublicGameOutcome(0, "Hal", 1))
    provider.reset_game()

    next_game = provider.policy(_decision(role="dropper", actor_name="Baku"))
    assert next_game == {13: pytest.approx(1.0)}
    assert provider.has_session_memory
    provider.observe(
        _reveal(
            self_role="dropper",
            self_name="Baku",
            opponent_name="Hal",
            opponent_action=13,
            game_index=1,
        )
    )

    provider.reset_session()
    assert not provider.has_session_memory
    assert provider.last_decision is None


def test_provider_fails_closed_before_stage_lookup_outside_pure_dth(
    tmp_path: Path,
) -> None:
    agent = _StageAgent()
    provider = PerfectHalPolicyProvider(tmp_path, agent=agent)
    with pytest.raises(ValueError, match="pure DTH only"):
        provider.policy(_decision(legal_seconds=tuple(range(1, ACTION_COUNT + 2))))
    assert agent.states == []


def test_provider_requires_one_public_reveal_per_decision(tmp_path: Path) -> None:
    provider = PerfectHalPolicyProvider(tmp_path, agent=_StageAgent())
    provider.policy(_decision())
    with pytest.raises(RuntimeError, match="twice before a reveal"):
        provider.policy(_decision())
    with pytest.raises(RuntimeError, match="unrevealed action"):
        provider.reset_game()


def test_diagnostics_are_json_serializable_and_explicitly_unrestricted(
    tmp_path: Path,
) -> None:
    provider = PerfectHalPolicyProvider(tmp_path, agent=_StageAgent())
    provider.policy(_decision())
    diagnostics = provider.experiment_diagnostics()
    assert diagnostics["pure_dth_only"] is True
    assert diagnostics["public_history_only"] is True
    assert diagnostics["unrestricted_best_response"] is True
    assert diagnostics["equilibrium_safety_blend"] is False
    assert "Perfect Hal" in provider.match_summary()
    json.dumps(diagnostics)
