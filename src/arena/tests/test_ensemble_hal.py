from dataclasses import replace
import json

import numpy as np
import pytest

from arena.policies.ensemble_hal import EnsembleHalConfig, EnsembleHalPolicyProvider, update_weights
from arena.tests.test_perfect_hal import _StageAgent, _decision, _reveal


def make_provider(matrix=None, config=EnsembleHalConfig()):
    return EnsembleHalPolicyProvider("unused", agent=_StageAgent(matrix), ensemble=config)


@pytest.mark.parametrize("role", ["dropper", "checker"])
def test_mix_frozen_proposals_and_score_both_against_reveal(role, monkeypatch):
    matrix = np.zeros((60, 60))
    matrix[0, 6], matrix[1, 6] = 0.8, -0.4
    matrix[6, 0], matrix[6, 1] = -0.6, 0.2
    provider = make_provider(matrix)
    # Preserve child lifecycle while controlling the two proposed policies.
    for child, policy in zip(provider.providers, ({1: 0.75, 2: 0.25}, {2: 1.0})):
        original = child.policy
        monkeypatch.setattr(child, "policy", lambda d, p=policy, original=original: (original(d), p)[1])
    provider.weights[role] = np.array([0.7, 0.3])
    assert provider.policy(_decision(role=role)) == pytest.approx({1: 0.525, 2: 0.475})
    assert len(provider.agent.states) == 1
    assert provider.reveals == 0
    assert provider.weights[role] == pytest.approx([0.7, 0.3])
    proposals = np.array(provider.latest["policies"])
    oriented = matrix if role == "dropper" else -matrix.T
    rewards = (proposals @ oriented[:, 6] + 1) / 2
    provider.observe(_reveal(self_role=role, self_action=2, opponent_action=7))
    assert provider.latest["revealed_rewards"] == pytest.approx(rewards)
    expected = update_weights(np.array([0.7, 0.3]), rewards, provider.config)
    assert provider.weights[role] == pytest.approx(expected)
    other_role = "checker" if role == "dropper" else "dropper"
    assert provider.weights[other_role] == pytest.approx([0.5, 0.5])
    assert [p.opponent_model.total_observations for p in provider.providers] == [1, 1]
    json.dumps(provider.experiment_diagnostics())


def test_public_history_lifecycle_and_invalid_reveals_do_not_learn():
    provider = make_provider()
    provider.policy(_decision())
    for operation in (provider.reset_game, provider.reset_session, lambda: provider.policy(_decision())):
        with pytest.raises(RuntimeError, match="unrevealed"):
            operation()
    for reveal in (_reveal(opponent_action=61), _reveal(self_role="checker"), _reveal(game_index=-1)):
        with pytest.raises(ValueError):
            provider.observe(reveal)
        assert [p.opponent_model.total_observations for p in provider.providers] == [0, 0]
    provider.observe(_reveal())
    with pytest.raises(RuntimeError, match="pending"):
        provider.observe(_reveal())
    provider.weights["dropper"] = np.array([0.8, 0.2])
    provider.reset_game()
    assert provider.has_session_memory
    assert provider.weights["dropper"] == pytest.approx([0.8, 0.2])
    provider.reset_session()
    assert not provider.has_session_memory
    assert provider.weights["dropper"] == pytest.approx([0.5, 0.5])


def test_leap_rejected_before_stage_lookup():
    provider = make_provider()
    with pytest.raises(ValueError, match="pure DTH"):
        provider.policy(replace(_decision(), turn_duration=61))
    assert provider.agent.states == []


def test_fixed_share_recovers_after_reversal_and_fixed_mix_stays_equal():
    config = EnsembleHalConfig()
    weights = np.full(2, 0.5)
    for _ in range(100):
        weights = update_weights(weights, np.array([1, 0]), config)
        assert min(weights) >= config.share / 2
    assert weights[0] > 0.97
    for _ in range(5):
        weights = update_weights(weights, np.array([0, 1]), config)
    assert weights[1] > 0.97
    assert update_weights(np.full(2, 0.5), np.array([1, 0]), EnsembleHalConfig(learning_rate=0)) == pytest.approx([0.5, 0.5])


def test_cli_and_browser_can_select_ensemble(monkeypatch):
    from arena import cli
    from arena.web.__main__ import build_parser
    from arena.policies import ensemble_hal
    monkeypatch.setattr(ensemble_hal, "CompleteDTHAgent", lambda *a: _StageAgent())
    args = ["--hal-agent", "perfect-hal", "--perfect-hal-model", "ensemble", "--pure-dth"]
    for parsed in (cli.build_parser().parse_args(["play", *args]), build_parser().parse_args(args)):
        assert isinstance(cli._make_perfect_hal_provider(parsed), EnsembleHalPolicyProvider)


def test_rewards_do_not_depend_on_the_action_sampled_for_hal():
    providers = [make_provider(np.arange(3600).reshape(60, 60) / 3600) for _ in range(2)]
    for provider, own_action in zip(providers, (1, 60)):
        provider.policy(_decision())
        provider.observe(_reveal(self_action=own_action, opponent_action=7))
    assert providers[0].latest["revealed_rewards"] == pytest.approx(providers[1].latest["revealed_rewards"])
    assert providers[0].weights["dropper"] == pytest.approx(providers[1].weights["dropper"])


def test_evaluation_clusters_emulator_replicates_and_counts_stops_as_nonwins():
    from arena.policies.evaluate_ensemble_hal import CONTROLLERS, summarize
    sessions = [{"source": "human_fitted", "family": "human_response", "cluster": 1,
                 "controllers": {name: [{"won": True if name == "ensemble" else None, "seat": seat}
                                        for seat in ("Hal", "Baku")] for name in CONTROLLERS}}
                for _ in range(4)]
    results = summarize(sessions, {"bootstrap_seed": 42, "bootstrap_replicates": 100})["human_fitted"]
    assert results["ensemble"]["wins"] == 8
    assert results["old"]["stopped"] == 8
    assert results["old"]["win_rate"] == 0
    comparison = results["ensemble"]["paired_win_rate_vs"]["old"]
    assert comparison == {"mean": 1.0, "interval_95": None, "identities": 1}
