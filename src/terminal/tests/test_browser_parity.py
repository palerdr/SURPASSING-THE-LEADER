"""The terminal and the local browser server agree.

Each test gives ``python -m terminal play`` and ``python -m arena.web`` the same
options, or the same seed, and compares what they build or record.
"""

from __future__ import annotations

import json

import pytest

from arena.policies.ensemble_hal import EnsembleHalPolicyProvider
from arena.testing import StageAgent as _StageAgent
from arena.tests.test_web_api import _StubHal, _client, _play_out


@pytest.mark.parametrize("choice,expected", [("v1", "PerfectHalOpponentModel"), ("bayesian-v2", "BayesianHalOpponentModel")])
def test_cli_and_browser_select_the_declared_model(monkeypatch, choice, expected):
    from terminal import cli
    from arena.web.__main__ import build_parser as web_parser
    from arena.policies import perfect_hal
    monkeypatch.setattr(perfect_hal, "CompleteDTHAgent", lambda *args, **kwargs: object())
    arguments = ["--hal-agent", "perfect-hal", "--pure-dth", "--perfect-hal-model", choice]
    for args in (cli.build_parser().parse_args(["play", *arguments]), web_parser().parse_args(arguments)):
        provider = cli._make_perfect_hal_provider(args)
        assert type(provider.opponent_model).__name__ == expected
    assert web_parser().parse_args([]).perfect_hal_model == "v1"


def test_cli_and_browser_can_select_ensemble(monkeypatch):
    from terminal import cli
    from arena.web.__main__ import build_parser
    from arena.policies import ensemble_hal
    monkeypatch.setattr(ensemble_hal, "CompleteDTHAgent", lambda *a: _StageAgent())
    args = ["--hal-agent", "perfect-hal", "--perfect-hal-model", "ensemble", "--pure-dth"]
    for parsed in (cli.build_parser().parse_args(["play", *args]), build_parser().parse_args(args)):
        assert isinstance(cli._make_perfect_hal_provider(parsed), EnsembleHalPolicyProvider)


def test_transcript_matches_the_cli_public_history_for_the_same_seed(tmp_path) -> None:
    from terminal import cli

    client = _client(max_half_rounds=3)
    snapshot = _play_out(client, client.get("/api/session").json())
    from_browser = client.get("/api/transcript").json()
    assert from_browser["current_game"]["phase"] == "game_over"

    fresh = _StubHal()
    original_make_hal = cli._make_hal
    original_human = cli._human_action
    cli._make_hal = lambda *_a, **_k: fresh
    cli._human_action = lambda *, actor, role, legal: legal[0]
    try:
        transcript = tmp_path / "cli.json"
        args = cli.build_parser().parse_args(
            ["play", "--seed", "41", "--max-half-rounds", "3", "--transcript", str(transcript)]
        )
        assert cli.command_play(args) == 0
    finally:
        cli._make_hal = original_make_hal
        cli._human_action = original_human
    from_cli = json.loads(transcript.read_text(encoding="utf-8"))["games"][0]
    assert from_browser["current_game"]["public_history"] == from_cli["public_history"]
    assert snapshot["half_rounds"] == from_cli["half_rounds"]
