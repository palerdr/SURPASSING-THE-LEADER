import os
import sys
import math
from types import SimpleNamespace

sys.path.insert(0, os.getcwd())

from stl.commands.compare_ladder import _aggregate_reports, _json_safe


def _counts(wins: int, games: int = 4):
    return {
        "games": games,
        "wins": wins,
        "draws": 0,
        "losses": games - wins,
        "win_rate": wins / games,
        "score_rate": wins / games,
    }


def _report(seed: int, champion_wins: int, candidate_wins: int):
    return {
        "seed": seed,
        "champion": {
            "overall": _counts(champion_wins),
            "opponents": {"safe": _counts(champion_wins)},
        },
        "candidate": {
            "overall": _counts(candidate_wins),
            "opponents": {"safe": _counts(candidate_wins)},
        },
    }


def test_aggregate_reports_sums_checkpoint_deltas():
    aggregate = _aggregate_reports(
        [
            _report(0, champion_wins=2, candidate_wins=3),
            _report(1, champion_wins=1, candidate_wins=2),
        ],
        ["safe"],
    )

    assert aggregate["champion"]["games"] == 8
    assert aggregate["champion"]["wins"] == 3
    assert aggregate["candidate"]["wins"] == 5
    assert aggregate["delta"]["wins_delta"] == 2
    assert aggregate["delta"]["win_rate_delta"] == 0.25
    assert aggregate["opponents"]["safe"]["delta"]["wins_delta"] == 2


def test_json_safe_converts_non_finite_floats():
    assert _json_safe({"llr": math.inf, "nested": [-math.inf]}) == {
        "llr": "inf",
        "nested": ["-inf"],
    }


def test_checkpoint_report_uses_fresh_agent_per_opponent(monkeypatch):
    made_agents = []
    calls = []

    def fake_make_agent(checkpoint, **kwargs):
        agent = object()
        made_agents.append((checkpoint, kwargs, agent))
        return agent

    def fake_make_choose_action(agent):
        return agent

    def fake_run_ladder(hal_choose_action, opponents, n_games, seed):
        assert len(opponents) == 1
        calls.append((hal_choose_action, opponents[0], n_games, seed))
        return {
            opponents[0]: MatchResult(
                games_played=n_games,
                hal_wins=1,
                baku_wins=0,
                draws=0,
                avg_game_length_half_rounds=1.0,
            )
        }

    monkeypatch.setattr(ladder_module, "_make_agent", fake_make_agent)
    monkeypatch.setattr(ladder_module, "make_choose_action", fake_make_choose_action)
    monkeypatch.setattr(ladder_module, "run_ladder", fake_run_ladder)
    args = SimpleNamespace(
        agent_iterations=30,
        policy_ensemble_size=5,
        policy_uniform_mix=0.4,
        resolve_at_critical=True,
        resolve_horizon=3,
        games=1,
    )

    report = _run_checkpoint_report(args, "candidate.pt", ["safe", "pattern_reader"], 7)

    assert report["overall"]["games"] == 2
    assert len(made_agents) == 2
    assert made_agents[0][1]["resolve_at_critical"] is True
    assert made_agents[0][1]["resolve_horizon"] == 3
    assert calls[0][0] is not calls[1][0]
    assert [call[1] for call in calls] == ["safe", "pattern_reader"]
