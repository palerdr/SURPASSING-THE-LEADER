import json
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.combine_checkpoint_ladder_reports import combine_reports


def _counts(wins: int, losses: int) -> dict:
    games = wins + losses
    return {
        "games": games,
        "wins": wins,
        "draws": 0,
        "losses": losses,
        "win_rate": wins / games,
        "score_rate": wins / games,
    }


def _seed_report(seed: int, *, champion_wins: int, candidate_wins: int) -> dict:
    champion = _counts(champion_wins, 2 - champion_wins)
    candidate = _counts(candidate_wins, 2 - candidate_wins)
    return {
        "seed": seed,
        "champion": {
            "overall": champion,
            "opponents": {"pattern_reader": champion},
        },
        "candidate": {
            "overall": candidate,
            "opponents": {"pattern_reader": candidate},
        },
        "delta": {
            "overall": {
                "wins_delta": candidate_wins - champion_wins,
                "losses_delta": champion_wins - candidate_wins,
                "win_rate_delta": (candidate_wins - champion_wins) / 2,
                "score_rate_delta": (candidate_wins - champion_wins) / 2,
            },
            "opponents": {},
        },
    }


def _report(seed_reports: list[dict]) -> dict:
    return {
        "config": {
            "champion_checkpoint": "champ.pt",
            "candidate_checkpoint": "cand.pt",
            "agent_iterations": 200,
            "policy_ensemble_size": 5,
            "policy_uniform_mix": 0.4,
            "resolve_at_critical": True,
            "resolve_horizon": 3,
            "games": 2,
            "seed": seed_reports[0]["seed"],
            "seeds": None,
            "opponents": "pattern_reader",
            "out": "old.json",
        },
        "seeds": [row["seed"] for row in seed_reports],
        "opponents": ["pattern_reader"],
        "seed_reports": seed_reports,
        "elapsed_seconds": 1.25,
    }


def _write(tmp_path, name: str, report: dict):
    path = tmp_path / name
    path.write_text(json.dumps(report), encoding="utf-8")
    return path


def test_combine_reports_sums_compatible_seed_reports(tmp_path):
    first = _write(
        tmp_path,
        "first.json",
        _report([_seed_report(0, champion_wins=0, candidate_wins=0)]),
    )
    second = _write(
        tmp_path,
        "second.json",
        _report(
            [
                _seed_report(1, champion_wins=1, candidate_wins=1),
                _seed_report(2, champion_wins=0, candidate_wins=1),
            ]
        ),
    )

    combined = combine_reports([first, second])

    assert combined["seeds"] == [0, 1, 2]
    assert combined["config"]["seeds"] == "0,1,2"
    assert combined["aggregate"]["champion"]["wins"] == 1
    assert combined["aggregate"]["candidate"]["wins"] == 2
    assert combined["aggregate"]["delta"]["wins_delta"] == 1
    assert combined["combined_from"] == [str(first), str(second)]


def test_combine_reports_rejects_duplicate_seed(tmp_path):
    first = _write(tmp_path, "first.json", _report([_seed_report(0, champion_wins=0, candidate_wins=0)]))
    second = _write(tmp_path, "second.json", _report([_seed_report(0, champion_wins=1, candidate_wins=1)]))

    with pytest.raises(ValueError, match="duplicate seed"):
        combine_reports([first, second])
