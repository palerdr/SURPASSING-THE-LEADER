import os
import sys
from dataclasses import asdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training.trace_divergence import analyze_trace_divergences, first_divergence_report
from training.tournament import _default_starting_game


def _record_to_json(record):
    row = asdict(record)
    row["result"] = record.result.value
    return row


def _row(*, winner: str, records, game_seed: int = 123, cause: str = "unknown"):
    return {
        "seed": 0,
        "game_index": 0,
        "game_seed": game_seed,
        "winner": winner,
        "cause": cause,
        "half_rounds": len(records),
        "history": [_record_to_json(record) for record in records],
    }


def _divergent_rows():
    champion_game = _default_starting_game(123)
    champion_first = champion_game.resolve_half_round(4, 60, survived_outcome=None)
    champion_second = champion_game.resolve_half_round(30, 2, survived_outcome=True)

    candidate_game = _default_starting_game(123)
    candidate_first = candidate_game.resolve_half_round(4, 60, survived_outcome=None)
    candidate_second = candidate_game.resolve_half_round(30, 60, survived_outcome=None)

    return (
        _row(
            winner="Hal",
            cause="baku_failed_check",
            records=[champion_first, champion_second],
        ),
        _row(
            winner="Baku",
            cause="hal_failed_check",
            records=[candidate_first, candidate_second],
        ),
    )


def _chance_only_rows():
    champion_game = _default_starting_game(123)
    champion_first = champion_game.resolve_half_round(4, 60, survived_outcome=None)
    champion_second = champion_game.resolve_half_round(30, 1, survived_outcome=False)

    candidate_game = _default_starting_game(123)
    candidate_first = candidate_game.resolve_half_round(4, 60, survived_outcome=None)
    candidate_second = candidate_game.resolve_half_round(30, 1, survived_outcome=True)

    return (
        _row(
            winner="Hal",
            cause="baku_failed_check",
            records=[champion_first, champion_second],
        ),
        _row(
            winner="Baku",
            cause="hal_failed_check",
            records=[candidate_first, candidate_second],
        ),
    )


def _mixed_action_rows():
    champion_game = _default_starting_game(123)
    champion_first = champion_game.resolve_half_round(3, 19, survived_outcome=None)

    candidate_game = _default_starting_game(123)
    candidate_first = candidate_game.resolve_half_round(4, 54, survived_outcome=None)

    return (
        _row(
            winner="Hal",
            cause="baku_failed_check",
            records=[champion_first],
        ),
        _row(
            winner="Baku",
            cause="hal_failed_check",
            records=[candidate_first],
        ),
    )


def test_first_divergence_report_replays_shared_pre_state():
    champion, candidate = _divergent_rows()

    report = first_divergence_report(champion, candidate)

    assert report is not None
    assert report["history_index"] == 1
    assert report["kind"] == "record"
    assert report["hal_role"] == "checker"
    assert report["shared_public_state"] is True
    assert report["champion_record"]["check_time"] == 2
    assert report["candidate_record"]["check_time"] == 60
    assert report["champion_pre_state"]["state"]["current_half"] == 2
    assert "check_time" in report["diff_fields"]


def test_analyze_trace_divergences_summarizes_regression_and_target_hint():
    champion, candidate = _divergent_rows()
    trace_report = {
        "config": {"opponent": "pattern_reader"},
        "summaries": {"champion": {"wins": 1}, "candidate": {"wins": 0}},
        "games": {"champion": [champion], "candidate": [candidate]},
    }

    report = analyze_trace_divergences(trace_report)

    assert report["summary"]["paired_games"] == 1
    assert report["summary"]["candidate_score_delta_total"] == -2
    assert report["summary"]["outcomes"] == {"candidate_regression": 1}
    assert report["summary"]["candidate_loss_causes"] == {"hal_failed_check": 1}
    assert report["summary"]["target_hints"] == 1
    assert report["target_hints"][0]["history_index"] == 1
    assert report["target_hints"][0]["trajectory"] == "candidate"
    assert report["target_hints"][0]["reason"] == (
        "candidate_regression_first_action_divergence"
    )
    assert report["paired_games"][0]["outcome_class"] == "candidate_regression"


def test_analyze_trace_divergences_does_not_target_chance_only_regression():
    champion, candidate = _chance_only_rows()
    trace_report = {
        "games": {"champion": [champion], "candidate": [candidate]},
    }

    report = analyze_trace_divergences(trace_report)

    assert report["summary"]["outcomes"] == {"candidate_regression": 1}
    assert report["summary"]["target_hints"] == 0
    assert report["summary"]["non_actionable_regressions"] == 1
    assert report["summary"]["non_actionable_regression_reasons"] == {
        "chance_only": 1
    }
    assert report["paired_games"][0]["first_divergence"]["diff_fields"] == [
        "result",
        "survived",
    ]
    assert report["paired_games"][0]["actionability"] == {
        "actionable": False,
        "reason": "chance_only",
    }


def test_analyze_trace_divergences_does_not_target_mixed_actor_actions():
    champion, candidate = _mixed_action_rows()
    trace_report = {
        "games": {"champion": [champion], "candidate": [candidate]},
    }

    report = analyze_trace_divergences(trace_report)

    assert report["summary"]["outcomes"] == {"candidate_regression": 1}
    assert report["summary"]["target_hints"] == 0
    assert report["summary"]["non_actionable_regressions"] == 1
    assert report["summary"]["non_actionable_regression_reasons"] == {
        "mixed_actor_action": 1
    }
    assert report["paired_games"][0]["first_divergence"]["diff_fields"] == [
        "drop_time",
        "check_time",
    ]
    assert report["paired_games"][0]["actionability"] == {
        "actionable": False,
        "reason": "mixed_actor_action",
    }
