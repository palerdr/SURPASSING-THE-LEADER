"""Tests of the Bayesian Perfect Hal v2 evaluator."""

from __future__ import annotations

import pytest

from hal_lab.experiments.perfect_hal_bayes_v2.evaluate_bayesian_hal import fit_priors, human_observation, split_games, summarize_sessions


def move(action):
    return {"dropper": "Hal", "checker": "Baku", "drop_second": 5, "check_second": action,
            "public_state_before": {"turn_duration": 60, "players": [
                {"name": name, "cylinder_seconds": 0, "ttd_seconds": 0} for name in ("Hal", "Baku")]}}


def test_holdout_games_and_identity_exclusion_prevent_prior_leakage():
    protocol = {"human_train_fraction": 0.6, "human_validation_fraction": 0.2, "minimum_player_games": 5}
    games = [{"player": p, "ordinal": p * 100 + i, "public_history": [move(4 if i < 6 else 59)]}
             for p in (1, 2) for i in range(10)]
    parts = split_games(games, protocol)
    assert [len(parts[1][s]) for s in ("train", "validation", "test")] == [6, 2, 2]
    a = fit_priors(parts, 0.5, excluded_player=1)
    for g in parts[1]["train"] + parts[2]["test"]:
        g["public_history"] = [move(60)] * 100
    b = fit_priors(parts, 0.5, excluded_player=1)
    assert a["checker"] == pytest.approx(b["checker"])
    assert a["checker"][3] > a["checker"][58]


def test_log_adapter_excludes_leap_and_never_supplies_current_action():
    recorded = move(5)
    first = human_observation(recorded)[0]
    recorded["check_second"] = 59
    assert human_observation(recorded)[0] == first
    assert first.native_state is None
    recorded["public_state_before"]["turn_duration"] = 61
    assert human_observation(recorded) is None


def test_human_emulator_variants_share_one_bootstrap_identity():
    from hal_lab.experiments.perfect_hal_bayes_v2.evaluate_bayesian_hal import CONTROLLERS
    sessions = [{"source": "human_fitted", "family": family, "cluster": 1,
                 "controllers": {name: [{"won": name != "exact", "seat": "Hal"},
                                         {"won": name != "exact", "seat": "Baku"}] for name in CONTROLLERS}}
                for family in ("categorical", "response")]
    protocol = {"bootstrap_seed": 5, "bootstrap_replicates": 50, "games_per_identity": 2}
    result = summarize_sessions(sessions, protocol)["human_fitted"]["bayes"]
    assert result["paired_vs_exact"]["identities"] == 1
    assert result["paired_vs_exact"]["interval_95"] is None
    assert result["paired_vs_exact"]["mean"] == 1
