"""The likelihood and exploitation gates on synthetic evaluations."""

import numpy as np
import pytest
from dagster import AssetKey, asset, materialize

from hal_loop import gates
from hal_loop.assets.checks import exploitation_gate, likelihood_gate
from hal_loop.assets.model import fit_from_histograms
from hal_loop.evaluate import evaluate_candidate
from hal_loop.prior import ACTIONS, ROLES, fit_role_prior, seconds_to_counts
from hal_loop.resources import PathsResource
from hal_loop.warehouse import Histograms


def peaked(center: int) -> np.ndarray:
    seconds = np.arange(1, ACTIONS + 1)
    weights = np.exp(-0.5 * ((seconds - center) / 2.0) ** 2) + 1e-3
    return weights / weights.sum()


def fitted(evaluation: dict) -> dict:
    """Add the card verdict that offline_evaluation copies into each evaluation."""
    return {**evaluation, "verdict": "fitted"}


def synthetic_histograms(seed: int = 5, players: int = 12) -> Histograms:
    """Players who drop near 44 and check near 12, split into train and validation."""
    rng = np.random.default_rng(seed)
    centers = {"dropper": 44, "checker": 12}
    counts = {}
    for p in range(players):
        for role in ROLES:
            for split, n in (("train", 20), ("validation", 8), ("test", 8)):
                moves = rng.choice(ACTIONS, size=n, p=peaked(centers[role])) + 1
                counts[(f"player-{p}", role, split)] = seconds_to_counts(moves)
    return Histograms(counts)


@pytest.fixture(scope="module")
def histograms():
    return synthetic_histograms()


def test_candidate_fitted_on_play_passes(histograms):
    candidate, selection = fit_from_histograms(histograms)
    assert all(selection[role]["strength"] >= 10 for role in ROLES)
    evaluation = fitted(evaluate_candidate(histograms, candidate, None))
    assert evaluation["production_source"] == "uniform"
    assert (evaluation["split"], evaluation["players"]) == ("test", 12)
    passed, description = gates.likelihood_gate(evaluation)
    assert passed, description
    assert evaluation["pooled"]["interval"][0] > 0


def test_misfit_candidate_fails_against_good_production(histograms):
    production = fit_role_prior(histograms.population("train"), 0.5)
    # The misfit candidate expects drops near 16 and checks near 48.
    wrong = {"dropper": seconds_to_counts([16] * 200), "checker": seconds_to_counts([48] * 200)}
    candidate = fit_role_prior(wrong, 0.5)
    evaluation = fitted(evaluate_candidate(histograms, candidate, production))
    assert evaluation["production_source"] == "current"
    passed, description = gates.likelihood_gate(evaluation)
    assert not passed and "does not beat" in description
    assert evaluation["pooled"]["difference"] < 0


def test_bootstrap_is_reproducible(histograms):
    candidate = fit_role_prior(histograms.population("train"), 0.5)
    first = evaluate_candidate(histograms, candidate, None)
    assert evaluate_candidate(histograms, candidate, None)["pooled"] == first["pooled"]


def test_likelihood_gate_requires_moves():
    empty = {"pooled": {"difference": None, "interval": [None, None], "moves": 0.0}, "players": 0,
             "production_source": "uniform", "verdict": "fitted"}
    assert gates.likelihood_gate(empty)[0] is False


def test_evaluation_scores_the_test_split_alone(histograms):
    candidate = fit_from_histograms(histograms)[0]
    evaluation = evaluate_candidate(histograms, candidate, None)
    test_moves = sum(counts.sum() for (_, _, split), counts in histograms.counts.items() if split == "test")
    assert evaluation["pooled"]["moves"] == test_moves
    # Without test moves, nothing is scored, although validation moves exist.
    no_test = Histograms({key: counts for key, counts in histograms.counts.items() if key[2] != "test"})
    assert evaluate_candidate(no_test, candidate, None)["pooled"]["moves"] == 0


def test_one_player_fails_the_gate(histograms):
    # One player's resamples are all identical, so the interval says nothing about the population.
    one = Histograms({key: counts for key, counts in histograms.counts.items() if key[0] == "player-0"})
    evaluation = fitted(evaluate_candidate(one, fit_from_histograms(histograms)[0], None))
    assert evaluation["players"] == 1
    passed, description = gates.likelihood_gate(evaluation)
    assert not passed and "too little scored data" in description


def test_gate_requires_minimum_moves(histograms):
    evaluation = fitted(evaluate_candidate(histograms, fit_from_histograms(histograms)[0], None))
    assert gates.likelihood_gate(evaluation)[0]
    short = {**evaluation, "pooled": {**evaluation["pooled"], "moves": gates.MINIMUM_SCORED_MOVES - 1.0}}
    assert not gates.likelihood_gate(short)[0]


@pytest.mark.parametrize("verdict", ["insufficient data", None])
def test_gate_requires_a_fitted_verdict(histograms, verdict):
    evaluation = evaluate_candidate(histograms, fit_from_histograms(histograms)[0], None)
    if verdict is not None:
        evaluation["verdict"] = verdict
    passed, description = gates.likelihood_gate(evaluation)
    assert not passed and "card verdict" in description


def test_exploitation_gate_tolerance():
    assert gates.exploitation_gate(0.495, 0.50)[0]
    assert not gates.exploitation_gate(0.48, 0.50)[0]


def test_checks_pass_and_skip_the_simulation(histograms, tmp_path):
    evaluation = fitted(evaluate_candidate(histograms, fit_from_histograms(histograms)[0], None))
    evaluation.update(version="abc123", directory=str(tmp_path))

    @asset(name="offline_evaluation")
    def synthetic_evaluation() -> dict:
        return evaluation

    result = materialize(
        [synthetic_evaluation, likelihood_gate, exploitation_gate],
        resources={"paths": PathsResource(data_dir=str(tmp_path))},
    )
    assert result.success
    checks = {event.check_name: event for event in result.get_asset_check_evaluations()}
    assert set(checks) == {"likelihood_gate"}
    assert checks["likelihood_gate"].passed
    assert checks["likelihood_gate"].asset_key == AssetKey("offline_evaluation")
