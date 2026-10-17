"""The role prior fit, its strength grid, its likelihood, and its live-loader compatibility."""

import numpy as np
import pytest

from arena.policies.adaptive import RoleDirichletOpponent
from hal_loop.prior import (
    ACTIONS,
    ROLES,
    STRENGTH_GRID,
    PlayerCounts,
    dirichlet_multinomial_log_likelihood,
    HistogramEmulator,
    fit_role_prior,
    role_log_likelihood,
    seconds_to_counts,
    select_strength,
    uniform_prior,
    validate_prior,
)


def peaked(center: int, width: float = 2.0) -> np.ndarray:
    seconds = np.arange(1, ACTIONS + 1)
    weights = np.exp(-0.5 * ((seconds - center) / width) ** 2) + 1e-3
    return weights / weights.sum()


def draw(rng, distribution, n) -> np.ndarray:
    return seconds_to_counts(rng.choice(ACTIONS, size=n, p=distribution) + 1)


def population(rng, centers, train=30, held_out=10):
    """Return the summed training counts and per-player counts for players with these centers."""
    players = [PlayerCounts(draw(rng, peaked(c), train), draw(rng, peaked(c), held_out)) for c in centers]
    return sum(p.train for p in players), players


def test_dirichlet_multinomial_matches_sequential_predictive():
    alpha = np.array([0.5, 1.5, 2.0])
    sequence = [0, 2, 2, 1, 0, 2]
    counts, sequential = np.zeros(3), 0.0
    for action in sequence:
        sequential += np.log((alpha[action] + counts[action]) / (alpha.sum() + counts.sum()))
        counts[action] += 1
    assert dirichlet_multinomial_log_likelihood(alpha, counts) == pytest.approx(sequential)


def test_fit_gives_valid_schema_that_the_live_loader_accepts():
    rng = np.random.default_rng(1)
    total, players = population(rng, [44] * 6)
    prior = fit_role_prior({"dropper": total, "checker": total[::-1]}, 0.5, players_by_role={"dropper": players})
    assert prior["schema_version"] == "adaptive-dth-role-prior-v1"
    for role in ROLES:
        mean = np.asarray(prior[role]["mean"])
        assert mean.shape == (ACTIONS,) and np.all(mean > 0) and mean.sum() == pytest.approx(1.0)
        assert prior[role]["strength"] in STRENGTH_GRID
    expected = (total + 0.5) / (total + 0.5).sum()
    assert np.allclose(prior["dropper"]["mean"], expected)
    model = validate_prior(prior)
    assert isinstance(model, RoleDirichletOpponent)
    assert np.allclose(model.predictive("dropper"), expected)


def test_validate_prior_rejects_a_zero_mean():
    prior = uniform_prior()
    prior["checker"]["mean"] = [0.0] + [1.0 / 59] * 59
    with pytest.raises(ValueError):
        validate_prior(prior)


def test_grid_prefers_a_strong_prior_for_alike_players_and_a_weak_one_for_distinct_players():
    rng = np.random.default_rng(2)
    alike_strength, alike_scores = select_strength(*population(rng, [44] * 8))
    distinct_strength, _ = select_strength(*population(rng, [5, 15, 25, 35, 45, 55, 10, 50]))
    assert set(alike_scores) == set(STRENGTH_GRID)
    assert alike_strength in STRENGTH_GRID and distinct_strength in STRENGTH_GRID
    assert alike_strength > distinct_strength


def test_grid_falls_back_without_held_out_moves():
    empty = PlayerCounts(np.ones(ACTIONS), np.zeros(ACTIONS))
    assert select_strength(np.ones(ACTIONS), [empty]) == (STRENGTH_GRID[0], {})


def test_log_likelihood_beats_uniform_on_peaked_play():
    rng = np.random.default_rng(3)
    distributions = {"dropper": peaked(44), "checker": peaked(12)}
    train = {role: draw(rng, distributions[role], 400) for role in ROLES}
    prior = fit_role_prior(train, 0.5)
    moves = {role: draw(rng, distributions[role], 50) for role in ROLES}
    history = {role: draw(rng, distributions[role], 10) for role in ROLES}
    for role in ROLES:
        assert role_log_likelihood(prior, role, moves[role]) > role_log_likelihood(uniform_prior(), role, moves[role])
        uniform = uniform_prior()
        assert role_log_likelihood(uniform, role, moves[role], history[role]) > role_log_likelihood(uniform, role, moves[role])


def test_histogram_emulator_samples_its_histogram():
    counts = {"dropper": np.zeros(ACTIONS), "checker": np.zeros(ACTIONS)}
    counts["dropper"][59] = 1000.0
    emulator = HistogramEmulator(counts)

    class Decision:
        role = "dropper"

    policy = emulator.policy(Decision())
    assert sorted(policy) == list(range(1, ACTIONS + 1))
    assert sum(policy.values()) == pytest.approx(1.0)
    assert max(policy, key=policy.get) == 60
