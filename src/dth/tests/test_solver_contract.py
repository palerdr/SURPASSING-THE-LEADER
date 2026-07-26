import pytest

from dth.solver import (
    TState,
    clear_solver_cache,
    horizon_one_cache_info,
    revival_model,
    solve,
    transition,
)


def test_successful_check_below_cylinder_cap_remains_live() -> None:
    assert transition((298, 7, 11, 13), 1, 1) == (
        (1.0, (11, 13, 299, 7)),
    )


@pytest.mark.parametrize(
    ("state", "drop", "check"),
    [
        ((299, 0, 0, 0), 1, 1),  # exactly 300
        ((299, 0, 0, 0), 1, 60),  # above 300 before the physical cap
        ((240, 0, 0, 0), 1, 60),  # exactly 300 by a 60-ST success
    ],
)
def test_successful_check_dumps_at_cylinder_cap(
    state: tuple[int, int, int, int],
    drop: int,
    check: int,
) -> None:
    assert transition(state, drop, check) == ((1.0, TState.W),)


def test_failed_check_dose_threshold_is_inclusive() -> None:
    assert revival_model(240, 0) == 0.0
    assert transition((240, 0, 0, 0), 2, 1) == ((1.0, TState.W),)


def test_resulting_total_ttd_exactly_300_remains_revival_eligible() -> None:
    probability = revival_model(0, 240)
    assert probability > 0.0
    assert transition((0, 240, 0, 0), 2, 1) == (
        (probability, (0, 0, 0, 300)),
        (1.0 - probability, TState.W),
    )


def test_resulting_total_ttd_above_300_is_fatal() -> None:
    assert revival_model(0, 241) == 0.0
    assert transition((0, 241, 0, 0), 2, 1) == ((1.0, TState.W),)


def test_horizon_one_does_not_memoize_cutoff_children() -> None:
    clear_solver_cache()
    solve((0, 0, 0, 0), 1)

    info = solve.cache_info()
    assert info.misses == 1
    assert info.currsize == 1


def test_horizon_one_reuses_exact_checker_equivalence_class() -> None:
    clear_solver_cache()
    first = solve((179, 60, 0, 0), 1)
    second = solve((179, 60, 241, 300), 1)

    assert first == second
    assert horizon_one_cache_info().misses == 1
