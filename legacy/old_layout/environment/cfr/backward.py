"""Analytic stage-transition map for backward induction (tablebase pilot).

At a half-round decision state, the 60x60 (61x60 in the leap turn) joint
action space collapses to at most 60 distinct outcomes:

- success cells (check >= drop) depend ONLY on ST = max(1, check - drop):
  the checker's cylinder grows by ST; at >= CYLINDER_MAX the injection is
  immediate and always fatal (death_duration = 300 => survival probability
  is exactly 0 in src/Referee.py).
- every fail cell (check < drop) produces the SAME death event:
  duration min(cylinder + 60, CYLINDER_MAX), survival probability taken
  from the engine's own referee (invariant G4: no reimplemented chance).

``analytic_stage_outcomes`` computes that outcome set by arithmetic plus
one referee call; ``verify_stage_outcomes_against_engine`` proves it
matches live ``Game.resolve_half_round`` probes outcome class by outcome
class. Backward-induction solvers assemble per-state payoff matrices from
these outcomes and child-value lookups in O(60) instead of ~9,000 engine
calls per node — the cost inversion that makes a full-game sweep feasible.

Everything here is EXACT: no action coarsening, terminal-only utilities,
chance probabilities from the engine referee. Interval-valued sweeps
(bounded unknowns at unsolved frontiers) live outside this namespace.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.Constants import (
    CYLINDER_MAX,
    DEATH_PROCEDURE_OVERHEAD,
    FAILED_CHECK_PENALTY,
    TURN_DURATION_NORMAL,
    WITHIN_ROUND_OVERHEAD,
)
from src.Game import Game

from .exact import ExactGameSnapshot, exact_public_state

MAX_ST = TURN_DURATION_NORMAL - 1  # check <= 60, drop >= 1 => ST in 1..59 always


@dataclass(frozen=True)
class StageSuccess:
    st: int
    checker_cylinder_after: float
    overflow: bool  # True => immediate injection, p = 0, dropper wins


@dataclass(frozen=True)
class StageFail:
    death_duration: float
    survival_probability: float  # engine referee value; 0.0 when duration hits the cap
    # Survive branch deltas: checker cylinder -> 0, ttd += duration,
    # deaths += 1, referee cprs += 1. Fatal branch: dropper wins.


@dataclass(frozen=True)
class StageOutcomes:
    dropper_name: str
    checker_name: str
    checker_is_hal: bool
    turn_duration: int
    drop_seconds: tuple[int, ...]
    check_seconds: tuple[int, ...]
    successes: tuple[StageSuccess, ...]  # indexed by st - 1
    fail: StageFail


def analytic_stage_outcomes(game: Game) -> StageOutcomes:
    """The stage's distinct outcomes, by arithmetic + one referee call."""
    from environment.legal_actions import legal_max_second

    dropper, checker = game.get_roles_for_half(game.current_half)
    turn_duration = game.get_turn_duration()
    drop_max = legal_max_second(dropper.name, "dropper", turn_duration)
    check_max = legal_max_second(checker.name, "checker", turn_duration)

    successes = []
    for st in range(1, MAX_ST + 1):
        cyl_after = checker.cylinder + st
        successes.append(
            StageSuccess(
                st=st,
                checker_cylinder_after=cyl_after,
                overflow=cyl_after >= CYLINDER_MAX,
            )
        )

    fail_duration = min(checker.cylinder + FAILED_CHECK_PENALTY, CYLINDER_MAX)
    p_survive = game.referee.compute_survival_probability(
        checker, death_duration=fail_duration
    )

    return StageOutcomes(
        dropper_name=dropper.name,
        checker_name=checker.name,
        checker_is_hal=checker.name.lower() == "hal",
        turn_duration=turn_duration,
        drop_seconds=tuple(range(1, drop_max + 1)),
        check_seconds=tuple(range(1, check_max + 1)),
        successes=tuple(successes),
        fail=StageFail(
            death_duration=float(fail_duration),
            survival_probability=float(p_survive),
        ),
    )


# ── Engine equivalence ─────────────────────────────────────────────────────


def _expected_clock_after(game: Game, *, death_duration: float | None) -> float:
    """Mirror the engine's clock bookkeeping for one resolved half-round."""
    clock = game.game_clock + game.get_turn_duration()
    if death_duration is not None:
        clock += death_duration + DEATH_PROCEDURE_OVERHEAD
    if game.current_half == 1:
        return clock + WITHIN_ROUND_OVERHEAD
    # Engine snap: next wall-clock minute (pre-leap multiples of 60; the
    # 8:59 minute holds 61 seconds, so post-leap minutes sit at 3601+60n).
    gc = int(clock)
    if gc < 3600:
        snapped = ((gc // 60) + 1) * 60
        return float(3601 if snapped == 3600 else snapped)
    if gc <= 3600:
        return 3601.0
    return float(3601 + (((gc - 3601) // 60) + 1) * 60)


def verify_stage_outcomes_against_engine(game: Game) -> None:
    """Assert the analytic map reproduces live engine probes exactly.

    Probes one representative joint action per outcome class (every ST,
    the tie cell, the fail cell with both forced outcomes) and compares
    full public states. Raises AssertionError on any mismatch.
    """
    outcomes = analytic_stage_outcomes(game)
    snap = ExactGameSnapshot(game)
    base = exact_public_state(game)
    checker_name = outcomes.checker_name

    def probe(drop: int, check: int, survived: bool | None):
        record = game.resolve_half_round(drop, check, survived_outcome=survived)
        state = exact_public_state(game)
        snap.restore(game)
        return record, state

    def check_field(state, name, expected, ctx):
        actual = getattr(state, name)
        assert actual == expected, (
            f"{ctx}: engine {name}={actual!r} != analytic {expected!r} "
            f"(base state {base!r})"
        )

    checker_is_p1 = base.p1_name == checker_name
    cyl_field = "p1_cylinder" if checker_is_p1 else "p2_cylinder"
    ttd_field = "p1_ttd" if checker_is_p1 else "p2_ttd"
    deaths_field = "p1_deaths" if checker_is_p1 else "p2_deaths"

    # Success classes: one probe per ST, plus the tie cell for ST=1.
    for success in outcomes.successes:
        st = success.st
        cells = [(1, 1 + st)] if st > 1 else [(1, 1), (1, 2)]
        for drop, check in cells:
            if check > max(outcomes.check_seconds):
                continue
            if success.overflow:
                record, state = probe(drop, check, False)
                assert record.death_duration == float(CYLINDER_MAX), (
                    f"overflow ST={st}: engine duration {record.death_duration}"
                )
                assert record.survival_probability == 0.0, (
                    f"overflow ST={st}: engine p={record.survival_probability} != 0"
                )
                check_field(state, "game_over", True, f"overflow ST={st}")
                check_field(state, "winner_name", outcomes.dropper_name, f"overflow ST={st}")
            else:
                record, state = probe(drop, check, None)
                assert record.st_gained == st, (
                    f"cell ({drop},{check}): engine ST {record.st_gained} != {st}"
                )
                check_field(state, cyl_field, success.checker_cylinder_after, f"success ST={st}")
                check_field(state, "game_over", False, f"success ST={st}")
                expected_clock = _expected_clock_after(game, death_duration=None)
                check_field(state, "game_clock", expected_clock, f"success ST={st}")

    # Fail class: one cell, both chance branches.
    fail = outcomes.fail
    drop, check = 2, 1
    record, state = probe(drop, check, True if fail.survival_probability > 0 else False)
    assert record.death_duration == fail.death_duration, (
        f"fail: engine duration {record.death_duration} != {fail.death_duration}"
    )
    assert record.survival_probability == fail.survival_probability, (
        f"fail: engine p={record.survival_probability} != {fail.survival_probability}"
    )
    if fail.survival_probability > 0:
        check_field(state, cyl_field, 0.0, "fail+survive")
        check_field(state, ttd_field, getattr(base, ttd_field) + fail.death_duration, "fail+survive")
        check_field(state, deaths_field, getattr(base, deaths_field) + 1, "fail+survive")
        check_field(state, "referee_cprs", base.referee_cprs + 1, "fail+survive")
        expected_clock = _expected_clock_after(game, death_duration=fail.death_duration)
        check_field(state, "game_clock", expected_clock, "fail+survive")
        # Fatal branch too.
        _, dead_state = probe(drop, check, False)
        check_field(dead_state, "game_over", True, "fail+die")
        check_field(dead_state, "winner_name", outcomes.dropper_name, "fail+die")
    else:
        check_field(state, "game_over", True, "fail (always fatal)")
        check_field(state, "winner_name", outcomes.dropper_name, "fail (always fatal)")


# ── Payoff-matrix assembly ─────────────────────────────────────────────────

_ST_INDEX_CACHE: dict = {}


def _st_index_matrix(n_drop: int, n_check: int) -> tuple[np.ndarray, np.ndarray]:
    """(fail_mask, st_index) for a drop x check grid of 1-based seconds."""
    key = (n_drop, n_check)
    if key not in _ST_INDEX_CACHE:
        drops = np.arange(1, n_drop + 1)[:, None]
        checks = np.arange(1, n_check + 1)[None, :]
        fail_mask = checks < drops
        st = np.maximum(1, checks - drops)  # only meaningful where not fail
        _ST_INDEX_CACHE[key] = (fail_mask, st - 1)
    return _ST_INDEX_CACHE[key]


def assemble_payoff_matrix(
    n_drop: int,
    n_check: int,
    success_values: np.ndarray,
    fail_value: float,
) -> np.ndarray:
    """Payoff matrix over 1-based (drop, check) grids from outcome values.

    ``success_values``: length-MAX_ST array, success_values[st-1] = value
    after a successful check with that ST (already +-1 for overflow
    terminals). ``fail_value``: the chance-weighted value of the shared
    fail cell (p * V_survive + (1-p) * terminal).
    """
    fail_mask, st_idx = _st_index_matrix(n_drop, n_check)
    matrix = np.where(fail_mask, fail_value, success_values[st_idx])
    return matrix.astype(np.float64)


def stage_matrix_from_values(
    outcomes: StageOutcomes,
    success_values: np.ndarray,
    fail_value: float,
) -> np.ndarray:
    """Hal-perspective payoff matrix for a state's StageOutcomes."""
    return assemble_payoff_matrix(
        len(outcomes.drop_seconds),
        len(outcomes.check_seconds),
        success_values,
        fail_value,
    )
