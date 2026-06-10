"""Interval-valued backward induction over one (ttd_hal, ttd_baku, cprs) epoch.

State convention (matches the Tier-0 pilot): V[bit, cyl_hal, cyl_baku],
bit 0 = Hal drops / Baku checks, bit 1 = Baku drops / Hal checks; values
are Hal-perspective. Within an epoch only cylinders move (success: the
checker's cylinder grows by ST; cylinders are the only resetting state),
so a single sweep by descending cylinder sum is exact — the post-leap
quotient is a DAG.

Transitions out of the epoch happen only on a SURVIVED death (checker's
ttd grows, cprs + 1): their values are supplied by ``survive_value`` —
either a solved deeper epoch's table or a certified bracket. The minimax
value of a matrix is monotone in its entries, so sweeping once with all
lower edges and once with all upper edges yields certified [lo, hi]
brackets at every state.

Death chance probabilities come from the engine referee per (player,
cylinder) — see ``survival_table`` — never reimplemented (G4).
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable

import numpy as np

from environment.cfr.backward import MAX_ST, assemble_payoff_matrix
from environment.cfr.exact import solve_minimax
from src.Constants import (
    CYLINDER_MAX,
    FAILED_CHECK_PENALTY,
    PHYSICALITY_BAKU,
    PHYSICALITY_HAL,
)
from src.Player import Player
from src.Referee import Referee

CYL = int(CYLINDER_MAX)

# survive_value(checker_is_hal, death_duration, next_bit, other_cylinder)
#   -> (lo, hi) Hal-perspective value bracket of the post-revival state
#      (checker cylinder reset to 0, checker ttd += duration, cprs + 1).
SurviveValueFn = Callable[[bool, int, int, int], tuple[float, float]]


@dataclass(frozen=True)
class EpochSpec:
    ttd_hal: float
    ttd_baku: float
    cprs: int

    @property
    def deaths_total_hint(self) -> int:
        """Engine invariant: cprs == total deaths."""
        return self.cprs


def survival_table(*, name: str, ttd: float, cprs: int) -> np.ndarray:
    """p_survive[cylinder] for this player's failed check, via the engine
    referee (G4). Index = checker cylinder BEFORE the +60 penalty."""
    physicality = PHYSICALITY_HAL if name.lower() == "hal" else PHYSICALITY_BAKU
    probe = Player(name=name, physicality=physicality)
    probe.ttd = float(ttd)
    referee = Referee()
    referee.cprs_performed = int(cprs)

    table = np.zeros(CYL, dtype=np.float64)
    for cyl in range(CYL):
        duration = min(cyl + FAILED_CHECK_PENALTY, CYLINDER_MAX)
        table[cyl] = referee.compute_survival_probability(
            probe, death_duration=duration
        )
    return table


def bracket_survive_value(lo: float = -1.0, hi: float = 1.0) -> SurviveValueFn:
    """Frontier bracket for unsolved deeper epochs."""

    def fn(checker_is_hal: bool, duration: int, next_bit: int, other_cyl: int):
        return (lo, hi)

    return fn


def solve_epoch(
    spec: EpochSpec,
    survive_value: SurviveValueFn,
    *,
    survival_overrides: tuple[np.ndarray, np.ndarray] | None = None,
    min_cyl: int = 0,
    progress: Callable[[str], None] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Solve one epoch; returns (V_lo, V_hi), each [2, CYL, CYL] float64.

    ``min_cyl`` restricts the sweep to states with BOTH cylinders >=
    min_cyl — a self-contained region (cylinders only grow), used by
    tests. Entries outside the region are NaN.

    ``survival_overrides`` replaces the engine-derived (hal, baku)
    survival tables — used by tests to force the all-deaths-fatal
    degenerate case, which must reproduce the Tier-0 limit game exactly.
    """
    if survival_overrides is not None:
        p_hal, p_baku = survival_overrides
    else:
        p_hal = survival_table(name="Hal", ttd=spec.ttd_hal, cprs=spec.cprs)
        p_baku = survival_table(name="Baku", ttd=spec.ttd_baku, cprs=spec.cprs)

    V_lo = np.full((2, CYL, CYL), np.nan, dtype=np.float64)
    V_hi = np.full((2, CYL, CYL), np.nan, dtype=np.float64)
    start = time.perf_counter()
    states = 0

    for cyl_sum in range(2 * (CYL - 1), 2 * min_cyl - 1, -1):
        ch_lo = max(min_cyl, cyl_sum - (CYL - 1))
        ch_hi = min(CYL - 1, cyl_sum - min_cyl)
        for ch in range(ch_lo, ch_hi + 1):
            cb = cyl_sum - ch
            for bit in (0, 1):
                checker_is_hal = bit == 1
                checker_cyl = ch if checker_is_hal else cb
                other_cyl = cb if checker_is_hal else ch
                terminal = -1.0 if checker_is_hal else 1.0  # checker dies
                next_bit = 1 - bit

                # Success outcome values per ST (same epoch; overflow terminal).
                succ_lo = np.full(MAX_ST, terminal)
                succ_hi = np.full(MAX_ST, terminal)
                room = (CYL - 1) - checker_cyl
                if room > 0:
                    take = min(MAX_ST, room)
                    if checker_is_hal:
                        succ_lo[:take] = V_lo[next_bit, ch + 1 : ch + 1 + take, cb]
                        succ_hi[:take] = V_hi[next_bit, ch + 1 : ch + 1 + take, cb]
                    else:
                        succ_lo[:take] = V_lo[next_bit, ch, cb + 1 : cb + 1 + take]
                        succ_hi[:take] = V_hi[next_bit, ch, cb + 1 : cb + 1 + take]

                # Shared fail outcome.
                p = (p_hal if checker_is_hal else p_baku)[checker_cyl]
                if p <= 0.0:
                    fail_lo = fail_hi = terminal
                else:
                    duration = min(checker_cyl + FAILED_CHECK_PENALTY, CYLINDER_MAX)
                    s_lo, s_hi = survive_value(
                        checker_is_hal, int(duration), next_bit, int(other_cyl)
                    )
                    fail_lo = p * s_lo + (1.0 - p) * terminal
                    fail_hi = p * s_hi + (1.0 - p) * terminal

                for table, succ, fail in (
                    (V_lo, succ_lo, fail_lo),
                    (V_hi, succ_hi, fail_hi),
                ):
                    matrix = assemble_payoff_matrix(60, 60, succ, fail)
                    if bit == 0:
                        _, value = solve_minimax(matrix)       # Hal (dropper) maximizes
                    else:
                        _, neg = solve_minimax(-matrix)        # Baku (dropper) minimizes
                        value = -neg
                    table[bit, ch, cb] = value
                states += 1

        if progress is not None and cyl_sum % 100 == 0:
            elapsed = time.perf_counter() - start
            progress(
                f"cyl_sum={cyl_sum} states={states} elapsed={elapsed:.0f}s "
                f"({1000 * elapsed / max(states, 1):.2f} ms/state)"
            )

    return V_lo, V_hi
