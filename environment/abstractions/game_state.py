"""Abstract game state for CFR tree traversal.

The abstraction maps continuous values into discrete buckets
so the CFR tree stays tractable. TTD and CPR are derived from
death counts using representative values rather than tracked exactly.
"""

from __future__ import annotations

from dataclasses import dataclass


# Cylinder buckets: 30 levels, 10s each (0, 10, 20, ..., 290+)
CYL_BUCKET_SIZE = 10
CYL_NUM_BUCKETS = 30

# Death count: cap at 3 (0, 1, 2, 3+)
MAX_DEATH_BUCKET = 3

# Clock buckets: 60s each
CLOCK_BUCKET_SIZE = 60

# Representative TTD per death count (derived from canon game timeline).
# Used for survival probability instead of tracking exact TTD in the state.
REPRESENTATIVE_TTD = {
    0: 0.0,
    1: 60.0,    # typical first death: 60s (failed check with empty cylinder)
    2: 144.0,   # 60 + 84 (canon: Hal's R2 death was 84s)
    3: 240.0,   # accumulated
}


@dataclass(frozen=True)
class AbstractState:
    """Hashable key for a CFR information set.

    TTD and CPR are not tracked — they're derived from death counts
    via REPRESENTATIVE_TTD to keep the state space tractable.
    """
    round_num: int
    half: int
    my_cyl: int
    opp_cyl: int
    my_deaths: int
    opp_deaths: int
    clock: int


def bucket_cylinder(cylinder: float) -> int:
    """Bucket a cylinder value (0-300) into an integer 0-29."""
    return min(int(cylinder // CYL_BUCKET_SIZE), CYL_NUM_BUCKETS - 1)


def bucket_deaths(deaths: int) -> int:
    """Bucket death count into 0-3."""
    return min(deaths, MAX_DEATH_BUCKET)


def bucket_clock(game_clock: float) -> int:
    """Bucket game clock into 60s intervals."""
    return int(game_clock // CLOCK_BUCKET_SIZE)


def representative_ttd(deaths: int) -> float:
    """Get representative TTD for a given death count."""
    return REPRESENTATIVE_TTD.get(min(deaths, MAX_DEATH_BUCKET), 240.0)


def total_cprs(my_deaths: int, opp_deaths: int) -> int:
    """Total CPRs performed = sum of all deaths across both players."""
    return my_deaths + opp_deaths


def make_abstract_state(
    round_num: int,
    half: int,
    my_cylinder: float,
    opp_cylinder: float,
    my_deaths: int,
    opp_deaths: int,
    game_clock: float,
) -> AbstractState:
    """Build an AbstractState from raw game values."""
    return AbstractState(
        round_num=round_num,
        half=half,
        my_cyl=bucket_cylinder(my_cylinder),
        opp_cyl=bucket_cylinder(opp_cylinder),
        my_deaths=bucket_deaths(my_deaths),
        opp_deaths=bucket_deaths(opp_deaths),
        clock=bucket_clock(game_clock),
    )
