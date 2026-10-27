from typing import Callable, Sequence, Tuple
from enum import IntEnum
import hashlib
import inspect
import numpy as np
from functools import cache
from scipy.optimize import linprog
from dataclasses import dataclass

DROPPER_ACTIONS = list(range(1, 61))
CHECKER_ACTIONS = list(range(1, 61))


SADDLE_GAP_TOLERANCE = 1e-6
_RAW_RADICES = (300, 301, 300, 301)
_FAILURE_DEAD_MIN_ST = 240


def successful_check(c,d): return d <= c
def st(c,d): return c - d + 1
def failed_check_dose(st): return st + 60
def overflow(st): return st >= 300
def survive_injection(st, ttd):
    dose = failed_check_dose(st)
    return dose < 300 and dose + ttd <= 300
def revival_model(st, ttd):
    if not survive_injection(st, ttd):
        return 0.0
    return 0.95 * (1.0 - st / 240.0) * 0.75 ** (ttd / 60.0)

type NTState = Tuple[int, int, int, int]
class TState(IntEnum):
    W = 1
    L = 0

type State = NTState | TState
type Branch = tuple[float, State]
type Distribution = tuple[Branch, ...]

def validate_live_state(raw: Sequence[int]) -> NTState:
    """Validate and canonicalize one role-canonical live DTH state."""

    if len(raw) != 4:
        raise ValueError(f"live state must have four coordinates, got {raw!r}")
    if any(isinstance(value, bool) or not isinstance(value, (int, np.integer)) for value in raw):
        raise ValueError(f"live state coordinates must be literal integers, got {raw!r}")
    state = tuple(int(value) for value in raw)
    checker_st, checker_ttd, dropper_st, dropper_ttd = state
    if not (0 <= checker_st < 300 and 0 <= dropper_st < 300):
        raise ValueError(f"live ST coordinates must be in 0..299, got {state!r}")
    if not (0 <= checker_ttd <= 300 and 0 <= dropper_ttd <= 300):
        raise ValueError(f"live TTD coordinates must be in 0..300, got {state!r}")
    return state


def validate_action(action: int, *, role: str) -> int:
    """Reject non-literal or out-of-contract action labels."""

    if isinstance(action, bool) or not isinstance(action, (int, np.integer)):
        raise ValueError(f"{role} action must be a literal integer in 1..60")
    normalized = int(action)
    if not 1 <= normalized <= 60:
        raise ValueError(f"{role} action must be in 1..60, got {action!r}")
    return normalized


def damage_rank(x: NTState) -> int:
    """Strictly increasing live-state rank for the complete DTH game."""

    return sum(validate_live_state(x))


def encode_raw_state_id(raw: Sequence[int]) -> int:
    """Pack every public-state coordinate losslessly into one integer."""

    state = validate_live_state(raw)
    encoded = 0
    multiplier = 1
    for value, radix in zip(reversed(state), reversed(_RAW_RADICES), strict=True):
        encoded += value * multiplier
        multiplier *= radix
    return encoded


def decode_raw_state_id(raw: int) -> NTState:
    """Inverse of :func:`encode_raw_state_id`; reject non-canonical integers."""

    if isinstance(raw, bool) or not isinstance(raw, (int, np.integer)):
        raise ValueError("packed state ID must be an integer")
    state_id = int(raw)
    if state_id < 0:
        raise ValueError("raw packed state ID cannot be negative")
    remainder = state_id
    values = [0, 0, 0, 0]
    for index in range(3, -1, -1):
        radix = _RAW_RADICES[index]
        values[index] = remainder % radix
        remainder //= radix
    if remainder:
        raise ValueError("packed state ID lies outside the live-state space")
    state = validate_live_state(values)
    if encode_raw_state_id(state) != state_id:
        raise ValueError("packed state ID is not canonical")
    return state


def failure_dead_quotient(raw: Sequence[int]) -> tuple[int, int] | None:
    """Return remaining ST capacities when both future failed checks are fatal.

    With both ST coordinates at least 240, every failed check is terminal.
    TTD can therefore never be read again.  Successful checks preserve TTD
    while swapping roles, so all states with the same two remaining capacities
    have identical transition-class matrices and values.
    """

    checker_st, _, dropper_st, _ = validate_live_state(raw)
    if checker_st < _FAILURE_DEAD_MIN_ST or dropper_st < _FAILURE_DEAD_MIN_ST:
        return None
    return 300 - checker_st, 300 - dropper_st


def canonical_state_id(raw: Sequence[int]) -> int:
    """Persistent key for the exact failure-dead quotient.

    Raw states use nonnegative mixed-radix IDs.  Quotient classes use a
    disjoint negative range, which makes schema corruption detectable without
    a separate type column.
    """

    state = validate_live_state(raw)
    quotient = failure_dead_quotient(state)
    if quotient is None:
        return encode_raw_state_id(state)
    checker_remaining, dropper_remaining = quotient
    return -1 - ((checker_remaining - 1) * 60 + (dropper_remaining - 1))


def state_from_canonical_id(raw: int) -> NTState:
    """Return the canonical public representative for a persistent state key."""

    if isinstance(raw, (bool, np.bool_)) or not isinstance(raw, (int, np.integer)):
        raise ValueError("canonical state ID must be a literal integer")
    state_id = int(raw)
    if state_id >= 0:
        return decode_raw_state_id(state_id)
    offset = -1 - state_id
    if not 0 <= offset < 60 * 60:
        raise ValueError("failure-dead quotient ID is outside its schema")
    checker_remaining, dropper_offset = divmod(offset, 60)
    return (
        299 - checker_remaining,
        0,
        299 - dropper_offset,
        0,
    )



def solver_schema_hash() -> str:
    """Stable hash of every rule that affects exact persistent solutions."""

    digest = hashlib.sha256()
    for rule in (
        revival_model,
        survive_injection,
        transition,
        validate_live_state,
        validate_action,
        failed_check_dose,
        overflow,
        st,
        successful_check,
    ):
        source = inspect.getsource(rule).encode("utf-8")
        digest.update(len(source).to_bytes(8, byteorder="big"))
        digest.update(source)
    digest.update(b"_FAILURE_DEAD_MIN_ST")
    digest.update(str(_FAILURE_DEAD_MIN_ST).encode("ascii"))
    return digest.hexdigest()


def reward(x: State):
    if x == 1:
        return 1
    elif x == 0:
        return -1
    else:
        return 0

def transition(x: NTState, d: int, c: int) -> Distribution:
    """From a state and given actions, give the distribution of child states"""
    x = validate_live_state(x)
    d = validate_action(d, role="dropper")
    c = validate_action(c, role="checker")
    sc, tc, sd, td = x
    # Check succeeds. Reaching the cylinder cap dumps a fatal 300-second dose.
    if successful_check(c,d):
        next_st = sc + st(c,d)
        if overflow(next_st):
            return (
                (1.0, TState.W),
            )
        return (
            (1.0, (sd, td, next_st, tc)),
            )
    #check fails
    else:
        p = revival_model(sc, tc)
        #die off injection
        if p == 0.0:
            return (
                (1.0, TState.W),
            )
        #get revived off injection
        else:
            return (
                    (p, (sd, td, 0, tc + sc + 60)),
                    (1.0 - p, TState.W),
                    )

@dataclass(frozen=True)
class Solution:
    value: float
    drop_policy: tuple[float, ...] | None
    check_policy: tuple[float, ...] | None
    saddle_gap: float



# E[Terminal or -V(s', h-1)]
def action_value(
        x: NTState,
        d: int,
        c: int,
        horizon: int,
) -> float:
    total = 0.0
    for probability, child in transition(x, d, c):
        if isinstance(child, TState):
            child_value = reward(child)
        elif horizon == 1:
            # A live cutoff is defined as a draw.  Calling ``solve(child, 0)``
            # here records thousands of otherwise useless horizon-zero cache
            # entries during a tablebase build.
            child_value = 0.0
        else:
            child_value = -value(child, horizon - 1)
        total += probability * child_value
    return total


def continuation_class_values(
    x: NTState,
    continuation_value: Callable[[NTState], float],
) -> tuple[tuple[float, ...], float]:
    """Evaluate the 60 success lags and one failed-check continuation class.

    The representatives ``(drop=1, check=lag)`` realize every inclusive
    successful lag exactly once.  Every failed cell has the same transition
    distribution, represented by ``(drop=2, check=1)``.  Live child values are
    supplied from the *next* Dropper's perspective and are therefore negated.
    """

    state = validate_live_state(x)

    def evaluate(distribution: Distribution) -> float:
        total = 0.0
        for probability, child in distribution:
            if isinstance(child, TState):
                branch_value = float(reward(child))
            else:
                branch_value = -float(continuation_value(child))
            total += probability * branch_value
        return total

    successful = tuple(
        evaluate(transition(state, 1, lag)) for lag in CHECKER_ACTIONS
    )
    failed = evaluate(transition(state, 2, 1))
    return successful, failed



def reconstruct_transition_class_matrix(
    successful_values: Sequence[float],
    failed_value: float,
) -> np.ndarray:
    """Reconstruct the literal 60x60 Dropper-row/Checker-column matrix."""

    if len(successful_values) != len(CHECKER_ACTIONS):
        raise ValueError("exactly 60 successful-check class values are required")
    success = np.asarray(successful_values, dtype=np.float64)
    if not np.all(np.isfinite(success)) or not np.isfinite(failed_value):
        raise ValueError("transition-class values must be finite")

    matrix = np.empty((len(DROPPER_ACTIONS), len(CHECKER_ACTIONS)), dtype=np.float64)
    for drop_index, drop in enumerate(DROPPER_ACTIONS):
        for check_index, check in enumerate(CHECKER_ACTIONS):
            matrix[drop_index, check_index] = (
                success[check - drop] if check >= drop else failed_value
            )
    return matrix



def payoff(x: NTState, horizon: int) -> np.ndarray:
    """Builds the payoff matrix for the current state
    with the continuation values for the given horizon"""
    state = validate_live_state(x)
    if horizon <= 0:
        raise ValueError(f"horizon must be positive, got {horizon}")
    matrix = np.empty(
        (len(DROPPER_ACTIONS), len(CHECKER_ACTIONS)),
        dtype=np.float64
    )

    for i, d in enumerate(DROPPER_ACTIONS):
        for j, c in enumerate(CHECKER_ACTIONS):
            matrix[i, j] = action_value(state, d, c, horizon)

    return matrix


def _solution_from_matrix(matrix: np.ndarray) -> Solution:
    matrix_value, p, q = solve_matrix(matrix)

    lower = np.min(matrix.T @ p)
    upper = np.max(matrix @ q)

    return Solution(
        value=matrix_value,
        drop_policy=tuple(p),
        check_policy=tuple(q),
        saddle_gap=upper - lower,
    )


@cache
def _solve_horizon_one(checker_st: int, checker_ttd: int) -> Solution:
    """Solve the exact H1 matrix for its complete behavioral equivalence class.

    With no continuation after a live H1 child, the current Dropper's ST and
    TTD cannot enter the stage payoff.  Retaining the original state-keyed
    public ``solve`` cache still gives every emitted tablebase row its ordinary
    raw-state identity; this private cache merely avoids duplicate LP work.
    """

    return _solution_from_matrix(payoff((checker_st, checker_ttd, 0, 0), 1))


@cache
def solve(x: NTState, horizon: int) -> Solution:
    """Solves a state for a given horizon"""
    x = validate_live_state(x)
    if horizon < 0:
        raise ValueError(f"horizon must be nonnegative, got {horizon}")
    if horizon == 0:
        return Solution(
            value=0.0,
            drop_policy=None,
            check_policy=None,
            saddle_gap=0.0,
        )
    if horizon == 1:
        return _solve_horizon_one(x[0], x[1])

    return _solution_from_matrix(payoff(x, horizon))


def clear_solver_cache() -> None:
    """Clear state and H1 behavioral-equivalence caches together."""

    solve.cache_clear()
    _solve_horizon_one.cache_clear()


def horizon_one_cache_info():
    """Expose H1 equivalence-cache accounting for exact build reports."""

    return _solve_horizon_one.cache_info()


def value(x: NTState, horizon: int) -> float:
    return solve(x, horizon).value


def solve_matrix(matrix: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    """Solve the minimax LP with HiGHS, the retained fallback and oracle."""
    rows, cols = matrix.shape
    # Dropper: maximize v subject to M.T @ p >= v.
    drop_result = linprog(
        c=np.concatenate([np.zeros(rows), [-1.0]]),
        A_ub=np.hstack([
            -matrix.T,
            np.ones((cols, 1)),
        ]),
        b_ub=np.zeros(cols),
        A_eq=np.hstack([
            np.ones((1, rows)),
            np.zeros((1, 1)),
        ]),
        b_eq=np.array([1.0]),
        bounds=[(0.0, None)] * rows + [(None, None)],
        method="highs",
    )

    if not drop_result.success:
        raise RuntimeError(
            f"Dropper LP failed: {drop_result.message}"
        )
    # Checker: minimize w subject to M @ q <= w.
    check_result = linprog(
        c=np.concatenate([np.zeros(cols), [1.0]]),
        A_ub=np.hstack([
            matrix,
            -np.ones((rows, 1)),
        ]),
        b_ub=np.zeros(rows),
        A_eq=np.hstack([
            np.ones((1, cols)),
            np.zeros((1, 1)),
        ]),
        b_eq=np.array([1.0]),
        bounds=[(0.0, None)] * cols + [(None, None)],
        method="highs",
    )

    if not check_result.success:
        raise RuntimeError(
            f"Checker LP failed: {check_result.message}"
        )
    #normalize the policies
    drop_policy = np.clip(drop_result.x[:-1], 0.0, None)
    check_policy = np.clip(check_result.x[:-1], 0.0, None)
    drop_policy /= drop_policy.sum()
    check_policy /= check_policy.sum()

    lower_bound = np.min(matrix.T @ drop_policy)
    upper_bound = np.max(matrix @ check_policy)
    saddle_gap = max(0.0, upper_bound - lower_bound)

    if saddle_gap > SADDLE_GAP_TOLERANCE:
        raise RuntimeError(
            f"LP saddle gap too large: {saddle_gap}"
        )
    matrix_value = (lower_bound + upper_bound) / 2.0

    return matrix_value, drop_policy, check_policy



def canonical_certificate_scalars(
    lower: float,
    value: float,
    upper: float,
) -> tuple[float, float, float]:
    """Restore the `lower <= value <= upper` ordering after ulp reversals.

    Independent primal and dual solves may reverse the two bounds by a few
    ulps, and the midpoint value can then sit a few ulps outside them.  The
    persistent store rejects any unordered certificate, so canonicalize the
    sub-tolerance numerical noise here; a genuine gap above the 1e-6
    threshold has already been rejected before this point.
    """

    if lower > upper:
        midpoint = (lower + upper) / 2.0
        lower = upper = midpoint
    return lower, min(max(value, lower), upper), upper



def complete_game_dependencies(x: NTState) -> tuple[NTState, ...]:
    """Enumerate unique live dependencies through the 61 exact classes."""

    state = validate_live_state(x)
    children: set[NTState] = set()
    for lag in CHECKER_ACTIONS:
        for _, child in transition(state, 1, lag):
            if isinstance(child, tuple):
                children.add(child)
    for _, child in transition(state, 2, 1):
        if isinstance(child, tuple):
            children.add(child)
    parent_rank = damage_rank(state)
    for child in children:
        if damage_rank(child) <= parent_rank:
            raise RuntimeError(
                f"complete-game transition does not increase damage rank: "
                f"{state!r} -> {child!r}"
            )
    return tuple(sorted(children, key=lambda child: (damage_rank(child), child)))
