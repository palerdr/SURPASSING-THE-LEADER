from typing import TYPE_CHECKING, Callable, Mapping, Sequence, Tuple
from enum import IntEnum
import hashlib
import json
import numpy as np
from functools import cache
from scipy.optimize import linprog
from dataclasses import dataclass

DROPPER_ACTIONS = list(range(1, 61))
CHECKER_ACTIONS = list(range(1, 61))
ST_SUPPORT = list(range(300))

# This identifier names the executable pure-DTH contract.  Persistent exact
# artifacts bind to both it and ``solver_schema_hash`` below, so a tablebase
# cannot silently survive a rules change.
TARGET_SCHEMA = "dth-v1-ttd-strict-overflow"
SOLVER_VERSION = "dth-complete-game-current"
COMPLETE_GAME_HORIZON = -1
SADDLE_GAP_TOLERANCE = 1e-6
STATE_ENCODING_VERSION = "dth-state-key-failure-dead-quotient-v1"
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
    q = st + 60
    st_term = 1.0 - (q / 300.0) ** 3
    ttd_term = 2.0 ** (-ttd / 240.0)
    return st_term * ttd_term

type NTState = Tuple[int, int, int, int]
class TState(IntEnum):
    W = 1
    L = 0

type State = NTState | TState
type Branch = tuple[float, State]
type Distribution = tuple[Branch, ...]

if TYPE_CHECKING:
    from dth.tablebase import CertifiedTablebase


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
    """Pack every public-state coordinate losslessly into one SQLite integer."""

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


def canonical_damage_rank(raw: Sequence[int]) -> int:
    """Strict topological rank for raw states plus failure-dead classes.

    Entering the quotient can erase as much as 600 points of now-dead TTD.
    Quotient ranks therefore occupy a disjoint band above every non-quotient
    live rank; successful lags still increase rank inside that band.
    """

    state = validate_live_state(raw)
    if failure_dead_quotient(state) is not None:
        representative = state_from_canonical_id(canonical_state_id(state))
        return 1200 + representative[0] + representative[2]
    return damage_rank(state)


def solver_schema_hash() -> str:
    """Stable hash of every rule that affects exact persistent solutions."""

    schema = {
        "actions": {"checker": [1, 60], "dropper": [1, 60]},
        "complete_game_horizon": COMPLETE_GAME_HORIZON,
        "failure": {
            "dose": "checker_st + 60",
            "fatal_dose": ">= 300",
            "fatal_total_ttd": "> 300",
            "revival": "(1 - (dose / 300)^3) * 2^(-ttd / 240)",
        },
        "role_orientation": "current_dropper",
        "schema": TARGET_SCHEMA,
        "success": {"condition": "checker >= dropper", "st": "checker-dropper+1"},
        "terminal_st": ">= 300",
    }
    encoded = json.dumps(schema, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def reward(x: State):
    if x == 1:
        return 1
    elif x == 0:
        return -1
    else:
        return 0
    
def transition(x: NTState, d: int, c: int) -> Distribution:
    """From a state and given actions, give the distribution of child states"""
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


@dataclass(frozen=True)
class CertifiedSolution:
    """A persistent, numerically certified finite or complete-game result."""

    state: NTState
    value: float
    drop_policy: tuple[float, ...]
    check_policy: tuple[float, ...]
    lower_bound: float
    upper_bound: float
    saddle_gap: float
    damage_rank: int
    scope: str
    horizon: int | None
    child_dependencies: int


@dataclass(frozen=True)
class ValueInterval:
    """A fail-closed Bellman enclosure from the current Dropper's view."""

    lower_bound: float
    upper_bound: float

    def __post_init__(self) -> None:
        if not (
            np.isfinite(self.lower_bound)
            and np.isfinite(self.upper_bound)
            and -1.0 <= self.lower_bound <= self.upper_bound <= 1.0
        ):
            raise ValueError(f"invalid value interval {self!r}")

    @property
    def exact(self) -> bool:
        return self.upper_bound - self.lower_bound <= SADDLE_GAP_TOLERANCE

    @property
    def midpoint(self) -> float:
        return (self.lower_bound + self.upper_bound) / 2.0


@dataclass
class CompleteGameMetrics:
    cache_hits: int = 0
    cache_misses: int = 0
    lp_solves: int = 0
    matrix_builds: int = 0
    child_dependencies: int = 0
    new_solutions: int = 0
    elapsed_seconds: float = 0.0
    dependency_construction_seconds: float = 0.0
    sqlite_lookup_and_certificate_seconds: float = 0.0
    matrix_reconstruction_seconds: float = 0.0
    highs_lp_solving_seconds: float = 0.0
    durable_commit_seconds: float = 0.0

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


def continuation_class_intervals(
    x: NTState,
    continuation_interval: Callable[[NTState], ValueInterval | None],
) -> tuple[tuple[ValueInterval, ...], ValueInterval]:
    """Evaluate Bellman classes with unknown children conservatively at [-1,1]."""

    state = validate_live_state(x)

    def evaluate(distribution: Distribution) -> ValueInterval:
        lower = 0.0
        upper = 0.0
        for probability, child in distribution:
            if isinstance(child, TState):
                branch = ValueInterval(float(reward(child)), float(reward(child)))
            else:
                child_interval = continuation_interval(child)
                if child_interval is None:
                    child_interval = ValueInterval(-1.0, 1.0)
                branch = ValueInterval(
                    -child_interval.upper_bound,
                    -child_interval.lower_bound,
                )
            lower += probability * branch.lower_bound
            upper += probability * branch.upper_bound
        return ValueInterval(
            max(-1.0, min(1.0, lower)),
            max(-1.0, min(1.0, upper)),
        )

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


def reconstruct_transition_class_interval_matrices(
    successful: Sequence[ValueInterval],
    failed: ValueInterval,
) -> tuple[np.ndarray, np.ndarray]:
    """Build elementwise lower and upper Bellman matrices."""

    lower = reconstruct_transition_class_matrix(
        [interval.lower_bound for interval in successful],
        failed.lower_bound,
    )
    upper = reconstruct_transition_class_matrix(
        [interval.upper_bound for interval in successful],
        failed.upper_bound,
    )
    if np.any(lower > upper):
        raise RuntimeError("Bellman interval matrices are not ordered")
    return lower, upper


def bellman_value_interval(
    x: NTState,
    continuation_interval: Callable[[NTState], ValueInterval | None],
) -> ValueInterval:
    """Propagate global child intervals through minimax monotonicity."""

    successful, failed = continuation_class_intervals(
        validate_live_state(x), continuation_interval
    )
    lower_matrix, upper_matrix = reconstruct_transition_class_interval_matrices(
        successful, failed
    )
    lower_value, _, _ = solve_matrix(lower_matrix)
    upper_value, _, _ = solve_matrix(upper_matrix)
    numerical_slack = 1e-10
    lower = max(-1.0, min(1.0, lower_value - numerical_slack))
    upper = max(-1.0, min(1.0, upper_value + numerical_slack))
    if lower > upper:
        if lower - upper <= 2.0 * numerical_slack:
            midpoint = (lower + upper) / 2.0
            lower = upper = midpoint
        else:
            raise RuntimeError("Bellman lower value exceeds its upper value")
    return ValueInterval(lower, upper)


def payoff_from_transition_classes(x: NTState, horizon: int) -> np.ndarray:
    """Optimized finite-horizon builder, retained beside ``payoff`` for parity."""

    if horizon <= 0:
        raise ValueError(f"horizon must be positive, got {horizon}")

    def child_value(child: NTState) -> float:
        return 0.0 if horizon == 1 else value(child, horizon - 1)

    successful, failed = continuation_class_values(x, child_value)
    return reconstruct_transition_class_matrix(successful, failed)


def payoff_from_value_lookup(
    x: NTState,
    horizon: int,
    values: Mapping[tuple[NTState, int], float],
) -> np.ndarray:
    """Reconstruct an exact finite matrix from an explicit child-value store."""

    state = validate_live_state(x)
    if horizon <= 0:
        raise ValueError(f"horizon must be positive, got {horizon}")

    def child_value(child: NTState) -> float:
        if horizon == 1:
            return 0.0
        key = (child, horizon - 1)
        try:
            return float(values[key])
        except KeyError as exc:
            raise KeyError(f"missing exact child value for {key!r}") from exc

    successful, failed = continuation_class_values(state, child_value)
    return reconstruct_transition_class_matrix(successful, failed)

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


def is_transition_class_matrix(
    matrix: np.ndarray,
    *,
    tolerance: float = 1e-12,
) -> bool:
    """Recognize the 61-class lower-triangular/diagonal DTH matrix."""

    values = np.asarray(matrix, dtype=np.float64)
    if values.shape != (60, 60) or not np.all(np.isfinite(values)):
        return False
    failed = values[1, 0]
    for row in range(60):
        for column in range(60):
            expected = (
                values[0, column - row] if column >= row else failed
            )
            if abs(float(values[row, column] - expected)) > tolerance:
                return False
    return True


def solve_full_support_structured_matrix(
    matrix: np.ndarray,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Solve a full-support DTH matrix by its equilibrium linear equations.

    This path is accepted only after structural recognition, strict positive
    support, and the same saddle-gap validation used by the HiGHS oracle.
    Singular, boundary-support, or numerically weak candidates fail closed so
    callers can fall back to :func:`solve_matrix`.
    """

    values = np.asarray(matrix, dtype=np.float64)
    if not is_transition_class_matrix(values):
        raise ValueError("matrix does not have the DTH transition-class structure")
    ones = np.ones(60, dtype=np.float64)
    augmented = np.block(
        [[values, -ones[:, None]], [ones[None, :], np.zeros((1, 1))]]
    )
    check_solution = np.linalg.solve(
        augmented,
        np.concatenate([np.zeros(60, dtype=np.float64), [1.0]]),
    )
    drop_augmented = np.block(
        [[values.T, -ones[:, None]], [ones[None, :], np.zeros((1, 1))]]
    )
    drop_solution = np.linalg.solve(
        drop_augmented,
        np.concatenate([np.zeros(60, dtype=np.float64), [1.0]]),
    )
    check = check_solution[:-1]
    drop = drop_solution[:-1]
    if (
        not np.all(np.isfinite(drop))
        or not np.all(np.isfinite(check))
        or np.min(drop) <= 1e-12
        or np.min(check) <= 1e-12
    ):
        raise RuntimeError("structured equilibrium is not strict full support")
    drop /= drop.sum()
    check /= check.sum()
    lower = float(np.min(values.T @ drop))
    upper = float(np.max(values @ check))
    gap = max(0.0, upper - lower)
    if gap > SADDLE_GAP_TOLERANCE:
        raise RuntimeError(f"structured saddle gap too large: {gap}")
    return (lower + upper) / 2.0, drop, check


def solve_certified_matrix(
    matrix: np.ndarray,
    *,
    prefer_structured: bool = True,
) -> tuple[float, np.ndarray, np.ndarray, str]:
    """Use the validated structured path when possible, else HiGHS."""

    if prefer_structured:
        try:
            value_, drop, check = solve_full_support_structured_matrix(matrix)
        except (ValueError, RuntimeError, np.linalg.LinAlgError):
            pass
        else:
            return value_, drop, check, "structured-full-support"
    value_, drop, check = solve_matrix(matrix)
    return value_, drop, check, "highs"


def _certify_matrix_solution(
    state: NTState,
    matrix: np.ndarray,
    *,
    scope: str,
    horizon: int | None,
    child_dependencies: int,
    backend_out: list[str] | None = None,
) -> CertifiedSolution:
    """Solve and retain the complete numerical certificate for one matrix."""

    normalized_state = validate_live_state(state)
    if matrix.shape != (len(DROPPER_ACTIONS), len(CHECKER_ACTIONS)):
        raise ValueError(f"expected a 60x60 matrix, got {matrix.shape}")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("exact payoff matrix contains a non-finite value")
    value_, drop, check, backend = solve_certified_matrix(matrix)
    if backend_out is not None:
        backend_out.append(backend)
    lower = float(np.min(matrix.T @ drop))
    upper = float(np.max(matrix @ check))

    def canonical_unit_interval(name: str, raw: float) -> float:
        if not -1.0 - 1e-10 <= raw <= 1.0 + 1e-10:
            raise RuntimeError(f"LP {name} lies outside [-1, 1]: {raw}")
        return min(1.0, max(-1.0, raw))

    lower = canonical_unit_interval("lower bound", lower)
    upper = canonical_unit_interval("upper bound", upper)
    value_ = canonical_unit_interval("value", float(value_))
    # Independent LP solves can reverse the two bounds by a few ulps.  The
    # mathematical saddle gap is nonnegative; canonicalize only that numerical
    # zero while retaining the existing 1e-6 rejection threshold.
    saddle_gap = max(0.0, upper - lower)
    if saddle_gap > SADDLE_GAP_TOLERANCE:
        raise RuntimeError(f"LP saddle gap too large: {saddle_gap}")
    return CertifiedSolution(
        state=normalized_state,
        value=float(value_),
        drop_policy=tuple(float(value) for value in drop),
        check_policy=tuple(float(value) for value in check),
        lower_bound=lower,
        upper_bound=upper,
        saddle_gap=saddle_gap,
        damage_rank=damage_rank(normalized_state),
        scope=scope,
        horizon=horizon,
        child_dependencies=child_dependencies,
    )


def certify_complete_game_matrix(
    x: NTState,
    matrix: np.ndarray,
    *,
    child_dependencies: int,
    backend_out: list[str] | None = None,
) -> CertifiedSolution:
    """Certify one complete-game matrix for rank-layer storage backends."""

    return _certify_matrix_solution(
        validate_live_state(x),
        matrix,
        scope="complete-game-exact",
        horizon=None,
        child_dependencies=child_dependencies,
        backend_out=backend_out,
    )


def certify_finite_horizon_solution(x: NTState, horizon: int) -> CertifiedSolution:
    """Wrap the unchanged finite-horizon solve in a persistent certificate."""

    state = validate_live_state(x)
    if horizon <= 0:
        raise ValueError("finite-horizon certification requires horizon >= 1")
    # ``payoff`` remains intentionally independent from the transition-class
    # builder, so this certificate also acts as an oracle for builder parity.
    matrix = payoff(state, horizon)
    certified = _certify_matrix_solution(
        state,
        matrix,
        scope="finite-horizon-exact",
        horizon=horizon,
        child_dependencies=0 if horizon == 1 else len(complete_game_dependencies(state)),
    )
    finite = solve(state, horizon)
    if abs(certified.value - finite.value) > 1e-10:
        raise RuntimeError("finite-horizon certificate disagrees with solve")
    return certified


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
