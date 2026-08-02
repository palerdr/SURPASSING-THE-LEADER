"""Research-only bounded resolver for pure DTH experiments.

Production play uses :class:`dth.agent.CompleteDTHAgent` exclusively. This
module survives as a dataset-labeling and model-evaluation surface: it can
compare depth-limited/network estimates with exact leaves from the completed
tablebase and records the measured saddle gap of the matrix it actually
solves. It never writes to or substitutes for the complete artifact.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np

from dth.cfr import solve_matrix_cfr_plus
from dth.solver import (
    NTState,
    complete_game_dependencies,
    continuation_class_values,
    reconstruct_transition_class_matrix,
    solve_certified_matrix,
    validate_live_state,
)
from dth.complete_tablebase import CompleteTablebase


Provenance = Literal[
    "complete-game-exact",
    "finite-horizon-exact",
    "approximate",
]


@dataclass(frozen=True)
class ResolveBudget:
    """Wall-clock and depth bounds one move may spend.

    This research-only resolver may use the completed tablebase for exact
    leaves; it never writes to the artifact.
    """

    deadline_seconds: float = 2.0
    max_depth: int = 3
    leaf_horizon: int = 4
    finite_fallback_horizon: int = 2
    class_matrix_leaf_limit: int = 1_000

    def __post_init__(self) -> None:
        if self.deadline_seconds <= 0.0:
            raise ValueError("deadline_seconds must be positive")
        if self.max_depth <= 0:
            raise ValueError("max_depth must be positive")
        if self.leaf_horizon <= 0:
            raise ValueError("leaf_horizon must be positive")
        if self.finite_fallback_horizon <= 0:
            raise ValueError("finite_fallback_horizon must be positive")
        if self.class_matrix_leaf_limit < 0:
            raise ValueError("class_matrix_leaf_limit must be nonnegative")


@dataclass(frozen=True)
class MoveDecision:
    """One move's certified answer and its provenance record."""

    state: NTState
    value: float
    drop_policy: tuple[float, ...]
    check_policy: tuple[float, ...]
    provenance: Provenance
    scope_detail: str
    horizon: int | None
    saddle_gap: float
    resolve_depth: int | None
    exact_leaf_fraction: float | None
    elapsed_seconds: float


@dataclass
class _ResolveOutcome:
    value: float
    drop_policy: np.ndarray
    check_policy: np.ndarray
    saddle_gap: float
    depth: int
    exact_leaves: int
    network_leaves: int


def solve_approximate_matrix(
    matrix: np.ndarray,
) -> tuple[float, np.ndarray, np.ndarray, float]:
    """Solve a network-leaf stage matrix and report its measured gap.

    Resolve matrices carry approximate leaf values, so the 1e-6 certificate
    firewall does not apply to them: an ill-conditioned matrix can leave the
    two independent LPs a few micro-units apart.  When the certified solvers
    reject such a matrix, CFR+ supplies the policies and the measured gap is
    reported on the move instead of being certified away.
    """

    try:
        value, drop, check, _ = solve_certified_matrix(matrix)
    except RuntimeError:
        # Gap rejections and outright LP failures alike: on an approximate
        # matrix both just mean the certified path cannot handle it, and
        # regret matching handles any finite matrix.
        solution = solve_matrix_cfr_plus(
            matrix, iterations=4096, gap_tolerance=1e-9
        )
        return (
            solution.value,
            solution.drop_policy,
            solution.check_policy,
            solution.saddle_gap,
        )
    lower = float(np.min(matrix.T @ drop))
    upper = float(np.max(matrix @ check))
    return value, drop, check, max(0.0, upper - lower)


@dataclass(frozen=True)
class _FiniteFallback:
    value: float
    drop_policy: tuple[float, ...]
    check_policy: tuple[float, ...]
    saddle_gap: float
    horizon: int | None
    cache_provenance: str


class NetworkLeafModel:
    """Torch checkpoint wrapper that batches scalar and class-matrix leaves."""

    def __init__(self, checkpoint_path: str | Path, device: str = "cpu") -> None:
        import torch

        from dth.network import DTHNetworkConfig, DTHPolicyValueNet

        self._torch = torch
        self.device = torch.device(device)
        payload = torch.load(
            Path(checkpoint_path), map_location=self.device, weights_only=False
        )
        config_values = dict(payload["model_config"])
        config_values.setdefault("transition_class_head", False)
        config_values.setdefault("play_value_head", False)
        self.config = DTHNetworkConfig(**config_values)
        self.model = DTHPolicyValueNet(self.config)
        self.model.load_state_dict(payload["state_dict"])
        self.model.to(self.device)
        self.model.eval()

    def values(self, states: list[NTState], horizon: int) -> np.ndarray:
        """Estimate leaf continuation values at the agent's query horizon.

        Resolve leaves stand in for complete-game continuations, so a
        checkpoint with a dedicated play head answers from it; the finite
        value head remains the fallback for older checkpoints.
        """

        torch = self._torch
        state_tensor = torch.tensor(states, dtype=torch.float32, device=self.device)
        horizon_tensor = torch.full(
            (len(states),), float(horizon), dtype=torch.float32, device=self.device
        )
        with torch.inference_mode():
            features = self.model.encode(state_tensor, horizon_tensor)
            if self.config.play_value_head:
                values = self.model.play_values(features)
            else:
                values, _, _ = self.model(features)
        return values.cpu().numpy().astype(np.float64)

    def class_matrix_values(self, states: list[NTState], horizon: int) -> np.ndarray:
        """Solve each leaf's predicted class matrix for a one-ply-deeper value."""

        torch = self._torch
        if not self.config.transition_class_head:
            raise ValueError("class-matrix leaves require transition_class_head")
        state_tensor = torch.tensor(states, dtype=torch.float32, device=self.device)
        horizon_tensor = torch.full(
            (len(states),), float(horizon), dtype=torch.float32, device=self.device
        )
        with torch.inference_mode():
            features = self.model.encode(state_tensor, horizon_tensor)
            class_values = self.model.transition_class_values(features)
        stacked = class_values.cpu().numpy().astype(np.float64)
        values = np.empty(len(states), dtype=np.float64)
        for index in range(len(states)):
            matrix = reconstruct_transition_class_matrix(
                stacked[index, :60], float(stacked[index, 60])
            )
            values[index], _, _, _ = solve_approximate_matrix(matrix)
        return values


class BoundedResolveAgent:
    """Research-only depth-limited resolver with complete-tablebase leaves."""

    def __init__(
        self,
        *,
        complete_path: str | Path | None = None,
        network: NetworkLeafModel | None = None,
        budget: ResolveBudget | None = None,
    ) -> None:
        self.budget = budget or ResolveBudget()
        self.network = network
        self._complete_path = None if complete_path is None else Path(complete_path)
        self._complete: CompleteTablebase | None = None

    def __enter__(self) -> "BoundedResolveAgent":
        if self._complete_path is not None:
            self._complete = CompleteTablebase(artifact_dir=self._complete_path)
        return self

    def __exit__(self, *details: object) -> None:
        self._complete = None

    def _exact_leaf_value(self, state: NTState) -> float | None:
        if self._complete is None:
            return None
        try:
            return float(self._complete.lookup(state)["value"])
        except LookupError:
            return None

    def decide(self, state: NTState) -> MoveDecision:
        started = time.monotonic()
        normalized = validate_live_state(state)

        complete = self._try_complete(normalized)
        if complete is not None:
            return MoveDecision(
                state=normalized,
                value=complete["value"],
                drop_policy=tuple(float(p) for p in complete["drop_policy"]),
                check_policy=tuple(float(p) for p in complete["check_policy"]),
                provenance="complete-game-exact",
                scope_detail="complete-tablebase",
                horizon=None,
                saddle_gap=complete["saddle_gap"],
                resolve_depth=None,
                exact_leaf_fraction=1.0,
                elapsed_seconds=time.monotonic() - started,
            )

        outcome = None
        if self.network is not None:
            outcome = self._iterative_resolve(normalized, started)
        if outcome is not None:
            leaves = outcome.exact_leaves + outcome.network_leaves
            return MoveDecision(
                state=normalized,
                value=outcome.value,
                drop_policy=tuple(float(p) for p in outcome.drop_policy),
                check_policy=tuple(float(p) for p in outcome.check_policy),
                provenance="approximate",
                scope_detail=f"bounded-resolve-depth-{outcome.depth}",
                horizon=None,
                saddle_gap=outcome.saddle_gap,
                resolve_depth=outcome.depth,
                exact_leaf_fraction=(
                    outcome.exact_leaves / leaves if leaves else 1.0
                ),
                elapsed_seconds=time.monotonic() - started,
            )

        finite = self._finite_fallback(normalized)
        return MoveDecision(
            state=normalized,
            value=finite.value,
            drop_policy=finite.drop_policy,
            check_policy=finite.check_policy,
            provenance="finite-horizon-exact",
            scope_detail=finite.cache_provenance,
            horizon=finite.horizon,
            saddle_gap=finite.saddle_gap,
            resolve_depth=None,
            exact_leaf_fraction=None,
            elapsed_seconds=time.monotonic() - started,
        )

    def _try_complete(self, state: NTState) -> dict | None:
        """The dense complete-game artifact, ahead of every other rung.

        Its domain is transition closed. Off-domain synthetic research states
        may continue through the explicitly requested approximate resolver.
        Certificate failures are never silently downgraded.
        """

        if self._complete is None:
            return None
        try:
            return self._complete.certificate(state)
        except LookupError:
            return None

    def _finite_fallback(self, state: NTState) -> "_FiniteFallback":
        # Research-only fallback: certify a bounded horizon live.
        from dth.solver import certify_finite_horizon_solution

        certificate = certify_finite_horizon_solution(
            state, self.budget.finite_fallback_horizon
        )
        return _FiniteFallback(
            value=certificate.value,
            drop_policy=certificate.drop_policy,
            check_policy=certificate.check_policy,
            saddle_gap=certificate.saddle_gap,
            horizon=certificate.horizon,
            cache_provenance="finite-live",
        )

    # Each extra resolve depth multiplies work by roughly the class branching
    # factor; a deeper attempt that cannot finish still bills its abort to the
    # move's latency, so do not start one without this much headroom.
    _NEXT_DEPTH_COST_FACTOR = 20.0

    def _iterative_resolve(
        self, root: NTState, started: float
    ) -> _ResolveOutcome | None:
        deadline = started + self.budget.deadline_seconds
        best: _ResolveOutcome | None = None
        previous_duration = 0.0
        for depth in range(1, self.budget.max_depth + 1):
            now = time.monotonic()
            if best is not None and (
                now >= deadline
                or deadline - now
                < self._NEXT_DEPTH_COST_FACTOR * previous_duration
            ):
                break
            depth_started = now
            try:
                candidate = self._resolve(root, depth, deadline)
            except TimeoutError:
                break
            best = candidate
            previous_duration = time.monotonic() - depth_started
        return best

    def resolve_labels(
        self,
        state: NTState,
        *,
        depth: int,
        deadline_seconds: float,
    ) -> dict[NTState, tuple[float, np.ndarray, np.ndarray, float]]:
        """Solve one bounded resolve and harvest every interior solution.

        Interior nodes are Bellman compositions over the whole slice below
        them, so each one is a depth-amplified training target; raw frontier
        leaves are excluded because they carry no information beyond the
        network's own output.  A deadline abort keeps the interiors already
        solved — the bottom-up sweep makes every committed row consistent.
        """

        if self.network is None:
            raise ValueError("resolve labelling requires a network")
        normalized = validate_live_state(state)
        if self._exact_leaf_value(normalized) is not None:
            # A certified state needs no depth-amplified label.
            return {}
        collected: dict[NTState, tuple[float, np.ndarray, np.ndarray, float]] = {}
        try:
            self._resolve(
                normalized,
                depth,
                time.monotonic() + deadline_seconds,
                collect=collected,
            )
        except TimeoutError:
            pass
        return collected

    def _resolve(
        self,
        root: NTState,
        depth: int,
        deadline: float,
        collect: dict[NTState, tuple[float, np.ndarray, np.ndarray, float]]
        | None = None,
    ) -> _ResolveOutcome:
        assert self.network is not None
        exact_hits: dict[NTState, float] = {}

        def exact_or_none(state: NTState) -> float | None:
            if state in exact_hits:
                return exact_hits[state]
            value = self._exact_leaf_value(state)
            if value is not None:
                exact_hits[state] = value
            return value

        levels: list[list[NTState]] = [[root]]
        for _ in range(depth):
            frontier: list[NTState] = []
            frontier_seen: set[NTState] = set()
            for index, state in enumerate(levels[-1]):
                # The deadline must bind inside a level too: a depth-three
                # frontier build alone can otherwise overrun by seconds.
                if index % 64 == 0 and time.monotonic() >= deadline:
                    raise TimeoutError
                if exact_or_none(state) is not None:
                    continue
                for child in complete_game_dependencies(state):
                    if child not in frontier_seen:
                        frontier_seen.add(child)
                        frontier.append(child)
            if time.monotonic() >= deadline:
                raise TimeoutError
            levels.append(frontier)

        network_leaves = [
            state
            for state in levels[depth]
            if exact_or_none(state) is None
        ]
        leaf_values: dict[NTState, float] = dict(exact_hits)
        if network_leaves:
            if (
                self.network.config.transition_class_head
                and len(network_leaves) <= self.budget.class_matrix_leaf_limit
            ):
                evaluated = self.network.class_matrix_values(
                    network_leaves, self.budget.leaf_horizon
                )
            else:
                evaluated = self.network.values(
                    network_leaves, self.budget.leaf_horizon
                )
            for state, value in zip(network_leaves, evaluated, strict=True):
                leaf_values[state] = float(np.clip(value, -1.0, 1.0))
        if time.monotonic() >= deadline:
            raise TimeoutError

        values: dict[NTState, float] = dict(leaf_values)
        solved_interior: set[NTState] = set()
        root_solution: tuple[float, np.ndarray, np.ndarray, float] | None = None
        for level_index in range(depth - 1, -1, -1):
            for index, state in enumerate(levels[level_index]):
                if (
                    index % 64 == 0
                    and level_index > 0
                    and time.monotonic() >= deadline
                ):
                    raise TimeoutError
                # A transposition solved at a deeper level keeps its deeper
                # value; exact hits stay authoritative; a network leaf value
                # is overwritten once the state's children are enumerated.
                if state in solved_interior or state in exact_hits:
                    continue
                successful, failed = continuation_class_values(
                    state, lambda child: values[child]
                )
                matrix = reconstruct_transition_class_matrix(successful, failed)
                value, drop, check, gap = solve_approximate_matrix(matrix)
                values[state] = value
                solved_interior.add(state)
                if collect is not None:
                    collect[state] = (value, drop, check, gap)
                if state == root and level_index == 0:
                    root_solution = (value, drop, check, gap)
            if time.monotonic() >= deadline and level_index > 0:
                raise TimeoutError

        if root_solution is None:
            raise RuntimeError("bounded resolve never solved its root")
        value, drop, check, gap = root_solution
        return _ResolveOutcome(
            value=value,
            drop_policy=drop,
            check_policy=check,
            saddle_gap=gap,
            depth=depth,
            exact_leaves=len(exact_hits),
            network_leaves=len(network_leaves),
        )
