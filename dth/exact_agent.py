"""One production exact API and one Hydra workflow for pure DTH."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import time
from typing import Any, Iterable, Literal, Sequence

import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf

from dth.reachability import (
    CensusError,
    RankLayerSolver,
    ReachabilityCensus,
    bellman_recertify,
    propagate_bellman_interval,
    reconstruct_policy_from_certified_children,
)
from dth.solver import (
    CertifiedSolution,
    NTState,
    certify_finite_horizon_solution,
    payoff,
    payoff_from_transition_classes,
    transition,
    validate_action,
    validate_live_state,
)
from dth.tablebase import CertifiedTablebase


Role = Literal["dropper", "checker"]


class ExactAgentDeadlineError(TimeoutError):
    """A bounded exact attempt ended without any certified answer."""


@dataclass(frozen=True)
class ExactAgentResult:
    state: NTState
    value: float
    drop_policy: tuple[float, ...]
    check_policy: tuple[float, ...]
    lower_bound: float
    upper_bound: float
    saddle_gap: float
    scope: Literal["complete-game-exact", "finite-horizon-exact"]
    horizon: int | None
    cache_provenance: Literal[
        "root-policy-cache",
        "root-policy-reconstruction",
        "finite-cache",
        "finite-expansion",
    ]
    elapsed_seconds: float

    @classmethod
    def from_certificate(
        cls,
        certificate: CertifiedSolution,
        *,
        cache_provenance: Literal[
            "root-policy-cache",
            "root-policy-reconstruction",
            "finite-cache",
            "finite-expansion",
        ],
        elapsed_seconds: float,
    ) -> "ExactAgentResult":
        return cls(
            state=certificate.state,
            value=certificate.value,
            drop_policy=certificate.drop_policy,
            check_policy=certificate.check_policy,
            lower_bound=certificate.lower_bound,
            upper_bound=certificate.upper_bound,
            saddle_gap=certificate.saddle_gap,
            scope=certificate.scope,  # type: ignore[arg-type]
            horizon=certificate.horizon,
            cache_provenance=cache_provenance,
            elapsed_seconds=elapsed_seconds,
        )


def _sample_policy(policy: Sequence[float], *, seed: int) -> int:
    values = np.asarray(policy, dtype=np.float64)
    if values.shape != (60,) or not np.all(np.isfinite(values)):
        raise ValueError("exact policy must be a finite length-60 vector")
    if np.any(values < -1e-12) or abs(float(values.sum()) - 1.0) > 1e-10:
        raise ValueError("exact policy must be normalized and nonnegative")
    return int(np.random.default_rng(seed).choice(np.arange(1, 61), p=values))


class ExactDTHAgent:
    """Value-first exact agent; policies are reconstructed only when queried."""

    def __init__(self, tablebase: CertifiedTablebase) -> None:
        self.tablebase = tablebase

    def prepare_finite_fallback(
        self,
        state: NTState,
        horizon: int,
    ) -> ExactAgentResult:
        started = time.monotonic()
        normalized = validate_live_state(state)
        cached = self.tablebase.get_finite(normalized, horizon)
        if cached is not None:
            return ExactAgentResult.from_certificate(
                cached,
                cache_provenance="finite-cache",
                elapsed_seconds=time.monotonic() - started,
            )
        solution = certify_finite_horizon_solution(normalized, horizon)
        self.tablebase.put_finite(solution)
        return ExactAgentResult.from_certificate(
            solution,
            cache_provenance="finite-expansion",
            elapsed_seconds=time.monotonic() - started,
        )

    def _finite_fallback(
        self,
        state: NTState,
        max_horizon: int,
    ) -> CertifiedSolution | None:
        for horizon in range(max_horizon, 0, -1):
            solution = self.tablebase.get_finite(state, horizon)
            if solution is not None:
                return solution
        return None

    def evaluate(
        self,
        state: NTState,
        *,
        deadline_seconds: float | None = None,
        finite_fallback_horizon: int = 4,
        allow_expansion: bool = True,
        max_new_solutions: int | None = None,
        census_max_states: int | None = 250_000,
        batch_size: int = 64,
        workers: int = 1,
        cache_root_policy: bool = True,
    ) -> ExactAgentResult:
        """Return a certified root policy or a pre-certified finite fallback.

        Any new complete-game work must have a time or state bound.  No child
        midpoint or approximate value is ever accepted by the rank solver.
        """

        if finite_fallback_horizon <= 0:
            raise ValueError("finite_fallback_horizon must be positive")
        started = time.monotonic()
        normalized = validate_live_state(state)
        cached_policy = self.tablebase.get_cached_policy(normalized)
        if cached_policy is not None:
            return ExactAgentResult.from_certificate(
                cached_policy,
                cache_provenance="root-policy-cache",
                elapsed_seconds=time.monotonic() - started,
            )
        stored = self.tablebase.get_complete_value(normalized)
        if stored is not None and stored.exact:
            solution = reconstruct_policy_from_certified_children(
                normalized, self.tablebase, cache=cache_root_policy
            )
            return ExactAgentResult.from_certificate(
                solution,
                cache_provenance="root-policy-reconstruction",
                elapsed_seconds=time.monotonic() - started,
            )
        if not allow_expansion:
            fallback = self._finite_fallback(normalized, finite_fallback_horizon)
            if fallback is None:
                raise ExactAgentDeadlineError("no certified result is available")
            return ExactAgentResult.from_certificate(
                fallback,
                cache_provenance="finite-cache",
                elapsed_seconds=time.monotonic() - started,
            )
        if deadline_seconds is None and max_new_solutions is None:
            raise ValueError(
                "new complete-game work requires deadline_seconds or "
                "max_new_solutions"
            )
        census = ReachabilityCensus(self.tablebase, [normalized])
        census_run = census.run(
            max_expansions=max_new_solutions,
            max_states=census_max_states,
            max_seconds=deadline_seconds,
        )
        if census_run.stop_reason != "complete":
            fallback = self._finite_fallback(normalized, finite_fallback_horizon)
            if fallback is None:
                raise ExactAgentDeadlineError(
                    "bounded census ended before a certified fallback was available"
                )
            return ExactAgentResult.from_certificate(
                fallback,
                cache_provenance="finite-cache",
                elapsed_seconds=time.monotonic() - started,
            )
        elapsed = time.monotonic() - started
        remaining_seconds = (
            None
            if deadline_seconds is None
            else max(0.0, deadline_seconds - elapsed)
        )
        RankLayerSolver(census).run(
            max_new_solutions=max_new_solutions,
            max_seconds=remaining_seconds,
            batch_size=batch_size,
            workers=workers,
        )
        stored = self.tablebase.get_complete_value(normalized)
        if stored is None or not stored.exact:
            fallback = self._finite_fallback(normalized, finite_fallback_horizon)
            if fallback is None:
                raise ExactAgentDeadlineError(
                    "bounded rank solve ended before a certified result"
                )
            return ExactAgentResult.from_certificate(
                fallback,
                cache_provenance="finite-cache",
                elapsed_seconds=time.monotonic() - started,
            )
        solution = reconstruct_policy_from_certified_children(
            normalized, self.tablebase, cache=cache_root_policy
        )
        return ExactAgentResult.from_certificate(
            solution,
            cache_provenance="root-policy-reconstruction",
            elapsed_seconds=time.monotonic() - started,
        )

    @staticmethod
    def sample_action(result: ExactAgentResult, *, role: Role, seed: int) -> int:
        policy = result.drop_policy if role == "dropper" else result.check_policy
        if role not in ("dropper", "checker"):
            raise ValueError(f"unknown role {role!r}")
        return validate_action(_sample_policy(policy, seed=seed), role=role)

    @staticmethod
    def replay_actions(
        result: ExactAgentResult,
        roles: Iterable[Role],
        *,
        seed: int,
    ) -> tuple[int, ...]:
        rng = np.random.default_rng(seed)
        actions: list[int] = []
        for role in roles:
            policy = result.drop_policy if role == "dropper" else result.check_policy
            actions.append(
                validate_action(
                    int(rng.choice(np.arange(1, 61), p=np.asarray(policy))),
                    role=role,
                )
            )
        return tuple(actions)


def opening_success_prefix_state(*, lag: int, half_turns: int) -> NTState:
    validate_action(lag, role="checker")
    if half_turns < 0:
        raise ValueError("opening prefix half_turns must be nonnegative")
    state: NTState = (0, 0, 0, 0)
    for _ in range(half_turns):
        branches = transition(state, 1, lag)
        live = [child for _, child in branches if isinstance(child, tuple)]
        if len(branches) != 1 or len(live) != 1:
            raise ValueError("opening prefix reaches a terminal transition")
        state = live[0]
    return state


def _roots(config: DictConfig) -> tuple[NTState, ...]:
    normalized = tuple(
        dict.fromkeys(validate_live_state(entry["state"]) for entry in config.roots)
    )
    if not normalized:
        raise ValueError("exact config requires at least one root")
    return normalized


def _builder_parity(
    roots: Sequence[NTState],
    horizons: Sequence[int],
) -> dict[str, object]:
    rows: list[dict[str, object]] = []
    for state in roots:
        for horizon in horizons:
            reference = payoff(state, int(horizon))
            optimized = payoff_from_transition_classes(state, int(horizon))
            rows.append(
                {
                    "state": list(state),
                    "horizon": int(horizon),
                    "max_abs_error": float(np.max(np.abs(reference - optimized))),
                }
            )
    maximum = max((float(row["max_abs_error"]) for row in rows), default=0.0)
    return {
        "rows": rows,
        "maximum_abs_error": maximum,
        "tolerance": 1e-12,
        "passed": maximum <= 1e-12,
    }


def run_exact(config: DictConfig) -> dict[str, Any]:
    """Run one fully parameterized, explicitly bounded exact workflow."""

    roots = _roots(config)
    resolved = OmegaConf.to_container(config, resolve=True)
    database_path = Path(str(config.database_path))
    report_path = Path(str(config.report_path))
    report_path.parent.mkdir(parents=True, exist_ok=True)
    census_cfg = config.census
    solve_cfg = config.solve
    if (
        census_cfg.get("max_expansions") is None
        and census_cfg.get("max_states") is None
        and census_cfg.get("max_seconds") is None
    ):
        raise ValueError("census must have at least one explicit bound")
    if bool(solve_cfg.enabled) and (
        solve_cfg.get("max_new_solutions") is None
        and solve_cfg.get("max_seconds") is None
    ):
        raise ValueError("solve must have an explicit state or time bound")

    opened = time.perf_counter()
    with CertifiedTablebase(database_path) as tablebase:
        verify_on_open_seconds = time.perf_counter() - opened
        before_bounds = {
            str(root): asdict(
                propagate_bellman_interval(root, tablebase, persist=True)
            )
            for root in roots
        }
        census = ReachabilityCensus(tablebase, roots)
        census_run = census.run(
            max_expansions=census_cfg.get("max_expansions"),
            max_states=census_cfg.get("max_states"),
            max_seconds=census_cfg.get("max_seconds"),
        )
        census_report = census.report(census_run)
        solve_report: dict[str, object] | None = None
        if bool(solve_cfg.enabled) and census.complete:
            solve_report = RankLayerSolver(census).run(
                max_new_solutions=solve_cfg.get("max_new_solutions"),
                max_seconds=solve_cfg.get("max_seconds"),
                batch_size=int(solve_cfg.batch_size),
                workers=int(solve_cfg.workers),
            )
        queries: list[dict[str, object]] = []
        if bool(config.policy_cache.enabled):
            for root in roots:
                stored = tablebase.get_complete_value(root)
                if stored is None or not stored.exact:
                    continue
                solution = reconstruct_policy_from_certified_children(
                    root, tablebase, cache=True
                )
                recertified = bellman_recertify(root, tablebase)
                queries.append(
                    {
                        "state": list(root),
                        "value": solution.value,
                        "saddle_gap": recertified.saddle_gap,
                    }
                )
        after_bounds = {}
        for root in roots:
            stored = tablebase.get_complete_value(root)
            interval = (
                propagate_bellman_interval(root, tablebase, persist=True)
                if stored is None or not stored.exact
                else stored.interval
            )
            after_bounds[str(root)] = asdict(interval)
        verification = tablebase.verify(full=True)
        report: dict[str, Any] = {
            "schema_version": "dth-exact-report-current",
            "resolved_config": resolved,
            "database_path": str(database_path),
            "builder_parity": _builder_parity(
                roots, [int(value) for value in config.parity_horizons]
            ),
            "intervals_before_census": before_bounds,
            "census": census_report,
            "solve": solve_report,
            "intervals_after_solve": after_bounds,
            "queried_root_certificates": queries,
            "verification": verification,
            "profiling": {
                "verify_on_open_seconds": verify_on_open_seconds,
                "sqlite_value_lookup_seconds": tablebase.metrics.value_lookup_seconds,
                "policy_deserialization_seconds": (
                    tablebase.metrics.policy_deserialization_seconds
                ),
                "policy_validation_seconds": (
                    tablebase.metrics.policy_validation_seconds
                ),
                "durable_commit_seconds": tablebase.metrics.durable_commit_seconds,
                "verify_seconds": tablebase.metrics.verify_seconds,
                "rank_solver": None if solve_report is None else solve_report["metrics"],
            },
            "complete_game_claim": bool(
                all(
                    tablebase.get_complete_value(root) is not None
                    and tablebase.get_complete_value(root).exact  # type: ignore[union-attr]
                    for root in roots
                )
                and len(queries) == len(roots)
            ),
        }
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
    )
    return report


@hydra.main(version_base="1.3", config_path="config", config_name="exact")
def main(config: DictConfig) -> None:
    print(json.dumps(run_exact(config), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
