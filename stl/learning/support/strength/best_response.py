"""Exact depth-limited best response against a frozen Markov policy.

Against a frozen policy pi for one seat, the other seat plus chance form a
single-agent decision problem; its exact optimum is THE exploitability
measurement (what poker projects can only approximate, this game's public
state lets us compute). The recursion is depth-limited; truncated frontier
states are bracketed in [-1, +1], so every result is a certified interval
[lo, hi] containing the true best-response value: Hal-perspective, with
the adversary minimizing when the adversary is Baku.

Two structural facts keep this affordable (both verified in src/Game.py):

- Within one half-round, all 61x61 normal joint actions collapse to at most ~62
  distinct successors: a successful check's child depends only on
  ST = check - drop, and every failed-check cell shares ONE death
  event (duration min(cylinder + 60, 300)). Children are therefore probed
  once per outcome class, not per joint cell.
- The game graph ratchets (cylinders strictly grow between deaths), so the
  memoized reachable set within a small depth stays modest.

Chance is expanded through the engine itself (resolve_half_round with
forced survived_outcome) — no reimplemented mechanics, per invariant G4.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable

import numpy as np

from stl.solver.exact import ExactGameSnapshot, exact_public_state, terminal_value
from stl.engine.actions import legal_max_second
from stl.engine.game import Game

# (game, role) -> (seconds, probabilities) for the frozen seat's mixture.
PolicyFn = Callable[[Game, str], tuple[tuple[int, ...], np.ndarray]]
FrontierFn = Callable[[Game], tuple[float, float] | None]

FRONTIER = (-1.0, 1.0)


@dataclass(frozen=True)
class BRResult:
    """Certified interval on the best-response value (Hal perspective)."""

    lo: float
    hi: float
    depth: int
    frozen_name: str
    adversary_name: str
    states_solved: int
    engine_expansions: int
    policy_queries: int
    frontier_hits: int
    tablebase_frontier_hits: int
    elapsed_seconds: float

    @property
    def width(self) -> float:
        return self.hi - self.lo


class _BRSolver:
    def __init__(
        self,
        policy: PolicyFn,
        frozen_name: str,
        *,
        support_mass: float,
        max_states: int,
        frontier_fn: FrontierFn | None,
    ) -> None:
        self.policy = policy
        self.frozen_name = frozen_name.lower()
        self.support_mass = float(support_mass)
        self.max_states = int(max_states)
        self.frontier_fn = frontier_fn
        self.memo: dict = {}
        self.states_solved = 0
        self.engine_expansions = 0
        self.policy_queries = 0
        self.frontier_hits = 0
        self.tablebase_frontier_hits = 0

    # ── outcome-class child expansion ────────────────────────────────

    def _child_interval(
        self,
        game: Game,
        snap: ExactGameSnapshot,
        drop: int,
        check: int,
        depth_to_go: int,
    ) -> tuple[float, float]:
        """Interval for the (drop, check) cell, expanding chance exactly."""
        probe = game.resolve_half_round(drop, check, survived_outcome=None)
        death_possible = probe.survived is not None
        p_survive = probe.survival_probability
        snap.restore(game)
        self.engine_expansions += 1

        if not death_possible:
            game.resolve_half_round(drop, check, survived_outcome=None)
            out = self._solve(game, depth_to_go - 1)
            snap.restore(game)
            return out

        lo = hi = 0.0
        for outcome, weight in ((True, float(p_survive)), (False, 1.0 - float(p_survive))):
            if weight <= 0.0:
                continue
            game.resolve_half_round(drop, check, survived_outcome=outcome)
            clo, chi = self._solve(game, depth_to_go - 1)
            snap.restore(game)
            lo += weight * clo
            hi += weight * chi
        return lo, hi

    def _solve(self, game: Game, depth_to_go: int) -> tuple[float, float]:
        tval = terminal_value(game, perspective_name="Hal")
        if tval is not None:
            return float(tval), float(tval)
        if depth_to_go <= 0:
            self.frontier_hits += 1
            if self.frontier_fn is not None:
                frontier = self.frontier_fn(game)
                if frontier is not None:
                    lo, hi = frontier
                    self.tablebase_frontier_hits += 1
                    return max(float(lo), -1.0), min(float(hi), 1.0)
            return FRONTIER

        key = (exact_public_state(game), depth_to_go)
        if key in self.memo:
            return self.memo[key]
        if self.states_solved >= self.max_states:
            raise RuntimeError(
                f"best_response_interval exceeded max_states={self.max_states}; "
                f"reduce depth or raise the cap"
            )
        self.states_solved += 1

        dropper, checker = game.get_roles_for_half(game.current_half)
        turn_duration = game.get_turn_duration()
        frozen_is_dropper = dropper.name.lower() == self.frozen_name
        frozen_role = "dropper" if frozen_is_dropper else "checker"
        adversary = checker if frozen_is_dropper else dropper
        adversary_role = "checker" if frozen_is_dropper else "dropper"
        adversary_is_hal = adversary.name.lower() == "hal"

        seconds, probs = self.policy(game, frozen_role)
        self.policy_queries += 1
        probs = np.asarray(probs, dtype=np.float64)
        slack = 0.0
        if self.support_mass < 1.0:
            order = np.argsort(probs)[::-1]
            keep_mask = np.zeros(len(probs), dtype=bool)
            acc = 0.0
            for idx in order:
                keep_mask[idx] = True
                acc += probs[idx]
                if acc >= self.support_mass:
                    break
            dropped = float(probs[~keep_mask].sum())
            slack = 2.0 * dropped  # values live in [-1,1]
            seconds = tuple(s for s, k in zip(seconds, keep_mask) if k)
            probs = probs[keep_mask]
            probs = probs / probs.sum()

        snap = ExactGameSnapshot(game)
        adversary_max = legal_max_second(adversary.name, adversary_role, turn_duration)

        # Outcome-class cache: success children keyed by ST, one fail child.
        success_cache: dict[int, tuple[float, float]] = {}
        fail_cache: tuple[float, float] | None = None

        def cell(drop: int, check: int) -> tuple[float, float]:
            nonlocal fail_cache
            if check >= drop:  # success; ties succeed with ST=0
                st = check - drop
                if st not in success_cache:
                    success_cache[st] = self._child_interval(game, snap, drop, check, depth_to_go)
                return success_cache[st]
            if fail_cache is None:
                fail_cache = self._child_interval(game, snap, drop, check, depth_to_go)
            return fail_cache

        best: tuple[float, float] | None = None
        for b in range(1, adversary_max + 1):
            lo_acc = 0.0
            hi_acc = 0.0
            for a, pa in zip(seconds, probs):
                drop, check = (a, b) if frozen_is_dropper else (b, a)
                clo, chi = cell(drop, check)
                lo_acc += pa * clo
                hi_acc += pa * chi
            cand = (lo_acc - slack, hi_acc + slack)
            if best is None:
                best = cand
            elif adversary_is_hal:
                best = (max(best[0], cand[0]), max(best[1], cand[1]))
            else:
                best = (min(best[0], cand[0]), min(best[1], cand[1]))

        assert best is not None
        best = (max(best[0], -1.0), min(best[1], 1.0))
        self.memo[key] = best
        return best


def best_response_interval(
    game: Game,
    policy: PolicyFn,
    *,
    depth: int,
    frozen_name: str = "Hal",
    support_mass: float = 1.0,
    max_states: int = 2_000_000,
    frontier_fn: FrontierFn | None = None,
) -> BRResult:
    """Certified interval on the adversary's best-response value vs ``policy``.

    ``policy`` must be a pure function of the public state (Markov) — the
    SolverAgent's state-seeded policy and the net policy both qualify.
    The returned interval brackets the TRUE depth-unlimited best-response
    value because truncated frontiers are bracketed in [-1, +1].
    """
    snap = ExactGameSnapshot(game)
    dropper, checker = game.get_roles_for_half(game.current_half)
    names = {dropper.name.lower(), checker.name.lower()}
    if frozen_name.lower() not in names:
        raise ValueError(f"frozen_name {frozen_name!r} not in game players {names}")
    adversary_name = (names - {frozen_name.lower()}).pop()

    solver = _BRSolver(
        policy,
        frozen_name,
        support_mass=support_mass,
        max_states=max_states,
        frontier_fn=frontier_fn,
    )
    start = time.perf_counter()
    lo, hi = solver._solve(game, depth)
    elapsed = time.perf_counter() - start
    snap.restore(game)

    return BRResult(
        lo=lo,
        hi=hi,
        depth=depth,
        frozen_name=frozen_name,
        adversary_name=adversary_name,
        states_solved=solver.states_solved,
        engine_expansions=solver.engine_expansions,
        policy_queries=solver.policy_queries,
        frontier_hits=solver.frontier_hits,
        tablebase_frontier_hits=solver.tablebase_frontier_hits,
        elapsed_seconds=elapsed,
    )
