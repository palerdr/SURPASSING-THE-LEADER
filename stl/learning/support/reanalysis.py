"""Phase H: tiered reanalysis of the Phase-G rejected pool.

Phase G's horizon-2/3 LP sweep rejects any state whose ``unresolved_probability``
exceeds ``REJECTED_UNRESOLVED_THRESHOLD`` (0.5) into a rejected pool serialized as
full ``ExactPublicState`` JSON. Those are the highest-variance states in the
sweep — the horizon was too shallow to pin their value down. Phase H rescues
them with a two-tier deepening:

    Tier 1 — exact-deepen (`resolve_subgame`, a candidate-restricted
        ``selective_solve`` at a deeper horizon). If the state now resolves
        (``unresolved_probability <= accept_unresolved``), emit an exact target
        (``SOURCE_EXACT_HORIZON_4``) — a record Phase G discarded, recovered at
        exact quality. The exact tier is audited against the full-width oracle
        (`audit_reanalysis`, charter gate G5) and its strategy's exploitability
        checked (gate G7).

    Tier 2 — MCTS fallback. States still unresolved even at the deeper exact
        horizon get a high-iteration ``mcts_search`` estimate
        (``SOURCE_REANALYSIS_MCTS``) with subgame re-solve at critical nodes and
        the best available leaf evaluator (a trained net when supplied, else
        ``TerminalOnlyEvaluator``).

Output is **additive**: a new target list to be MERGED with — never overwrite —
the Phase-G corpus. This module imports from ``environment.cfr`` (allowed:
training may depend on the rigorous core) but adds nothing to it, so the
``environment/cfr`` firewall is untouched.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from stl.solver.diagnostics import diagnose_exact_strategy
from stl.solver.evaluator import LeafEvaluator, TerminalOnlyEvaluator
from stl.solver.exact import ExactSearchConfig, solve_exact_finite_horizon, terminal_value
from stl.solver.selective import SelectiveSearchResult, selective_solve
from stl.solver.subgame_resolve import resolve_subgame
from stl.learning.model import extract_features
from stl.engine.game import Game

from stl.learning.targets import (
    SOURCE_EXACT_HORIZON_4,
    SOURCE_REANALYSIS_MCTS,
    ValueTarget,
    _legal_policy_vectors,
    _strategy_vectors,
    game_from_public_state,
)


REANALYSIS_EXACT_HORIZON = 4
# Tier-1 success: deep solve leaves at most this much horizon-cutoff mass, so the
# exact value is well-pinned (the unresolved mass contributes 0 to the breakdown,
# so a near-zero residual keeps the exact target trustworthy).
REANALYSIS_ACCEPT_UNRESOLVED = 0.05
REANALYSIS_MCTS_ITERS = 4000
# Charter gate G5: selective value must track full-width exact within this gap.
REANALYSIS_AUDIT_MAX_GAP = 0.05


@dataclass(frozen=True)
class ReanalysisOutcome:
    """One rejected state's reanalysis result, for corpus + logging."""

    tier: str  # SOURCE_EXACT_HORIZON_4 | SOURCE_REANALYSIS_MCTS
    value_for_hal: float
    unresolved_after: float
    target: ValueTarget


def _exact_target(game: Game, result: SelectiveSearchResult, config: ExactSearchConfig) -> ValueTarget:
    drop_dist, check_dist = _strategy_vectors(
        drop_seconds=result.drop_seconds,
        check_seconds=result.check_seconds,
        dropper_strategy=result.dropper_strategy,
        checker_strategy=result.checker_strategy,
    )
    _, _, drop_mask, check_mask = _legal_policy_vectors(game, config)
    return ValueTarget(
        features=extract_features(game),
        value=float(result.value_for_hal),
        source=SOURCE_EXACT_HORIZON_4,
        horizon=REANALYSIS_EXACT_HORIZON,
        dropper_dist=drop_dist,
        checker_dist=check_dist,
        dropper_legal_mask=drop_mask,
        checker_legal_mask=check_mask,
        unresolved_probability=float(result.unresolved_probability),
    )


def reanalyze_state(
    game: Game,
    *,
    config: ExactSearchConfig | None = None,
    exact_horizon: int = REANALYSIS_EXACT_HORIZON,
    accept_unresolved: float = REANALYSIS_ACCEPT_UNRESOLVED,
    mcts_iters: int = REANALYSIS_MCTS_ITERS,
    evaluator: LeafEvaluator | None = None,
    rng: np.random.Generator | None = None,
) -> ReanalysisOutcome:
    """Tier-1 exact-deepen; tier-2 high-iter MCTS fallback.

    Returns the outcome (tier, value, residual unresolved mass, and the additive
    ValueTarget). ``evaluator`` is the MCTS leaf evaluator; defaults to
    ``TerminalOnlyEvaluator`` so reanalysis runs without a trained net.
    """
    config = config or ExactSearchConfig()

    # Tier 1: deeper EXACT solve (selective candidate set, no learned frontier).
    deep = resolve_subgame(game, horizon=exact_horizon, config=config)
    if deep.unresolved_probability <= accept_unresolved:
        return ReanalysisOutcome(
            tier=SOURCE_EXACT_HORIZON_4,
            value_for_hal=float(deep.value_for_hal),
            unresolved_after=float(deep.unresolved_probability),
            target=_exact_target(game, deep, config),
        )

    # Tier 2: high-iter MCTS fallback for states still unresolved at depth.
    from stl.solver.mcts import MCTSConfig, make_node, mcts_search

    leaf = evaluator or TerminalOnlyEvaluator()
    if rng is None:
        rng = np.random.default_rng(0)
    mcts_config = MCTSConfig(
        iterations=mcts_iters, exploration_c=1.0, evaluator=None, use_tablebase=False
    )
    root = make_node(game, config, evaluator=leaf)
    result = mcts_search(
        game=game,
        config=mcts_config,
        evaluator=leaf,
        rng=rng,
        exact_config=config,
        subgame_resolve_at_critical=True,
    )
    drop_dist, check_dist = _strategy_vectors(
        drop_seconds=root.drop_seconds,
        check_seconds=root.check_seconds,
        dropper_strategy=result.root_strategy_dropper,
        checker_strategy=result.root_strategy_checker,
    )
    _, _, drop_mask, check_mask = _legal_policy_vectors(game, config)
    target = ValueTarget(
        features=extract_features(game),
        value=float(result.root_value_for_hal),
        source=SOURCE_REANALYSIS_MCTS,
        horizon=mcts_iters,
        dropper_dist=drop_dist,
        checker_dist=check_dist,
        dropper_legal_mask=drop_mask,
        checker_legal_mask=check_mask,
        unresolved_probability=float(deep.unresolved_probability),
    )
    return ReanalysisOutcome(
        tier=SOURCE_REANALYSIS_MCTS,
        value_for_hal=float(result.root_value_for_hal),
        unresolved_after=float(deep.unresolved_probability),
        target=target,
    )


def reanalyze_pool(
    pool: list[dict],
    *,
    config: ExactSearchConfig | None = None,
    exact_horizon: int = REANALYSIS_EXACT_HORIZON,
    accept_unresolved: float = REANALYSIS_ACCEPT_UNRESOLVED,
    mcts_iters: int = REANALYSIS_MCTS_ITERS,
    evaluator: LeafEvaluator | None = None,
    seed: int = 0,
    limit: int | None = None,
) -> list[ReanalysisOutcome]:
    """Reanalyze a rejected pool (list of ExactPublicState dicts) into additive
    outcomes. Already-terminal entries are skipped (they should not appear in a
    rejected pool, but the guard keeps the contract clean)."""
    config = config or ExactSearchConfig()
    rng = np.random.default_rng(seed)
    states = pool if limit is None else pool[:limit]
    outcomes: list[ReanalysisOutcome] = []
    for snapshot in states:
        game = game_from_public_state(snapshot)
        if terminal_value(game, perspective_name=config.perspective_name) is not None:
            continue
        outcomes.append(
            reanalyze_state(
                game,
                config=config,
                exact_horizon=exact_horizon,
                accept_unresolved=accept_unresolved,
                mcts_iters=mcts_iters,
                evaluator=evaluator,
                rng=rng,
            )
        )
    return outcomes


@dataclass(frozen=True)
class ReanalysisAudit:
    """G5/G7 audit summary over a sample of exact-tier reanalysis outcomes."""

    sampled: int
    exact_tier_total: int
    max_value_gap: float
    max_nash_gap: float
    failures: list[str]  # human-readable state descriptors exceeding a gate

    @property
    def passed(self) -> bool:
        return not self.failures


def audit_reanalysis(
    pool: list[dict],
    *,
    config: ExactSearchConfig | None = None,
    exact_horizon: int = REANALYSIS_EXACT_HORIZON,
    accept_unresolved: float = REANALYSIS_ACCEPT_UNRESOLVED,
    max_gap: float = REANALYSIS_AUDIT_MAX_GAP,
    sample: int = 8,
) -> ReanalysisAudit:
    """Charter gates G5 + G7 on the EXACT reanalysis tier.

    For up to ``sample`` rejected states that the exact tier resolves, re-solve
    full-width at the same horizon and check (G5) the selective value tracks the
    full-width oracle within ``max_gap`` and (G7) the full-width strategy's
    ``nash_gap`` is ~0. The audit is SAMPLED because full-width depth-4 on
    contested states is the expensive path; the sample size is reported so a
    bounded audit is never mistaken for full coverage (charter: no silent caps).
    """
    config = config or ExactSearchConfig()
    sampled = 0
    exact_total = 0
    max_value_gap = 0.0
    max_nash_gap = 0.0
    failures: list[str] = []

    for snapshot in pool:
        game = game_from_public_state(snapshot)
        if terminal_value(game, perspective_name=config.perspective_name) is not None:
            continue
        selective = resolve_subgame(game, horizon=exact_horizon, config=config)
        if selective.unresolved_probability > accept_unresolved:
            continue  # not an exact-tier rescue; tier-2 MCTS is judged elsewhere
        exact_total += 1
        if sampled >= sample:
            continue

        full = solve_exact_finite_horizon(game, exact_horizon, config)
        value_gap = abs(selective.value_for_hal - full.value_for_hal)
        max_value_gap = max(max_value_gap, value_gap)
        descriptor = f"clock={game.game_clock},half={game.current_half},p2cyl={game.player2.cylinder}"
        if value_gap > max_gap:
            failures.append(f"G5 value_gap {value_gap:.4f} > {max_gap} at {descriptor}")

        if full.payoff_for_hal is not None:
            diag = diagnose_exact_strategy(game, full, perspective_name=config.perspective_name)
            max_nash_gap = max(max_nash_gap, diag.nash_gap)
            if diag.nash_gap > 1e-6:
                failures.append(f"G7 nash_gap {diag.nash_gap:.2e} at {descriptor}")
        sampled += 1

    return ReanalysisAudit(
        sampled=sampled,
        exact_tier_total=exact_total,
        max_value_gap=max_value_gap,
        max_nash_gap=max_nash_gap,
        failures=failures,
    )
