"""Tests for Phase H tiered reanalysis (training/reanalysis.py).

Kept fast: the exact-tier + audit tests use near-overflow states that resolve at
shallow depth (so full-width audit is cheap). The genuinely contested states that
populate a real rejected pool are deliberately NOT solved at depth 4 here — that
is the heavy production path, exercised by the pilot/CLI, not the unit suite.
"""

from __future__ import annotations

import os
import sys
from dataclasses import asdict

import pytest

sys.path.insert(0, os.getcwd())

from stl.solver.exact import ExactSearchConfig, exact_public_state
from stl.solver.selective import selective_solve
from stl.learning.model import FEATURE_DIM
from stl.learning.reanalysis import (
    SOURCE_EXACT_HORIZON_4,
    SOURCE_REANALYSIS_MCTS,
    audit_reanalysis,
    reanalyze_pool,
    reanalyze_state,
)
from stl.learning.targets import VALID_SOURCES, _build_game, game_from_public_state

CFG = ExactSearchConfig()


def _forced_overflow_game():
    # Baku checker at cyl=299: EVERY joint action drives an overflow/fail death of
    # duration 300 (p=0 → permanent), so every cell is terminal at depth 1. The
    # exact tier resolves it to +1 with no survive-line recursion, keeping the
    # full-width audit fast. (Near-but-not-fully-forced cylinders have surviving
    # ST<=2 lines that recurse deeply at full width — the heavy production path.)
    return _build_game(baku_cylinder=299.0, hal_cylinder=0.0, clock=720.0, current_half=1)


def test_reanalysis_sources_are_registered():
    assert SOURCE_EXACT_HORIZON_4 in VALID_SOURCES
    assert SOURCE_REANALYSIS_MCTS in VALID_SOURCES


def test_game_from_public_state_round_trips_all_fields():
    """Reconstruction must round-trip every public-state field, from both the
    ExactPublicState object and its JSON-dict form (the load_rejected_pool form)."""
    g = _build_game(
        baku_cylinder=180.0, hal_cylinder=120.0, clock=3450.0, current_half=1,
        baku_deaths=1, hal_deaths=1, referee_cprs=3,
    )
    g.player1.ttd = 60.0
    g.player2.ttd = 120.0
    s = exact_public_state(g)

    assert exact_public_state(game_from_public_state(s)) == s
    assert exact_public_state(game_from_public_state(asdict(s))) == s


def test_unresolved_probability_nonincreasing_with_depth():
    """Deeper search resolves at least as much terminal mass — never less. This
    monotonicity is what makes deeper reanalysis a *rescue* of unresolved states.
    Uses a low-cylinder state where baku HAS a safe check (cyl<=240 → checking 60
    survives every drop), so real unresolved mass survives to horizon 1, and
    shallow horizons (1→2) keep it fast."""
    game = _build_game(baku_cylinder=240.0, hal_cylinder=0.0, clock=720.0, current_half=1)
    u1 = selective_solve(game, 1, CFG).unresolved_probability
    u2 = selective_solve(game, 2, CFG).unresolved_probability
    assert u1 > 0.0  # genuinely unresolved at shallow depth (baku survives)
    assert u2 <= u1 + 1e-9


def test_reanalyze_state_emits_valid_additive_target():
    out = reanalyze_state(_forced_overflow_game(), config=CFG, mcts_iters=100)
    assert out.tier in (SOURCE_EXACT_HORIZON_4, SOURCE_REANALYSIS_MCTS)
    assert -1.0 <= out.value_for_hal <= 1.0

    t = out.target
    assert t.source == out.tier
    assert t.features.shape == (FEATURE_DIM,)
    assert t.dropper_dist.shape == (61,)
    assert t.checker_dist.shape == (61,)
    assert t.dropper_legal_mask.shape == (61,)
    assert t.checker_legal_mask.shape == (61,)


def test_exact_tier_rescue_audits_clean_against_full_width():
    """A near-overflow state is rescued by the exact tier; its selective value
    must match the full-width oracle (G5) with ~0 exploitability (G7)."""
    game = _forced_overflow_game()
    out = reanalyze_state(game, config=CFG)
    assert out.tier == SOURCE_EXACT_HORIZON_4
    assert out.unresolved_after <= 0.05

    audit = audit_reanalysis([asdict(exact_public_state(game))], config=CFG, sample=4)
    assert audit.exact_tier_total >= 1
    assert audit.passed, audit.failures
    assert audit.max_value_gap <= 0.05
    assert audit.max_nash_gap <= 1e-6


def test_mcts_fallback_tier_fires_when_exact_does_not_resolve():
    """Tier-2 coverage. An impossible accept threshold routes even a fully
    resolved state to the high-iter MCTS fallback (on real data this fires for
    states still unresolved at the deeper exact horizon), producing a
    SOURCE_REANALYSIS_MCTS target. Forced state + tiny iters keeps it fast."""
    out = reanalyze_state(
        _forced_overflow_game(), config=CFG, accept_unresolved=-1.0, mcts_iters=64
    )
    assert out.tier == SOURCE_REANALYSIS_MCTS
    assert out.target.source == SOURCE_REANALYSIS_MCTS
    assert -1.0 <= out.value_for_hal <= 1.0


def test_reanalyze_pool_skips_terminal_states():
    """A rejected pool should not contain terminals, but if one slips in it is
    skipped (terminals are already exact, not reanalysis candidates)."""
    nonterminal = asdict(exact_public_state(_forced_overflow_game()))
    terminal = dict(nonterminal)
    terminal["game_over"] = True
    terminal["winner_name"] = "Hal"
    terminal["loser_name"] = "Baku"

    outcomes = reanalyze_pool([nonterminal, terminal], config=CFG, mcts_iters=50)
    assert len(outcomes) == 1
    assert outcomes[0].target.source in (SOURCE_EXACT_HORIZON_4, SOURCE_REANALYSIS_MCTS)
