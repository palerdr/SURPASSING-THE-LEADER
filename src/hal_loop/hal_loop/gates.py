"""Promotion gates as pure decisions over evaluation results.

The constants below are the gate protocol. The README quotes them, and the
likelihood gate, the propose op, and the tests read them from here.
"""

from __future__ import annotations

from collections.abc import Mapping

EXPLOITATION_TOLERANCE = 0.01
REQUIRED_VERDICT = "fitted"
MINIMUM_SCORED_PLAYERS = 5
MINIMUM_SCORED_MOVES = 100


def verdict_problem(verdict: str | None) -> str | None:
    """Return why a card verdict blocks promotion, or None when it allows promotion."""
    if verdict != REQUIRED_VERDICT:
        return f"the card verdict is {verdict!r}, not {REQUIRED_VERDICT!r}"
    return None


def likelihood_gate(evaluation: Mapping) -> tuple[bool, str]:
    """Pass when the card verdict is fitted, the test split has enough data, and the candidate beats production.

    The pooled candidate-minus-production difference per move must be positive,
    and the lower bound of its 95% interval must be positive.
    """
    problem = verdict_problem(evaluation.get("verdict"))
    if problem:
        return False, problem
    pooled = evaluation["pooled"]
    players, moves = evaluation["players"], pooled["moves"]
    if players < MINIMUM_SCORED_PLAYERS or moves < MINIMUM_SCORED_MOVES:
        return False, (
            f"too little scored data: {players} players and {moves:g} moves, "
            f"against minimums of {MINIMUM_SCORED_PLAYERS} players and {MINIMUM_SCORED_MOVES} moves"
        )
    difference = pooled["difference"]
    low, high = pooled["interval"]
    if difference is None or low is None:
        return False, "no scored moves"
    passed = difference > 0 and low > 0
    verdict = "beats" if passed else "does not beat"
    return passed, (
        f"candidate {verdict} {evaluation['production_source']} production on {players} players and {moves:g} moves: "
        f"{difference:+.5f} nats per move, 95% interval [{low:+.5f}, {high:+.5f}]"
    )


def exploitation_gate(candidate_score: float, equilibrium_score: float, tolerance: float = EXPLOITATION_TOLERANCE) -> tuple[bool, str]:
    """Pass when the adaptive candidate scores at most `tolerance` below the equilibrium provider."""
    passed = candidate_score >= equilibrium_score - tolerance
    return passed, (
        f"adaptive mean score {candidate_score:.4f} against equilibrium {equilibrium_score:.4f} "
        f"(tolerance {tolerance}); {'pass' if passed else 'fail'}"
    )
