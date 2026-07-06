"""Sequential Probability Ratio Test for candidate-vs-reference matches.

Implements the *simplified GSPRT* of Michel Van den Bergh (the
approximation underlying fishtest's simpler cousins, e.g. OpenBench /
cutechess-style gating): the trinomial (W, D, L) record is collapsed to
a per-game score in {0, 0.5, 1}, and the generalized log-likelihood
ratio between two Elo hypotheses is approximated by

    s      = (W + 0.5 * D) / N
    sigma2 = (W*(1-s)^2 + D*(0.5-s)^2 + L*(0-s)^2) / N
    s_i    = 1 / (1 + 10 ** (-elo_i / 400))          (elo -> score)
    LLR    ~= (s1 - s0) * (2*s - s0 - s1) * N / (2 * sigma2)

with the classical Wald bounds

    lower = log(beta / (1 - alpha))    -> "reject"  (H0: elo <= elo0)
    upper = log((1 - beta) / alpha)    -> "accept"  (H1: elo >= elo1)

Reference: M. Van den Bergh, "GSPRT approximation of the sequential
probability ratio test" (fishtest wiki / mathematics writeup). This is
the score-and-variance normal approximation, not the full trinomial
MLE version fishtest currently ships, but it shares the same bounds and
asymptotics and is standard for small gating harnesses.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

__all__ = ["SPRTState", "sprt_verdict", "elo_to_score", "sprt_llr"]


# Below this many games a zero-variance record (all wins / all draws /
# all losses) is not allowed to terminate the test: 1-2 lucky games
# should never gate a candidate.
_MIN_GAMES_FOR_DEGENERATE_VERDICT = 5


def elo_to_score(elo: float) -> float:
    """Expected score of an ``elo``-stronger player (logistic model)."""
    return 1.0 / (1.0 + 10.0 ** (-elo / 400.0))


def sprt_llr(
    wins: int,
    draws: int,
    losses: int,
    *,
    elo0: float = 0.0,
    elo1: float = 20.0,
) -> float:
    """Simplified-GSPRT log-likelihood ratio for H1(elo1) vs H0(elo0).

    Returns 0.0 when no games have been played and ``+/-inf`` for
    degenerate (zero-variance) records that sit outside [s0, s1] — the
    caller decides whether enough games back such a verdict.
    """
    n = wins + draws + losses
    if n == 0:
        return 0.0

    s = (wins + 0.5 * draws) / n
    sigma2 = (
        wins * (1.0 - s) ** 2
        + draws * (0.5 - s) ** 2
        + losses * (0.0 - s) ** 2
    ) / n

    s0 = elo_to_score(elo0)
    s1 = elo_to_score(elo1)

    if sigma2 <= 0.0:
        # All games had the identical result. The normal approximation
        # collapses; classify by where the (constant) score sits.
        lo, hi = min(s0, s1), max(s0, s1)
        if s >= hi:
            return math.inf
        if s <= lo:
            return -math.inf
        return 0.0

    return (s1 - s0) * (2.0 * s - s0 - s1) * n / (2.0 * sigma2)


def sprt_verdict(
    wins: int,
    draws: int,
    losses: int,
    *,
    elo0: float = 0.0,
    elo1: float = 20.0,
    alpha: float = 0.05,
    beta: float = 0.05,
) -> str:
    """Run the simplified GSPRT on a (W, D, L) record.

    Returns one of:
      - ``"accept"``   — H1 accepted (candidate is at least ``elo1`` strong),
      - ``"reject"``   — H0 accepted (candidate is at most ``elo0`` strong),
      - ``"continue"`` — neither bound crossed; play more games.

    Degenerate records (zero score variance, i.e. every game had the
    same result) only terminate once at least
    ``_MIN_GAMES_FOR_DEGENERATE_VERDICT`` games have been played, so a
    couple of identical results can never gate a candidate by
    themselves.
    """
    if not (0.0 < alpha < 1.0 and 0.0 < beta < 1.0):
        raise ValueError(f"alpha/beta must lie in (0, 1); got {alpha}, {beta}")
    if elo1 <= elo0:
        raise ValueError(f"elo1 must exceed elo0; got elo0={elo0}, elo1={elo1}")
    if min(wins, draws, losses) < 0:
        raise ValueError("W/D/L counts must be non-negative")

    n = wins + draws + losses
    llr = sprt_llr(wins, draws, losses, elo0=elo0, elo1=elo1)

    if math.isinf(llr) and n < _MIN_GAMES_FOR_DEGENERATE_VERDICT:
        return "continue"

    lower = math.log(beta / (1.0 - alpha))
    upper = math.log((1.0 - beta) / alpha)

    if llr >= upper:
        return "accept"
    if llr <= lower:
        return "reject"
    return "continue"


@dataclass
class SPRTState:
    """Accumulating W/D/L record for one candidate-vs-reference pairing.

    Feed it results as they arrive and poll :meth:`verdict`; the test is
    sequential, so checking after every game is the intended usage.
    """

    wins: int = 0
    draws: int = 0
    losses: int = 0

    @property
    def n(self) -> int:
        return self.wins + self.draws + self.losses

    def record(self, result: str) -> None:
        """Record one game result: ``"win"``, ``"draw"``, or ``"loss"``."""
        if result == "win":
            self.wins += 1
        elif result == "draw":
            self.draws += 1
        elif result == "loss":
            self.losses += 1
        else:
            raise ValueError(f"result must be win/draw/loss, got {result!r}")

    def record_match(self, wins: int, draws: int, losses: int) -> None:
        """Fold an aggregated (W, D, L) block into the running record."""
        if min(wins, draws, losses) < 0:
            raise ValueError("W/D/L counts must be non-negative")
        self.wins += wins
        self.draws += draws
        self.losses += losses

    def llr(self, *, elo0: float = 0.0, elo1: float = 20.0) -> float:
        return sprt_llr(
            self.wins, self.draws, self.losses, elo0=elo0, elo1=elo1
        )

    def verdict(
        self,
        *,
        elo0: float = 0.0,
        elo1: float = 20.0,
        alpha: float = 0.05,
        beta: float = 0.05,
    ) -> str:
        return sprt_verdict(
            self.wins,
            self.draws,
            self.losses,
            elo0=elo0,
            elo1=elo1,
            alpha=alpha,
            beta=beta,
        )
