"""Certified tablebase construction (plan Phase 3).

Backward induction over the post-leap clock-free quotient, one epoch
(ttd_hal, ttd_baku, cprs) at a time. Where children leave the solved
region (a survived death entering an unsolved epoch), their values are
BRACKETED and the sweep propagates certified [lo, hi] intervals — the
declared approximation layer, which is why this package lives OUTSIDE
the exact-only ``environment/cfr`` namespace. The per-state transition
structure itself comes from the engine-equivalence-verified analytic map
(``environment/cfr/backward.py``); survival probabilities come from the
engine referee (invariant G4).
"""

from .epoch_sweep import EpochSpec, solve_epoch, survival_table
from .tier_a import TierAEvaluator, TierAInterval, TierALookup, TierALookupResult, frontier_interval_fn

__all__ = [
    "EpochSpec",
    "solve_epoch",
    "survival_table",
    "TierAEvaluator",
    "TierAInterval",
    "TierALookup",
    "TierALookupResult",
    "frontier_interval_fn",
]
