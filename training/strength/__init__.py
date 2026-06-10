"""Playing-strength measurement: best-response exploitability, match gates.

This package exists because value-MSE on a self-generated ruler does not
measure playing strength. The honest defensibility metric for a frozen
Markov policy in this game is an exact best-response computation — the
opponent-plus-chance fold into an MDP, solvable by depth-limited DP with
certified interval brackets at the truncation frontier.
"""

from .best_response import BRResult, best_response_interval
from .policies import agent_policy, fixed_second_policy, net_policy, uniform_policy

__all__ = [
    "BRResult",
    "best_response_interval",
    "agent_policy",
    "fixed_second_policy",
    "net_policy",
    "uniform_policy",
]

# ── Match gate / SPRT layer (tickets 9+10) — appended exports only ──
from .sprt import SPRTState, sprt_verdict
from .match_gate import (
    LadderEntry,
    gate_report,
    reset_per_game,
    run_ladder,
    run_ladder_entries,
    wilson_interval,
)

__all__ += [
    "SPRTState",
    "sprt_verdict",
    "LadderEntry",
    "gate_report",
    "reset_per_game",
    "run_ladder",
    "run_ladder_entries",
    "wilson_interval",
]
