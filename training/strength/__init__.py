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
