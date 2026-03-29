"""Opponent implementations used by the environment layer.

This package contains simple baselines, scripted teachers, model-backed
opponents, and weighted leagues that mix them together.
"""

from .base import Opponent
from .league import LeagueEntry, WeightedOpponentLeague
from .random_bot import RandomBot
from .safe_bot import BridgePressureBot, SafeBot, LeapAwareSafeBot
