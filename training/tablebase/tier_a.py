"""Runtime access to the generated Tier A interval tablebase.

Tier A is an interval-valued post-leap quotient over states with <= 1
total death. It is deliberately outside ``environment.cfr``: these values
are certified brackets useful for search frontiers, not part of the
terminal-only exact oracle.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

from environment.cfr.evaluator import LeafEvaluation, normalize_leaf_evaluation
from environment.cfr.exact import terminal_value
from environment.legal_actions import legal_max_second
from src.Constants import LS_WINDOW_END
from src.Game import Game


DEFAULT_TIER_A_DIR = Path(__file__).resolve().parents[2] / "checkpoints" / "tablebase" / "tier_a"


@dataclass(frozen=True)
class TierAInterval:
    lo: float
    hi: float
    source: str
    bit: int
    hal_cylinder: int
    baku_cylinder: int

    @property
    def width(self) -> float:
        return self.hi - self.lo

    @property
    def midpoint(self) -> float:
        return 0.5 * (self.lo + self.hi)


@dataclass(frozen=True)
class TierALookupResult:
    interval: TierAInterval | None
    miss_reason: str | None = None

    @property
    def hit(self) -> bool:
        return self.interval is not None


def _player_by_name(game: Game, name: str):
    for player in (game.player1, game.player2):
        if player.name.lower() == name.lower():
            return player
    return None


def _int_index(value: float, *, name: str) -> tuple[int | None, str | None]:
    idx = int(round(float(value)))
    if abs(float(value) - idx) > 1e-9:
        return None, f"{name}_non_integer"
    if not (0 <= idx < 300):
        return None, f"{name}_out_of_range"
    return idx, None


def _bit_for_current_roles(game: Game) -> tuple[int | None, str | None]:
    if game.game_over:
        return None, "terminal"
    dropper, checker = game.get_roles_for_half(game.current_half)
    if dropper.name.lower() == "hal" and checker.name.lower() == "baku":
        return 0, None
    if dropper.name.lower() == "baku" and checker.name.lower() == "hal":
        return 1, None
    return None, "unsupported_roles"


class TierALookup:
    """Lazy lookup for ``checkpoints/tablebase/tier_a`` interval artifacts."""

    def __init__(self, root: str | os.PathLike[str] = DEFAULT_TIER_A_DIR, *, verify: bool = False) -> None:
        self.root = Path(root)
        self._cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        if verify:
            self.verify_manifest()

    def verify_manifest(self) -> dict[str, str]:
        manifest_path = self.root / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(f"Tier A manifest not found: {manifest_path}")
        with manifest_path.open() as fh:
            manifest: dict[str, str] = json.load(fh)
        for name, expected in manifest.items():
            path = self.root / name
            if not path.exists():
                raise FileNotFoundError(f"Tier A artifact listed in manifest is missing: {path}")
            with path.open("rb") as fh:
                actual = hashlib.sha256(fh.read()).hexdigest()
            if actual != expected:
                raise ValueError(f"Tier A artifact hash mismatch for {name}: {actual} != {expected}")
        return manifest

    def _load(self, name: str) -> tuple[np.ndarray, np.ndarray]:
        if name not in self._cache:
            path = self.root / name
            if not path.exists():
                raise FileNotFoundError(f"Tier A artifact not found: {path}")
            with np.load(path) as data:
                self._cache[name] = (
                    data["lo"].astype(np.float64),
                    data["hi"].astype(np.float64),
                )
        return self._cache[name]

    def _artifact_for_game(self, game: Game) -> tuple[str | None, str | None]:
        if game.game_clock <= float(LS_WINDOW_END):
            return None, "not_post_leap"
        if game.get_turn_duration() != 60:
            return None, "non_normal_turn"
        if game.referee.cprs_performed != game.player1.deaths + game.player2.deaths:
            return None, "cprs_death_mismatch"

        hal = _player_by_name(game, "Hal")
        baku = _player_by_name(game, "Baku")
        if hal is None or baku is None:
            return None, "missing_hal_or_baku"

        total_deaths = int(hal.deaths + baku.deaths)
        if total_deaths == 0:
            if game.referee.cprs_performed != 0 or hal.ttd != 0.0 or baku.ttd != 0.0:
                return None, "unsupported_d0_epoch"
            return "d0.npz", None

        if total_deaths == 1 and game.referee.cprs_performed == 1:
            if hal.deaths == 1 and baku.deaths == 0 and baku.ttd == 0.0:
                ttd, reason = _int_index(hal.ttd, name="hal_ttd")
                if reason is not None:
                    return None, reason
                if not (60 <= ttd <= 299):
                    return None, "hal_ttd_out_of_tier_a_range"
                return f"d1_hal_{ttd}.npz", None
            if baku.deaths == 1 and hal.deaths == 0 and hal.ttd == 0.0:
                ttd, reason = _int_index(baku.ttd, name="baku_ttd")
                if reason is not None:
                    return None, reason
                if not (60 <= ttd <= 299):
                    return None, "baku_ttd_out_of_tier_a_range"
                return f"d1_baku_{ttd}.npz", None
            return None, "unsupported_d1_epoch"

        return None, "too_many_deaths"

    def lookup(self, game: Game) -> TierALookupResult:
        terminal = terminal_value(game, perspective_name="Hal")
        if terminal is not None:
            return TierALookupResult(
                TierAInterval(float(terminal), float(terminal), "terminal", -1, -1, -1)
            )

        bit, reason = _bit_for_current_roles(game)
        if reason is not None:
            return TierALookupResult(None, reason)

        hal = _player_by_name(game, "Hal")
        baku = _player_by_name(game, "Baku")
        if hal is None or baku is None:
            return TierALookupResult(None, "missing_hal_or_baku")
        ch, reason = _int_index(hal.cylinder, name="hal_cylinder")
        if reason is not None:
            return TierALookupResult(None, reason)
        cb, reason = _int_index(baku.cylinder, name="baku_cylinder")
        if reason is not None:
            return TierALookupResult(None, reason)

        artifact, reason = self._artifact_for_game(game)
        if reason is not None:
            return TierALookupResult(None, reason)
        if not (self.root / artifact).exists():
            return TierALookupResult(None, "artifact_missing")

        lo, hi = self._load(artifact)
        low = float(lo[bit, ch, cb])
        high = float(hi[bit, ch, cb])
        if not np.isfinite(low) or not np.isfinite(high):
            return TierALookupResult(None, "non_finite_interval")
        if low > high + 1e-7:
            return TierALookupResult(None, "unordered_interval")
        return TierALookupResult(TierAInterval(low, high, artifact, bit, ch, cb))


def uniform_policy_for_game(game: Game) -> tuple[np.ndarray, np.ndarray]:
    if game.game_over:
        return np.zeros(61, dtype=np.float64), np.zeros(61, dtype=np.float64)
    dropper, checker = game.get_roles_for_half(game.current_half)
    turn_duration = game.get_turn_duration()
    drop_max = legal_max_second(dropper.name, "dropper", turn_duration)
    check_max = legal_max_second(checker.name, "checker", turn_duration)
    drop = np.zeros(61, dtype=np.float64)
    check = np.zeros(61, dtype=np.float64)
    if drop_max > 0:
        drop[:drop_max] = 1.0 / drop_max
    if check_max > 0:
        check[:check_max] = 1.0 / check_max
    return drop, check


class TierAEvaluator:
    """Leaf evaluator wrapper that short-circuits low-width Tier A hits."""

    def __init__(
        self,
        fallback: Callable[[Game], LeafEvaluation | float],
        *,
        lookup: TierALookup | None = None,
        max_width: float = 0.0,
        use_midpoint_for_wide: bool = False,
        preserve_fallback_policy: bool = True,
    ) -> None:
        self.fallback = fallback
        self.lookup = lookup or TierALookup()
        self.max_width = float(max_width)
        self.use_midpoint_for_wide = bool(use_midpoint_for_wide)
        self.preserve_fallback_policy = bool(preserve_fallback_policy)
        self.hits = 0
        self.wide_hits = 0
        self.misses: dict[str, int] = {}

    def __call__(self, game: Game) -> LeafEvaluation:
        result = self.lookup.lookup(game)
        if result.interval is not None:
            interval = result.interval
            if interval.width <= self.max_width or self.use_midpoint_for_wide:
                self.hits += 1
                if self.preserve_fallback_policy and not game.game_over:
                    _, drop, check = normalize_leaf_evaluation(self.fallback(game), game)
                else:
                    drop, check = uniform_policy_for_game(game)
                return float(interval.midpoint), drop, check
            self.wide_hits += 1
            self.misses["wide_interval"] = self.misses.get("wide_interval", 0) + 1
        else:
            reason = result.miss_reason or "unknown"
            self.misses[reason] = self.misses.get(reason, 0) + 1
        return normalize_leaf_evaluation(self.fallback(game), game)


def frontier_interval_fn(
    lookup: TierALookup | None = None,
    *,
    max_width: float | None = None,
) -> Callable[[Game], tuple[float, float] | None]:
    table = lookup or TierALookup()

    def fn(game: Game) -> tuple[float, float] | None:
        result = table.lookup(game)
        if result.interval is None:
            return None
        if max_width is not None and result.interval.width > max_width:
            return None
        return result.interval.lo, result.interval.hi

    return fn
