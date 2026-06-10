"""Anti-determinism exploiter: punishes any opponent that repeats seconds.

A frozen *mixed* policy is safe against this bot; a policy that has
collapsed to a pure (or near-pure) action is maximally punished:

  - If the target keeps CHECKING at second c, drop at c+1 — the check
    happens before the handkerchief is down, a guaranteed failed check.
  - If the target keeps DROPPING at second d, check at exactly d — a
    guaranteed success with the minimum possible squandered time.

This makes the bot the cheap stand-in for the best-response probe in
match play: a deterministic agent's win-rate collapses against it while
a well-mixed agent should score about as it does against ``safe``.

The bot reads only ``game.history`` (opponents are privileged) and only
records from THIS game, so ``reset()`` has nothing to clear.
"""

from __future__ import annotations

from collections import Counter

from environment.legal_actions import clamp_action
from src.Game import Game

from .base import Opponent

# Need at least this many observations of the target in a role before
# trusting the mode; below it, fall back to the scripted defaults.
_MIN_SAMPLES = 3

# The mode must cover at least this fraction of the samples to count as
# "the target is repeating itself".
_MIN_MODAL_FREQUENCY = 0.5

# Fallback drop second when no pattern has emerged (mid-turn keeps the
# checker honest without giving away large ST).
_FALLBACK_DROP = 30

# Fallback check second: the safe last-normal-second check.
_FALLBACK_CHECK = 60


def _modal_second(samples: list[int]) -> tuple[int | None, float]:
    """(modal second, modal frequency); ties break to the smallest second."""
    if not samples:
        return None, 0.0
    counts = Counter(samples)
    best_count = max(counts.values())
    modal = min(sec for sec, cnt in counts.items() if cnt == best_count)
    return modal, best_count / len(samples)


class PatternReaderBaku(Opponent):
    """Baku-seat opponent that mirrors the target's repeated seconds.

    Construct with the *target's* player name (default ``"Hal"`` — the
    player this bot is NOT). Name comparison is case-insensitive to
    match the engine's conventions.
    """

    def __init__(self, target_name: str = "Hal") -> None:
        self.target_name = target_name

    # ── helpers ───────────────────────────────────────────────────────

    def _target_check_seconds(self, game: Game) -> list[int]:
        target = self.target_name.lower()
        return [
            rec.check_time
            for rec in game.history
            if rec.checker.lower() == target
        ]

    def _target_drop_seconds(self, game: Game) -> list[int]:
        target = self.target_name.lower()
        return [
            rec.drop_time
            for rec in game.history
            if rec.dropper.lower() == target
        ]

    # ── Opponent interface ────────────────────────────────────────────

    def choose_action(self, game: Game, role: str, turn_duration: int) -> int:
        if role == "dropper":
            samples = self._target_check_seconds(game)
            modal, freq = _modal_second(samples)
            if (
                len(samples) >= _MIN_SAMPLES
                and modal is not None
                and freq >= _MIN_MODAL_FREQUENCY
            ):
                # Drop one second AFTER the modal check: if the target
                # repeats, it checks before the drop -> guaranteed fail.
                second = modal + 1
            else:
                second = _FALLBACK_DROP
            return clamp_action(
                second, actor="baku", role="dropper", turn_duration=turn_duration
            )

        samples = self._target_drop_seconds(game)
        modal, freq = _modal_second(samples)
        if (
            len(samples) >= _MIN_SAMPLES
            and modal is not None
            and freq >= _MIN_MODAL_FREQUENCY
        ):
            # Check at exactly the modal drop: success with minimal ST.
            second = modal
        else:
            second = _FALLBACK_CHECK
        return clamp_action(
            second, actor="baku", role="checker", turn_duration=turn_duration
        )

    def reset(self) -> None:
        """No persistent state: history is per-game. Kept as explicit no-op."""
