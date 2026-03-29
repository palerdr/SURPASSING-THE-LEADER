"""
Drop The Handkerchief — Referee

The Referee (Yakou) manages the death/revival process.

The referee has ONE job in the engine: determine whether a player survives
a death episode. This involves:

1. Computing the survival probability from four independent factors.
2. Rolling against that probability.
3. Tracking CPR fatigue (how many revivals have been performed total).

The referee is a GLOBAL resource — fatigue accumulates regardless of
which player died. If Baku dies twice and Hal dies once, Yakou has
performed 3 CPRs and is exhausted for whoever needs revival next.

Survival Probability:
    P = base_curve(this_death_duration)
      x cardiac_modifier(player.ttd before this death)
      x referee_modifier(self.cprs_performed)
      x player.physicality

Where:
    base_curve(t)           = max(0, 1 - (t / 300)^k)
    cardiac_modifier(ttd)   = α^(ttd / 60)
    referee_modifier(n)     = max(β_min, β^n)
"""

from __future__ import annotations
import random
from dataclasses import dataclass
from .Player import Player
from .Constants import BASE_CURVE_K, CARDIAC_DECAY, REFEREE_DECAY, REFEREE_FLOOR


@dataclass
class Referee:
    cprs_performed: int = 0

    def compute_survival_probability(self, player: Player, death_duration: float) -> float:
        """
        Compute the probability that `player` survives a death of `death_duration` seconds.

        Args:
            player: The player who is dying. Use player.ttd (BEFORE this death
                    is added) for cardiac modifier, and player.physicality for baseline.
            death_duration: How long the player will be dead THIS time (seconds).
                            This is the cylinder contents at time of injection.

        Returns:
            Float in [0.0, 1.0] — probability of successful revival.

        Implementation notes:
            - base_curve uses death_duration (THIS episode's danger)
            - cardiac uses player.ttd (accumulated PRIOR damage to the heart)
            - referee uses self.cprs_performed (global fatigue)
            - physicality is player.physicality (constant per player)
            - Multiply all four. Clamp to [0.0, 1.0].
            - At death_duration >= 300: always return 0.0 regardless of other factors.
        """
        def death_curve(t):
            return max(0, 1- (t/300)**BASE_CURVE_K)
        def cardiac_modifier(ttd):
            return CARDIAC_DECAY**(ttd/60)
        def referee_modifier(n):
            return max(REFEREE_FLOOR, REFEREE_DECAY**n)
        # def clamp(num, min_value, max_value):
        #     return max(min(num, max_value), min_value)
        death_pr = death_curve(death_duration) * cardiac_modifier(player.ttd) * referee_modifier(self.cprs_performed) * player.physicality

        return 0.0 if death_duration >= 300 else death_pr

    def attempt_revival(self, player: Player, death_duration: float, rng: random.Random | None = None) -> bool:
        """
        Attempt to revive a player. Rolls against survival probability.

        Args:
            player: The dying player. NOTE: player.ttd should reflect state
                    BEFORE this death (on_death hasn't been called yet, or if it has,
                    you need to subtract this death_duration back out for the cardiac calc).
            death_duration: Duration of this death episode.
            rng: Optional seeded RNG for deterministic testing. If None, use random.random().

        Returns:
            True if revival succeeds, False if player dies permanently.

        Side effects:
            - Increments self.cprs_performed (always, regardless of outcome).
            - On the Player side, the caller (Game) should handle on_death/on_revival/on_permanent_death.

        Design note:
            The referee doesn't mutate the player — it only reads player state and
            returns a bool. The Game object orchestrates the state transitions.
        """
        if rng is None:
            rng = random.Random()
            
        prob = self.compute_survival_probability(player, death_duration)
        roll = rng.random()
        survived = prob > roll
        self.cprs_performed += 1
        return survived