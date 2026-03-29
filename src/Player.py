from __future__ import annotations
from dataclasses import dataclass, field
from .Constants import CYLINDER_MAX, TURN_DURATION_NORMAL

@dataclass
class Player:
    name: str
    physicality: float = 1.0

    # ── Mutable game state ──
    cylinder: float = 0.0
    ttd: float = 0.0
    deaths: int = 0
    alive: bool = True

    # ── Per-death history (for analysis / debugging) ──
    # Each entry is the duration of one death episode in seconds.
    death_history: list[float] = field(default_factory=list)

    @property
    def safe_strategies_remaining(self) -> int:
        """
        How many consecutive safe-strategy checks can this player do
        without the cylinder hitting CYLINDER_MAX?

        Safe if: cylinder + 60 < CYLINDER_MAX (strictly less — 300 triggers injection).

        Returns the number of times the player can absorb a worst-case safe check.
        """
        return max(0, int((CYLINDER_MAX - 1 - self.cylinder) // TURN_DURATION_NORMAL))
    

    def add_to_cylinder(self, amount: float) -> bool:
        """
        Add NDD to the cylinder.

        Returns True if cylinder >= CYLINDER_MAX after addition,
        meaning immediate injection is triggered.
        """
        self.cylinder += amount
        return self.cylinder >= CYLINDER_MAX

    def on_death(self, death_duration: float) -> None:
        """
        Called when this player dies.
        - Increment deaths count
        - Add death_duration to ttd
        - Append to death_history
        - Set alive = False
        """
        self.deaths += 1
        self.ttd += death_duration
        self.death_history.append(death_duration)
        self.alive = False

    def on_revival(self) -> None:
        """
        Called when revival succeeds.
        - Reset cylinder to 0
        - Set alive = True
        """
        self.cylinder = 0
        self.alive = True

    def on_permanent_death(self) -> None:
        """
        Called when revival fails. Player is dead for good.
        - alive stays False
        - This is a terminal state for the player.
        """
        self.alive = False
        
