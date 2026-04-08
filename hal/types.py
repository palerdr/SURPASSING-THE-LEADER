from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto


@dataclass(frozen=True)
class Bucket:
    lo: int
    hi: int
    label: str

class MemoryMode(Enum):
    NORMAL = auto()
    PRE_AMNESIA = auto()
    AMNESIA = auto()
    RECOVERED = auto()

#hal's belief state
@dataclass(frozen=True)
class BeliefState:
    baku_check_history: tuple[int, ...] = ()
    baku_drop_history: tuple[int, ...] = ()
    exploitation_mode: bool = False
    baku_predicted_bucket_probs: tuple[float, ...] | None = None
    baku_entropy: float = 2.0


#hal's agg state
@dataclass(frozen=True)
class HalState:
    memory: MemoryMode = MemoryMode.NORMAL
    belief: BeliefState = field(default_factory=BeliefState)
    leap_deduced: bool = False
