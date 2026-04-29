"""When we reach a leaf node we need a value for that position
[-1, 1] from Hal's perspective, that flows back up the tree"""
from typing import Protocol
from .utility import terminal_value
from src.Game import Game

class LeafEvaluator(Protocol):
          def __call__(self, game: Game) -> float: ...

class TerminalOnlyEvaluator:
    """Returns terminal_value or 0.0 for unresolved."""
    def __init__(self, perspective_name: str = "Hal") -> None:
          self.perspective = perspective_name
    
    def __call__(self, game: Game) -> float:
          tval = terminal_value(game, perspective_name=self.perspective)
          if tval is not None:
                return tval
          else:
                return 0.0
    

class TablebaseEvaluator:
    """Wraps another evaluator; checks tablebase first."""

class ValueNetEvaluator:
    """Wraps the trained value net behind a clean interface."""