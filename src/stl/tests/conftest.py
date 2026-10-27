"""Shared fixtures for Drop The Handkerchief tests."""

import pytest

from stl.engine.game import Player
from stl.engine.game import Referee
from stl.engine.game import Game
from stl.engine.game import PHYSICALITY_HAL, PHYSICALITY_BAKU


@pytest.fixture
def hal():
    return Player(name="Hal", physicality=PHYSICALITY_HAL)

@pytest.fixture
def baku():
    return Player(name="Baku", physicality=PHYSICALITY_BAKU)

@pytest.fixture
def referee():
    return Referee()

@pytest.fixture
def game(hal, baku, referee):
    """Game with Hal dropping first, seeded for determinism."""
    g = Game(player1=hal, player2=baku, referee=referee, first_dropper=hal)
    g.seed(42)
    return g

