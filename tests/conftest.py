"""Shared fixtures for Drop The Handkerchief tests."""

import sys
import os
import pytest

# Add project root so `src` package resolves
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.Player import Player
from src.Referee import Referee
from src.Game import Game
from src.Constants import PHYSICALITY_HAL, PHYSICALITY_BAKU


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

@pytest.fixture
def game_at_r1(game):
    """Game with clock set to 720 (8:12 AM) — manga R1 start."""
    game.game_clock = 720.0
    return game
