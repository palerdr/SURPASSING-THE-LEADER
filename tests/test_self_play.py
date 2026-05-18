import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from environment.legal_actions import IllegalActionError
from environment.opponents.base import Opponent
from hal.self_play import play_one_game


class _IllegalHal:
    def reset(self):
        pass

    def choose_action(self, game, role, turn_duration):
        del game, role, turn_duration
        return 61


class _AlwaysOneOpponent(Opponent):
    def reset(self):
        pass

    def choose_action(self, game, role, turn_duration):
        del game, role, turn_duration
        return 1


def test_self_play_rejects_illegal_action():
    with pytest.raises(IllegalActionError):
        play_one_game(_IllegalHal(), _AlwaysOneOpponent(), seed=0)
