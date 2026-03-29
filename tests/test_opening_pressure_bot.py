import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from environment.opponents.safe_bot import BridgePressureBot
from src.Constants import TURN_DURATION_LEAP, TURN_DURATION_NORMAL


def test_bridge_pressure_bot_uses_second_two_in_opening():
    bot = BridgePressureBot()

    assert bot.choose_action(game=None, role="dropper", turn_duration=TURN_DURATION_NORMAL) == 2
    assert bot.choose_action(game=None, role="checker", turn_duration=TURN_DURATION_NORMAL) == 2


def test_bridge_pressure_bot_keeps_leap_dropper_sixty_one():
    bot = BridgePressureBot()

    assert bot.choose_action(game=None, role="dropper", turn_duration=TURN_DURATION_LEAP) == 61
