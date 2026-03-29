"""Opponent that uses a trained MaskablePPO model to choose actions.

This bridges RL inference back into the bot interface — a trained Hal
checkpoint can play as the opponent for Baku's training, and vice versa.
"""

from __future__ import annotations

from sb3_contrib import MaskablePPO

from src.Game import Game
from .base import Opponent
from ..awareness import (
    AwarenessConfig,
    LeapAwareness,
    build_action_mask,
    exposes_leap_features,
    initial_awareness_for_role,
    update_awareness,
)
from ..observation import build_observation


class SelectorOpponent(Opponent):
    """Wraps a ControllerSelector (or any object with .predict/.reset) as an opponent.

    This enables the promoted Baku selector to serve as a frozen opponent
    during Hal training or evaluation.
    """

    def __init__(self, selector, role: str, awareness_config: AwarenessConfig | None = None):
        self.selector = selector
        self.role = role.lower()
        self.awareness_config = awareness_config or AwarenessConfig()
        self.awareness = initial_awareness_for_role(role)
        self._processed_history_len = 0

    def _sync_awareness(self, game: Game) -> LeapAwareness:
        while self._processed_history_len < len(game.history):
            record = game.history[self._processed_history_len]
            self.awareness = update_awareness(
                self.awareness,
                controlled_role_name=self.role,
                config=self.awareness_config,
                game=game,
                record=record,
            )
            self._processed_history_len += 1
        return self.awareness

    def choose_action(self, game: Game, role: str, turn_duration: int) -> int:
        awareness = self._sync_awareness(game)
        am_hal = self.role.lower() == "hal"
        me = game.player1 if am_hal else game.player2
        opp = game.player2 if am_hal else game.player1

        obs = build_observation(game, me, opp, exposes_leap_features(awareness))
        mask = build_action_mask(
            role=role,
            is_leap_turn=game.is_leap_second_turn(),
            awareness=awareness,
        )
        action, _ = self.selector.predict(obs, action_masks=mask, deterministic=True)
        return int(action) + 1

    def reset(self) -> None:
        self.awareness = initial_awareness_for_role(self.role)
        self._processed_history_len = 0
        if hasattr(self.selector, "reset"):
            self.selector.reset()


class ModelOpponent(Opponent):
    """Wraps a saved MaskablePPO model as an opponent bot.

    Args:
        model_path: Path to the saved model (without .zip extension).
        role: "hal" or "baku" — which player this model was trained as.
    """

    def __init__(
        self,
        model_path: str,
        role: str,
        awareness_config: AwarenessConfig | None = None,
    ):
        self.model_path = model_path
        self.model = MaskablePPO.load(model_path)
        self.role = role.lower()
        self.awareness_config = awareness_config or AwarenessConfig()
        self.awareness = initial_awareness_for_role(role)
        self._processed_history_len = 0

    def _sync_awareness(self, game: Game) -> LeapAwareness:
        while self._processed_history_len < len(game.history):
            record = game.history[self._processed_history_len]
            self.awareness = update_awareness(
                self.awareness,
                controlled_role_name=self.role,
                config=self.awareness_config,
                game=game,
                record=record,
            )
            self._processed_history_len += 1

        return self.awareness

    def choose_action(self, game: Game, role: str, turn_duration: int) -> int:
        """
        Args:
            game: Current game state.
            role: "dropper" or "checker" (what the env assigned us this half).
            turn_duration: 60 or 61.

        Returns:
            Game second (int, 1-indexed).
        """
        awareness = self._sync_awareness(game)
        am_hal = self.role.lower() == "hal"
        if am_hal:
            me = game.player1
            opp = game.player2
        else:
            me = game.player2
            opp = game.player1

        obs = build_observation(game, me, opp, exposes_leap_features(awareness))

        mask = build_action_mask(
            role=role,
            is_leap_turn=game.is_leap_second_turn(),
            awareness=awareness,
        )
        
        action, _ = self.model.predict(obs, action_masks=mask, deterministic=True)

        return int(action) + 1
    

    def reset(self) -> None:
        self.awareness = initial_awareness_for_role(self.role)
        self._processed_history_len = 0
