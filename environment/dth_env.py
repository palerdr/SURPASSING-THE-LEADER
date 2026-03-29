"""Drop The Handkerchief — Gymnasium Environment.

This is the interface between the RL agent and the game engine.
SB3's PPO calls env.reset() and env.step(action) in a loop.

One step = one half-round. The agent always controls one player (Hal or Baku),
and the opponent is controlled by a bot. Roles alternate each half-round.

Action space: Discrete(61)
    action 0  → second 1
    action 59 → second 60
    action 60 → second 61 (only legal for dropper during leap turn)

Observation space: Box(0, 1, shape=(20,)) — see observation.py

The environment uses SB3's action masking via the action_masks() method.
MaskablePPO will call this each step to know which actions are legal.
"""

from __future__ import annotations

from collections.abc import Callable

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from src.Game import Game
from src.Player import Player
from src.Referee import Referee
from src.Constants import (
    PHYSICALITY_HAL,
    PHYSICALITY_BAKU,
)

from .awareness import (
    AwarenessConfig,
    LeapAwareness,
    build_action_mask,
    exposes_leap_features,
    initial_awareness_for_role,
    update_awareness,
)
from .observation import OBS_SIZE, build_observation
from .reward import compute_route_shaping_bonus, shaped_reward, sparse_reward
from .route_stages import current_route_stage_flags
from .opponents.base import Opponent
from .scenarios import (
    apply_scenario,
    scenario_from_options,
    validate_named_scenario_semantics,
    validate_scenario_reachability,
)


class DTHEnv(gym.Env):
    """Gymnasium environment for Drop The Handkerchief."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        opponent: Opponent,
        agent_role: str = "hal",
        seed: int | None = None,
        use_shaping: bool = False,
        shaping_preset: str = "light",
        awareness_config: AwarenessConfig | None = None,
        scenario_sampler: Callable[[np.random.Generator], dict | None] | None = None,
        max_steps: int | None = None,
    ):
        super().__init__()

        self.opponent = opponent
        self.agent_role = agent_role
        self._seed = seed
        self.use_shaping = use_shaping
        self.shaping_preset = shaping_preset
        self.awareness_config = awareness_config or AwarenessConfig()
        self.scenario_sampler = scenario_sampler
        self.max_steps = max_steps

        self.action_space = spaces.Discrete(61)
        self.observation_space = spaces.Box(
            low = 0.0, high = 1.0, shape = (OBS_SIZE,), dtype = np.float32
        )

        self.game = None
        self.agent = None
        self.opp_player = None
        self.awareness = initial_awareness_for_role(agent_role)
        self.episode_steps = 0
        self.current_scenario_name = "opening"
        self._awarded_route_stages: set[str] = set()



    def reset(self, *, seed=None, options=None):
        super().reset(seed = seed)
        game_seed = seed if seed is not None else self._seed
        if options is None and self.scenario_sampler is not None:
            sampled = self.scenario_sampler(self.np_random)
            options = None if sampled is None else {"scenario": sampled}

        scenario = scenario_from_options(options)

        HAL = Player("Hal", physicality = PHYSICALITY_HAL)
        BAKU = Player("Baku", physicality = PHYSICALITY_BAKU) 
        YAKOU = Referee()
        
        first_dropper = HAL
        self.game = Game(player1 = HAL, player2 =  BAKU, referee = YAKOU, first_dropper = first_dropper)

        if game_seed is not None:
            self.game.seed(game_seed)

        if self.agent_role.lower() == "hal":
            self.agent = HAL
            self.opp_player = BAKU
        else:
            self.agent = BAKU
            self.opp_player = HAL

        awareness_override = None
        if scenario is not None:
            validate_scenario_reachability(self.game, scenario)
            apply_scenario(self.game, HAL, BAKU, scenario)
            validate_named_scenario_semantics(self.game, scenario)
            awareness_override = scenario.awareness
            self.current_scenario_name = scenario.name
        else:
            self.current_scenario_name = "opening"
        
        self.awareness = initial_awareness_for_role(self.agent_role)
        if awareness_override is not None:
            self.awareness = awareness_override
        self.episode_steps = 0
        self._awarded_route_stages = {
            stage_name
            for stage_name, active in current_route_stage_flags(self.game).items()
            if active
        }

        self.opponent.reset()

        init_obs = build_observation(
            self.game,
            self.agent,
            self.opp_player,
            exposes_leap_features(self.awareness),
        )
        info_dict = {
            "game_clock": self.game.game_clock,
            "awareness": self.awareness.value,
            "scenario_name": self.current_scenario_name,
            "episode_steps": self.episode_steps,
        }
        return init_obs, info_dict







    def action_masks(self) -> np.ndarray:
        """
        Returns:
            np.ndarray of shape (61,), dtype bool.
        """
        assert self.game is not None
        assert self.agent is not None

        dropper, checker = self.game.get_roles_for_half(self.game.current_half)
        role = "dropper" if self.agent is dropper else "checker"
        return build_action_mask(
            role=role,
            is_leap_turn=self.game.is_leap_second_turn(),
            awareness=self.awareness,
        )
        

    def step(self, action: int):
        """

        Args:
            action: Integer in [0, 60]. Maps to game second (action + 1).

        Returns:
            Tuple of (obs, reward, terminated, truncated, info).
        """
        assert self.game is not None
        assert self.agent is not None
        assert self.opp_player is not None
        self.episode_steps += 1
        
        agent_second = action + 1

        D, C = self.game.get_roles_for_half(self.game.current_half)
        turn_duration = self.game.get_turn_duration()

        agent_is_dropper = (self.agent is D)

        opp_role = "checker" if agent_is_dropper else "dropper"

        opp_second = self.opponent.choose_action(self.game, opp_role, turn_duration)

        if agent_is_dropper:
            drop_time = agent_second
            check_time = opp_second
        else:
            drop_time = opp_second
            check_time = agent_second

        record = self.game.play_half_round(drop_time, check_time)
        self.awareness = update_awareness(
            self.awareness,
            controlled_role_name=self.agent_role,
            config=self.awareness_config,
            game=self.game,
            record=record,
        )
        
        obs = build_observation(
            self.game,
            self.agent,
            self.opp_player,
            exposes_leap_features(self.awareness),
        )
        terminated = self.game.game_over
        agent_won = self.game.winner is self.agent

        if self.use_shaping:
            route_bonus, self._awarded_route_stages = compute_route_shaping_bonus(
                self.game,
                self._awarded_route_stages,
                preset=self.shaping_preset,
            )
            reward = shaped_reward(terminated, agent_won, route_bonus)
        else:
            reward = sparse_reward(terminated, agent_won)
        truncated = False
        truncated_reason = None
        if self.max_steps is not None and self.episode_steps >= self.max_steps and not terminated:
            truncated = True
            reward = 0.0
            truncated_reason = "max_steps"
        info = {
            "game_clock": self.game.game_clock,
            "result": record.result,
            "awareness": self.awareness.value,
            "scenario_name": self.current_scenario_name,
            "episode_steps": self.episode_steps,
        }
        if truncated_reason is not None:
            info["truncated_reason"] = truncated_reason

        return (obs, reward, terminated, truncated, info)
