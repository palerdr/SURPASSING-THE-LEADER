"""Modular opening-head architecture for Baku.

Two architecture families:
  Family A: Frozen opening model (BC-only) + trainable late model (PPO)
  Family B: Trainable opening model (BC + opening PPO) + trainable late model (PPO)

Both use a two-model wrapper (ModularBakuSelector) that routes early turns
to the opening model and late turns to the late model, with bp routing to
the existing bp_specialist via the T2 classifier.
"""

from __future__ import annotations

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sb3_contrib import MaskablePPO

import gymnasium as gym
from gymnasium import spaces

from environment.dth_env import DTHEnv
from environment.observation import OBS_SIZE
from environment.opponents.factory import create_scripted_opponent
from training.behavior_clone import TraceSample, behavior_clone_policy
from training.expert_selector import build_classifier


class ModularBakuSelector:
    """Two-model modular Baku selector with separate opening and late parameters.

    Routing uses the observation's round_index and half_index to determine
    the actual game turn, ensuring correct behavior for non-opening scenarios
    (e.g., r9_pre_leap starts at round 8, not turn 0).

    - Classify bp vs hal at classify_turn (episode counter)
    - If bp: bp_specialist for all remaining turns
    - If hal (or pre-classification) and game_turn < opening_horizon: opening_model
    - Otherwise: late_model
    """

    def __init__(
        self,
        bp_specialist: MaskablePPO,
        opening_model: MaskablePPO,
        late_model: MaskablePPO,
        classifier: LogisticRegression,
        opening_horizon: int = 8,
        classify_turn: int = 2,
    ):
        self.bp_specialist = bp_specialist
        self.opening_model = opening_model
        self.late_model = late_model
        self.classifier = classifier
        self.opening_horizon = opening_horizon
        self.classify_turn = classify_turn
        self._turn = 0
        self._is_bp: bool | None = None

    @staticmethod
    def _game_turn_from_obs(obs: np.ndarray) -> int:
        """Extract the actual game turn from observation features.

        obs[8] = round_index / 10.0, obs[9] = half_index (0 or 1).
        game_turn = round_index * 2 + half_index.
        """
        obs_1d = obs.flatten() if obs.ndim > 1 else obs
        game_round = round(float(obs_1d[8]) * 10)
        game_half = round(float(obs_1d[9]))
        return game_round * 2 + game_half

    @property
    def policy(self):
        return self.late_model.policy

    def reset(self):
        self._turn = 0
        self._is_bp = None

    def predict(self, obs, *, action_masks=None, deterministic=True):
        if self._is_bp is None and self._turn >= self.classify_turn:
            obs_2d = obs.reshape(1, -1) if obs.ndim == 1 else obs
            label = int(self.classifier.predict(obs_2d)[0])
            self._is_bp = label == 0  # 0=bp, 1=hal

        self._turn += 1

        if self._is_bp:
            return self.bp_specialist.predict(
                obs, action_masks=action_masks, deterministic=deterministic
            )

        game_turn = self._game_turn_from_obs(obs)
        if game_turn < self.opening_horizon:
            return self.opening_model.predict(
                obs, action_masks=action_masks, deterministic=deterministic
            )

        return self.late_model.predict(
            obs, action_masks=action_masks, deterministic=deterministic
        )


class OpeningAutoPlayEnv(gym.Env):
    """Env wrapper that auto-plays opening turns with a frozen model.

    The RL agent (late model) only sees post-opening observations.
    If the game ends during the opening auto-play, the env re-rolls a new
    episode so the agent always starts from a post-opening state.
    """

    metadata = {"render_modes": []}

    def __init__(
        self, opening_model: MaskablePPO, opening_horizon: int, **env_kwargs
    ):
        super().__init__()
        self.opening_model = opening_model
        self.opening_horizon = opening_horizon
        self.env_kwargs = env_kwargs
        self.env = DTHEnv(**env_kwargs)
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space
        self._reroll_offset = 0

    def reset(self, *, seed=None, options=None):
        for attempt in range(200):
            if seed is not None:
                effective_seed = seed + (self._reroll_offset + attempt * 997 if attempt > 0 else 0)
            else:
                effective_seed = None

            obs, info = self.env.reset(seed=effective_seed, options=options)
            survived = True

            for _t in range(self.opening_horizon):
                mask = self.env.action_masks()
                action, _ = self.opening_model.predict(
                    obs, action_masks=mask, deterministic=True
                )
                obs, _r, term, trunc, info = self.env.step(int(action))
                if term or trunc:
                    survived = False
                    break

            if survived:
                self._reroll_offset += 1
                return obs, info

        raise RuntimeError(
            "Opening model dies every episode — cannot train late model"
        )

    def step(self, action):
        return self.env.step(action)

    def action_masks(self):
        return self.env.action_masks()

    @property
    def game(self):
        return self.env.game

    @property
    def agent(self):
        return self.env.agent

    @property
    def opp_player(self):
        return self.env.opp_player


class DenseRewardWrapper(gym.Wrapper):
    """Adds dense per-turn reward on top of existing sparse/shaped reward.

    The reward_fn receives (game, agent, opp_player) after each step
    and returns an additional float reward that is scaled and added.
    """

    def __init__(self, env, reward_fn, reward_scale: float = 1.0):
        super().__init__(env)
        self.reward_fn = reward_fn
        self.reward_scale = reward_scale

    def step(self, action):
        obs, reward, term, trunc, info = self.env.step(action)
        game = self._get_game()
        if game and game.history:
            agent = self._get_agent()
            opp = self._get_opp()
            if agent and opp:
                dense = self.reward_fn(game, agent, opp)
                reward += dense * self.reward_scale
        return obs, reward, term, trunc, info

    def _get_game(self):
        env = self.env
        while env is not None:
            if hasattr(env, "game"):
                return env.game
            env = getattr(env, "env", None)
        return None

    def _get_agent(self):
        env = self.env
        while env is not None:
            if hasattr(env, "agent"):
                return env.agent
            env = getattr(env, "env", None)
        return None

    def _get_opp(self):
        env = self.env
        while env is not None:
            if hasattr(env, "opp_player"):
                return env.opp_player
            env = getattr(env, "env", None)
        return None

    def action_masks(self):
        return self.env.action_masks()


class OpeningTruncatedEnv(gym.Env):
    """Env that terminates after opening_horizon turns.

    Used for PPO training of the opening model (Family B).
    Reward: +1 for surviving to opening_horizon, -1 for death during opening.
    """

    metadata = {"render_modes": []}

    def __init__(self, opening_horizon: int, **env_kwargs):
        super().__init__()
        self.opening_horizon = opening_horizon
        self.env = DTHEnv(**env_kwargs)
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space
        self._turn = 0

    def reset(self, *, seed=None, options=None):
        self._turn = 0
        return self.env.reset(seed=seed, options=options)

    def step(self, action):
        self._turn += 1
        obs, _reward, term, trunc, info = self.env.step(action)

        if term:
            return obs, -1.0, True, False, info

        if self._turn >= self.opening_horizon:
            return obs, 1.0, False, True, info

        return obs, 0.0, False, False, info

    def action_masks(self):
        return self.env.action_masks()


class RecurrentModularBakuSelector:
    """ModularBakuSelector with a recurrent (LSTM/GRU) late head.

    Manages the LSTM hidden state across turns. The hidden state is
    initialized when the late model first activates (after opening turns),
    and carried forward for subsequent late-model turns.
    """

    def __init__(
        self,
        bp_specialist,
        opening_model,
        late_model,  # RecurrentPPO
        classifier,
        opening_horizon: int = 8,
        classify_turn: int = 2,
    ):
        self.bp_specialist = bp_specialist
        self.opening_model = opening_model
        self.late_model = late_model
        self.classifier = classifier
        self.opening_horizon = opening_horizon
        self.classify_turn = classify_turn
        self._turn = 0
        self._is_bp = None
        self._lstm_state = None
        self._lstm_started = False

    @staticmethod
    def _game_turn_from_obs(obs):
        obs_1d = obs.flatten() if obs.ndim > 1 else obs
        return round(float(obs_1d[8]) * 10) * 2 + round(float(obs_1d[9]))

    @property
    def policy(self):
        return self.late_model.policy

    def reset(self):
        self._turn = 0
        self._is_bp = None
        self._lstm_state = None
        self._lstm_started = False

    def predict(self, obs, *, action_masks=None, deterministic=True):
        if self._is_bp is None and self._turn >= self.classify_turn:
            obs_2d = obs.reshape(1, -1) if obs.ndim == 1 else obs
            label = int(self.classifier.predict(obs_2d)[0])
            self._is_bp = label == 0

        self._turn += 1

        if self._is_bp:
            return self.bp_specialist.predict(
                obs, action_masks=action_masks, deterministic=deterministic
            )

        game_turn = self._game_turn_from_obs(obs)
        if game_turn < self.opening_horizon:
            return self.opening_model.predict(
                obs, action_masks=action_masks, deterministic=deterministic
            )

        # Recurrent late model
        episode_start = np.array([not self._lstm_started])
        self._lstm_started = True
        action, self._lstm_state = self.late_model.predict(
            obs, state=self._lstm_state, episode_start=episode_start,
            deterministic=deterministic,
        )
        # Enforce action legality (RecurrentPPO has no native masking)
        if action_masks is not None and not action_masks[int(action)]:
            action = np.array(min(int(action), 59))
        return action, None


class ActionClampEnv(gym.Wrapper):
    """Clamps illegal actions to the nearest legal action.

    Required for RecurrentPPO which does not support native action masking.
    In DTH, the only illegal action is action 60 (second 61) on non-leap turns.
    Clamping to action 59 (second 60) is the minimal correction.
    """

    def step(self, action):
        act = int(action)
        mask = self.env.action_masks()
        if not mask[act]:
            act = min(act, len(mask) - 2)  # clamp to second-to-last legal
        return self.env.step(act)

    def action_masks(self):
        return self.env.action_masks()


class LateCurriculumEnv(gym.Env):
    """Mixed env for late-head training with scenario augmentation.

    With probability p_opening: auto-play the opening with the frozen model,
    then hand off to the RL agent (like OpeningAutoPlayEnv).
    With probability 1-p_opening: start from a random mid-game scenario,
    so the agent gets full episodes of mid/late-game turns.

    This gives the late model both post-opening data AND rich mid-game data.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        opening_model: MaskablePPO,
        opening_horizon: int,
        p_opening: float = 0.5,
        late_scenarios: list[str] | None = None,
        **env_kwargs,
    ):
        super().__init__()
        self.opening_model = opening_model
        self.opening_horizon = opening_horizon
        self.p_opening = p_opening
        self.late_scenarios = late_scenarios or ["round7_pressure", "round8_bridge"]
        self.env = DTHEnv(**env_kwargs)
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space
        self._rng = np.random.default_rng(env_kwargs.get("seed"))
        self._reroll_offset = 0

    def reset(self, *, seed=None, options=None):
        from training.curriculum import get_scenario

        if self._rng.random() < self.p_opening:
            # Opening autoplay path
            for attempt in range(200):
                if seed is not None:
                    eff = seed + (self._reroll_offset + attempt * 997 if attempt > 0 else 0)
                else:
                    eff = None
                obs, info = self.env.reset(seed=eff, options=None)
                survived = True
                for _ in range(self.opening_horizon):
                    mask = self.env.action_masks()
                    action, _ = self.opening_model.predict(
                        obs, action_masks=mask, deterministic=True
                    )
                    obs, _, term, trunc, info = self.env.step(int(action))
                    if term or trunc:
                        survived = False
                        break
                if survived:
                    self._reroll_offset += 1
                    return obs, info
            raise RuntimeError("Opening dies every episode")
        else:
            # Late scenario path
            scenario_name = str(self._rng.choice(self.late_scenarios))
            scenario = get_scenario(scenario_name)
            return self.env.reset(seed=seed, options={"scenario": scenario})

    def step(self, action):
        return self.env.step(action)

    def action_masks(self):
        return self.env.action_masks()

    @property
    def game(self):
        return self.env.game

    @property
    def agent(self):
        return self.env.agent

    @property
    def opp_player(self):
        return self.env.opp_player


def collect_opening_samples(
    base_model_path: str,
    opponent_names: list[str],
    action_overrides: dict[int, int],
    opening_horizon: int,
    seeds: range = range(200),
) -> list[TraceSample]:
    """Collect BC training samples from the patched opening path.

    Plays games with the base model, applying action_overrides at specified
    turns. Records (obs, mask, action) for all turns in [0, opening_horizon).
    """
    model = MaskablePPO.load(base_model_path)
    samples: list[TraceSample] = []

    for opp_name in opponent_names:
        for seed in seeds:
            env = DTHEnv(
                opponent=create_scripted_opponent(opp_name),
                agent_role="baku",
                seed=seed,
            )
            obs, _ = env.reset()

            for turn in range(opening_horizon):
                mask = env.action_masks()

                if turn in action_overrides:
                    action_idx = action_overrides[turn] - 1  # action_second to 0-indexed
                else:
                    action_idx = int(
                        model.predict(obs, action_masks=mask, deterministic=True)[0]
                    )

                samples.append(
                    TraceSample(
                        observation=obs.copy().astype(np.float32),
                        action_mask=mask.copy(),
                        action_index=action_idx,
                    )
                )

                obs, _, term, trunc, _ = env.step(action_idx)
                if term or trunc:
                    break

    return samples


def verify_opening_accuracy(
    model: MaskablePPO, samples: list[TraceSample]
) -> dict[str, float]:
    """Check deterministic prediction accuracy of the opening model."""
    correct = 0
    total = len(samples)
    with torch.no_grad():
        for s in samples:
            obs_t, _ = model.policy.obs_to_tensor(s.observation.reshape(1, -1))
            mask_t = torch.as_tensor(
                s.action_mask.reshape(1, -1), device=model.policy.device
            )
            dist = model.policy.get_distribution(obs_t, action_masks=mask_t)
            pred = dist.distribution.probs.argmax(dim=1).item()
            if pred == s.action_index:
                correct += 1
    return {"correct": correct, "total": total, "accuracy": correct / total if total else 0}


def train_opening_model_bc(
    base_model_path: str,
    opponent_names: list[str],
    action_overrides: dict[int, int],
    opening_horizon: int,
    epochs: int = 80,
    lr: float = 1e-3,
    seeds: range = range(200),
    seed: int = 42,
) -> tuple[MaskablePPO, list[TraceSample]]:
    """Train a MaskablePPO via BC on the patched opening sequence.

    Returns (trained_model, samples) so caller can verify accuracy.
    """
    samples = collect_opening_samples(
        base_model_path, opponent_names, action_overrides, opening_horizon, seeds
    )
    print(
        f"Opening BC: {len(samples)} samples, "
        f"{len(seeds)} seeds × {len(opponent_names)} opponents, "
        f"horizon={opening_horizon}"
    )

    env = DTHEnv(
        opponent=create_scripted_opponent(opponent_names[0]),
        agent_role="baku",
        seed=0,
    )
    opening_model = MaskablePPO("MlpPolicy", env, verbose=0, seed=seed)

    behavior_clone_policy(
        opening_model,
        samples,
        epochs=epochs,
        batch_size=64,
        learning_rate=lr,
    )

    stats = verify_opening_accuracy(opening_model, samples)
    print(
        f"Opening BC accuracy: {stats['correct']}/{stats['total']} "
        f"({stats['accuracy']:.1%})"
    )

    return opening_model, samples


def build_modular_selector(
    bp_specialist_path: str,
    base_model_path: str,
    opening_model: MaskablePPO,
    late_model: MaskablePPO,
    opening_horizon: int = 8,
    classify_turn: int = 2,
) -> ModularBakuSelector:
    """Assemble the full modular selector from components."""
    bp_specialist = MaskablePPO.load(bp_specialist_path)
    clf = build_classifier(
        base_model_path, ["bridge_pressure", "hal_death_trade"], classify_turn
    )
    return ModularBakuSelector(
        bp_specialist=bp_specialist,
        opening_model=opening_model,
        late_model=late_model,
        classifier=clf,
        opening_horizon=opening_horizon,
        classify_turn=classify_turn,
    )
