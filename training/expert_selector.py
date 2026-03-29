"""Minimal expert-selector: routes to the right specialist based on opponent classification."""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sb3_contrib import MaskablePPO

from environment.dth_env import DTHEnv
from environment.opponents.factory import create_scripted_opponent


class ExpertSelector:
    """Wraps multiple frozen specialists with a learned classifier.

    Usage mirrors MaskablePPO.predict() so it can drop into evaluate.py.
    For turns < classify_turn, uses the default model.
    At classify_turn, classifies the opponent and locks in the specialist.

    Per-expert action overrides can be specified via ``action_overrides``:
    a dict mapping label -> {turn_index: action_second}. When the active
    expert has an override for the current turn, that action is returned
    instead of querying the model.
    """

    def __init__(
        self,
        specialists: dict[int, MaskablePPO],  # label -> model
        default_label: int,
        classifier: LogisticRegression,
        classify_turn: int = 2,
        action_overrides: dict[int, dict[int, int]] | None = None,
    ):
        self.specialists = specialists
        self.default_label = default_label
        self.classifier = classifier
        self.classify_turn = classify_turn
        self.action_overrides = action_overrides or {}
        self._turn = 0
        self._locked_label: int | None = None

    @property
    def policy(self):
        """Expose policy for obs_to_tensor compatibility."""
        return self.specialists[self.default_label].policy

    def reset(self):
        self._turn = 0
        self._locked_label = None

    def predict(self, obs, *, action_masks=None, deterministic=True):
        if self._locked_label is not None:
            label = self._locked_label
        elif self._turn >= self.classify_turn:
            obs_2d = obs.reshape(1, -1) if obs.ndim == 1 else obs
            label = int(self.classifier.predict(obs_2d)[0])
            self._locked_label = label
        else:
            label = self.default_label

        turn = self._turn
        self._turn += 1

        overrides = self.action_overrides.get(label, {})
        if turn in overrides:
            action = np.array(overrides[turn] - 1)  # action_second -> 0-indexed
            return action, None

        model = self.specialists[label]
        return model.predict(obs, action_masks=action_masks, deterministic=deterministic)


def build_classifier(
    model_path: str,
    opponents: list[str],
    classify_turn: int = 2,
    games_per_opponent: int = 100,
) -> LogisticRegression:
    """Train a logistic regression to classify opponent type from observations."""
    model = MaskablePPO.load(model_path)
    obs_list, labels = [], []

    for opp_idx, opp_name in enumerate(opponents):
        for seed in range(games_per_opponent):
            opp = create_scripted_opponent(opp_name)
            env = DTHEnv(opponent=opp, agent_role="baku", seed=seed)
            obs, _ = env.reset()
            alive = True
            for t in range(classify_turn + 1):
                mask = env.action_masks()
                if t == classify_turn:
                    obs_list.append(obs.copy())
                    labels.append(opp_idx)
                    break
                action, _ = model.predict(obs, action_masks=mask, deterministic=True)
                obs, _, term, trunc, _ = env.step(int(action))
                if term or trunc:
                    alive = False
                    break
            if not alive:
                continue

    X, y = np.array(obs_list), np.array(labels)
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X, y)
    return clf


class OpeningController:
    """Learned early-opening controller that handles a bounded turn prefix.

    Active from ``start_turn`` to ``end_turn`` (inclusive, 0-indexed).
    Outside that range, returns None and the caller falls through to the
    base model.  The controller is trained on (observation, action) pairs
    from the hal-family patched path.
    """

    def __init__(self, obs_action_map: dict[tuple, int], start_turn: int, end_turn: int):
        self.obs_action_map = obs_action_map  # rounded-obs-tuple -> action_second
        self.start_turn = start_turn
        self.end_turn = end_turn

    def query(self, obs: np.ndarray, turn: int) -> int | None:
        """Return action_second if this turn is in scope, else None."""
        if turn < self.start_turn or turn > self.end_turn:
            return None
        key = tuple(obs.round(6))
        return self.obs_action_map.get(key)


class FeatureRuleController:
    """Robust opening controller using discrete feature matching.

    Uses (role, round_bucket, half) to identify turns instead of exact
    float observation vectors.  Survives base-model weight drift because
    the routing features (role/round/half) are derived from game structure,
    not from learned policy outputs.
    """

    def __init__(
        self,
        feature_action_map: dict[tuple[int, int, int], int],
        start_turn: int,
        end_turn: int,
    ):
        self.feature_action_map = feature_action_map  # (role, round_bucket, half) -> action_second
        self.start_turn = start_turn
        self.end_turn = end_turn

    @staticmethod
    def _extract_key(obs: np.ndarray) -> tuple[int, int, int]:
        role = round(float(obs[7]))
        round_bucket = round(float(obs[8]) * 10)
        half = round(float(obs[9]))
        return (role, round_bucket, half)

    def query(self, obs: np.ndarray, turn: int) -> int | None:
        if turn < self.start_turn or turn > self.end_turn:
            return None
        key = self._extract_key(obs)
        return self.feature_action_map.get(key)


class MLPOpeningController:
    """Small MLP opening controller that generalizes across nearby observations.

    A 2-layer neural network maps observation -> action distribution.
    Robust to small observation perturbations because the mapping is smooth.
    """

    def __init__(self, weights: list[np.ndarray], biases: list[np.ndarray],
                 start_turn: int, end_turn: int, n_actions: int = 61):
        self.weights = weights
        self.biases = biases
        self.start_turn = start_turn
        self.end_turn = end_turn
        self.n_actions = n_actions

    def query(self, obs: np.ndarray, turn: int) -> int | None:
        if turn < self.start_turn or turn > self.end_turn:
            return None
        x = obs.astype(np.float32)
        for w, b in zip(self.weights[:-1], self.biases[:-1]):
            x = np.maximum(0, x @ w + b)  # ReLU
        logits = x @ self.weights[-1] + self.biases[-1]
        return int(np.argmax(logits[:self.n_actions])) + 1


def build_feature_rule_controller(
    base_model_path: str,
    opponent_names: list[str],
    action_overrides: dict[int, int],
    controller_turns: tuple[int, int],
) -> FeatureRuleController:
    """Build a feature-rule controller from patched-path data."""
    model = MaskablePPO.load(base_model_path)
    feature_map: dict[tuple[int, int, int], int] = {}

    for opp_name in opponent_names:
        env = DTHEnv(opponent=create_scripted_opponent(opp_name), agent_role="baku", seed=42)
        obs, _ = env.reset()
        for turn in range(controller_turns[1] + 1):
            mask = env.action_masks()
            if turn in action_overrides:
                action_second = action_overrides[turn]
            else:
                action_second = int(model.predict(obs, action_masks=mask, deterministic=True)[0]) + 1

            if controller_turns[0] <= turn <= controller_turns[1]:
                key = FeatureRuleController._extract_key(obs)
                feature_map[key] = action_second

            obs, _, term, trunc, _ = env.step(action_second - 1)
            if term or trunc:
                break

    return FeatureRuleController(feature_map, controller_turns[0], controller_turns[1])


def build_mlp_opening_controller(
    base_model_path: str,
    opponent_names: list[str],
    action_overrides: dict[int, int],
    controller_turns: tuple[int, int],
    hidden_size: int = 32,
    epochs: int = 200,
    lr: float = 1e-3,
) -> MLPOpeningController:
    """Train a small MLP controller on patched-path observations."""
    import torch
    import torch.nn as nn

    model = MaskablePPO.load(base_model_path)
    obs_list, action_list = [], []

    for opp_name in opponent_names:
        for seed in range(200):
            env = DTHEnv(opponent=create_scripted_opponent(opp_name), agent_role="baku", seed=seed)
            obs, _ = env.reset()
            for turn in range(controller_turns[1] + 1):
                mask = env.action_masks()
                if turn in action_overrides:
                    action_second = action_overrides[turn]
                else:
                    action_second = int(model.predict(obs, action_masks=mask, deterministic=True)[0]) + 1

                if controller_turns[0] <= turn <= controller_turns[1]:
                    obs_list.append(obs.copy())
                    action_list.append(action_second - 1)

                obs, _, term, trunc, _ = env.step(action_second - 1)
                if term or trunc:
                    break

    X = torch.tensor(np.array(obs_list), dtype=torch.float32)
    y = torch.tensor(action_list, dtype=torch.long)

    net = nn.Sequential(
        nn.Linear(X.shape[1], hidden_size), nn.ReLU(),
        nn.Linear(hidden_size, 61),
    )
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    for _ in range(epochs):
        logits = net(X)
        loss = loss_fn(logits, y)
        opt.zero_grad()
        loss.backward()
        opt.step()

    weights, biases = [], []
    for layer in net:
        if isinstance(layer, nn.Linear):
            weights.append(layer.weight.detach().numpy().T)
            biases.append(layer.bias.detach().numpy())

    return MLPOpeningController(weights, biases, controller_turns[0], controller_turns[1])


def build_opening_controller(
    base_model_path: str,
    opponent_names: list[str],
    action_overrides: dict[int, int],
    controller_turns: tuple[int, int],
    seeds: range = range(200),
) -> OpeningController:
    """Build an opening controller by recording the patched-path observations.

    Plays games with ``action_overrides`` applied, records every (obs, action)
    at turns within ``controller_turns``, then builds a lookup controller.
    For turns without an override, the base-model action is used.
    """
    model = MaskablePPO.load(base_model_path)
    obs_action_map: dict[tuple, int] = {}

    for opp_name in opponent_names:
        for seed in seeds:
            env = DTHEnv(opponent=create_scripted_opponent(opp_name), agent_role="baku", seed=seed)
            obs, _ = env.reset()
            for turn in range(controller_turns[1] + 1):
                mask = env.action_masks()
                if turn in action_overrides:
                    action_second = action_overrides[turn]
                else:
                    action_second = int(model.predict(obs, action_masks=mask, deterministic=True)[0]) + 1

                if controller_turns[0] <= turn <= controller_turns[1]:
                    key = tuple(obs.round(6))
                    obs_action_map[key] = action_second

                obs, _, term, trunc, _ = env.step(action_second - 1)
                if term or trunc:
                    break

    return OpeningController(obs_action_map, controller_turns[0], controller_turns[1])


class ControllerSelector:
    """Selector that uses an opening controller for early hal-family turns,
    then hands back to the base model.  Zero hard-coded action overrides.

    Flow:
    - T0 to classify_turn-1: default model
    - At classify_turn: classify bp vs hal
    - If bp: lock in bp_specialist for all remaining turns
    - If hal and turn in [controller.start, controller.end]: use controller
    - If hal and turn > controller.end: fall through to base model
    """

    def __init__(
        self,
        bp_specialist: MaskablePPO,
        base_model: MaskablePPO,
        classifier: LogisticRegression,
        hal_controller: OpeningController,
        classify_turn: int = 2,
    ):
        self.bp_specialist = bp_specialist
        self.base_model = base_model
        self.classifier = classifier
        self.hal_controller = hal_controller
        self.classify_turn = classify_turn
        self._turn = 0
        self._is_bp: bool | None = None

    @property
    def policy(self):
        return self.base_model.policy

    def reset(self):
        self._turn = 0
        self._is_bp = None

    def predict(self, obs, *, action_masks=None, deterministic=True):
        # Classify at classify_turn
        if self._is_bp is None and self._turn >= self.classify_turn:
            obs_2d = obs.reshape(1, -1) if obs.ndim == 1 else obs
            label = int(self.classifier.predict(obs_2d)[0])
            self._is_bp = label == 0  # 0 = bp, 1 = hal

        turn = self._turn
        self._turn += 1

        # Route bp to specialist
        if self._is_bp:
            return self.bp_specialist.predict(obs, action_masks=action_masks, deterministic=deterministic)

        # Hal family: try controller first, fall through to base model
        if self._is_bp is not None:  # classified as hal
            ctrl_action = self.hal_controller.query(obs, turn)
            if ctrl_action is not None:
                return np.array(ctrl_action - 1), None

        # Default: base model
        return self.base_model.predict(obs, action_masks=action_masks, deterministic=deterministic)


def build_controller_selector(
    bp_specialist_path: str,
    base_model_path: str,
    hal_overrides: dict[int, int],
    controller_turns: tuple[int, int] = (2, 4),
    classify_turn: int = 2,
    controller_type: str = "lookup",
) -> ControllerSelector:
    """Build the complete controller-based selector.

    controller_type: "lookup" (exact obs), "feature_rule" (robust), or "mlp" (learned).
    """
    bp_specialist = MaskablePPO.load(bp_specialist_path)
    base_model = MaskablePPO.load(base_model_path)
    clf = build_classifier(base_model_path, ["bridge_pressure", "hal_death_trade"], classify_turn)

    if controller_type == "lookup":
        hal_controller = build_opening_controller(
            base_model_path, ["hal_death_trade", "hal_pressure"],
            hal_overrides, controller_turns,
        )
    elif controller_type == "feature_rule":
        hal_controller = build_feature_rule_controller(
            base_model_path, ["hal_death_trade", "hal_pressure"],
            hal_overrides, controller_turns,
        )
    elif controller_type == "mlp":
        hal_controller = build_mlp_opening_controller(
            base_model_path, ["hal_death_trade", "hal_pressure"],
            hal_overrides, controller_turns,
        )
    else:
        raise ValueError(f"Unknown controller_type: {controller_type}")

    return ControllerSelector(bp_specialist, base_model, clf, hal_controller, classify_turn)


def build_expert_selector(
    specialist_paths: dict[str, str],
    default_opponent: str,
    base_model_path: str,
    classify_turn: int = 2,
) -> ExpertSelector:
    """Build a complete expert selector from checkpoint paths.

    specialist_paths: {opponent_name: checkpoint_path}
    default_opponent: which specialist to use before classification
    """
    opponents = list(specialist_paths.keys())
    default_label = opponents.index(default_opponent)

    specialists = {}
    for idx, opp_name in enumerate(opponents):
        specialists[idx] = MaskablePPO.load(specialist_paths[opp_name])

    clf = build_classifier(base_model_path, opponents, classify_turn)
    return ExpertSelector(specialists, default_label, clf, classify_turn)
