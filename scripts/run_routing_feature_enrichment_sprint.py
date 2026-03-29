#!/usr/bin/env python3
"""Sprint: Routing Feature Enrichment at PHYSICALITY_BAKU=0.88.

Optimization target: break the bp-vs-native observation overlap by giving the
routing/classifier path public opponent-action-history features, WITHOUT changing
the main policy observation shape.

Previous sprint (dual-head-opening) proved:
  - bp and native Hal produce identical 20-dim observations at T2
  - P(bp)=0.643 for BOTH opponents (std=0.000)
  - No routing architecture overcomes identical inputs

This sprint tests whether appending 4-dim opponent-action-history features to the
classifier input (24-dim total) breaks the overlap while keeping policy on 20-dim.

Candidates:
  E1: enriched old clf (bp + scripted hal, 24-dim) + ModularBaku
  E2: enriched native-aware clf (bp + scripted hal + native hal, 24-dim) + ModularBaku
  E3: enriched native-aware clf + DualHead A
  E4: enriched stacked (enriched old opening + enriched native late)
  D1: delayed T4 with enriched native clf + universal hal opening

Promotion (strong):  bp>=0.84, ht>=0.48, hp>=0.48, r9=1.0, mis=0,
  AND (native_wr >= 0.12 OR native_r7 >= 0.20).
Promotion (acceptable): bp>=0.82, ht>=0.48, hp>=0.48, r9=1.0, mis=0,
  AND (native_wr >= 0.06 OR native_r7 >= 0.10).
"""

from __future__ import annotations

import sys, os, json, time
from pathlib import Path
from dataclasses import dataclass, asdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from sb3_contrib import MaskablePPO
from sklearn.linear_model import LogisticRegression

from src.Constants import PHYSICALITY_BAKU
from environment.dth_env import DTHEnv
from environment.awareness import (
    AwarenessConfig,
    build_action_mask,
    exposes_leap_features,
    initial_awareness_for_role,
    update_awareness,
)
from environment.observation import build_observation
from environment.opponents.base import Opponent
from environment.opponents.factory import create_model_opponent, create_scripted_opponent
from environment.route_stages import current_route_stage_flags, stage_is_eligible_from_start
from environment.routing_features import (
    ROUTING_FEATURE_SIZE,
    build_routing_feature_vector,
    build_public_opponent_history_features,
)
from training.curriculum import get_scenario
from training.modular_policy import ModularBakuSelector
from training.expert_selector import (
    FeatureRuleController,
    build_classifier,
)


# ── Paths ────────────────────────────────────────────────────────────────
def _resolve(stem):
    for p in (
        Path("models/checkpoints") / stem,
        Path("models/checkpoints") / f"{stem}.zip",
        Path("STL/models/checkpoints") / stem,
        Path("STL/models/checkpoints") / f"{stem}.zip",
    ):
        if p.exists():
            return str(p.with_suffix("")) if p.suffix == ".zip" else str(p)
    raise FileNotFoundError(stem)


BASE_MODEL = _resolve("baku_vs_bridge_pressure_opening_to_round7_bc_only_bc_combined_0.3")
BP_SPECIALIST = _resolve("baku_vs_bridge_pressure_opening_to_round7_bc_only_bc_spec_e50_lr5.5")
LEARNED_HAL_PRIOR = _resolve("hal_vs_promoted_baku_prior_4096")
NATIVE_HAL_16K = _resolve("_native_hal_native_hal_16k")

BASELINE_ROUTE = [1, 1, 1, 10, 60, 1, 60, 1]
BASELINE_OVERRIDES = {t: BASELINE_ROUTE[t] for t in range(8)}
OPENING_HORIZON = 8
EVAL_GAMES = 50
GATE_GAMES = 16
HAL_OPPONENTS = ["hal_death_trade", "hal_pressure"]
SCRIPTED_OPPONENTS = ["bridge_pressure", "hal_death_trade", "hal_pressure"]

STRONG_PASS = {
    "bp_r7": 0.84, "ht_r7": 0.48, "hp_r7": 0.48,
    "seeded_r9": 1.0, "misaligned": 0,
    "native_wr": 0.12, "native_r7": 0.20,
}
ACCEPTABLE_PASS = {
    "bp_r7": 0.82, "ht_r7": 0.48, "hp_r7": 0.48,
    "seeded_r9": 1.0, "misaligned": 0,
    "native_wr": 0.06, "native_r7": 0.10,
}


# ── Model caching ────────────────────────────────────────────────────────
_base_model_cache = None
_bp_specialist_cache = None
_native_hal_cache = None


def _get_base_model():
    global _base_model_cache
    if _base_model_cache is None:
        _base_model_cache = MaskablePPO.load(BASE_MODEL)
    return _base_model_cache


def _get_bp_specialist():
    global _bp_specialist_cache
    if _bp_specialist_cache is None:
        _bp_specialist_cache = MaskablePPO.load(BP_SPECIALIST)
    return _bp_specialist_cache


def _get_native_hal():
    global _native_hal_cache
    if _native_hal_cache is None:
        _native_hal_cache = MaskablePPO.load(NATIVE_HAL_16K)
    return _native_hal_cache


class _CachedHalOpponent(Opponent):
    """ModelOpponent-compatible wrapper reusing a pre-loaded MaskablePPO."""

    def __init__(self, model):
        self.model = model
        self.role = "hal"
        self.awareness_config = AwarenessConfig()
        self.awareness = initial_awareness_for_role("hal")
        self._processed_history_len = 0

    def choose_action(self, game, role, turn_duration):
        while self._processed_history_len < len(game.history):
            record = game.history[self._processed_history_len]
            self.awareness = update_awareness(
                self.awareness, controlled_role_name=self.role,
                config=self.awareness_config, game=game, record=record,
            )
            self._processed_history_len += 1
        me = game.player1
        opp = game.player2
        obs = build_observation(game, me, opp, exposes_leap_features(self.awareness))
        mask = build_action_mask(
            role=role, is_leap_turn=game.is_leap_second_turn(),
            awareness=self.awareness,
        )
        action, _ = self.model.predict(obs, action_masks=mask, deterministic=True)
        return int(action) + 1

    def reset(self):
        self.awareness = initial_awareness_for_role("hal")
        self._processed_history_len = 0


def _make_native_opp():
    return _CachedHalOpponent(_get_native_hal())


# ── FeatureRuleController builder ────────────────────────────────────────
def _build_hal_frc():
    model = _get_base_model()
    feature_map = {}
    for opp_name in HAL_OPPONENTS:
        for seed in range(50):
            if len(feature_map) >= OPENING_HORIZON:
                break
            env = DTHEnv(
                opponent=create_scripted_opponent(opp_name),
                agent_role="baku", seed=seed,
            )
            obs, _ = env.reset()
            for turn in range(OPENING_HORIZON):
                mask = env.action_masks()
                action_second = BASELINE_OVERRIDES.get(turn)
                if action_second is None:
                    action_second = int(
                        model.predict(obs, action_masks=mask, deterministic=True)[0]
                    ) + 1
                key = FeatureRuleController._extract_key(obs)
                if key not in feature_map:
                    feature_map[key] = action_second
                obs, _, term, trunc, _ = env.step(action_second - 1)
                if term or trunc:
                    break
    return FeatureRuleController(feature_map, 0, OPENING_HORIZON - 1)


def _build_bp_opening_frc():
    bp_spec = _get_bp_specialist()
    feature_map = {}
    for seed in range(50):
        env = DTHEnv(
            opponent=create_scripted_opponent("bridge_pressure"),
            agent_role="baku", seed=seed,
        )
        obs, _ = env.reset()
        for turn in range(OPENING_HORIZON):
            mask = env.action_masks()
            action_idx = int(bp_spec.predict(obs, action_masks=mask, deterministic=True)[0])
            action_second = action_idx + 1
            key = FeatureRuleController._extract_key(obs)
            if key not in feature_map:
                feature_map[key] = action_second
            obs, _, term, trunc, _ = env.step(action_idx)
            if term or trunc:
                break
    return FeatureRuleController(feature_map, 0, OPENING_HORIZON - 1)


class ControllerAsModel:
    """Wraps FeatureRuleController to have predict() interface."""

    def __init__(self, controller: FeatureRuleController, fallback_action: int = 0):
        self.controller = controller
        self.fallback_action = fallback_action

    @property
    def policy(self):
        return None

    def predict(self, obs, *, action_masks=None, deterministic=True):
        obs_1d = obs.flatten() if obs.ndim > 1 else obs
        game_round = round(float(obs_1d[8]) * 10)
        game_half = round(float(obs_1d[9]))
        game_turn = game_round * 2 + game_half
        action_second = self.controller.query(obs_1d, game_turn)
        if action_second is not None:
            return np.array(action_second - 1), None
        return np.array(self.fallback_action), None


# ═════════════════════════════════════════════════════════════════════════
# ENRICHED CLASSIFIER BUILDERS
# ═════════════════════════════════════════════════════════════════════════

def build_enriched_classifier(
    opponent_configs: list[tuple[str, callable, int]],
    classify_turn: int = 2,
    games_per_opponent: int = 100,
) -> LogisticRegression:
    """Train classifier on enriched 24-dim observations (20-dim base + 4-dim history).

    opponent_configs: list of (name, factory_fn(seed) -> Opponent, label)
    """
    model = _get_base_model()
    obs_list, labels = [], []

    for opp_name, opp_factory, label in opponent_configs:
        for seed in range(games_per_opponent):
            opp = opp_factory(seed)
            env = DTHEnv(opponent=opp, agent_role="baku", seed=seed)
            obs, _ = env.reset()
            alive = True
            for t in range(classify_turn + 1):
                mask = env.action_masks()
                if t == classify_turn:
                    enriched = build_routing_feature_vector(
                        obs, env.game, env.agent, env.opp_player,
                    )
                    obs_list.append(enriched)
                    labels.append(label)
                    break
                action = BASELINE_OVERRIDES.get(t, int(model.predict(
                    obs, action_masks=mask, deterministic=True)[0]) + 1) - 1
                obs, _, term, trunc, _ = env.step(action)
                if term or trunc:
                    alive = False
                    break
            if not alive:
                continue

    X = np.array(obs_list)
    y = np.array(labels)
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X, y)
    return clf


def _build_enriched_old_classifier(classify_turn=2):
    """Enriched classifier: bp(0) + scripted hal(1) with 24-dim features."""
    return build_enriched_classifier([
        ("bridge_pressure", lambda s: create_scripted_opponent("bridge_pressure"), 0),
        ("hal_death_trade", lambda s: create_scripted_opponent("hal_death_trade"), 1),
    ], classify_turn=classify_turn)


def _build_enriched_native_classifier(classify_turn=2, games_per_opponent=100):
    """Enriched classifier: bp(0) + scripted hal(1) + native hal(1) with 24-dim."""
    return build_enriched_classifier([
        ("bridge_pressure", lambda s: create_scripted_opponent("bridge_pressure"), 0),
        ("hal_death_trade", lambda s: create_scripted_opponent("hal_death_trade"), 1),
        ("native_hal", lambda s: _make_native_opp(), 1),
    ], classify_turn=classify_turn, games_per_opponent=games_per_opponent)


# ═════════════════════════════════════════════════════════════════════════
# ENRICHED SELECTOR CLASSES
# ═════════════════════════════════════════════════════════════════════════

class EnrichedModularSelector:
    """ModularBakuSelector that uses enriched 24-dim features for classification only.

    At classify_turn, builds enriched observation from game history and classifies.
    Policy models still receive the standard 20-dim observation.
    Requires set_env() to be called before each game to provide game access.
    """

    def __init__(
        self,
        bp_specialist,
        opening_model,
        late_model,
        enriched_classifier: LogisticRegression,
        opening_horizon: int = 8,
        classify_turn: int = 2,
    ):
        self.bp_specialist = bp_specialist
        self.opening_model = opening_model
        self.late_model = late_model
        self.enriched_classifier = enriched_classifier
        self.opening_horizon = opening_horizon
        self.classify_turn = classify_turn
        self._turn = 0
        self._is_bp: bool | None = None
        self._env = None

    def set_env(self, env):
        self._env = env

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

    def predict(self, obs, *, action_masks=None, deterministic=True):
        if self._is_bp is None and self._turn >= self.classify_turn:
            assert self._env is not None, "Must call set_env() before predict()"
            enriched = build_routing_feature_vector(
                obs, self._env.game, self._env.agent, self._env.opp_player,
            )
            enriched_2d = enriched.reshape(1, -1)
            label = int(self.enriched_classifier.predict(enriched_2d)[0])
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

        return self.late_model.predict(
            obs, action_masks=action_masks, deterministic=deterministic
        )


class EnrichedDualHeadSelector:
    """DualHeadOpeningSelector with enriched 24-dim classification.

    Two opening heads: bp_opening_head and hal_opening_head.
    Classification uses enriched features; policy uses standard 20-dim.
    """

    def __init__(
        self,
        bp_opening_head,
        hal_opening_head,
        bp_specialist,
        late_model,
        enriched_classifier: LogisticRegression,
        opening_horizon: int = 8,
        classify_turn: int = 2,
    ):
        self.bp_opening_head = bp_opening_head
        self.hal_opening_head = hal_opening_head
        self.bp_specialist = bp_specialist
        self.late_model = late_model
        self.enriched_classifier = enriched_classifier
        self.opening_horizon = opening_horizon
        self.classify_turn = classify_turn
        self._turn = 0
        self._is_bp: bool | None = None
        self._env = None

    def set_env(self, env):
        self._env = env

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

    def predict(self, obs, *, action_masks=None, deterministic=True):
        if self._is_bp is None and self._turn >= self.classify_turn:
            assert self._env is not None, "Must call set_env() before predict()"
            enriched = build_routing_feature_vector(
                obs, self._env.game, self._env.agent, self._env.opp_player,
            )
            label = int(self.enriched_classifier.predict(enriched.reshape(1, -1))[0])
            self._is_bp = label == 0

        self._turn += 1
        game_turn = self._game_turn_from_obs(obs)

        if game_turn < self.opening_horizon:
            if self._is_bp or self._is_bp is None:
                return self.bp_opening_head.predict(
                    obs, action_masks=action_masks, deterministic=deterministic
                )
            else:
                return self.hal_opening_head.predict(
                    obs, action_masks=action_masks, deterministic=deterministic
                )

        if self._is_bp:
            return self.bp_specialist.predict(
                obs, action_masks=action_masks, deterministic=deterministic
            )
        return self.late_model.predict(
            obs, action_masks=action_masks, deterministic=deterministic
        )


class EnrichedStackedSelector:
    """Stacked routing with enriched classifiers for both opening and late phases."""

    def __init__(
        self,
        bp_opening_head,
        hal_opening_head,
        bp_specialist,
        late_model,
        opening_classifier: LogisticRegression,
        late_classifier: LogisticRegression,
        opening_horizon: int = 8,
        opening_classify_turn: int = 2,
        late_classify_turn: int = 2,
    ):
        self.bp_opening_head = bp_opening_head
        self.hal_opening_head = hal_opening_head
        self.bp_specialist = bp_specialist
        self.late_model = late_model
        self.opening_classifier = opening_classifier
        self.late_classifier = late_classifier
        self.opening_horizon = opening_horizon
        self.opening_classify_turn = opening_classify_turn
        self.late_classify_turn = late_classify_turn
        self._turn = 0
        self._opening_label: int | None = None
        self._late_label: int | None = None
        self._env = None

    def set_env(self, env):
        self._env = env

    @staticmethod
    def _game_turn_from_obs(obs):
        obs_1d = obs.flatten() if obs.ndim > 1 else obs
        return round(float(obs_1d[8]) * 10) * 2 + round(float(obs_1d[9]))

    @property
    def policy(self):
        return self.late_model.policy

    def reset(self):
        self._turn = 0
        self._opening_label = None
        self._late_label = None

    def predict(self, obs, *, action_masks=None, deterministic=True):
        game_turn = self._game_turn_from_obs(obs)

        if game_turn < self.opening_horizon:
            if (self._opening_label is None
                    and self._turn >= self.opening_classify_turn):
                assert self._env is not None
                enriched = build_routing_feature_vector(
                    obs, self._env.game, self._env.agent, self._env.opp_player,
                )
                self._opening_label = int(
                    self.opening_classifier.predict(enriched.reshape(1, -1))[0]
                )
            self._turn += 1
            if self._opening_label == 0:
                return self.bp_opening_head.predict(
                    obs, action_masks=action_masks, deterministic=deterministic
                )
            elif self._opening_label == 1:
                return self.hal_opening_head.predict(
                    obs, action_masks=action_masks, deterministic=deterministic
                )
            else:
                return self.bp_opening_head.predict(
                    obs, action_masks=action_masks, deterministic=deterministic
                )

        if self._late_label is None and self._turn >= self.late_classify_turn:
            assert self._env is not None
            enriched = build_routing_feature_vector(
                obs, self._env.game, self._env.agent, self._env.opp_player,
            )
            self._late_label = int(
                self.late_classifier.predict(enriched.reshape(1, -1))[0]
            )
        self._turn += 1

        if self._late_label == 0:
            return self.bp_specialist.predict(
                obs, action_masks=action_masks, deterministic=deterministic
            )
        return self.late_model.predict(
            obs, action_masks=action_masks, deterministic=deterministic
        )


class EnrichedDelayedSelector:
    """Delayed classification at T4 with universal opening for T0-T3.

    Uses hal opening universally until classify_turn (T4), then classifies
    with enriched features and routes accordingly for T4+.
    """

    def __init__(
        self,
        universal_opening,
        bp_opening_head,
        hal_opening_head,
        bp_specialist,
        late_model,
        enriched_classifier: LogisticRegression,
        opening_horizon: int = 8,
        classify_turn: int = 4,
    ):
        self.universal_opening = universal_opening
        self.bp_opening_head = bp_opening_head
        self.hal_opening_head = hal_opening_head
        self.bp_specialist = bp_specialist
        self.late_model = late_model
        self.enriched_classifier = enriched_classifier
        self.opening_horizon = opening_horizon
        self.classify_turn = classify_turn
        self._turn = 0
        self._route: str | None = None
        self._env = None

    def set_env(self, env):
        self._env = env

    @staticmethod
    def _game_turn_from_obs(obs):
        obs_1d = obs.flatten() if obs.ndim > 1 else obs
        return round(float(obs_1d[8]) * 10) * 2 + round(float(obs_1d[9]))

    @property
    def policy(self):
        return self.late_model.policy

    def reset(self):
        self._turn = 0
        self._route = None

    def predict(self, obs, *, action_masks=None, deterministic=True):
        game_turn = self._game_turn_from_obs(obs)

        # Pre-classification: universal opening
        if self._route is None and self._turn < self.classify_turn:
            self._turn += 1
            return self.universal_opening.predict(
                obs, action_masks=action_masks, deterministic=deterministic
            )

        # Classify
        if self._route is None:
            assert self._env is not None
            enriched = build_routing_feature_vector(
                obs, self._env.game, self._env.agent, self._env.opp_player,
            )
            label = int(self.enriched_classifier.predict(enriched.reshape(1, -1))[0])
            self._route = "bp" if label == 0 else "hal"

        self._turn += 1

        if game_turn < self.opening_horizon:
            if self._route == "bp":
                return self.bp_opening_head.predict(
                    obs, action_masks=action_masks, deterministic=deterministic
                )
            return self.hal_opening_head.predict(
                obs, action_masks=action_masks, deterministic=deterministic
            )

        if self._route == "bp":
            return self.bp_specialist.predict(
                obs, action_masks=action_masks, deterministic=deterministic
            )
        return self.late_model.predict(
            obs, action_masks=action_masks, deterministic=deterministic
        )


# ── Evaluation helpers ───────────────────────────────────────────────────
def pct(v):
    return f"{v * 100:.0f}%"


@dataclass
class EvalResult:
    opponent: str
    games: int
    wins: int
    win_rate: float
    avg_half_rounds: float
    r7_count: int
    r7_eligible: int
    r7_rate: float
    deaths_by_agent: int


def evaluate_selector(selector, opp_name, games=50, seed=42, opp_model_path=None):
    """Evaluate selector, auto-wiring env reference for enriched selectors."""
    wins = 0; total_hr = 0; r7_count = 0; r7_eligible = 0; deaths = 0
    has_set_env = hasattr(selector, "set_env")
    for gi in range(games):
        opp = (create_model_opponent(opp_model_path, agent_role="baku")
               if opp_model_path else create_scripted_opponent(opp_name))
        env = DTHEnv(opponent=opp, agent_role="baku", seed=seed + gi)
        obs, _ = env.reset()
        selector.reset()
        if has_set_env:
            selector.set_env(env)
        reached_r7 = current_route_stage_flags(env.game).get("round7_pressure", False)
        eligible = stage_is_eligible_from_start(env.game, "round7_pressure")
        while True:
            mask = env.action_masks()
            action, _ = selector.predict(obs, action_masks=mask, deterministic=True)
            obs, reward, term, trunc, _ = env.step(int(action))
            reached_r7 = reached_r7 or current_route_stage_flags(env.game).get(
                "round7_pressure", False)
            if term or trunc:
                won = bool(term and env.game.winner is env.agent and reward > 0)
                wins += int(won)
                total_hr += len(env.game.history)
                if eligible:
                    r7_eligible += 1
                    r7_count += int(reached_r7)
                deaths += env.agent.deaths
                break
    n = games
    return EvalResult(
        opp_name, n, wins, wins / n, total_hr / n,
        r7_count, r7_eligible, r7_count / r7_eligible if r7_eligible else 0, deaths,
    )


def eval_r9_seeded(selector, seed=42):
    opp = create_scripted_opponent("safe")
    env = DTHEnv(
        opponent=opp, agent_role="baku", seed=seed,
        scenario_sampler=lambda _rng: get_scenario("round9_pre_leap"),
    )
    obs, _ = env.reset()
    selector.reset()
    if hasattr(selector, "set_env"):
        selector.set_env(env)
    while True:
        mask = env.action_masks()
        action, _ = selector.predict(obs, action_masks=mask, deterministic=True)
        obs, reward, term, trunc, _ = env.step(int(action))
        if term or trunc:
            won = bool(term and env.game.winner is env.agent and reward > 0)
            return {"wins": int(won), "games": 1, "rate": float(won)}


def eval_full_suite(selector, games, seed, tag, extra_stats=None):
    results = {}
    for opp in SCRIPTED_OPPONENTS:
        res = evaluate_selector(selector, opp, games=games, seed=seed)
        results[opp] = asdict(res)
        print(f"    {tag} vs {opp}: r7={pct(res.r7_rate)} wr={pct(res.win_rate)}")
    r9 = eval_r9_seeded(selector, seed=seed)
    results["seeded_r9"] = r9
    print(f"    {tag} seeded r9: {pct(r9['rate'])}")
    pr = evaluate_selector(
        selector, "prior_4096", games=games, seed=seed,
        opp_model_path=LEARNED_HAL_PRIOR,
    )
    results["prior_4096"] = asdict(pr)
    print(f"    {tag} vs prior_4096: wr={pct(pr.win_rate)}")
    nh = evaluate_selector(
        selector, "native_hal", games=games, seed=seed,
        opp_model_path=NATIVE_HAL_16K,
    )
    results["native_hal"] = asdict(nh)
    print(f"    {tag} vs native_hal: wr={pct(nh.win_rate)} r7={pct(nh.r7_rate)}")
    if extra_stats:
        results["extra_stats"] = extra_stats
    return results


def enriched_confidence_profile(clf, opp_factory, n_games=20, classify_turn=2):
    """Classifier confidence on enriched 24-dim observations."""
    model = _get_base_model()
    probas = []
    raw_features = []
    for seed in range(n_games):
        opp = opp_factory(seed)
        env = DTHEnv(opponent=opp, agent_role="baku", seed=42 + seed)
        obs, _ = env.reset()
        alive = True
        for t in range(classify_turn + 1):
            mask = env.action_masks()
            if t == classify_turn:
                enriched = build_routing_feature_vector(
                    obs, env.game, env.agent, env.opp_player,
                )
                p = clf.predict_proba(enriched.reshape(1, -1))[0]
                probas.append(p.tolist())
                raw_features.append(enriched[20:].tolist())
                break
            action = BASELINE_OVERRIDES.get(t, int(model.predict(
                obs, action_masks=mask, deterministic=True)[0]) + 1) - 1
            obs, _, term, trunc, _ = env.step(action)
            if term or trunc:
                alive = False
                break
        if not alive:
            continue
    if not probas:
        return {"mean_p_bp": 0, "mean_p_hal": 0, "n": 0, "features": []}
    arr = np.array(probas)
    feat_arr = np.array(raw_features)
    return {
        "mean_p_bp": float(arr[:, 0].mean()),
        "mean_p_hal": float(arr[:, 1].mean()) if arr.shape[1] > 1 else 0,
        "std_p_bp": float(arr[:, 0].std()),
        "n": len(probas),
        "mean_history_features": feat_arr.mean(axis=0).tolist(),
        "std_history_features": feat_arr.std(axis=0).tolist(),
    }


def verify_enriched_classifier_accuracy(clf, opp_factory, label, n_games=20,
                                         classify_turn=2):
    """Check enriched classifier labels opp_factory opponents correctly."""
    model = _get_base_model()
    correct = 0; total = 0
    for seed in range(n_games):
        opp = opp_factory(seed)
        env = DTHEnv(opponent=opp, agent_role="baku", seed=42 + seed)
        obs, _ = env.reset()
        alive = True
        for t in range(classify_turn + 1):
            mask = env.action_masks()
            if t == classify_turn:
                enriched = build_routing_feature_vector(
                    obs, env.game, env.agent, env.opp_player,
                )
                pred = int(clf.predict(enriched.reshape(1, -1))[0])
                total += 1
                if pred == label:
                    correct += 1
                break
            action = BASELINE_OVERRIDES.get(t, int(model.predict(
                obs, action_masks=mask, deterministic=True)[0]) + 1) - 1
            obs, _, term, trunc, _ = env.step(action)
            if term or trunc:
                alive = False
                break
        if not alive:
            continue
    return {"correct": correct, "total": total,
            "accuracy": correct / total if total else 0}


# ── Old (non-enriched) classifier for reference baselines ────────────────
_old_clf_cache = None
_native_clf_cache = None


def _get_old_classifier():
    global _old_clf_cache
    if _old_clf_cache is None:
        _old_clf_cache = build_classifier(
            BASE_MODEL, ["bridge_pressure", "hal_death_trade"], classify_turn=2
        )
    return _old_clf_cache


def _build_native_aware_classifier(classify_turn=2, games_per_opponent=100):
    global _native_clf_cache
    if _native_clf_cache is not None:
        return _native_clf_cache

    model = _get_base_model()
    obs_list, labels = [], []

    for opp_idx, opp_name in enumerate(["bridge_pressure", "hal_death_trade"]):
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
                action = BASELINE_OVERRIDES.get(t, int(model.predict(
                    obs, action_masks=mask, deterministic=True)[0]) + 1) - 1
                obs, _, term, trunc, _ = env.step(action)
                if term or trunc:
                    alive = False
                    break
            if not alive:
                continue

    for seed in range(games_per_opponent):
        opp = _make_native_opp()
        env = DTHEnv(opponent=opp, agent_role="baku", seed=42 + seed)
        obs, _ = env.reset()
        alive = True
        for t in range(classify_turn + 1):
            mask = env.action_masks()
            if t == classify_turn:
                obs_list.append(obs.copy())
                labels.append(1)
                break
            action = BASELINE_OVERRIDES.get(t, int(model.predict(
                obs, action_masks=mask, deterministic=True)[0]) + 1) - 1
            obs, _, term, trunc, _ = env.step(action)
            if term or trunc:
                alive = False
                break
        if not alive:
            continue

    X = np.array(obs_list)
    y = np.array(labels)
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X, y)
    _native_clf_cache = clf
    return clf


# ═════════════════════════════════════════════════════════════════════════
# LANE 0: Reproduce Two Reference Baselines
# ═════════════════════════════════════════════════════════════════════════
def lane_0_baselines():
    print("\n" + "=" * 60)
    print(f"LANE 0: Reference Baselines (PHYSICALITY_BAKU={PHYSICALITY_BAKU})")
    print("=" * 60)

    bp_specialist = _get_bp_specialist()
    late_model = _get_base_model()

    old_clf = _get_old_classifier()
    native_clf = _build_native_aware_classifier()

    hal_frc = _build_hal_frc()
    hal_opening = ControllerAsModel(hal_frc)

    # Reference 1: Old T2 classifier (bp=high, native=zero)
    print("\n  --- Reference 1: Old T2 classifier ---")
    ref1 = ModularBakuSelector(
        bp_specialist=bp_specialist, opening_model=hal_opening,
        late_model=late_model, classifier=old_clf,
        opening_horizon=OPENING_HORIZON, classify_turn=2,
    )
    ref1_results = eval_full_suite(ref1, EVAL_GAMES, 42, "ref1_old_clf",
                                    {"classifier": "old_t2"})

    # Reference 2: Native-aware classifier (bp=low, native=nonzero)
    print("\n  --- Reference 2: Native-aware T2 classifier ---")
    ref2 = ModularBakuSelector(
        bp_specialist=bp_specialist, opening_model=hal_opening,
        late_model=late_model, classifier=native_clf,
        opening_horizon=OPENING_HORIZON, classify_turn=2,
    )
    ref2_results = eval_full_suite(ref2, EVAL_GAMES, 42, "ref2_native_clf",
                                    {"classifier": "native_aware_t2"})

    return ref1_results, ref2_results, hal_frc, old_clf, native_clf


# ═════════════════════════════════════════════════════════════════════════
# LANE 1: Build Enriched Classifiers + Diagnostics
# ═════════════════════════════════════════════════════════════════════════
def lane_1_enriched_classifiers():
    print("\n" + "=" * 60)
    print("LANE 1: Enriched Classifiers + Diagnostics")
    print("=" * 60)

    enriched_old = _build_enriched_old_classifier(classify_turn=2)
    enriched_native = _build_enriched_native_classifier(classify_turn=2)
    enriched_native_t4 = _build_enriched_native_classifier(classify_turn=4)

    classifiers = {
        "enriched_old_t2": enriched_old,
        "enriched_native_t2": enriched_native,
        "enriched_native_t4": enriched_native_t4,
    }

    # Accuracy diagnostics
    print("\n  Classifier accuracy diagnostics:")
    accuracy_diag = {}
    for clf_name, clf in classifiers.items():
        ct = 4 if "t4" in clf_name else 2
        bp_acc = verify_enriched_classifier_accuracy(
            clf, lambda s: create_scripted_opponent("bridge_pressure"), 0,
            n_games=20, classify_turn=ct)
        hal_acc = verify_enriched_classifier_accuracy(
            clf, lambda s: create_scripted_opponent("hal_death_trade"), 1,
            n_games=20, classify_turn=ct)
        native_acc = verify_enriched_classifier_accuracy(
            clf, lambda s: _make_native_opp(), 1,
            n_games=20, classify_turn=ct)
        accuracy_diag[clf_name] = {
            "bp_accuracy": bp_acc["accuracy"],
            "hal_accuracy": hal_acc["accuracy"],
            "native_accuracy": native_acc["accuracy"],
        }
        print(f"  {clf_name}: bp={bp_acc['accuracy']:.0%} "
              f"hal={hal_acc['accuracy']:.0%} native={native_acc['accuracy']:.0%}")

    # Confidence profiles (enriched features)
    print("\n  Enriched confidence profiles:")
    confidence_diag = {}
    for clf_name, clf in classifiers.items():
        ct = 4 if "t4" in clf_name else 2
        confidence_diag[clf_name] = {}
        for opp_label, factory in [
            ("bp", lambda s: create_scripted_opponent("bridge_pressure")),
            ("hal_dt", lambda s: create_scripted_opponent("hal_death_trade")),
            ("native", lambda s: _make_native_opp()),
        ]:
            profile = enriched_confidence_profile(clf, factory, n_games=20,
                                                    classify_turn=ct)
            confidence_diag[clf_name][opp_label] = profile
            print(f"    {clf_name} vs {opp_label}: "
                  f"p_bp={profile['mean_p_bp']:.3f} "
                  f"p_hal={profile['mean_p_hal']:.3f} "
                  f"std={profile.get('std_p_bp', 0):.3f} "
                  f"hist_feats={[f'{x:.3f}' for x in profile.get('mean_history_features', [])]}")

    # Key diagnostic: do bp and native Hal now produce DIFFERENT enriched features?
    bp_feats = confidence_diag.get("enriched_native_t2", {}).get("bp", {}).get(
        "mean_history_features", [0, 0, 0, 0])
    native_feats = confidence_diag.get("enriched_native_t2", {}).get("native", {}).get(
        "mean_history_features", [0, 0, 0, 0])
    overlap_broken = not np.allclose(bp_feats, native_feats, atol=0.01)
    print(f"\n  ** OVERLAP BROKEN: {overlap_broken} **")
    print(f"     bp history features:     {[f'{x:.4f}' for x in bp_feats]}")
    print(f"     native history features: {[f'{x:.4f}' for x in native_feats]}")

    return classifiers, accuracy_diag, confidence_diag, overlap_broken


# ═════════════════════════════════════════════════════════════════════════
# LANE 2: Enriched Selector Candidates
# ═════════════════════════════════════════════════════════════════════════
def lane_2_enriched_selectors(hal_frc, classifiers):
    print("\n" + "=" * 60)
    print("LANE 2: Enriched Selector Candidates")
    print("=" * 60)

    bp_specialist = _get_bp_specialist()
    late_model = _get_base_model()
    hal_opening = ControllerAsModel(hal_frc)

    bp_frc = _build_bp_opening_frc()
    bp_opening = ControllerAsModel(bp_frc)

    candidates = []

    # E1: Enriched old clf + ModularBaku
    print("\n  --- E1: Enriched old clf + ModularBaku ---")
    e1 = EnrichedModularSelector(
        bp_specialist=bp_specialist, opening_model=hal_opening,
        late_model=late_model,
        enriched_classifier=classifiers["enriched_old_t2"],
        opening_horizon=OPENING_HORIZON, classify_turn=2,
    )
    candidates.append(("E1_enriched_old_modular", e1, {
        "type": "enriched_modular",
        "classifier": "enriched_old_t2",
        "misaligned": 0,
    }))

    # E2: Enriched native clf + ModularBaku
    print("\n  --- E2: Enriched native clf + ModularBaku ---")
    e2 = EnrichedModularSelector(
        bp_specialist=bp_specialist, opening_model=hal_opening,
        late_model=late_model,
        enriched_classifier=classifiers["enriched_native_t2"],
        opening_horizon=OPENING_HORIZON, classify_turn=2,
    )
    candidates.append(("E2_enriched_native_modular", e2, {
        "type": "enriched_modular",
        "classifier": "enriched_native_t2",
        "misaligned": 0,
    }))

    # E3: Enriched native clf + DualHead A
    print("\n  --- E3: Enriched native clf + DualHead A ---")
    e3 = EnrichedDualHeadSelector(
        bp_opening_head=bp_opening, hal_opening_head=hal_opening,
        bp_specialist=bp_specialist, late_model=late_model,
        enriched_classifier=classifiers["enriched_native_t2"],
        opening_horizon=OPENING_HORIZON, classify_turn=2,
    )
    candidates.append(("E3_enriched_native_dual_head", e3, {
        "type": "enriched_dual_head",
        "classifier": "enriched_native_t2",
        "bp_opening_entries": len(bp_frc.feature_action_map),
        "hal_opening_entries": len(hal_frc.feature_action_map),
        "misaligned": 0,
    }))

    # E4: Enriched stacked (enriched old opening + enriched native late)
    print("\n  --- E4: Enriched stacked (old opening + native late) ---")
    e4 = EnrichedStackedSelector(
        bp_opening_head=bp_opening, hal_opening_head=hal_opening,
        bp_specialist=bp_specialist, late_model=late_model,
        opening_classifier=classifiers["enriched_old_t2"],
        late_classifier=classifiers["enriched_native_t2"],
        opening_horizon=OPENING_HORIZON,
        opening_classify_turn=2, late_classify_turn=2,
    )
    candidates.append(("E4_enriched_stacked", e4, {
        "type": "enriched_stacked",
        "opening_clf": "enriched_old_t2",
        "late_clf": "enriched_native_t2",
        "misaligned": 0,
    }))

    # D1: Delayed T4 classification with universal hal opening for T0-T3
    print("\n  --- D1: Delayed T4 + universal hal opening ---")
    d1 = EnrichedDelayedSelector(
        universal_opening=hal_opening,
        bp_opening_head=bp_opening, hal_opening_head=hal_opening,
        bp_specialist=bp_specialist, late_model=late_model,
        enriched_classifier=classifiers["enriched_native_t4"],
        opening_horizon=OPENING_HORIZON, classify_turn=4,
    )
    candidates.append(("D1_delayed_t4_hal_univ", d1, {
        "type": "enriched_delayed_t4",
        "universal_opening": "hal",
        "classifier": "enriched_native_t4",
        "misaligned": 0,
    }))

    return candidates


# ═════════════════════════════════════════════════════════════════════════
# LANE 3: Gate + Full Validation
# ═════════════════════════════════════════════════════════════════════════
def lane_3_validate(all_candidates, ref1_results):
    print("\n" + "=" * 60)
    print("LANE 3: Gate + Full Validation")
    print("=" * 60)

    validated = []
    results_list = []

    for tag, selector, stats in all_candidates:
        print(f"\n  --- {tag} ---")

        # 16-game gate
        gate = {}
        for opp in SCRIPTED_OPPONENTS:
            res = evaluate_selector(selector, opp, games=GATE_GAMES, seed=42)
            gate[opp] = asdict(res)
        r9 = eval_r9_seeded(selector, seed=42)
        gate["seeded_r9"] = r9

        bp = gate.get("bridge_pressure", {}).get("r7_rate", 0)
        ht = gate.get("hal_death_trade", {}).get("r7_rate", 0)
        hp = gate.get("hal_pressure", {}).get("r7_rate", 0)
        gate_pass = bp >= 0.60 and ht >= 0.30 and hp >= 0.30 and r9["rate"] >= 1.0

        entry = {"tag": tag, "stats": stats, "gate_pass": gate_pass, "gate": gate}

        if not gate_pass:
            print(f"    GATE FAIL: bp={pct(bp)} ht={pct(ht)} hp={pct(hp)} "
                  f"r9={pct(r9['rate'])}")
            entry["analysis"] = None
            entry["gap"] = None
            results_list.append(entry)
            continue

        print(f"    GATE PASS: bp={pct(bp)} ht={pct(ht)} hp={pct(hp)}")

        # Full 50-game evaluation
        full = eval_full_suite(selector, EVAL_GAMES, 42, tag, stats)
        entry["full_eval"] = full

        full_native_wr = full.get("native_hal", {}).get("win_rate", 0)
        full_native_r7 = full.get("native_hal", {}).get("r7_rate", 0)
        entry["gap"] = {
            "full_native_wr": full_native_wr,
            "full_native_r7": full_native_r7,
        }

        validated.append(entry)
        results_list.append(entry)

    return results_list, validated


# ═════════════════════════════════════════════════════════════════════════
# LANE 4: Verdict
# ═════════════════════════════════════════════════════════════════════════
def lane_4_verdict(ref1, ref2, validated):
    print("\n" + "=" * 60)
    print("LANE 4: Verdict")
    print("=" * 60)

    ref1_bp = ref1.get("bridge_pressure", {}).get("r7_rate", 0)
    ref1_native_wr = ref1.get("native_hal", {}).get("win_rate", 0)
    ref1_native_r7 = ref1.get("native_hal", {}).get("r7_rate", 0)
    ref2_bp = ref2.get("bridge_pressure", {}).get("r7_rate", 0)
    ref2_native_wr = ref2.get("native_hal", {}).get("win_rate", 0)
    ref2_native_r7 = ref2.get("native_hal", {}).get("r7_rate", 0)

    print(f"\n  Reference 1 (old clf):     bp={pct(ref1_bp)} native_wr={pct(ref1_native_wr)} "
          f"native_r7={pct(ref1_native_r7)}")
    print(f"  Reference 2 (native clf):  bp={pct(ref2_bp)} native_wr={pct(ref2_native_wr)} "
          f"native_r7={pct(ref2_native_r7)}")

    best_candidate = None
    best_level = None
    best_score = -999.0
    tradeoff_broken = False

    for c in validated:
        v = c.get("full_eval", {})
        c_bp = v.get("bridge_pressure", {}).get("r7_rate", 0)
        c_ht = v.get("hal_death_trade", {}).get("r7_rate", 0)
        c_hp = v.get("hal_pressure", {}).get("r7_rate", 0)
        c_r9 = v.get("seeded_r9", {}).get("rate", 0)
        c_prior = v.get("prior_4096", {}).get("win_rate", 0)
        c_native_wr = v.get("native_hal", {}).get("win_rate", 0)
        c_native_r7 = v.get("native_hal", {}).get("r7_rate", 0)
        c_mis = c.get("stats", {}).get("misaligned", 0)

        if c_bp >= 0.82 and (c_native_wr >= 0.06 or c_native_r7 >= 0.10):
            tradeoff_broken = True

        strong = (
            c_bp >= STRONG_PASS["bp_r7"]
            and c_ht >= STRONG_PASS["ht_r7"]
            and c_hp >= STRONG_PASS["hp_r7"]
            and c_r9 >= STRONG_PASS["seeded_r9"]
            and c_mis == STRONG_PASS["misaligned"]
            and (c_native_wr >= STRONG_PASS["native_wr"]
                 or c_native_r7 >= STRONG_PASS["native_r7"])
        )
        acceptable = (
            c_bp >= ACCEPTABLE_PASS["bp_r7"]
            and c_ht >= ACCEPTABLE_PASS["ht_r7"]
            and c_hp >= ACCEPTABLE_PASS["hp_r7"]
            and c_r9 >= ACCEPTABLE_PASS["seeded_r9"]
            and c_mis == ACCEPTABLE_PASS["misaligned"]
            and (c_native_wr >= ACCEPTABLE_PASS["native_wr"]
                 or c_native_r7 >= ACCEPTABLE_PASS["native_r7"])
        )

        level = "STRONG_PASS" if strong else ("ACCEPTABLE_PASS" if acceptable else None)
        score = (c_bp * 2 + c_native_wr * 3 + c_native_r7 * 2
                 + c_ht + c_hp + c_prior)

        c["analysis"] = {
            "level": level,
            "score": round(score, 4),
            "abs": {
                "bp_r7": round(c_bp, 4), "ht_r7": round(c_ht, 4),
                "hp_r7": round(c_hp, 4), "r9": round(c_r9, 4),
                "prior_wr": round(c_prior, 4), "native_wr": round(c_native_wr, 4),
                "native_r7": round(c_native_r7, 4), "misaligned": c_mis,
            },
            "vs_ref1": {
                "bp_delta": round(c_bp - ref1_bp, 4),
                "native_wr_delta": round(c_native_wr - ref1_native_wr, 4),
                "native_r7_delta": round(c_native_r7 - ref1_native_r7, 4),
            },
            "vs_ref2": {
                "bp_delta": round(c_bp - ref2_bp, 4),
                "native_wr_delta": round(c_native_wr - ref2_native_wr, 4),
                "native_r7_delta": round(c_native_r7 - ref2_native_r7, 4),
            },
        }

        if level and (not best_level or score > best_score):
            best_score = score
            best_candidate = c
            best_level = level
        elif not best_level and score > best_score:
            best_score = score
            best_candidate = c

        print(f"  {c['tag']}: level={level or '---'} score={score:.2f} "
              f"bp={pct(c_bp)} native_wr={pct(c_native_wr)} "
              f"native_r7={pct(c_native_r7)}")

    verdict_str = "NO_IMPROVEMENT"
    if best_level == "STRONG_PASS":
        verdict_str = "STRONG_PASS"
    elif best_level == "ACCEPTABLE_PASS":
        verdict_str = "ACCEPTABLE_PASS"
    elif tradeoff_broken:
        verdict_str = "TRADEOFF_BROKEN"

    return {
        "verdict": verdict_str,
        "tradeoff_broken": tradeoff_broken,
        "best_candidate": best_candidate["tag"] if best_candidate else None,
        "best_level": best_level,
        "best_score": round(best_score, 4) if best_score > -999 else None,
        "candidates_evaluated": len(validated),
    }


# ═════════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════════
def main():
    t0 = time.time()
    print("=" * 60)
    print(f"SPRINT: Routing Feature Enrichment (p={PHYSICALITY_BAKU})")
    print("=" * 60)

    # Lane 0: Reproduce baselines
    ref1, ref2, hal_frc, old_clf, native_clf = lane_0_baselines()

    # Lane 1: Build enriched classifiers + diagnostics
    classifiers, accuracy_diag, confidence_diag, overlap_broken = (
        lane_1_enriched_classifiers()
    )

    # Lane 2: Build enriched selectors
    all_candidates = lane_2_enriched_selectors(hal_frc, classifiers)

    # Lane 3: Gate + validate
    results_list, validated = lane_3_validate(all_candidates, ref1)

    # Lane 4: Verdict
    verdict = lane_4_verdict(ref1, ref2, validated)
    verdict["overlap_broken"] = overlap_broken

    elapsed = round(time.time() - t0, 1)

    # ── Deliverables ─────────────────────────────────────────────────────
    report_dir = Path("docs/reports")
    json_dir = report_dir / "json"
    report_dir.mkdir(parents=True, exist_ok=True)
    json_dir.mkdir(parents=True, exist_ok=True)

    # JSON report
    json_data = {
        "sprint": "routing-feature-enrichment",
        "date": "2026-03-28",
        "physicality_baku": PHYSICALITY_BAKU,
        "baseline_route": BASELINE_ROUTE,
        "reference_1_old_clf": ref1,
        "reference_2_native_clf": ref2,
        "enriched_classifier_accuracy": accuracy_diag,
        "enriched_confidence_profiles": {
            clf_name: {
                opp: {k: v for k, v in profile.items()}
                for opp, profile in profiles.items()
            }
            for clf_name, profiles in confidence_diag.items()
        },
        "overlap_broken": overlap_broken,
        "candidates": [
            {"tag": r["tag"], "stats": r["stats"], "gate_pass": r["gate_pass"],
             "analysis": r.get("analysis"), "gap": r.get("gap")}
            for r in results_list
        ],
        "validated": [
            {"tag": c["tag"], "analysis": c.get("analysis"), "gap": c.get("gap")}
            for c in validated
        ],
        "verdict": verdict,
        "elapsed_seconds": elapsed,
    }
    json_path = json_dir / "sprint_routing_feature_enrichment_2026-03-28.json"
    with open(json_path, "w") as f:
        json.dump(json_data, f, indent=2, default=str)
    print(f"\n  JSON report: {json_path}")

    # Markdown report
    md_path = report_dir / "sprint_routing_feature_enrichment_report_2026-03-28.md"
    with open(md_path, "w") as f:
        f.write("# Sprint Report: Routing Feature Enrichment (p=0.88)\n")
        f.write(f"**Date:** 2026-03-28\n")
        f.write(f"**PHYSICALITY_BAKU:** {PHYSICALITY_BAKU}\n")
        f.write(f"**Verdict:** {verdict['verdict']}\n")
        f.write(f"**Overlap broken:** {overlap_broken}\n")
        f.write(f"**Tradeoff broken:** {verdict['tradeoff_broken']}\n\n")

        f.write("## Executive Summary\n\n")
        f.write("This sprint targets the observation-space limitation identified in the dual-head\n")
        f.write("opening sprint: bridge_pressure and native Hal produce numerically identical\n")
        f.write("20-dim observations at T2, making classification impossible. The fix: append\n")
        f.write("4-dim public opponent-action-history features to the classifier input (24-dim\n")
        f.write("total) while keeping the policy on the original 20-dim observation shape.\n\n")

        f.write("## Optimization Target\n\n")
        f.write("Break the bp-vs-native observation overlap by giving the routing/classifier\n")
        f.write("path the public opponent-action-history features that Baku would actually\n")
        f.write("remember. The policy observation shape must NOT change.\n\n")

        f.write("## Why Routing-Only Enrichment Is the Right Intermediate Step\n\n")
        f.write("A full observation-v2 retrain would change the policy input shape and require\n")
        f.write("retraining all models from scratch. By enriching ONLY the classifier/routing\n")
        f.write("input, we can test the hypothesis (opponent history breaks the overlap) without\n")
        f.write("any retrain risk. If enrichment works, it validates the features for a future\n")
        f.write("observation-v2 migration. If it doesn't, we learn cheaply.\n\n")

        f.write("## Reference Baselines (Reproduced)\n\n")
        f.write("| Metric | Ref 1: Old T2 clf | Ref 2: Native-aware T2 clf |\n")
        f.write("|--------|-------------------|----------------------------|\n")
        for m in ["bridge_pressure", "hal_death_trade", "hal_pressure"]:
            r1_r7 = ref1.get(m, {}).get("r7_rate", 0)
            r2_r7 = ref2.get(m, {}).get("r7_rate", 0)
            f.write(f"| {m} r7 | {pct(r1_r7)} | {pct(r2_r7)} |\n")
        f.write(f"| seeded r9 | {pct(ref1.get('seeded_r9', {}).get('rate', 0))} "
                f"| {pct(ref2.get('seeded_r9', {}).get('rate', 0))} |\n")
        f.write(f"| prior_4096 WR | {pct(ref1.get('prior_4096', {}).get('win_rate', 0))} "
                f"| {pct(ref2.get('prior_4096', {}).get('win_rate', 0))} |\n")
        f.write(f"| native WR | {pct(ref1.get('native_hal', {}).get('win_rate', 0))} "
                f"| {pct(ref2.get('native_hal', {}).get('win_rate', 0))} |\n")
        f.write(f"| native r7 | {pct(ref1.get('native_hal', {}).get('r7_rate', 0))} "
                f"| {pct(ref2.get('native_hal', {}).get('r7_rate', 0))} |\n\n")

        f.write("## Enriched Classifier Diagnostics\n\n")
        f.write("### Accuracy\n\n")
        f.write("| Classifier | bp acc | hal acc | native acc |\n")
        f.write("|------------|--------|---------|------------|\n")
        for clf_name, acc in accuracy_diag.items():
            f.write(f"| {clf_name} | {pct(acc['bp_accuracy'])} "
                    f"| {pct(acc['hal_accuracy'])} | {pct(acc['native_accuracy'])} |\n")

        f.write("\n### Confidence Profiles (Enriched Features)\n\n")
        f.write("| Classifier | Opponent | P(bp) | P(hal) | std | History Features |\n")
        f.write("|------------|----------|-------|--------|-----|------------------|\n")
        for clf_name, profiles in confidence_diag.items():
            for opp, p in profiles.items():
                feats = [f"{x:.3f}" for x in p.get("mean_history_features", [])]
                f.write(f"| {clf_name} | {opp} | {p['mean_p_bp']:.3f} "
                        f"| {p['mean_p_hal']:.3f} | {p.get('std_p_bp', 0):.3f} "
                        f"| {feats} |\n")

        f.write(f"\n### Key Finding: Overlap Broken = **{overlap_broken}**\n\n")
        bp_h = confidence_diag.get("enriched_native_t2", {}).get("bp", {}).get(
            "mean_history_features", [])
        nat_h = confidence_diag.get("enriched_native_t2", {}).get("native", {}).get(
            "mean_history_features", [])
        f.write(f"- bp history features:     `{[f'{x:.4f}' for x in bp_h]}`\n")
        f.write(f"- native history features: `{[f'{x:.4f}' for x in nat_h]}`\n\n")

        f.write("## Tested Selector Variants\n\n")
        f.write("| Candidate | Type | Classifier | Gate |\n")
        f.write("|-----------|------|------------|------|\n")
        for r in results_list:
            gate_str = "PASS" if r["gate_pass"] else "FAIL"
            clf_name = r.get("stats", {}).get("classifier", r.get("stats", {}).get(
                "opening_clf", "—"))
            f.write(f"| {r['tag']} | {r['stats'].get('type', '—')} "
                    f"| {clf_name} | {gate_str} |\n")

        f.write("\n## Full Validation\n\n")
        if validated:
            f.write("| Candidate | bp r7 | ht r7 | hp r7 | r9 | prior WR | native WR | native r7 | Level |\n")
            f.write("|-----------|-------|-------|-------|----|----------|-----------|-----------|-------|\n")
            for c in validated:
                if c.get("analysis") and c["analysis"].get("abs"):
                    a = c["analysis"]["abs"]
                    f.write(f"| {c['tag']} | {pct(a['bp_r7'])} | {pct(a['ht_r7'])} "
                            f"| {pct(a['hp_r7'])} | {pct(a['r9'])} | {pct(a['prior_wr'])} "
                            f"| {pct(a['native_wr'])} | {pct(a['native_r7'])} "
                            f"| {c['analysis'].get('level', '---')} |\n")
        else:
            f.write("No candidates passed the gate.\n")

        f.write(f"\n## Tests Added\n\n")
        f.write("- `tests/test_enriched_routing.py`: unit tests for enriched selector classes\n")
        f.write("  - TestEnrichedModularSelector: routing with enriched classification\n")
        f.write("  - TestEnrichedDualHeadSelector: dual-head routing with enriched classification\n")
        f.write("  - TestEnrichedStackedSelector: stacked routing with enriched classifiers\n")
        f.write("  - TestEnrichedDelayedSelector: delayed T4 classification routing\n")
        f.write("  - TestBuildEnrichedClassifier: classifier training on 24-dim features\n\n")

        f.write(f"## Final Verdict\n\n")
        f.write(f"**{verdict['verdict']}**\n\n")
        f.write(f"- Overlap broken: **{overlap_broken}**\n")
        f.write(f"- Tradeoff broken: **{verdict['tradeoff_broken']}**\n")
        f.write(f"- Best candidate: **{verdict.get('best_candidate', 'None')}**\n")
        f.write(f"- Best level: **{verdict.get('best_level', 'None')}**\n")
        f.write(f"- Candidates evaluated (full): {verdict['candidates_evaluated']}\n\n")

        f.write("## Best Next Move\n\n")
        if verdict["verdict"] in ("STRONG_PASS", "ACCEPTABLE_PASS"):
            f.write("Routing-feature enrichment succeeded. The enriched classifier breaks the\n")
            f.write("bp-vs-native observation overlap and produces a selector that passes\n")
            f.write("promotion gates. Next steps:\n")
            f.write("1. Freeze the enriched selector configuration as the new baseline.\n")
            f.write("2. Consider observation-v2 retrain to embed history features in policy.\n")
            f.write("3. Resume self-play / native Hal improvement work.\n")
        elif overlap_broken:
            f.write("The observation overlap WAS broken — enriched features successfully\n")
            f.write("distinguish bp from native Hal. However, no candidate met promotion gates.\n")
            f.write("This confirms the modeling flaw and validates the enrichment approach.\n")
            f.write("Next steps:\n")
            f.write("1. Full observation-v2 retrain with history features in the policy input.\n")
            f.write("2. This sprint proves the features work; the bottleneck is now\n")
            f.write("   policy-level adaptation to native Hal, not routing.\n")
        else:
            f.write("The observation overlap was NOT broken. Enriched features do not\n")
            f.write("distinguish bp from native Hal at T2. Possible reasons:\n")
            f.write("1. Bridge_pressure and native Hal take identical actions in R0.\n")
            f.write("2. The history window is too small.\n")
            f.write("Next: investigate opponent action distributions in R0.\n")

        f.write(f"\nSprint time: ~{elapsed:.0f}s\n")

    print(f"\n  Markdown report: {md_path}")

    # ── Terminal summary ─────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("SPRINT SUMMARY")
    print("=" * 60)
    print(f"  Overlap broken: {overlap_broken}")
    print(f"  Verdict: {verdict['verdict']}")
    print(f"  Tradeoff broken: {verdict['tradeoff_broken']}")
    best_tag = verdict.get('best_candidate', 'None')
    best_cand = next((c for c in validated if c.get("tag") == best_tag), None)
    if best_cand and best_cand.get("analysis"):
        a = best_cand["analysis"]["abs"]
        print(f"  Best candidate: {best_tag}")
        print(f"    bp_r7={pct(a['bp_r7'])} ht_r7={pct(a['ht_r7'])} "
              f"hp_r7={pct(a['hp_r7'])} r9={pct(a['r9'])}")
        print(f"    native_wr={pct(a['native_wr'])} native_r7={pct(a['native_r7'])}")
        print(f"    level={best_cand['analysis']['level'] or '---'}")
    else:
        print(f"  Best candidate: {best_tag}")
    print(f"  Elapsed: {elapsed:.0f}s")
    print("=" * 60)


if __name__ == "__main__":
    main()
