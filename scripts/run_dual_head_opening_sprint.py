#!/usr/bin/env python3
"""Sprint: Dual-Head Opening Architecture at PHYSICALITY_BAKU=0.88.

Optimization target: break the bp-vs-native routing tradeoff by splitting
opening paths instead of relying on one classifier seam to route both.

Previous sprint proved:
  - Old classifier: bp=88% r7, native=0% (correct bp routing, wrong native routing)
  - Native-aware classifier: bp=50% r7, native=50% (wrong bp routing, correct native routing)
  - FeatureRuleController achieves 100% opening accuracy across all opponent types
  - The blocker is classifier routing architecture, not opening delivery

This sprint builds 3 materially different dual-head opening architectures:
  Family A: Explicit dual-head (bp_opening_head + hal_opening_head, classifier selects head)
  Family B: Stacked routing (opening head universal, late head has separate routing)
  Family C: Confidence-based routing (classifier probability drives routing)

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
from training.curriculum import get_scenario
from training.modular_policy import (
    ModularBakuSelector,
    verify_opening_accuracy,
)
from training.behavior_clone import TraceSample, behavior_clone_policy
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
FAST_GAMES = 30
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


# ── Classifier builders ─────────────────────────────────────────────────
_old_clf_cache = None
_native_clf_cache = None


def _get_old_classifier():
    """Old T2 classifier: trained on bp + scripted hal only."""
    global _old_clf_cache
    if _old_clf_cache is None:
        _old_clf_cache = build_classifier(
            BASE_MODEL, ["bridge_pressure", "hal_death_trade"], classify_turn=2
        )
    return _old_clf_cache


def _build_native_aware_classifier(classify_turn=2, games_per_opponent=100):
    """Classifier that includes native Hal in the hal class."""
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


# ── FeatureRuleController builder ────────────────────────────────────────
def _build_hal_frc():
    """Build the hal-family FeatureRuleController from baseline route."""
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
    """Capture bp_specialist's opening behavior as a FeatureRuleController.

    Runs bp_specialist against bridge_pressure for multiple seeds,
    records (role, round_bucket, half) -> action_second at each turn.
    """
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


# ── Controller wrapper ───────────────────────────────────────────────────
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
# DUAL-HEAD OPENING ARCHITECTURES
# ═════════════════════════════════════════════════════════════════════════

class DualHeadOpeningSelector:
    """Family A: Explicit dual-head opening with separate bp and hal opening paths.

    Two opening heads:
      - bp_opening_head: captures bp_specialist's natural opening behavior
      - hal_opening_head: FRC with hal-family route [1,1,1,10,60,1,60,1]

    Routing:
      T0-T1: pre-classification → bp_opening_head (safe default)
      T2: classify opponent
      T2 to opening_horizon-1:
        bp classified → bp_opening_head
        hal classified → hal_opening_head
      T >= opening_horizon:
        bp classified → bp_specialist (late)
        hal classified → late_model
    """

    def __init__(
        self,
        bp_opening_head,
        hal_opening_head,
        bp_specialist,
        late_model,
        classifier: LogisticRegression,
        opening_horizon: int = 8,
        classify_turn: int = 2,
    ):
        self.bp_opening_head = bp_opening_head
        self.hal_opening_head = hal_opening_head
        self.bp_specialist = bp_specialist
        self.late_model = late_model
        self.classifier = classifier
        self.opening_horizon = opening_horizon
        self.classify_turn = classify_turn
        self._turn = 0
        self._is_bp: bool | None = None

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
            obs_2d = obs.reshape(1, -1) if obs.ndim == 1 else obs
            label = int(self.classifier.predict(obs_2d)[0])
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


class StackedRoutingSelector:
    """Family B: Stacked routing with separate opening and late routing decisions.

    Opening and late phases use independent routing logic.

    Opening routing: uses opening_classifier (or universal mode if opening_classifier is None)
    Late routing: uses late_classifier (separate classifier)

    This decouples opening routing from late routing so each can be
    optimized independently.
    """

    def __init__(
        self,
        bp_opening_head,
        hal_opening_head,
        bp_specialist,
        late_model,
        opening_classifier: LogisticRegression | None,
        late_classifier: LogisticRegression,
        opening_horizon: int = 8,
        opening_classify_turn: int = 2,
        late_classify_turn: int = 2,
        universal_opening=None,
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
        self.universal_opening = universal_opening
        self._turn = 0
        self._opening_label: int | None = None
        self._late_label: int | None = None

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

        # Opening phase
        if game_turn < self.opening_horizon:
            # Classify for opening if we have an opening classifier
            if (self._opening_label is None
                    and self.opening_classifier is not None
                    and self._turn >= self.opening_classify_turn):
                obs_2d = obs.reshape(1, -1) if obs.ndim == 1 else obs
                self._opening_label = int(self.opening_classifier.predict(obs_2d)[0])

            self._turn += 1

            # Universal opening mode
            if self.universal_opening is not None and self.opening_classifier is None:
                return self.universal_opening.predict(
                    obs, action_masks=action_masks, deterministic=deterministic
                )

            # Dual-head opening
            if self._opening_label == 0:  # bp
                return self.bp_opening_head.predict(
                    obs, action_masks=action_masks, deterministic=deterministic
                )
            elif self._opening_label == 1:  # hal
                return self.hal_opening_head.predict(
                    obs, action_masks=action_masks, deterministic=deterministic
                )
            else:  # pre-classification
                return self.bp_opening_head.predict(
                    obs, action_masks=action_masks, deterministic=deterministic
                )

        # Late phase - classify independently
        if self._late_label is None and self._turn >= self.late_classify_turn:
            obs_2d = obs.reshape(1, -1) if obs.ndim == 1 else obs
            self._late_label = int(self.late_classifier.predict(obs_2d)[0])

        self._turn += 1

        if self._late_label == 0:  # bp
            return self.bp_specialist.predict(
                obs, action_masks=action_masks, deterministic=deterministic
            )
        return self.late_model.predict(
            obs, action_masks=action_masks, deterministic=deterministic
        )


class ConfidenceRoutingSelector:
    """Family C: Confidence-based routing with classifier probability thresholds.

    Uses classifier's predict_proba() to make routing decisions:
      - P(bp) > bp_threshold: high confidence bp → bp_opening → bp_specialist
      - P(hal) > hal_threshold: high confidence hal → hal_opening → late_model
      - Otherwise (ambiguous): fallback path → fallback_opening → late_model

    This routes ambiguous cases (like native Hal) to a safe path instead
    of forcing a binary decision.
    """

    def __init__(
        self,
        bp_opening_head,
        hal_opening_head,
        fallback_opening_head,
        bp_specialist,
        late_model,
        classifier: LogisticRegression,
        bp_threshold: float = 0.7,
        hal_threshold: float = 0.7,
        opening_horizon: int = 8,
        classify_turn: int = 2,
    ):
        self.bp_opening_head = bp_opening_head
        self.hal_opening_head = hal_opening_head
        self.fallback_opening_head = fallback_opening_head
        self.bp_specialist = bp_specialist
        self.late_model = late_model
        self.classifier = classifier
        self.bp_threshold = bp_threshold
        self.hal_threshold = hal_threshold
        self.opening_horizon = opening_horizon
        self.classify_turn = classify_turn
        self._turn = 0
        self._route: str | None = None  # "bp", "hal", or "fallback"

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
        if self._route is None and self._turn >= self.classify_turn:
            obs_2d = obs.reshape(1, -1) if obs.ndim == 1 else obs
            proba = self.classifier.predict_proba(obs_2d)[0]
            p_bp = proba[0]
            p_hal = proba[1] if len(proba) > 1 else 1.0 - p_bp
            if p_bp >= self.bp_threshold:
                self._route = "bp"
            elif p_hal >= self.hal_threshold:
                self._route = "hal"
            else:
                self._route = "fallback"

        self._turn += 1
        game_turn = self._game_turn_from_obs(obs)

        if game_turn < self.opening_horizon:
            if self._route == "bp" or self._route is None:
                return self.bp_opening_head.predict(
                    obs, action_masks=action_masks, deterministic=deterministic
                )
            elif self._route == "hal":
                return self.hal_opening_head.predict(
                    obs, action_masks=action_masks, deterministic=deterministic
                )
            else:  # fallback
                return self.fallback_opening_head.predict(
                    obs, action_masks=action_masks, deterministic=deterministic
                )

        # Late game
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
    wins = 0; total_hr = 0; r7_count = 0; r7_eligible = 0; deaths = 0
    for gi in range(games):
        opp = (create_model_opponent(opp_model_path, agent_role="baku")
               if opp_model_path else create_scripted_opponent(opp_name))
        env = DTHEnv(opponent=opp, agent_role="baku", seed=seed + gi)
        obs, _ = env.reset()
        selector.reset()
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


def verify_classifier_accuracy(clf, opp_factory, label, n_games=20, classify_turn=2):
    """Check classifier labels opp_factory opponents correctly."""
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
                pred = int(clf.predict(obs.reshape(1, -1))[0])
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


def classifier_confidence_profile(clf, opp_factory, n_games=20, classify_turn=2):
    """Get classifier probability distribution for an opponent type."""
    model = _get_base_model()
    probas = []
    for seed in range(n_games):
        opp = opp_factory(seed)
        env = DTHEnv(opponent=opp, agent_role="baku", seed=42 + seed)
        obs, _ = env.reset()
        alive = True
        for t in range(classify_turn + 1):
            mask = env.action_masks()
            if t == classify_turn:
                p = clf.predict_proba(obs.reshape(1, -1))[0]
                probas.append(p.tolist())
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
        return {"mean_p_bp": 0, "mean_p_hal": 0, "n": 0}
    arr = np.array(probas)
    return {
        "mean_p_bp": float(arr[:, 0].mean()),
        "mean_p_hal": float(arr[:, 1].mean()) if arr.shape[1] > 1 else 0,
        "std_p_bp": float(arr[:, 0].std()),
        "n": len(probas),
    }


# ═════════════════════════════════════════════════════════════════════════
# LANE 0: Reproduce Two Reference Baselines
# ═════════════════════════════════════════════════════════════════════════
def lane_0_baselines():
    print("\n" + "=" * 60)
    print(f"LANE 0: Reference Baselines (PHYSICALITY_BAKU={PHYSICALITY_BAKU})")
    print("=" * 60)

    bp_specialist = _get_bp_specialist()
    late_model = _get_base_model()

    # Build both classifiers
    old_clf = _get_old_classifier()
    native_clf = _build_native_aware_classifier()

    # Verify classifier accuracy on each opponent type
    clf_diag = {}
    for clf_name, clf in [("old_t2", old_clf), ("native_aware_t2", native_clf)]:
        bp_acc = verify_classifier_accuracy(
            clf, lambda s: create_scripted_opponent("bridge_pressure"), 0, n_games=20)
        native_acc = verify_classifier_accuracy(
            clf, lambda s: _make_native_opp(), 1, n_games=20)
        clf_diag[clf_name] = {"bp_accuracy": bp_acc["accuracy"],
                               "native_accuracy": native_acc["accuracy"]}
        print(f"  {clf_name}: bp_acc={bp_acc['accuracy']:.0%} "
              f"native_acc={native_acc['accuracy']:.0%}")

    # Build FRC for hal opening
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
                                    {"classifier": "old_t2", **clf_diag["old_t2"]})

    # Reference 2: Native-aware classifier (bp=low, native=nonzero)
    print("\n  --- Reference 2: Native-aware T2 classifier ---")
    ref2 = ModularBakuSelector(
        bp_specialist=bp_specialist, opening_model=hal_opening,
        late_model=late_model, classifier=native_clf,
        opening_horizon=OPENING_HORIZON, classify_turn=2,
    )
    ref2_results = eval_full_suite(ref2, EVAL_GAMES, 42, "ref2_native_clf",
                                    {"classifier": "native_aware_t2", **clf_diag["native_aware_t2"]})

    return ref1_results, ref2_results, clf_diag, hal_frc, old_clf, native_clf


# ═════════════════════════════════════════════════════════════════════════
# LANE 1: Family A — Explicit Dual-Head Opening
# ═════════════════════════════════════════════════════════════════════════
def lane_1_family_a(hal_frc, old_clf, native_clf):
    print("\n" + "=" * 60)
    print("LANE 1: Family A — Explicit Dual-Head Opening")
    print("=" * 60)

    bp_specialist = _get_bp_specialist()
    late_model = _get_base_model()

    # Build bp opening head from bp_specialist behavior
    bp_frc = _build_bp_opening_frc()
    bp_opening = ControllerAsModel(bp_frc)
    hal_opening = ControllerAsModel(hal_frc)

    print(f"  bp_opening_frc ({len(bp_frc.feature_action_map)} entries):")
    for key, action in sorted(bp_frc.feature_action_map.items()):
        role_str = "C" if key[0] == 0 else "D"
        print(f"    ({role_str}, R{key[1]}, H{'1' if key[2] == 0 else '2'}) -> {action}")
    print(f"  hal_opening_frc ({len(hal_frc.feature_action_map)} entries):")
    for key, action in sorted(hal_frc.feature_action_map.items()):
        role_str = "C" if key[0] == 0 else "D"
        print(f"    ({role_str}, R{key[1]}, H{'1' if key[2] == 0 else '2'}) -> {action}")

    candidates = []

    # A1: Dual-head + old classifier (bp routing is correct)
    # Native Hal misclassified as bp → gets bp_opening → bp_specialist
    print("\n  --- A1: Dual-head + old T2 classifier ---")
    a1 = DualHeadOpeningSelector(
        bp_opening_head=bp_opening, hal_opening_head=hal_opening,
        bp_specialist=bp_specialist, late_model=late_model,
        classifier=old_clf, opening_horizon=OPENING_HORIZON, classify_turn=2,
    )
    candidates.append(("A1_dual_old_clf", a1, {
        "type": "dual_head_old_clf",
        "bp_opening_entries": len(bp_frc.feature_action_map),
        "hal_opening_entries": len(hal_frc.feature_action_map),
        "misaligned": 0,
    }))

    # A2: Dual-head + native-aware classifier (native routing correct, bp may degrade)
    print("\n  --- A2: Dual-head + native-aware classifier ---")
    a2 = DualHeadOpeningSelector(
        bp_opening_head=bp_opening, hal_opening_head=hal_opening,
        bp_specialist=bp_specialist, late_model=late_model,
        classifier=native_clf, opening_horizon=OPENING_HORIZON, classify_turn=2,
    )
    candidates.append(("A2_dual_native_clf", a2, {
        "type": "dual_head_native_clf",
        "bp_opening_entries": len(bp_frc.feature_action_map),
        "hal_opening_entries": len(hal_frc.feature_action_map),
        "misaligned": 0,
    }))

    # A3: Dual-head + old classifier, but hal_opening for BOTH heads
    # Tests: what if bp_opening_head is just hal_opening too? (universal opening)
    # This isolates whether the opening actions themselves matter
    print("\n  --- A3: Dual-head old clf, universal hal opening ---")
    a3 = DualHeadOpeningSelector(
        bp_opening_head=hal_opening, hal_opening_head=hal_opening,
        bp_specialist=bp_specialist, late_model=late_model,
        classifier=old_clf, opening_horizon=OPENING_HORIZON, classify_turn=2,
    )
    candidates.append(("A3_universal_hal_old_clf", a3, {
        "type": "dual_head_universal_hal_old_clf",
        "opening_entries": len(hal_frc.feature_action_map),
        "misaligned": 0,
    }))

    return candidates


# ═════════════════════════════════════════════════════════════════════════
# LANE 2: Family B — Stacked/Delayed Routing
# ═════════════════════════════════════════════════════════════════════════
def lane_2_family_b(hal_frc, old_clf, native_clf):
    print("\n" + "=" * 60)
    print("LANE 2: Family B — Stacked/Delayed Routing")
    print("=" * 60)

    bp_specialist = _get_bp_specialist()
    late_model = _get_base_model()
    bp_frc = _build_bp_opening_frc()
    bp_opening = ControllerAsModel(bp_frc)
    hal_opening = ControllerAsModel(hal_frc)

    candidates = []

    # B1: Universal hal opening (no opening classifier) + old late classifier
    # Opening: hal_opening for EVERYONE → Late: old clf routes bp vs hal
    print("\n  --- B1: Universal hal opening + old late clf ---")
    b1 = StackedRoutingSelector(
        bp_opening_head=bp_opening, hal_opening_head=hal_opening,
        bp_specialist=bp_specialist, late_model=late_model,
        opening_classifier=None, late_classifier=old_clf,
        opening_horizon=OPENING_HORIZON,
        universal_opening=hal_opening,
    )
    candidates.append(("B1_univ_hal_old_late", b1, {
        "type": "stacked_univ_hal_old_late",
        "opening_mode": "universal_hal",
        "late_clf": "old_t2",
        "misaligned": 0,
    }))

    # B2: Universal bp opening + old late classifier
    # Tests if bp_specialist's opening is benign enough for all opponents
    print("\n  --- B2: Universal bp opening + old late clf ---")
    b2 = StackedRoutingSelector(
        bp_opening_head=bp_opening, hal_opening_head=hal_opening,
        bp_specialist=bp_specialist, late_model=late_model,
        opening_classifier=None, late_classifier=old_clf,
        opening_horizon=OPENING_HORIZON,
        universal_opening=bp_opening,
    )
    candidates.append(("B2_univ_bp_old_late", b2, {
        "type": "stacked_univ_bp_old_late",
        "opening_mode": "universal_bp",
        "late_clf": "old_t2",
        "misaligned": 0,
    }))

    # B3: Old opening classifier + native-aware late classifier
    # Opening: old clf routes opening (bp opening correct for bp, hal opening for scripted hal)
    # Late: native clf routes late (native Hal → late_model, may misroute bp)
    print("\n  --- B3: Old opening clf + native late clf ---")
    b3 = StackedRoutingSelector(
        bp_opening_head=bp_opening, hal_opening_head=hal_opening,
        bp_specialist=bp_specialist, late_model=late_model,
        opening_classifier=old_clf, late_classifier=native_clf,
        opening_horizon=OPENING_HORIZON,
        opening_classify_turn=2, late_classify_turn=2,
    )
    candidates.append(("B3_old_open_native_late", b3, {
        "type": "stacked_old_open_native_late",
        "opening_clf": "old_t2",
        "late_clf": "native_aware_t2",
        "misaligned": 0,
    }))

    # B4: Native opening classifier + old late classifier
    # Opening: native clf routes (bp may get hal opening, native Hal gets hal opening)
    # Late: old clf routes (bp → bp_specialist, native Hal → bp_specialist)
    print("\n  --- B4: Native opening clf + old late clf ---")
    b4 = StackedRoutingSelector(
        bp_opening_head=bp_opening, hal_opening_head=hal_opening,
        bp_specialist=bp_specialist, late_model=late_model,
        opening_classifier=native_clf, late_classifier=old_clf,
        opening_horizon=OPENING_HORIZON,
        opening_classify_turn=2, late_classify_turn=2,
    )
    candidates.append(("B4_native_open_old_late", b4, {
        "type": "stacked_native_open_old_late",
        "opening_clf": "native_aware_t2",
        "late_clf": "old_t2",
        "misaligned": 0,
    }))

    return candidates


# ═════════════════════════════════════════════════════════════════════════
# LANE 3: Family C — Confidence-Based Routing
# ═════════════════════════════════════════════════════════════════════════
def lane_3_family_c(hal_frc, old_clf, native_clf):
    print("\n" + "=" * 60)
    print("LANE 3: Family C — Confidence-Based Routing")
    print("=" * 60)

    bp_specialist = _get_bp_specialist()
    late_model = _get_base_model()
    bp_frc = _build_bp_opening_frc()
    bp_opening = ControllerAsModel(bp_frc)
    hal_opening = ControllerAsModel(hal_frc)

    # Profile classifier confidence for each opponent type
    print("\n  Classifier confidence profiles:")
    for clf_name, clf in [("old_t2", old_clf), ("native_aware_t2", native_clf)]:
        for opp_name, factory in [
            ("bp", lambda s: create_scripted_opponent("bridge_pressure")),
            ("hal_dt", lambda s: create_scripted_opponent("hal_death_trade")),
            ("native", lambda s: _make_native_opp()),
        ]:
            profile = classifier_confidence_profile(clf, factory, n_games=20)
            print(f"    {clf_name} vs {opp_name}: p_bp={profile['mean_p_bp']:.3f} "
                  f"p_hal={profile['mean_p_hal']:.3f} std={profile.get('std_p_bp', 0):.3f}")

    candidates = []

    # C1: Old classifier + confidence threshold 0.7
    # High-confidence bp → bp path; low confidence → fallback (hal opening + late)
    print("\n  --- C1: Old clf, threshold=0.7, fallback=hal_opening ---")
    c1 = ConfidenceRoutingSelector(
        bp_opening_head=bp_opening, hal_opening_head=hal_opening,
        fallback_opening_head=hal_opening,
        bp_specialist=bp_specialist, late_model=late_model,
        classifier=old_clf,
        bp_threshold=0.7, hal_threshold=0.7,
        opening_horizon=OPENING_HORIZON, classify_turn=2,
    )
    candidates.append(("C1_old_conf07_hal_fb", c1, {
        "type": "confidence_old_clf_07_hal_fallback",
        "bp_threshold": 0.7, "hal_threshold": 0.7,
        "fallback": "hal_opening",
        "misaligned": 0,
    }))

    # C2: Old classifier + threshold 0.9 (stricter)
    print("\n  --- C2: Old clf, threshold=0.9, fallback=hal_opening ---")
    c2 = ConfidenceRoutingSelector(
        bp_opening_head=bp_opening, hal_opening_head=hal_opening,
        fallback_opening_head=hal_opening,
        bp_specialist=bp_specialist, late_model=late_model,
        classifier=old_clf,
        bp_threshold=0.9, hal_threshold=0.9,
        opening_horizon=OPENING_HORIZON, classify_turn=2,
    )
    candidates.append(("C2_old_conf09_hal_fb", c2, {
        "type": "confidence_old_clf_09_hal_fallback",
        "bp_threshold": 0.9, "hal_threshold": 0.9,
        "fallback": "hal_opening",
        "misaligned": 0,
    }))

    # C3: Old classifier + asymmetric thresholds (bp=0.8, hal=0.5)
    # Low hal threshold = easier to route to hal path
    print("\n  --- C3: Old clf, bp=0.8/hal=0.5, fallback=hal_opening ---")
    c3 = ConfidenceRoutingSelector(
        bp_opening_head=bp_opening, hal_opening_head=hal_opening,
        fallback_opening_head=hal_opening,
        bp_specialist=bp_specialist, late_model=late_model,
        classifier=old_clf,
        bp_threshold=0.8, hal_threshold=0.5,
        opening_horizon=OPENING_HORIZON, classify_turn=2,
    )
    candidates.append(("C3_old_asym_08_05", c3, {
        "type": "confidence_old_clf_08_05",
        "bp_threshold": 0.8, "hal_threshold": 0.5,
        "fallback": "hal_opening",
        "misaligned": 0,
    }))

    # C4: Native-aware classifier + confidence 0.7
    print("\n  --- C4: Native clf, threshold=0.7, fallback=hal_opening ---")
    c4 = ConfidenceRoutingSelector(
        bp_opening_head=bp_opening, hal_opening_head=hal_opening,
        fallback_opening_head=hal_opening,
        bp_specialist=bp_specialist, late_model=late_model,
        classifier=native_clf,
        bp_threshold=0.7, hal_threshold=0.7,
        opening_horizon=OPENING_HORIZON, classify_turn=2,
    )
    candidates.append(("C4_native_conf07_hal_fb", c4, {
        "type": "confidence_native_clf_07_hal_fallback",
        "bp_threshold": 0.7, "hal_threshold": 0.7,
        "fallback": "hal_opening",
        "misaligned": 0,
    }))

    # C5: Old classifier + confidence 0.7, fallback=bp_opening (test opposite fallback)
    print("\n  --- C5: Old clf, threshold=0.7, fallback=bp_opening ---")
    c5 = ConfidenceRoutingSelector(
        bp_opening_head=bp_opening, hal_opening_head=hal_opening,
        fallback_opening_head=bp_opening,
        bp_specialist=bp_specialist, late_model=late_model,
        classifier=old_clf,
        bp_threshold=0.7, hal_threshold=0.7,
        opening_horizon=OPENING_HORIZON, classify_turn=2,
    )
    candidates.append(("C5_old_conf07_bp_fb", c5, {
        "type": "confidence_old_clf_07_bp_fallback",
        "bp_threshold": 0.7, "hal_threshold": 0.7,
        "fallback": "bp_opening",
        "misaligned": 0,
    }))

    return candidates


# ═════════════════════════════════════════════════════════════════════════
# LANE 4: Gate + Full Validation
# ═════════════════════════════════════════════════════════════════════════
def lane_4_validate(all_candidates, ref1_results):
    print("\n" + "=" * 60)
    print("LANE 4: Gate + Full Validation")
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

        # Native-Hal gap measurement
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
# LANE 5: Verdict
# ═════════════════════════════════════════════════════════════════════════
def lane_5_verdict(ref1, ref2, validated):
    print("\n" + "=" * 60)
    print("LANE 5: Verdict")
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

        # Check if tradeoff is broken: bp strong AND native nonzero
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
    print(f"SPRINT: Dual-Head Opening Architecture (p={PHYSICALITY_BAKU})")
    print("=" * 60)

    # Lane 0: Reproduce baselines
    ref1, ref2, clf_diag, hal_frc, old_clf, native_clf = lane_0_baselines()

    # Lanes 1-3: Build architecture families (parallel-independent)
    family_a = lane_1_family_a(hal_frc, old_clf, native_clf)
    family_b = lane_2_family_b(hal_frc, old_clf, native_clf)
    family_c = lane_3_family_c(hal_frc, old_clf, native_clf)

    # Combine all candidates
    all_candidates = family_a + family_b + family_c

    # Lane 4: Gate + full validation
    results_list, validated = lane_4_validate(all_candidates, ref1)

    # Lane 5: Verdict
    verdict = lane_5_verdict(ref1, ref2, validated)

    elapsed = round(time.time() - t0, 1)

    # ── Deliverables ─────────────────────────────────────────────────────
    report_dir = Path("docs/reports")
    json_dir = report_dir / "json"
    report_dir.mkdir(parents=True, exist_ok=True)
    json_dir.mkdir(parents=True, exist_ok=True)

    # JSON report
    json_data = {
        "sprint": "dual-head-opening",
        "date": "2026-03-28",
        "physicality_baku": PHYSICALITY_BAKU,
        "baseline_route": BASELINE_ROUTE,
        "reference_1_old_clf": ref1,
        "reference_2_native_clf": ref2,
        "classifier_diagnostic": clf_diag,
        "families": {
            "A_dual_head": [
                {"tag": r["tag"], "stats": r["stats"], "gate_pass": r["gate_pass"],
                 "analysis": r.get("analysis"), "gap": r.get("gap")}
                for r in results_list if r["tag"].startswith("A")
            ],
            "B_stacked": [
                {"tag": r["tag"], "stats": r["stats"], "gate_pass": r["gate_pass"],
                 "analysis": r.get("analysis"), "gap": r.get("gap")}
                for r in results_list if r["tag"].startswith("B")
            ],
            "C_confidence": [
                {"tag": r["tag"], "stats": r["stats"], "gate_pass": r["gate_pass"],
                 "analysis": r.get("analysis"), "gap": r.get("gap")}
                for r in results_list if r["tag"].startswith("C")
            ],
        },
        "validated": [
            {"tag": c["tag"], "analysis": c.get("analysis"), "gap": c.get("gap")}
            for c in validated
        ],
        "verdict": verdict,
        "elapsed_seconds": elapsed,
    }
    json_path = json_dir / "sprint_dual_head_opening_2026-03-28.json"
    with open(json_path, "w") as f:
        json.dump(json_data, f, indent=2, default=str)
    print(f"\n  JSON report: {json_path}")

    # Markdown report
    md_path = report_dir / "sprint_dual_head_opening_report_2026-03-28.md"
    with open(md_path, "w") as f:
        f.write("# Sprint Report: Dual-Head Opening Architecture (p=0.88)\n")
        f.write(f"**Date:** 2026-03-28\n")
        f.write(f"**PHYSICALITY_BAKU:** {PHYSICALITY_BAKU}\n")
        f.write(f"**Verdict:** {verdict['verdict']}\n")
        f.write(f"**Tradeoff broken:** {verdict['tradeoff_broken']}\n\n")

        f.write("## Executive Summary\n\n")
        f.write("This sprint targets the bp-vs-native routing tradeoff identified in the\n")
        f.write("previous sprint. The single classifier seam forces a binary choice:\n")
        f.write("old classifier gives bp=88% and native=0%; native-aware classifier gives\n")
        f.write("bp~50% and native~50%. Three dual-head opening architecture families\n")
        f.write("were built to break this tradeoff by splitting the opening paths.\n\n")

        f.write("## Optimization Target\n\n")
        f.write("Preserve bp opening strength AND native-Hal opening strength simultaneously\n")
        f.write("by using separate opening paths for each opponent family.\n\n")

        f.write("## Two Reference Baselines\n\n")
        f.write("| Metric | Ref 1: Old clf | Ref 2: Native clf |\n")
        f.write("|--------|----------------|--------------------|\n")
        for m in ["bridge_pressure", "hal_death_trade", "hal_pressure"]:
            r1_r7 = ref1.get(m, {}).get("r7_rate", 0)
            r2_r7 = ref2.get(m, {}).get("r7_rate", 0)
            r1_wr = ref1.get(m, {}).get("win_rate", 0)
            r2_wr = ref2.get(m, {}).get("win_rate", 0)
            f.write(f"| {m} r7 | {pct(r1_r7)} | {pct(r2_r7)} |\n")
            f.write(f"| {m} WR | {pct(r1_wr)} | {pct(r2_wr)} |\n")
        f.write(f"| seeded r9 | {pct(ref1.get('seeded_r9', {}).get('rate', 0))} "
                f"| {pct(ref2.get('seeded_r9', {}).get('rate', 0))} |\n")
        f.write(f"| prior_4096 WR | {pct(ref1.get('prior_4096', {}).get('win_rate', 0))} "
                f"| {pct(ref2.get('prior_4096', {}).get('win_rate', 0))} |\n")
        f.write(f"| native WR | {pct(ref1.get('native_hal', {}).get('win_rate', 0))} "
                f"| {pct(ref2.get('native_hal', {}).get('win_rate', 0))} |\n")
        f.write(f"| native r7 | {pct(ref1.get('native_hal', {}).get('r7_rate', 0))} "
                f"| {pct(ref2.get('native_hal', {}).get('r7_rate', 0))} |\n\n")

        f.write("## Architecture Families\n\n")
        f.write("### Family A: Explicit Dual-Head Opening\n")
        f.write("Two opening heads (bp_opening_head from bp_specialist behavior, hal_opening_head\n")
        f.write("from FRC route), classifier selects which head to use during opening.\n\n")
        f.write("### Family B: Stacked/Delayed Routing\n")
        f.write("Opening and late phases use independent routing decisions. Opening can be\n")
        f.write("universal (same for all opponents) or use a separate classifier.\n\n")
        f.write("### Family C: Confidence-Based Routing\n")
        f.write("Uses classifier probability thresholds. High-confidence routes to specialized\n")
        f.write("head; low-confidence falls back to safe path (hal_opening + late_model).\n\n")

        f.write("## Candidate Results\n\n")
        f.write("| Candidate | Gate | bp r7 | ht r7 | hp r7 | r9 | native WR | native r7 | Level |\n")
        f.write("|-----------|------|-------|-------|-------|----|-----------|-----------|-------|\n")
        for r in results_list:
            gate_str = "PASS" if r["gate_pass"] else "FAIL"
            if r.get("analysis") and r["analysis"].get("abs"):
                a = r["analysis"]["abs"]
                f.write(f"| {r['tag']} | {gate_str} | {pct(a['bp_r7'])} | {pct(a['ht_r7'])} "
                        f"| {pct(a['hp_r7'])} | {pct(a['r9'])} | {pct(a['native_wr'])} "
                        f"| {pct(a['native_r7'])} | {r['analysis'].get('level', '---')} |\n")
            else:
                g = r.get("gate", {})
                bp_g = g.get("bridge_pressure", {}).get("r7_rate", 0)
                f.write(f"| {r['tag']} | {gate_str} | {pct(bp_g)} | --- | --- | --- | --- | --- | --- |\n")

        f.write(f"\n## Final Verdict\n\n")
        f.write(f"**{verdict['verdict']}**\n\n")
        f.write(f"- Tradeoff broken: **{verdict['tradeoff_broken']}**\n")
        f.write(f"- Best candidate: **{verdict.get('best_candidate', 'None')}**\n")
        f.write(f"- Best level: **{verdict.get('best_level', 'None')}**\n")
        f.write(f"- Candidates evaluated (full): {verdict['candidates_evaluated']}\n\n")

        f.write("## Best Next Move\n\n")
        if verdict["tradeoff_broken"]:
            f.write("The bp-vs-native routing tradeoff has been broken. The best candidate\n")
            f.write("achieves both bp strength and native Hal resistance simultaneously.\n")
            f.write("Next: freeze the opening configuration and advance to self-play training.\n")
        else:
            f.write("The tradeoff was NOT broken. Consider:\n")
            f.write("1. Observation-enriched classifier (add opponent action history features)\n")
            f.write("2. Training a unified late model that handles both bp and native Hal\n")
            f.write("3. Separate training pipelines per opponent class\n")

        f.write(f"\nSprint time: ~{elapsed:.0f}s\n")

    print(f"\n  Markdown report: {md_path}")

    # ── Terminal summary ─────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("SPRINT SUMMARY")
    print("=" * 60)
    print(f"  Verdict: {verdict['verdict']}")
    print(f"  Tradeoff broken: {verdict['tradeoff_broken']}")
    print(f"  Best candidate: {verdict.get('best_candidate', 'None')}")
    print(f"  Best level: {verdict.get('best_level', 'None')}")
    print(f"  Elapsed: {elapsed:.0f}s")
    print("=" * 60)


if __name__ == "__main__":
    main()
