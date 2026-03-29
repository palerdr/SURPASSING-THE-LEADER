#!/usr/bin/env python3
"""Sprint: Opening vs Native Hal at PHYSICALITY_BAKU=0.88.

Optimization target: Improve Baku's OPENING response (T0-T7) against the
strongest scaffold-native Hal while preserving corrected scripted baseline.

The prior sprint showed: Baku WR=0%, r7_rate=0% vs native_hal_16k.
Late-head tuning produced NO_IMPROVEMENT across 3 ladder families.
This sprint tests whether opening adaptation can break the 0% ceiling.

Lanes:
  0: Reproduce corrected p=0.88 baseline (scripted + prior + native)
  1: Opening failure diagnosis vs native Hal (trace games)
  2: Counterfactual audit T0-T7 vs native Hal
  3: Three opening-improvement families:
     A – Counterfactual-guided route construction
     B – Population-based discrete opening evolution
     C – Greedy coordinate-descent per-turn optimization
  4: Full validation of serious candidates (50-game)
  5: Verdict + deliverables

Promotion (strong):  bp>=0.84, ht>=0.48, hp>=0.48, r9=1.0, mis=0,
  AND (native_wr delta >= +12pp OR native_r7_rate: 0%->>=20%).
Promotion (acceptable): bp>=0.82, ht>=0.48, hp>=0.48, r9=1.0, mis=0,
  AND (native_wr delta >= +6pp OR native_r7_rate: 0%->>=10%).
"""

from __future__ import annotations

import sys, os, json, time, collections, itertools
from pathlib import Path
from dataclasses import dataclass, asdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from sb3_contrib import MaskablePPO

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
    build_modular_selector,
    collect_opening_samples,
    verify_opening_accuracy,
)
from training.behavior_clone import behavior_clone_policy
from training.expert_selector import build_classifier


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
CF_GAMES = 30
EVO_GAMES = 20
HAL_OPPONENTS = ["hal_death_trade", "hal_pressure"]
SCRIPTED_OPPONENTS = ["bridge_pressure", "hal_death_trade", "hal_pressure"]

CF_ACTIONS = [1, 5, 10, 15, 20, 30, 40, 50, 55, 60]
TURN_ROLES = ["C", "D", "C", "D", "C", "D", "C", "D"]

STRONG_PASS = {
    "bp_r7": 0.84, "ht_r7": 0.48, "hp_r7": 0.48,
    "seeded_r9": 1.0, "misaligned": 0,
    "native_wr_delta": 0.12, "native_r7_rate": 0.20,
}
ACCEPTABLE_PASS = {
    "bp_r7": 0.82, "ht_r7": 0.48, "hp_r7": 0.48,
    "seeded_r9": 1.0, "misaligned": 0,
    "native_wr_delta": 0.06, "native_r7_rate": 0.10,
}


# ── Model caching ────────────────────────────────────────────────────────
_base_model_cache = None
_native_hal_cache = None


def _get_base_model():
    global _base_model_cache
    if _base_model_cache is None:
        _base_model_cache = MaskablePPO.load(BASE_MODEL)
    return _base_model_cache


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


def _make_fast_native_opp():
    return _CachedHalOpponent(_get_native_hal())


# ── Evaluation helpers ───────────────────────────────────────────────────
def pct(v):
    return f"{v*100:.0f}%"


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


def eval_route_vs_native(overrides, n_games=CF_GAMES, base_seed=42):
    """Fast eval: opening overrides + base model late vs cached native Hal."""
    model = _get_base_model()
    wins = 0; total = 0; r7_count = 0; total_hr = 0
    for gi in range(n_games):
        opp = _make_fast_native_opp()
        env = DTHEnv(opponent=opp, agent_role="baku", seed=base_seed + gi)
        obs, _ = env.reset()
        done = False
        reached_r7 = current_route_stage_flags(env.game).get("round7_pressure", False)
        last_reward = 0.0

        for t in range(OPENING_HORIZON):
            if done:
                break
            mask = env.action_masks()
            action_idx = (overrides[t] - 1) if t in overrides else int(
                model.predict(obs, action_masks=mask, deterministic=True)[0])
            obs, last_reward, term, trunc, _ = env.step(action_idx)
            reached_r7 = reached_r7 or current_route_stage_flags(env.game).get(
                "round7_pressure", False)
            if term or trunc:
                done = True

        while not done:
            mask = env.action_masks()
            action_idx = int(model.predict(obs, action_masks=mask, deterministic=True)[0])
            obs, last_reward, term, trunc, _ = env.step(action_idx)
            reached_r7 = reached_r7 or current_route_stage_flags(env.game).get(
                "round7_pressure", False)
            if term or trunc:
                done = True

        total += 1
        r7_count += int(reached_r7)
        total_hr += len(env.game.history)
        won = bool(term and env.game.winner is env.agent and last_reward > 0)
        wins += int(won)

    return {
        "wins": wins, "games": total,
        "win_rate": wins / total if total else 0,
        "r7_count": r7_count, "r7_rate": r7_count / total if total else 0,
        "avg_hr": total_hr / total if total else 0,
    }


def score_native(res):
    """Composite score for ranking routes against native Hal."""
    return res["win_rate"] * 2 + res["r7_rate"] + res["avg_hr"] / 30


# ── Opening model builder ───────────────────────────────────────────────
_opening_model_cache = {}


def build_opening_model_from_overrides(overrides):
    route_key = tuple(sorted(overrides.items()))
    if route_key in _opening_model_cache:
        return _opening_model_cache[route_key]
    samples = collect_opening_samples(
        BASE_MODEL, HAL_OPPONENTS, overrides, OPENING_HORIZON, seeds=range(200),
    )
    env = DTHEnv(
        opponent=create_scripted_opponent("hal_death_trade"),
        agent_role="baku", seed=0,
    )
    opening_model = MaskablePPO("MlpPolicy", env, verbose=0, seed=42)
    behavior_clone_policy(
        opening_model, samples, epochs=80, batch_size=64, learning_rate=1e-3,
    )
    accuracy = verify_opening_accuracy(opening_model, samples)
    bc_stats = {
        "accuracy": accuracy["accuracy"],
        "misaligned": accuracy["total"] - accuracy["correct"],
        "correct": accuracy["correct"],
        "total": accuracy["total"],
    }
    result = (opening_model, bc_stats)
    _opening_model_cache[route_key] = result
    return result


def build_selector_for_route(overrides):
    opening_model, bc_stats = build_opening_model_from_overrides(overrides)
    late_model = MaskablePPO.load(BASE_MODEL)
    selector = build_modular_selector(
        bp_specialist_path=BP_SPECIALIST,
        base_model_path=BASE_MODEL,
        opening_model=opening_model,
        late_model=late_model,
        opening_horizon=OPENING_HORIZON,
    )
    return selector, bc_stats


def eval_full_suite(selector, games, seed, tag, bc_stats):
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
    results["bc_stats"] = bc_stats
    return results


# ══════════════════════════════════════════════════════════════════════════
# LANE 0: Baseline Reproduction
# ══════════════════════════════════════════════════════════════════════════
def lane_0_baseline():
    print("\n" + "=" * 60)
    print(f"LANE 0: p=0.88 Baseline (PHYSICALITY_BAKU={PHYSICALITY_BAKU})")
    print(f"Route: {BASELINE_ROUTE}")
    print("=" * 60)

    selector, bc_stats = build_selector_for_route(BASELINE_OVERRIDES)
    print(f"  BC: {bc_stats['accuracy']:.1%} ({bc_stats['correct']}/{bc_stats['total']}), "
          f"misaligned={bc_stats['misaligned']}")
    results = eval_full_suite(selector, EVAL_GAMES, 42, "baseline", bc_stats)
    return results


# ══════════════════════════════════════════════════════════════════════════
# LANE 1: Opening Failure Diagnosis vs Native Hal
# ══════════════════════════════════════════════════════════════════════════
def lane_1_diagnose():
    print("\n" + "=" * 60)
    print("LANE 1: Opening Failure Diagnosis vs Native Hal")
    print("=" * 60)

    model = _get_base_model()
    game_summaries = []

    for gi in range(20):
        opp = _make_fast_native_opp()
        env = DTHEnv(opponent=opp, agent_role="baku", seed=42 + gi)
        obs, _ = env.reset()
        done = False
        last_reward = 0.0
        first_death_turn = None

        for t in range(OPENING_HORIZON):
            if done:
                break
            mask = env.action_masks()
            action_idx = BASELINE_OVERRIDES[t] - 1
            prev_deaths = env.agent.deaths
            obs, last_reward, term, trunc, _ = env.step(action_idx)
            if env.agent.deaths > prev_deaths and first_death_turn is None:
                first_death_turn = t
            if term or trunc:
                done = True

        t_idx = OPENING_HORIZON
        while not done:
            mask = env.action_masks()
            action_idx = int(model.predict(obs, action_masks=mask, deterministic=True)[0])
            prev_deaths = env.agent.deaths
            obs, last_reward, term, trunc, _ = env.step(action_idx)
            if env.agent.deaths > prev_deaths and first_death_turn is None:
                first_death_turn = t_idx
            t_idx += 1
            if term or trunc:
                done = True

        won = bool(term and env.game.winner is env.agent and last_reward > 0)
        game_summaries.append({
            "game": gi, "won": won,
            "half_rounds": len(env.game.history),
            "baku_deaths": env.agent.deaths,
            "baku_ttd": env.agent.ttd,
            "first_death_turn": first_death_turn,
            "hal_deaths": env.game.player1.deaths,
        })

    avg_hr = np.mean([g["half_rounds"] for g in game_summaries])
    avg_baku_deaths = np.mean([g["baku_deaths"] for g in game_summaries])
    avg_hal_deaths = np.mean([g["hal_deaths"] for g in game_summaries])
    win_count = sum(g["won"] for g in game_summaries)
    first_deaths = [g["first_death_turn"] for g in game_summaries
                    if g["first_death_turn"] is not None]

    print(f"\n  20-game trace vs native_hal_16k:")
    print(f"    Baku wins:         {win_count}/20")
    print(f"    Avg half-rounds:   {avg_hr:.1f}")
    print(f"    Avg Baku deaths:   {avg_baku_deaths:.1f}")
    print(f"    Avg Hal deaths:    {avg_hal_deaths:.1f}")
    if first_deaths:
        print(f"    First death turn:  min={min(first_deaths)} max={max(first_deaths)} "
              f"median={np.median(first_deaths):.0f}")
        opening_deaths = [t for t in first_deaths if t < OPENING_HORIZON]
        late_deaths = [t for t in first_deaths if t >= OPENING_HORIZON]
        print(f"    In opening (T0-7): {len(opening_deaths)}/{len(first_deaths)}")
        print(f"    In late (T8+):     {len(late_deaths)}/{len(first_deaths)}")

    return {
        "games": game_summaries,
        "summary": {
            "win_rate": win_count / 20,
            "avg_half_rounds": round(float(avg_hr), 2),
            "avg_baku_deaths": round(float(avg_baku_deaths), 2),
            "avg_hal_deaths": round(float(avg_hal_deaths), 2),
            "first_death_turns": first_deaths,
            "opening_death_fraction": (
                len([t for t in first_deaths if t < OPENING_HORIZON]) /
                max(1, len(first_deaths))
            ),
        },
    }


# ══════════════════════════════════════════════════════════════════════════
# LANE 2: Counterfactual Audit T0-T7 vs Native Hal
# ══════════════════════════════════════════════════════════════════════════
def lane_2_counterfactual():
    print("\n" + "=" * 60)
    print("LANE 2: Counterfactual Audit T0-T7 vs Native Hal")
    print("=" * 60)

    bl = eval_route_vs_native(BASELINE_OVERRIDES, n_games=CF_GAMES)
    bl_score = score_native(bl)
    print(f"\n  Baseline vs native: wr={pct(bl['win_rate'])} "
          f"r7={pct(bl['r7_rate'])} hr={bl['avg_hr']:.1f} score={bl_score:.4f}")

    audit = {"baseline": bl, "turns": {}}
    improvements = []

    for turn in range(OPENING_HORIZON):
        bl_action = BASELINE_ROUTE[turn]
        role = TURN_ROLES[turn]
        print(f"\n  T{turn} ({role}, baseline={bl_action}):")

        turn_data = {}
        best_action = bl_action
        best_score = bl_score

        for action in CF_ACTIONS:
            if action == bl_action:
                turn_data[action] = {"native": bl, "note": "baseline"}
                continue

            ov = dict(BASELINE_OVERRIDES)
            ov[turn] = action
            res = eval_route_vs_native(ov, n_games=CF_GAMES)
            turn_data[action] = {"native": res}

            sc = score_native(res)
            delta_wr = res["win_rate"] - bl["win_rate"]
            delta_r7 = res["r7_rate"] - bl["r7_rate"]
            delta_hr = res["avg_hr"] - bl["avg_hr"]
            marker = ""
            if delta_wr > 0 or delta_r7 > 0:
                marker = f" ↑wr={delta_wr:+.0%} r7={delta_r7:+.0%} hr={delta_hr:+.1f}"
            elif delta_hr > 0.5:
                marker = f" hr={delta_hr:+.1f}"

            print(f"    a={action:2d}: wr={pct(res['win_rate'])} r7={pct(res['r7_rate'])} "
                  f"hr={res['avg_hr']:.1f}{marker}")

            if sc > best_score:
                best_score = sc
                best_action = action

        improvement = best_score - bl_score
        audit["turns"][f"T{turn}"] = {
            "role": role, "baseline_action": bl_action,
            "best_action": best_action, "improvement": round(improvement, 4),
            "results": {str(k): {kk: vv for kk, vv in v.items() if kk != "note"}
                        for k, v in turn_data.items()},
        }

        if best_action != bl_action:
            print(f"    ** T{turn} best: a={best_action} (+{improvement:.4f})")
            improvements.append((turn, best_action, improvement))
        else:
            print(f"    ** T{turn}: baseline optimal")

    audit["improvements"] = [(t, a, round(imp, 4)) for t, a, imp in improvements]
    print(f"\n  Total improvements found: {len(improvements)}")
    return audit


# ══════════════════════════════════════════════════════════════════════════
# LANE 3A: Counterfactual-Guided Route Construction
# ══════════════════════════════════════════════════════════════════════════
def lane_3a_cf_guided(audit):
    print("\n" + "=" * 60)
    print("LANE 3A: Counterfactual-Guided Routes")
    print("=" * 60)

    improvements = audit.get("improvements", [])
    candidates = []

    if not improvements:
        print("  No CF improvements — skip 3A")
        return candidates

    # A1: All improvements combined
    ov = dict(BASELINE_OVERRIDES)
    diffs = []
    for turn, action, imp in improvements:
        ov[turn] = action
        diffs.append(f"T{turn}={action}")
    tag = "A1_all_cf"
    print(f"\n  --- {tag}: {', '.join(diffs)} ---")
    res = eval_route_vs_native(ov, n_games=CF_GAMES)
    route = [ov.get(t, BASELINE_ROUTE[t]) for t in range(OPENING_HORIZON)]
    print(f"    wr={pct(res['win_rate'])} r7={pct(res['r7_rate'])} hr={res['avg_hr']:.1f}")
    candidates.append({"tag": tag, "overrides": dict(ov), "route": route,
                       "native": res, "diffs": diffs})

    # A2: Top individual deviations
    imps_sorted = sorted(improvements, key=lambda x: -x[2])
    for i, (turn, action, imp) in enumerate(imps_sorted[:3]):
        tag = f"A2_top{i+1}_T{turn}eq{action}"
        ov = dict(BASELINE_OVERRIDES)
        ov[turn] = action
        print(f"\n  --- {tag} ---")
        res = eval_route_vs_native(ov, n_games=CF_GAMES)
        route = [ov.get(t, BASELINE_ROUTE[t]) for t in range(OPENING_HORIZON)]
        print(f"    wr={pct(res['win_rate'])} r7={pct(res['r7_rate'])} hr={res['avg_hr']:.1f}")
        candidates.append({"tag": tag, "overrides": dict(ov), "route": route,
                           "native": res, "diffs": [f"T{turn}={action}"]})

    # A3: Pairwise combinations
    if len(imps_sorted) >= 2:
        for (t1, a1, _), (t2, a2, _) in itertools.combinations(imps_sorted[:3], 2):
            tag = f"A3_T{t1}eq{a1}_T{t2}eq{a2}"
            ov = dict(BASELINE_OVERRIDES)
            ov[t1] = a1; ov[t2] = a2
            print(f"\n  --- {tag} ---")
            res = eval_route_vs_native(ov, n_games=CF_GAMES)
            route = [ov.get(t, BASELINE_ROUTE[t]) for t in range(OPENING_HORIZON)]
            print(f"    wr={pct(res['win_rate'])} r7={pct(res['r7_rate'])} hr={res['avg_hr']:.1f}")
            candidates.append({"tag": tag, "overrides": dict(ov), "route": route,
                               "native": res, "diffs": [f"T{t1}={a1}", f"T{t2}={a2}"]})

    return candidates


# ══════════════════════════════════════════════════════════════════════════
# LANE 3B: Population-Based Discrete Opening Evolution
# ══════════════════════════════════════════════════════════════════════════
def lane_3b_evolution():
    print("\n" + "=" * 60)
    print("LANE 3B: Population-Based Opening Evolution")
    print("=" * 60)

    rng = np.random.default_rng(42)
    action_pool = np.array([1, 5, 10, 15, 20, 30, 40, 50, 55, 60])
    pop_size = 16
    n_gens = 4
    elite_k = 4

    # Initialize: 4 mutants of baseline + 12 random
    population = []
    for i in range(4):
        route = list(BASELINE_ROUTE)
        t = int(rng.integers(0, OPENING_HORIZON))
        route[t] = int(rng.choice(action_pool))
        population.append(route)
    for _ in range(12):
        population.append([int(rng.choice(action_pool)) for _ in range(OPENING_HORIZON)])

    best_route = list(BASELINE_ROUTE)
    best_fitness = -1.0
    best_res = None

    for gen in range(n_gens):
        print(f"\n  Gen {gen+1}/{n_gens} (pop={len(population)}):")

        scored = []
        for route in population:
            ov = {t: route[t] for t in range(OPENING_HORIZON)}
            res = eval_route_vs_native(ov, n_games=EVO_GAMES)
            f = score_native(res)
            scored.append((f, route, res))
            if f > best_fitness:
                best_fitness = f
                best_route = list(route)
                best_res = res

        scored.sort(key=lambda x: -x[0])
        top = scored[0]
        print(f"    Best: {top[1]} f={top[0]:.4f} wr={pct(top[2]['win_rate'])} "
              f"r7={pct(top[2]['r7_rate'])} hr={top[2]['avg_hr']:.1f}")

        if gen == n_gens - 1:
            break

        elites = [s[1] for s in scored[:elite_k]]
        children = [list(e) for e in elites]
        while len(children) < pop_size:
            p1, p2 = rng.choice(elite_k, 2, replace=False)
            child = [elites[p1][t] if rng.random() < 0.5 else elites[p2][t]
                     for t in range(OPENING_HORIZON)]
            if rng.random() < 0.3:
                for _ in range(int(rng.integers(1, 3))):
                    ti = int(rng.integers(0, OPENING_HORIZON))
                    child[ti] = int(rng.choice(action_pool))
            children.append(child)
        population = children

    # Re-evaluate best with more games
    ov = {t: best_route[t] for t in range(OPENING_HORIZON)}
    res = eval_route_vs_native(ov, n_games=CF_GAMES)
    diffs = [f"T{t}={best_route[t]}" for t in range(8) if best_route[t] != BASELINE_ROUTE[t]]
    print(f"\n  Evo best: {best_route}")
    print(f"    Re-eval ({CF_GAMES}g): wr={pct(res['win_rate'])} "
          f"r7={pct(res['r7_rate'])} hr={res['avg_hr']:.1f}")

    candidates = [{"tag": "B_evo_best", "overrides": ov, "route": best_route,
                   "native": res, "diffs": diffs}]

    # Also include last-gen best if different
    last_best = scored[0][1]
    if last_best != best_route:
        ov2 = {t: last_best[t] for t in range(OPENING_HORIZON)}
        res2 = eval_route_vs_native(ov2, n_games=CF_GAMES)
        diffs2 = [f"T{t}={last_best[t]}" for t in range(8) if last_best[t] != BASELINE_ROUTE[t]]
        candidates.append({"tag": "B_evo_lastgen", "overrides": ov2,
                           "route": list(last_best), "native": res2, "diffs": diffs2})

    return candidates


# ══════════════════════════════════════════════════════════════════════════
# LANE 3C: Greedy Coordinate-Descent Per-Turn Optimization
# ══════════════════════════════════════════════════════════════════════════
def lane_3c_greedy():
    print("\n" + "=" * 60)
    print("LANE 3C: Greedy Coordinate-Descent Optimization")
    print("=" * 60)

    current_route = list(BASELINE_ROUTE)

    for pass_num in range(2):
        print(f"\n  Pass {pass_num+1}/2:")
        changed = False

        for turn in range(OPENING_HORIZON):
            role = TURN_ROLES[turn]
            current_action = current_route[turn]

            ov_cur = {t: current_route[t] for t in range(OPENING_HORIZON)}
            res_cur = eval_route_vs_native(ov_cur, n_games=CF_GAMES)
            sc_cur = score_native(res_cur)

            best_action = current_action
            best_score = sc_cur

            for action in CF_ACTIONS:
                if action == current_action:
                    continue
                ov = dict(ov_cur)
                ov[turn] = action
                res = eval_route_vs_native(ov, n_games=CF_GAMES)
                sc = score_native(res)
                if sc > best_score:
                    best_score = sc
                    best_action = action

            if best_action != current_action:
                delta = best_score - sc_cur
                print(f"    T{turn} ({role}): {current_action} -> {best_action} (+{delta:.4f})")
                current_route[turn] = best_action
                changed = True
            else:
                print(f"    T{turn} ({role}): {current_action} (opt)")

        if not changed:
            print(f"  Pass {pass_num+1}: converged")
            break

    ov = {t: current_route[t] for t in range(OPENING_HORIZON)}
    res = eval_route_vs_native(ov, n_games=CF_GAMES)
    diffs = [f"T{t}={current_route[t]}" for t in range(8)
             if current_route[t] != BASELINE_ROUTE[t]]
    print(f"\n  Final: {current_route}")
    print(f"    wr={pct(res['win_rate'])} r7={pct(res['r7_rate'])} hr={res['avg_hr']:.1f}")
    return [{"tag": "C_greedy", "overrides": ov, "route": current_route,
             "native": res, "diffs": diffs}]


# ══════════════════════════════════════════════════════════════════════════
# LANE 4: Full Validation
# ══════════════════════════════════════════════════════════════════════════
def lane_4_validate(all_candidates, baseline):
    print("\n" + "=" * 60)
    print("LANE 4: Full Validation")
    print("=" * 60)

    seen = set()
    unique = []
    for c in all_candidates:
        rk = tuple(c["route"])
        if rk in seen or rk == tuple(BASELINE_ROUTE):
            continue
        seen.add(rk)
        unique.append(c)

    print(f"  Unique non-baseline candidates: {len(unique)}")
    if not unique:
        print("  No candidates to validate.")
        return []

    validated = []
    for c in unique:
        tag = c["tag"]
        route = c["route"]
        ov = c["overrides"]
        diffs = c.get("diffs", [])

        print(f"\n  --- {tag}: {diffs} ---")

        selector, bc_stats = build_selector_for_route(ov)
        c["bc_stats"] = bc_stats
        print(f"    BC: {bc_stats['accuracy']:.1%} mis={bc_stats['misaligned']}")

        if bc_stats["misaligned"] > 0:
            print(f"    SKIP: misaligned={bc_stats['misaligned']}")
            c["gate_pass"] = False
            continue

        # Gate (16-game scripted)
        gate_results = {}
        for opp in SCRIPTED_OPPONENTS:
            res = evaluate_selector(selector, opp, games=GATE_GAMES, seed=42)
            gate_results[opp] = asdict(res)
        r9 = eval_r9_seeded(selector, seed=42)
        gate_results["seeded_r9"] = r9

        bp = gate_results.get("bridge_pressure", {}).get("r7_rate", 0)
        ht = gate_results.get("hal_death_trade", {}).get("r7_rate", 0)
        hp = gate_results.get("hal_pressure", {}).get("r7_rate", 0)
        gate_pass = bp >= 0.70 and ht >= 0.30 and hp >= 0.30 and r9["rate"] >= 1.0

        if not gate_pass:
            print(f"    GATE FAIL: bp={pct(bp)} ht={pct(ht)} hp={pct(hp)} r9={pct(r9['rate'])}")
            c["gate_pass"] = False
            continue

        print(f"    GATE PASS: bp={pct(bp)} ht={pct(ht)} hp={pct(hp)}")
        c["gate_pass"] = True

        full = eval_full_suite(selector, EVAL_GAMES, 42, tag, bc_stats)
        c["full_eval"] = full
        validated.append(c)

    return validated


# ══════════════════════════════════════════════════════════════════════════
# LANE 5: Verdict
# ══════════════════════════════════════════════════════════════════════════
def lane_5_verdict(baseline, validated):
    print("\n" + "=" * 60)
    print("LANE 5: Verdict")
    print("=" * 60)

    bl_bp = baseline.get("bridge_pressure", {}).get("r7_rate", 0)
    bl_ht = baseline.get("hal_death_trade", {}).get("r7_rate", 0)
    bl_hp = baseline.get("hal_pressure", {}).get("r7_rate", 0)
    bl_native_wr = baseline.get("native_hal", {}).get("win_rate", 0)
    bl_native_r7 = baseline.get("native_hal", {}).get("r7_rate", 0)

    best_candidate = None
    best_level = None
    best_score = -999.0
    any_improvement = False

    for c in validated:
        v = c.get("full_eval", {})
        c_bp = v.get("bridge_pressure", {}).get("r7_rate", 0)
        c_ht = v.get("hal_death_trade", {}).get("r7_rate", 0)
        c_hp = v.get("hal_pressure", {}).get("r7_rate", 0)
        c_r9 = v.get("seeded_r9", {}).get("rate", 0)
        c_prior = v.get("prior_4096", {}).get("win_rate", 0)
        c_native_wr = v.get("native_hal", {}).get("win_rate", 0)
        c_native_r7 = v.get("native_hal", {}).get("r7_rate", 0)
        c_mis = v.get("bc_stats", {}).get("misaligned", 0)

        wr_delta = c_native_wr - bl_native_wr
        r7_delta = c_native_r7 - bl_native_r7

        if wr_delta > 0.02 or r7_delta > 0.02:
            any_improvement = True

        strong = (
            c_bp >= STRONG_PASS["bp_r7"]
            and c_ht >= STRONG_PASS["ht_r7"]
            and c_hp >= STRONG_PASS["hp_r7"]
            and c_r9 >= STRONG_PASS["seeded_r9"]
            and c_mis == STRONG_PASS["misaligned"]
            and (wr_delta >= STRONG_PASS["native_wr_delta"]
                 or c_native_r7 >= STRONG_PASS["native_r7_rate"])
        )
        acceptable = (
            c_bp >= ACCEPTABLE_PASS["bp_r7"]
            and c_ht >= ACCEPTABLE_PASS["ht_r7"]
            and c_hp >= ACCEPTABLE_PASS["hp_r7"]
            and c_r9 >= ACCEPTABLE_PASS["seeded_r9"]
            and c_mis == ACCEPTABLE_PASS["misaligned"]
            and (wr_delta >= ACCEPTABLE_PASS["native_wr_delta"]
                 or c_native_r7 >= ACCEPTABLE_PASS["native_r7_rate"])
        )

        level = "STRONG_PASS" if strong else ("ACCEPTABLE_PASS" if acceptable else None)
        score = wr_delta * 3 + r7_delta * 2 + (c_ht - bl_ht) * 0.5 + (c_hp - bl_hp) * 0.5

        c["analysis"] = {
            "level": level, "score": round(score, 4),
            "deltas": {
                "bp_r7": round(c_bp - bl_bp, 4),
                "ht_r7": round(c_ht - bl_ht, 4),
                "hp_r7": round(c_hp - bl_hp, 4),
                "prior_wr": round(c_prior - baseline.get("prior_4096", {}).get("win_rate", 0), 4),
                "native_wr": round(wr_delta, 4),
                "native_r7": round(r7_delta, 4),
            },
            "abs": {
                "bp_r7": round(c_bp, 4), "ht_r7": round(c_ht, 4),
                "hp_r7": round(c_hp, 4), "r9": round(c_r9, 4),
                "prior_wr": round(c_prior, 4), "native_wr": round(c_native_wr, 4),
                "native_r7": round(c_native_r7, 4), "misaligned": c_mis,
            },
        }

        if level and (not best_level or score > best_score):
            best_score = score
            best_candidate = c
            best_level = level
        elif not best_level and score > best_score:
            best_score = score
            best_candidate = c

        print(f"  {c['tag']}: level={level or '---'} score={score:.4f} "
              f"native_wr={wr_delta:+.0%} native_r7={r7_delta:+.0%}")

    verdict_str = "NO_IMPROVEMENT"
    if best_level == "STRONG_PASS":
        verdict_str = "STRONG_PASS"
    elif best_level == "ACCEPTABLE_PASS":
        verdict_str = "ACCEPTABLE_PASS"
    elif any_improvement:
        verdict_str = "PARTIAL_IMPROVEMENT"

    return {
        "verdict": verdict_str,
        "best_candidate": best_candidate["tag"] if best_candidate else None,
        "best_route": best_candidate["route"] if best_candidate else None,
        "best_level": best_level,
        "best_score": round(best_score, 4),
        "any_improvement": any_improvement,
        "candidates_evaluated": len(validated),
    }


# ══════════════════════════════════════════════════════════════════════════
# Deliverables
# ══════════════════════════════════════════════════════════════════════════
def write_report(path, baseline, diagnosis, audit, cands_a, cands_b, cands_c,
                 validated, verdict, elapsed):
    bl = baseline
    bl_bp = bl.get("bridge_pressure", {}).get("r7_rate", 0)
    bl_ht = bl.get("hal_death_trade", {}).get("r7_rate", 0)
    bl_hp = bl.get("hal_pressure", {}).get("r7_rate", 0)
    bl_r9 = bl.get("seeded_r9", {}).get("rate", 0)
    bl_prior = bl.get("prior_4096", {}).get("win_rate", 0)
    bl_nwr = bl.get("native_hal", {}).get("win_rate", 0)
    bl_nr7 = bl.get("native_hal", {}).get("r7_rate", 0)
    bl_mis = bl.get("bc_stats", {}).get("misaligned", 0)
    diag = diagnosis.get("summary", {})

    with open(path, "w") as f:
        f.write(f"# Sprint Report: Opening vs Native Hal at p=0.88\n")
        f.write(f"**Date:** 2026-03-28\n")
        f.write(f"**PHYSICALITY_BAKU:** {PHYSICALITY_BAKU}\n")
        f.write(f"**Verdict:** {verdict['verdict']}\n\n")

        f.write("## Executive Summary\n\n")
        f.write("This sprint targets Baku's OPENING response (T0-T7) against the strongest\n")
        f.write("scaffold-native Hal (native_hal_16k). The prior sprint showed Baku WR=0%\n")
        f.write("and r7_rate=0% against native Hal, with late-head tuning producing no signal.\n")
        f.write("Three materially different opening-improvement families were tested:\n")
        f.write("counterfactual-guided search, population-based evolution, and greedy\n")
        f.write("coordinate descent.\n\n")

        f.write("## Optimization Target\n\n")
        f.write("Improve opening response to raise Baku win rate or r7_rate vs native Hal,\n")
        f.write("while preserving scripted baseline (bp>=0.84, ht>=0.48, hp>=0.48, r9=100%).\n\n")

        f.write("## Why Opening Progress Matters\n\n")
        f.write("The strongest native Hal defeats Baku before round 7 (r7_rate=0%). ")
        f.write("Late-head adaptation produced zero improvement across 3+ families. ")
        f.write("If the opening can be improved, it would unlock meaningful self-play iteration. ")
        f.write("If not, the opening action space is saturated under current calibration.\n\n")

        f.write("## Baseline Table\n\n")
        f.write("| Metric | Value |\n|--------|-------|\n")
        f.write(f"| Route | `{BASELINE_ROUTE}` |\n")
        f.write(f"| BC accuracy | {bl.get('bc_stats', {}).get('accuracy', 0):.1%} |\n")
        f.write(f"| Misaligned | {bl_mis} |\n")
        f.write(f"| bp r7 | {pct(bl_bp)} |\n")
        f.write(f"| ht r7 | {pct(bl_ht)} |\n")
        f.write(f"| hp r7 | {pct(bl_hp)} |\n")
        f.write(f"| seeded r9 | {pct(bl_r9)} |\n")
        f.write(f"| prior_4096 WR | {pct(bl_prior)} |\n")
        f.write(f"| native Hal WR | {pct(bl_nwr)} |\n")
        f.write(f"| native Hal r7 | {pct(bl_nr7)} |\n\n")

        f.write("## Opening Failure Analysis vs Native Hal\n\n")
        f.write(f"- Avg half-rounds: {diag.get('avg_half_rounds', 'N/A')}\n")
        f.write(f"- Avg Baku deaths: {diag.get('avg_baku_deaths', 'N/A')}\n")
        f.write(f"- Avg Hal deaths: {diag.get('avg_hal_deaths', 'N/A')}\n")
        fdt = diag.get("first_death_turns", [])
        if fdt:
            f.write(f"- First Baku death turn: median={np.median(fdt):.0f}, "
                    f"range=[{min(fdt)}, {max(fdt)}]\n")
            f.write(f"- Opening death fraction: "
                    f"{diag.get('opening_death_fraction', 0):.0%}\n")
        f.write("\n")

        f.write("## Counterfactual Audit vs Native Hal\n\n")
        improvements = audit.get("improvements", [])
        if improvements:
            f.write(f"Found {len(improvements)} single-turn improvement(s):\n\n")
            for t, a, imp in improvements:
                f.write(f"- T{t} ({TURN_ROLES[t]}): baseline={BASELINE_ROUTE[t]}"
                        f" -> {a} (+{imp:.4f})\n")
        else:
            f.write("No single-turn improvements found. Baseline is locally "
                    "optimal vs native Hal.\n")
        f.write("\n")

        f.write("## Search/Evolution/Distillation Families\n\n")

        f.write("### Family A: Counterfactual-Guided\n\n")
        if cands_a:
            for c in cands_a:
                nat = c.get("native", {})
                f.write(f"- **{c['tag']}**: diffs={c.get('diffs', [])} "
                        f"wr={pct(nat.get('win_rate', 0))} "
                        f"r7={pct(nat.get('r7_rate', 0))}\n")
        else:
            f.write("No candidates (no CF improvements found).\n")
        f.write("\n")

        f.write("### Family B: Population Evolution\n\n")
        for c in cands_b:
            nat = c.get("native", {})
            f.write(f"- **{c['tag']}**: route={c.get('route', [])} "
                    f"wr={pct(nat.get('win_rate', 0))} "
                    f"r7={pct(nat.get('r7_rate', 0))}\n")
        f.write("\n")

        f.write("### Family C: Greedy Descent\n\n")
        for c in cands_c:
            nat = c.get("native", {})
            f.write(f"- **{c['tag']}**: route={c.get('route', [])} "
                    f"wr={pct(nat.get('win_rate', 0))} "
                    f"r7={pct(nat.get('r7_rate', 0))}\n")
        f.write("\n")

        f.write("## Full Validation\n\n")
        if validated:
            f.write("| Candidate | bp r7 | ht r7 | hp r7 | r9 | prior WR "
                    "| native WR | native r7 | mis | Level |\n")
            f.write("|-----------|-------|-------|-------|----|----------"
                    "|-----------|-----------|-----|-------|\n")
            f.write(f"| baseline | {pct(bl_bp)} | {pct(bl_ht)} | {pct(bl_hp)}"
                    f" | {pct(bl_r9)} | {pct(bl_prior)} | {pct(bl_nwr)}"
                    f" | {pct(bl_nr7)} | {bl_mis} | --- |\n")
            for c in validated:
                a = c.get("analysis", {}).get("abs", {})
                lvl = c.get("analysis", {}).get("level", "---") or "---"
                f.write(f"| {c['tag']} | {pct(a.get('bp_r7', 0))}"
                        f" | {pct(a.get('ht_r7', 0))}"
                        f" | {pct(a.get('hp_r7', 0))}"
                        f" | {pct(a.get('r9', 0))}"
                        f" | {pct(a.get('prior_wr', 0))}"
                        f" | {pct(a.get('native_wr', 0))}"
                        f" | {pct(a.get('native_r7', 0))}"
                        f" | {a.get('misaligned', 0)} | {lvl} |\n")
        else:
            f.write("No candidates passed gate or none found.\n")
        f.write("\n")

        f.write("## Final Verdict\n\n")
        f.write(f"**{verdict['verdict']}**\n\n")
        f.write(f"- Best candidate: {verdict.get('best_candidate', 'none')}\n")
        f.write(f"- Level: {verdict.get('best_level', 'none')}\n")
        f.write(f"- Score: {verdict.get('best_score', 0)}\n")
        f.write(f"- Any improvement signal: "
                f"{'YES' if verdict.get('any_improvement') else 'NO'}\n")
        f.write(f"- Sprint time: {elapsed:.0f}s\n\n")

        f.write("## Best Next Move\n\n")
        if verdict["verdict"] in ("STRONG_PASS", "ACCEPTABLE_PASS"):
            f.write("Promote the best opening route and retrain native Hal "
                    "against it.\n")
            f.write("This creates a real self-play response loop.\n")
        elif verdict.get("any_improvement"):
            f.write("Partial improvement detected but below promotion threshold.\n")
            f.write("Consider: longer evolution, wider action space, or "
                    "architectural changes.\n")
        else:
            f.write("No opening adaptation improved against native Hal.\n")
            f.write("The current opening action space appears saturated "
                    "under p=0.88.\n")
            f.write("Consider: (1) stochastic openings via entropy-regularized "
                    "BC, (2) expanded\n")
            f.write("observation space with opponent modeling, (3) recurrent "
                    "opening policy that\n")
            f.write("conditions on opponent's early actions, or (4) breaking "
                    "the deterministic\n")
            f.write("argmax ceiling with temperature-based action selection.\n")


def terminal_summary(baseline, diagnosis, audit, validated, verdict, elapsed):
    print("\n" + "=" * 60)
    print("TERMINAL SUMMARY")
    print("=" * 60)

    bl_nwr = baseline.get("native_hal", {}).get("win_rate", 0)
    bl_nr7 = baseline.get("native_hal", {}).get("r7_rate", 0)
    bl_bp = baseline.get("bridge_pressure", {}).get("r7_rate", 0)
    bl_ht = baseline.get("hal_death_trade", {}).get("r7_rate", 0)
    bl_hp = baseline.get("hal_pressure", {}).get("r7_rate", 0)

    print(f"  PHYSICALITY_BAKU:     {PHYSICALITY_BAKU}")
    print(f"  Baseline route:       {BASELINE_ROUTE}")
    print(f"  Baseline vs native:   wr={pct(bl_nwr)} r7={pct(bl_nr7)}")
    print(f"  Baseline scripted:    bp={pct(bl_bp)} ht={pct(bl_ht)} hp={pct(bl_hp)}")
    print(f"  ──────────────────────────────────────")

    if validated:
        best = max(validated, key=lambda c: c.get("analysis", {}).get("score", -999))
        a = best.get("analysis", {})
        d = a.get("deltas", {})
        ab = a.get("abs", {})
        print(f"  Best candidate:       {best['tag']}")
        print(f"  Route:                {best.get('route', 'N/A')}")
        print(f"  Native WR:            {pct(ab.get('native_wr', 0))} "
              f"(delta {d.get('native_wr', 0):+.0%})")
        print(f"  Native r7:            {pct(ab.get('native_r7', 0))} "
              f"(delta {d.get('native_r7', 0):+.0%})")
        print(f"  Scripted:             bp={pct(ab.get('bp_r7', 0))} "
              f"ht={pct(ab.get('ht_r7', 0))} hp={pct(ab.get('hp_r7', 0))}")
    else:
        print(f"  Best candidate:       none")

    print(f"  ──────────────────────────────────────")
    print(f"  CF improvements:      {len(audit.get('improvements', []))}")
    print(f"  Candidates validated: {verdict.get('candidates_evaluated', 0)}")
    print(f"  Verdict:              {verdict['verdict']}")
    print(f"  Sprint time:          {elapsed:.0f}s")
    print("=" * 60)


# ══════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════
def main():
    start = time.time()
    print("=" * 60)
    print(f"SPRINT: Opening vs Native Hal at PHYSICALITY_BAKU={PHYSICALITY_BAKU}")
    print(f"Date: 2026-03-28")
    print(f"Target: Improve T0-T7 against native_hal_16k")
    print("=" * 60)

    # Lane 0
    baseline = lane_0_baseline()

    # Lane 1
    diagnosis = lane_1_diagnose()

    # Lane 2
    audit = lane_2_counterfactual()

    # Lane 3: Three families
    cands_a = lane_3a_cf_guided(audit)
    cands_b = lane_3b_evolution()
    cands_c = lane_3c_greedy()
    all_candidates = cands_a + cands_b + cands_c
    print(f"\n  Total candidates: {len(all_candidates)}")

    # Lane 4
    validated = lane_4_validate(all_candidates, baseline)

    # Lane 5
    verdict = lane_5_verdict(baseline, validated)

    elapsed = time.time() - start

    # ── Deliverables ─────────────────────────────────────────────────────
    output = {
        "sprint": "opening-vs-native-hal",
        "date": "2026-03-28",
        "physicality_baku": PHYSICALITY_BAKU,
        "baseline_route": BASELINE_ROUTE,
        "native_hal": "native_hal_16k",
        "baseline": baseline,
        "diagnosis": diagnosis.get("summary", {}),
        "counterfactual_audit": {
            "improvements": audit.get("improvements", []),
            "baseline_native": audit.get("baseline", {}),
        },
        "families": {
            "A_cf_guided": [{k: v for k, v in c.items()
                             if k not in ("overrides",)} for c in cands_a],
            "B_evolution": [{k: v for k, v in c.items()
                             if k not in ("overrides",)} for c in cands_b],
            "C_greedy": [{k: v for k, v in c.items()
                          if k not in ("overrides",)} for c in cands_c],
        },
        "validated": [{k: v for k, v in c.items()
                       if k not in ("overrides",)} for c in validated],
        "verdict": verdict,
        "elapsed_seconds": round(elapsed),
    }

    json_dir = Path("docs/reports/json")
    json_dir.mkdir(parents=True, exist_ok=True)
    json_path = json_dir / "sprint_opening_vs_native_hal_2026-03-28.json"
    with open(json_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nJSON: {json_path}")

    md_path = Path("docs/reports") / "sprint_opening_vs_native_hal_report_2026-03-28.md"
    write_report(md_path, baseline, diagnosis, audit, cands_a, cands_b, cands_c,
                 validated, verdict, elapsed)
    print(f"Report: {md_path}")

    terminal_summary(baseline, diagnosis, audit, validated, verdict, elapsed)


if __name__ == "__main__":
    main()
