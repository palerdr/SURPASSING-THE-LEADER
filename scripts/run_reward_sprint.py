#!/usr/bin/env python3
"""Sprint: Late-Head Reward Engineering — full execution.

Tests whether denser strategic late-game rewards can produce real
adaptation signal on top of the stable modular scaffold.
"""

from __future__ import annotations

import sys
import os
import json
import time
from pathlib import Path
from dataclasses import dataclass, asdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from sb3_contrib import MaskablePPO

from environment.dth_env import DTHEnv
from environment.opponents.factory import create_model_opponent, create_scripted_opponent
from environment.opponents.league import WeightedOpponentLeague, LeagueEntry
from environment.route_stages import current_route_stage_flags, stage_is_eligible_from_start
from src.Game import HalfRoundResult
from training.curriculum import get_scenario
from training.modular_policy import (
    DenseRewardWrapper,
    OpeningAutoPlayEnv,
    build_modular_selector,
    train_opening_model_bc,
)
from training.expert_selector import build_classifier

# ── Paths ──────────────────────────────────────────────────────────────
def _resolve(stem):
    for p in (Path("models/checkpoints") / stem, Path("models/checkpoints") / f"{stem}.zip",
              Path("STL/models/checkpoints") / stem, Path("STL/models/checkpoints") / f"{stem}.zip"):
        if p.exists():
            return str(p.with_suffix("")) if p.suffix == ".zip" else str(p)
    raise FileNotFoundError(stem)

BASE_MODEL = _resolve("baku_vs_bridge_pressure_opening_to_round7_bc_only_bc_combined_0.3")
BP_SPECIALIST = _resolve("baku_vs_bridge_pressure_opening_to_round7_bc_only_bc_spec_e50_lr5.5")
LEARNED_HAL = _resolve("hal_vs_promoted_baku_prior_4096")
HAL_OVERRIDES = {3: 10}
HAL_OPPONENTS = ["hal_death_trade", "hal_pressure"]
ALL_OPPONENTS = ["bridge_pressure", "hal_death_trade", "hal_pressure"]
OPENING_HORIZON = 8

_opening_model_cache = None
def get_opening_model():
    global _opening_model_cache
    if _opening_model_cache is None:
        m, _ = train_opening_model_bc(BASE_MODEL, HAL_OPPONENTS, HAL_OVERRIDES, OPENING_HORIZON, epochs=80, lr=1e-3)
        _opening_model_cache = m
    return _opening_model_cache


# ── Reward functions ───────────────────────────────────────────────────

def w1_survival_pressure(game, agent, opp):
    """W1: Cylinder advantage + check quality."""
    rec = game.history[-1]
    r = 0.0
    # Cylinder advantage (opponent having more NDD is good for us)
    r += 0.1 * (opp.cylinder - agent.cylinder) / 300.0
    # As checker: reward success, penalize failure
    if rec.checker == agent.name:
        if rec.result == HalfRoundResult.CHECK_SUCCESS:
            r += 0.05
            r -= 0.03 * (rec.st_gained / 60.0)  # penalize high ST
        elif rec.result in (HalfRoundResult.CHECK_FAIL_SURVIVED, HalfRoundResult.CHECK_FAIL_DIED):
            r -= 0.15
    # As dropper: reward opponent failure or high ST
    if rec.dropper == agent.name:
        if rec.result in (HalfRoundResult.CHECK_FAIL_SURVIVED, HalfRoundResult.CHECK_FAIL_DIED):
            r += 0.1  # opponent failed check
        elif rec.result == HalfRoundResult.CHECK_SUCCESS:
            r += 0.03 * (rec.st_gained / 60.0)  # opponent gained high ST
    return r


def w2_clock_route(game, agent, opp):
    """W2: Clock management + safe-check budget."""
    rec = game.history[-1]
    r = 0.0
    # Clock progression toward leap (reward being closer)
    r += 0.02 * min(game.game_clock / 3600.0, 1.0)
    # Safe-check budget advantage
    safe_diff = agent.safe_strategies_remaining - opp.safe_strategies_remaining
    r += 0.03 * safe_diff / 5.0
    # Penalize agent death events (they waste time and add clock overhead)
    if rec.checker == agent.name and rec.result in (
        HalfRoundResult.CHECK_FAIL_SURVIVED, HalfRoundResult.CHECK_FAIL_DIED,
        HalfRoundResult.CYLINDER_OVERFLOW_SURVIVED, HalfRoundResult.CYLINDER_OVERFLOW_DIED,
    ):
        r -= 0.1
    return r


def w3_light_strategic(game, agent, opp):
    """W3: Light survival signal (for use with heavy lhal league)."""
    rec = game.history[-1]
    r = 0.0
    r += 0.05 * (opp.cylinder - agent.cylinder) / 300.0
    if rec.checker == agent.name and "FAIL" in rec.result.name:
        r -= 0.1
    return r


# ── Evaluation helpers ─────────────────────────────────────────────────

@dataclass
class EvalResult:
    opponent: str; games: int; wins: int; win_rate: float
    avg_half_rounds: float; r7_count: int; r7_eligible: int; r7_rate: float
    deaths_by_agent: int


def evaluate_selector(selector, opponent_name, games=24, seed=42, opponent_model_path=None):
    wins = 0; total_hr = 0; r7_count = 0; r7_eligible = 0; deaths = 0
    for gi in range(games):
        opp = (create_model_opponent(opponent_model_path, agent_role="hal")
               if opponent_model_path else create_scripted_opponent(opponent_name))
        env = DTHEnv(opponent=opp, agent_role="baku", seed=seed + gi)
        obs, _ = env.reset()
        selector.reset()
        reached_r7 = current_route_stage_flags(env.game).get("round7_pressure", False)
        eligible = stage_is_eligible_from_start(env.game, "round7_pressure")
        while True:
            mask = env.action_masks()
            action, _ = selector.predict(obs, action_masks=mask, deterministic=True)
            obs, reward, term, trunc, _ = env.step(int(action))
            reached_r7 = reached_r7 or current_route_stage_flags(env.game).get("round7_pressure", False)
            if term or trunc:
                won = bool(term and env.game.winner is env.agent and reward > 0)
                wins += int(won); total_hr += len(env.game.history)
                if eligible: r7_eligible += 1; r7_count += int(reached_r7)
                deaths += env.agent.deaths
                break
    n = games
    return EvalResult(opponent=opponent_name, games=n, wins=wins, win_rate=wins/n,
                      avg_half_rounds=total_hr/n, r7_count=r7_count,
                      r7_eligible=r7_eligible, r7_rate=r7_count/r7_eligible if r7_eligible else 0,
                      deaths_by_agent=deaths)


def eval_r9_seeded_selector(selector, seed=42):
    opp = create_scripted_opponent("safe")
    env = DTHEnv(opponent=opp, agent_role="baku", seed=seed,
                 scenario_sampler=lambda _rng: get_scenario("round9_pre_leap"))
    obs, _ = env.reset()
    selector.reset()
    while True:
        mask = env.action_masks()
        action, _ = selector.predict(obs, action_masks=mask, deterministic=True)
        obs, reward, term, trunc, _ = env.step(int(action))
        if term or trunc:
            won = bool(term and env.game.winner is env.agent and reward > 0)
            return {"wins": int(won), "games": 1, "rate": float(won)}


def eval_suite(selector, games=24, seed=42, tag="", lhal_games=0):
    results = {}
    pct = lambda v: f"{v*100:.0f}%"
    for opp in ALL_OPPONENTS:
        res = evaluate_selector(selector, opp, games=games, seed=seed)
        results[opp] = asdict(res)
        print(f"  {tag} vs {opp}: r7={pct(res.r7_rate)} wins={pct(res.win_rate)} avg_hr={res.avg_half_rounds:.1f}")

    r9 = eval_r9_seeded_selector(selector, seed=seed)
    results["seeded_r9"] = r9
    print(f"  {tag} seeded r9: {r9['wins']}/{r9['games']} ({pct(r9['rate'])})")

    if lhal_games > 0:
        lh = evaluate_selector(selector, "learned_hal", games=lhal_games, seed=seed,
                                opponent_model_path=LEARNED_HAL)
        results["learned_hal"] = asdict(lh)
        print(f"  {tag} vs learned_hal: wins={pct(lh.win_rate)} deaths={lh.deaths_by_agent} avg_hr={lh.avg_half_rounds:.1f}")
    return results


# ── Training helpers ───────────────────────────────────────────────────

def make_league(lhal_weight=0.5, seed=42):
    return WeightedOpponentLeague([
        LeagueEntry(label="ht", weight=1.0, opponent=create_scripted_opponent("hal_death_trade")),
        LeagueEntry(label="hp", weight=1.0, opponent=create_scripted_opponent("hal_pressure")),
        LeagueEntry(label="lhal", weight=lhal_weight, opponent=create_model_opponent(LEARNED_HAL, agent_role="hal")),
    ], seed=seed)


def train_with_reward(reward_fn, reward_scale, timesteps, lhal_weight=0.5,
                        use_base_shaping=True, shaping_preset="bridge", seed=42):
    opening_model = get_opening_model()
    league = make_league(lhal_weight=lhal_weight, seed=seed)
    base_env = OpeningAutoPlayEnv(
        opening_model=opening_model, opening_horizon=OPENING_HORIZON,
        opponent=league, agent_role="baku", seed=seed,
        use_shaping=use_base_shaping, shaping_preset=shaping_preset,
    )
    env = DenseRewardWrapper(base_env, reward_fn=reward_fn, reward_scale=reward_scale)
    model = MaskablePPO.load(BASE_MODEL, env=env)
    model.learn(total_timesteps=timesteps)
    return model


def build_and_eval(late_model, tag, games=24, lhal_games=24):
    opening_model = get_opening_model()
    selector = build_modular_selector(
        bp_specialist_path=BP_SPECIALIST, base_model_path=BASE_MODEL,
        opening_model=opening_model, late_model=late_model,
        opening_horizon=OPENING_HORIZON,
    )
    return eval_suite(selector, games=games, tag=tag, lhal_games=lhal_games)


# ── Lane E: Action-difference audit ───────────────────────────────────

def action_diff_audit(late_model, tag, n_games=10, seed=42):
    """Compare deterministic actions of late_model vs base model at turns 8+."""
    opening_model = get_opening_model()
    base_late = MaskablePPO.load(BASE_MODEL)
    diffs = 0
    total = 0
    first_diff_turn = None

    for gi in range(n_games):
        for opp_name in HAL_OPPONENTS:
            opp = create_scripted_opponent(opp_name)
            env = DTHEnv(opponent=opp, agent_role="baku", seed=seed + gi)
            obs, _ = env.reset()
            term = False
            trunc = False

            # Auto-play opening
            for t in range(OPENING_HORIZON):
                mask = env.action_masks()
                a, _ = opening_model.predict(obs, action_masks=mask, deterministic=True)
                obs, _, term, trunc, _ = env.step(int(a))
                if term or trunc:
                    break

            if term or trunc:
                continue

            # Compare late-model actions post-opening
            turn = OPENING_HORIZON
            while True:
                mask = env.action_masks()
                a_new, _ = late_model.predict(obs, action_masks=mask, deterministic=True)
                a_base, _ = base_late.predict(obs, action_masks=mask, deterministic=True)
                total += 1
                if int(a_new) != int(a_base):
                    diffs += 1
                    if first_diff_turn is None:
                        first_diff_turn = turn

                obs, _, term, trunc, _ = env.step(int(a_new))
                turn += 1
                if term or trunc:
                    break

    pct = diffs / total if total else 0
    print(f"  {tag} action audit: {diffs}/{total} diffs ({pct:.1%}), first diff at turn {first_diff_turn}")
    return {"diffs": diffs, "total": total, "diff_rate": pct, "first_diff_turn": first_diff_turn}


# ── Lane 0 ─────────────────────────────────────────────────────────────

def lane_0():
    print("\n" + "=" * 60)
    print("LANE 0: Baseline Reproduction (corrected harness)")
    print("=" * 60)
    base_late = MaskablePPO.load(BASE_MODEL)
    res = build_and_eval(base_late, "baseline", games=24, lhal_games=24)
    return res


# ── Lane A+B: Build candidates + static gate ─────────────────────────

def lane_ab():
    print("\n" + "=" * 60)
    print("LANE A+B: Reward Families + Static Gate")
    print("=" * 60)

    candidates = []

    # W1: Survival/Pressure — two scales
    for scale, stag in [(1.0, "s1"), (3.0, "s3")]:
        tag = f"W1_{stag}"
        print(f"\n--- {tag}: survival/pressure, scale={scale}, 32k steps ---")
        late = train_with_reward(w1_survival_pressure, reward_scale=scale, timesteps=32768,
                                  use_base_shaping=True, shaping_preset="bridge")
        candidates.append({
            "tag": tag,
            "family": "W1",
            "late_model": late,
            "reward_fn": w1_survival_pressure,
            "reward_scale": scale,
            "lhal_weight": 0.5,
            "shaping_preset": "bridge",
        })

    # W2: Clock/Route — two scales
    for scale, stag in [(1.0, "s1"), (3.0, "s3")]:
        tag = f"W2_{stag}"
        print(f"\n--- {tag}: clock/route, scale={scale}, 32k steps ---")
        late = train_with_reward(w2_clock_route, reward_scale=scale, timesteps=32768,
                                  use_base_shaping=True, shaping_preset="bridge")
        candidates.append({
            "tag": tag,
            "family": "W2",
            "late_model": late,
            "reward_fn": w2_clock_route,
            "reward_scale": scale,
            "lhal_weight": 0.5,
            "shaping_preset": "bridge",
        })

    # W3: Light strategic + heavy lhal
    for lhal_w, stag in [(4.0, "4x"), (8.0, "8x")]:
        tag = f"W3_lhal{stag}"
        print(f"\n--- {tag}: light strategic + lhal_weight={lhal_w}, 32k steps ---")
        late = train_with_reward(w3_light_strategic, reward_scale=1.0, timesteps=32768,
                                  lhal_weight=lhal_w, use_base_shaping=True, shaping_preset="light")
        candidates.append({
            "tag": tag,
            "family": "W3",
            "late_model": late,
            "reward_fn": w3_light_strategic,
            "reward_scale": 1.0,
            "lhal_weight": lhal_w,
            "shaping_preset": "light",
        })

    # Static gate
    print("\n--- Static Safety Gate ---")
    passing = []
    for cand in candidates:
        tag = cand["tag"]
        res = build_and_eval(cand["late_model"], tag, games=24, lhal_games=24)
        cand["static_results"] = res
        bp = res.get("bridge_pressure", {}).get("r7_rate", 0)
        ht = res.get("hal_death_trade", {}).get("r7_rate", 0)
        hp = res.get("hal_pressure", {}).get("r7_rate", 0)
        r9 = res.get("seeded_r9", {}).get("rate", 0)
        lh = res.get("learned_hal", {}).get("win_rate", 0)
        gate = bp >= 0.76 and ht >= 0.32 and hp >= 0.32 and r9 >= 1.0
        cand["static_pass"] = gate
        pct = lambda v: f"{v*100:.0f}%"
        status = "PASS" if gate else "FAIL"
        print(f"  >> {tag}: {status} (bp={pct(bp)} ht={pct(ht)} hp={pct(hp)} r9={pct(r9)} lhal={pct(lh)})")
        if gate:
            passing.append(cand)

    return candidates, passing


# ── Lane C+D: Stress + validation ─────────────────────────────────────

def lane_cd(passing):
    print("\n" + "=" * 60)
    print("LANE C+D: Stress + Validation")
    print("=" * 60)

    for cand in passing:
        tag = cand["tag"]
        cand["stress_results"] = {}

        for steps in [2048, 8192]:
            print(f"\n  Stress {steps} for {tag}...")
            league = make_league(lhal_weight=cand.get("lhal_weight", 0.5))
            base_env = OpeningAutoPlayEnv(
                opening_model=get_opening_model(), opening_horizon=OPENING_HORIZON,
                opponent=league, agent_role="baku", seed=42,
                use_shaping=True, shaping_preset=cand.get("shaping_preset", "bridge"),
            )
            env = DenseRewardWrapper(
                base_env,
                reward_fn=cand["reward_fn"],
                reward_scale=cand.get("reward_scale", 1.0),
            )
            model = MaskablePPO.load(BASE_MODEL, env=env)
            model.learn(total_timesteps=steps)
            sel = build_modular_selector(
                bp_specialist_path=BP_SPECIALIST, base_model_path=BASE_MODEL,
                opening_model=get_opening_model(), late_model=model,
                opening_horizon=OPENING_HORIZON,
            )
            res = eval_suite(sel, games=24, tag=f"{tag}_stress{steps}", lhal_games=24)
            bp = res.get("bridge_pressure", {}).get("r7_rate", 0)
            ht = res.get("hal_death_trade", {}).get("r7_rate", 0)
            hp = res.get("hal_pressure", {}).get("r7_rate", 0)
            r9 = res.get("seeded_r9", {}).get("rate", 0)
            lh = res.get("learned_hal", {}).get("win_rate", 0)
            gate = bp >= 0.76 and ht >= 0.32 and hp >= 0.32 and r9 >= 1.0
            cand["stress_results"][steps] = {"results": res, "gate_pass": gate}
            pct = lambda v: f"{v*100:.0f}%"
            print(f"  >> {tag} @ {steps}: {'PASS' if gate else 'FAIL'} (bp={pct(bp)} ht={pct(ht)} hp={pct(hp)} r9={pct(r9)} lhal={pct(lh)})")

        # 50-game validation
        print(f"\n  50-game validation for {tag}...")
        res50 = build_and_eval(cand["late_model"], f"{tag}_50g", games=50, lhal_games=50)
        cand["full_validation"] = res50


# ── Lane E: Action-difference audit ───────────────────────────────────

def lane_e(passing):
    print("\n" + "=" * 60)
    print("LANE E: Action-Difference Audit")
    print("=" * 60)
    for cand in passing:
        tag = cand["tag"]
        audit = action_diff_audit(cand["late_model"], tag)
        cand["action_audit"] = audit


# ── Promotion ─────────────────────────────────────────────────────────

def check_promotion(candidates):
    best = None; best_bar = None
    for cand in candidates:
        v = cand.get("full_validation", cand.get("static_results", {}))
        bp_s = v.get("bridge_pressure", {}).get("r7_rate", 0)
        ht_s = v.get("hal_death_trade", {}).get("r7_rate", 0)
        hp_s = v.get("hal_pressure", {}).get("r7_rate", 0)
        r9_s = v.get("seeded_r9", {}).get("rate", 0)
        lh_s = v.get("learned_hal", {}).get("win_rate", 0)
        has_action_diff = cand.get("action_audit", {}).get("diffs", 0) > 0

        s8 = cand.get("stress_results", {}).get(8192, {}).get("results", {})
        s2 = cand.get("stress_results", {}).get(2048, {}).get("results", {})

        static_ok = bp_s >= 0.80 and ht_s >= 0.36 and hp_s >= 0.36 and r9_s >= 1.0
        stress8_ok = (s8.get("bridge_pressure", {}).get("r7_rate", 0) >= 0.76
                      and s8.get("hal_death_trade", {}).get("r7_rate", 0) >= 0.32
                      and s8.get("hal_pressure", {}).get("r7_rate", 0) >= 0.32
                      and s8.get("seeded_r9", {}).get("rate", 0) >= 1.0)
        improved = lh_s >= 0.24 or ht_s >= 0.40 or hp_s >= 0.40
        primary = static_ok and stress8_ok and improved and has_action_diff

        stress2_ok = (s2.get("bridge_pressure", {}).get("r7_rate", 0) >= 0.76
                      and s2.get("hal_death_trade", {}).get("r7_rate", 0) >= 0.32
                      and s2.get("hal_pressure", {}).get("r7_rate", 0) >= 0.32
                      and s2.get("seeded_r9", {}).get("rate", 0) >= 1.0)
        modest = lh_s >= 0.20 or ht_s >= 0.38 or hp_s >= 0.38
        secondary = static_ok and stress2_ok and modest and has_action_diff

        if primary and best_bar != "primary":
            best = cand; best_bar = "primary"
        elif secondary and best is None:
            best = cand; best_bar = "secondary"

    return {"best_candidate": best["tag"] if best else None, "promotion_bar": best_bar,
            "primary_met": best_bar == "primary", "secondary_met": best_bar in ("primary", "secondary")}


# ── Main ──────────────────────────────────────────────────────────────

def main():
    start = time.time()
    print("=" * 60)
    print("SPRINT: Late-Head Reward Engineering")
    print("=" * 60)

    lane0 = lane_0()
    all_candidates, static_passing = lane_ab()

    if not static_passing:
        print("\n!! ALL CANDIDATES FAIL STATIC GATE !!")
        promotion = {"best_candidate": None, "promotion_bar": None, "primary_met": False, "secondary_met": False}
    else:
        lane_cd(static_passing)
        lane_e(static_passing)
        promotion = check_promotion(static_passing)

    out = {
        "sprint": "late-head-reward-engineering", "date": "2026-03-27",
        "lane0": lane0,
        "candidates": [
            {k: v for k, v in c.items() if k not in {"late_model", "reward_fn"}}
            for c in all_candidates
        ],
        "promotion": promotion,
    }
    out_dir = Path("docs/reports/json"); out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "sprint_late_head_reward_engineering_2026-03-27.json", "w") as f:
        json.dump(out, f, indent=2, default=str)

    elapsed = time.time() - start
    print(f"\nSprint completed in {elapsed:.0f}s")
    print(f"Promotion: {promotion.get('promotion_bar') or 'FAIL'}")
    if promotion.get("best_candidate"):
        print(f"Best candidate: {promotion['best_candidate']}")


if __name__ == "__main__":
    main()
