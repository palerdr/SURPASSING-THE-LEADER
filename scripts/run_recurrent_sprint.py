#!/usr/bin/env python3
"""Sprint: Recurrent Late Head — full execution.

Tests whether an LSTM late head can break the MlpPolicy deterministic
argmax ceiling while keeping the frozen opening head intact.
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
from sb3_contrib import MaskablePPO, RecurrentPPO

from environment.dth_env import DTHEnv
from environment.opponents.factory import create_model_opponent, create_scripted_opponent
from environment.opponents.league import WeightedOpponentLeague, LeagueEntry
from environment.route_stages import current_route_stage_flags, stage_is_eligible_from_start
from training.curriculum import get_scenario
from training.modular_policy import (
    ActionClampEnv,
    LateCurriculumEnv,
    OpeningAutoPlayEnv,
    RecurrentModularBakuSelector,
    build_modular_selector,
    train_opening_model_bc,
)
from training.expert_selector import build_classifier


# ── Paths ──────────────────────────────────────────────────────────────
def _resolve(stem: str) -> str:
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
        model, _ = train_opening_model_bc(
            base_model_path=BASE_MODEL, opponent_names=HAL_OPPONENTS,
            action_overrides=HAL_OVERRIDES, opening_horizon=OPENING_HORIZON,
            epochs=80, lr=1e-3,
        )
        _opening_model_cache = model
    return _opening_model_cache


# ── Evaluation helpers ─────────────────────────────────────────────────

@dataclass
class EvalResult:
    opponent: str; games: int; wins: int; win_rate: float
    avg_half_rounds: float; r7_count: int; r7_eligible: int; r7_rate: float
    deaths_by_agent: int


def evaluate_selector(selector, opponent_name, games=24, seed=42,
                       opponent_model_path=None):
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


def eval_r9_seeded(model, seed=42):
    opp = create_scripted_opponent("safe")
    env = DTHEnv(opponent=opp, agent_role="baku", seed=seed,
                 scenario_sampler=lambda _rng: get_scenario("round9_pre_leap"))
    obs, _ = env.reset()
    # Use base model for r9 (late-game scenario)
    while True:
        mask = env.action_masks()
        action, _ = model.predict(obs, action_masks=mask, deterministic=True)
        obs, reward, term, trunc, _ = env.step(int(action))
        if term or trunc:
            won = bool(term and env.game.winner is env.agent and reward > 0)
            return {"wins": int(won), "games": 1, "rate": float(won)}


def eval_r9_seeded_selector(selector, seed=42):
    """Evaluate seeded r9 with the full selector, not the base model.

    This is the correct regression check for recurrent late-head candidates,
    because the scenario starts beyond the opening horizon and should exercise
    the late model through the selector path.
    """
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

    # Seeded r9 should exercise the selector's late-model path.
    r9 = eval_r9_seeded_selector(selector, seed=seed)
    results["seeded_r9"] = r9
    print(f"  {tag} seeded r9: {r9['wins']}/{r9['games']} ({pct(r9['rate'])})")

    if lhal_games > 0:
        lh = evaluate_selector(selector, "learned_hal", games=lhal_games, seed=seed,
                                opponent_model_path=LEARNED_HAL)
        results["learned_hal"] = asdict(lh)
        print(f"  {tag} vs learned_hal: wins={pct(lh.win_rate)} deaths={lh.deaths_by_agent} avg_hr={lh.avg_half_rounds:.1f}")
    return results


# ── League + training helpers ──────────────────────────────────────────

def make_hal_league(lhal_weight=0.5, seed=42):
    return WeightedOpponentLeague([
        LeagueEntry(label="ht", weight=1.0, opponent=create_scripted_opponent("hal_death_trade")),
        LeagueEntry(label="hp", weight=1.0, opponent=create_scripted_opponent("hal_pressure")),
        LeagueEntry(label="lhal", weight=lhal_weight, opponent=create_model_opponent(LEARNED_HAL, agent_role="hal")),
    ], seed=seed)


def train_recurrent_late(opening_model, timesteps, league, hidden_size=64,
                          n_lstm_layers=1, seed=42, use_shaping=True,
                          shaping_preset="bridge", use_curriculum=False,
                          p_opening=0.5):
    """Train a RecurrentPPO late model on the opening-autoplay env."""
    if use_curriculum:
        base_env = LateCurriculumEnv(
            opening_model=opening_model, opening_horizon=OPENING_HORIZON,
            p_opening=p_opening, late_scenarios=["round7_pressure", "round8_bridge"],
            opponent=league, agent_role="baku", seed=seed,
            use_shaping=use_shaping, shaping_preset=shaping_preset,
        )
    else:
        base_env = OpeningAutoPlayEnv(
            opening_model=opening_model, opening_horizon=OPENING_HORIZON,
            opponent=league, agent_role="baku", seed=seed,
            use_shaping=use_shaping, shaping_preset=shaping_preset,
        )
    env = ActionClampEnv(base_env)

    model = RecurrentPPO(
        "MlpLstmPolicy", env,
        learning_rate=3e-4, n_steps=512, batch_size=64, n_epochs=10,
        gamma=0.99, verbose=1, seed=seed,
        policy_kwargs=dict(lstm_hidden_size=hidden_size, n_lstm_layers=n_lstm_layers),
    )
    model.learn(total_timesteps=timesteps)
    return model


def build_recurrent_selector(recurrent_late_model):
    opening_model = get_opening_model()
    bp_specialist = MaskablePPO.load(BP_SPECIALIST)
    clf = build_classifier(BASE_MODEL, ["bridge_pressure", "hal_death_trade"], classify_turn=2)
    return RecurrentModularBakuSelector(
        bp_specialist=bp_specialist, opening_model=opening_model,
        late_model=recurrent_late_model, classifier=clf,
        opening_horizon=OPENING_HORIZON, classify_turn=2,
    )


def build_and_eval(late_model, tag, games=24, lhal_games=24):
    selector = build_recurrent_selector(late_model)
    return eval_suite(selector, games=games, tag=tag, lhal_games=lhal_games)


# ── Lane 0: Baseline + feasibility ────────────────────────────────────

def lane_0():
    print("\n" + "=" * 60)
    print("LANE 0: Recurrent Feasibility + Baseline")
    print("=" * 60)

    print("\nRecurrent stack:")
    print("  Algorithm: RecurrentPPO (sb3_contrib)")
    print("  Policy: MlpLstmPolicy")
    print("  Action masking: ActionClampEnv wrapper (clamps action 60 on non-leap turns)")
    print("  Integration: RecurrentModularBakuSelector manages LSTM state")

    # MLP baseline for comparison
    print("\n--- MLP baseline (for comparison) ---")
    opening_model = get_opening_model()
    base_late = MaskablePPO.load(BASE_MODEL)
    mlp_sel = build_modular_selector(
        bp_specialist_path=BP_SPECIALIST, base_model_path=BASE_MODEL,
        opening_model=opening_model, late_model=base_late,
        opening_horizon=OPENING_HORIZON,
    )
    mlp_res = eval_suite(mlp_sel, games=24, tag="mlp_baseline", lhal_games=24)
    return {"mlp_baseline": mlp_res}


# ── Lane A: Recurrent late-head families ──────────────────────────────

def lane_a():
    print("\n" + "=" * 60)
    print("LANE A: Recurrent Late-Head Families")
    print("=" * 60)

    opening_model = get_opening_model()
    candidates = []

    # R1: LSTM h=64, different budgets
    for steps, stag in [(65536, "64k"), (131072, "128k")]:
        tag = f"R1_{stag}"
        print(f"\n--- {tag}: LSTM h=64, {steps} steps, bridge shaping ---")
        league = make_hal_league(lhal_weight=0.5)
        late = train_recurrent_late(opening_model, timesteps=steps, league=league,
                                     hidden_size=64, use_shaping=True, shaping_preset="bridge")
        candidates.append({"tag": tag, "family": "R1", "late_model": late,
                           "ppo_steps": steps, "hidden_size": 64})

    # R2: LSTM h=128
    for steps, stag in [(65536, "64k"), (131072, "128k")]:
        tag = f"R2_{stag}"
        print(f"\n--- {tag}: LSTM h=128, {steps} steps, bridge shaping ---")
        league = make_hal_league(lhal_weight=0.5)
        late = train_recurrent_late(opening_model, timesteps=steps, league=league,
                                     hidden_size=128, use_shaping=True, shaping_preset="bridge")
        candidates.append({"tag": tag, "family": "R2", "late_model": late,
                           "ppo_steps": steps, "hidden_size": 128})

    # R3: LSTM h=64 with scenario curriculum
    for p_open, stag in [(0.5, "p50"), (0.3, "p30")]:
        tag = f"R3_{stag}"
        print(f"\n--- {tag}: LSTM h=64, curriculum p_opening={p_open}, 65k steps ---")
        league = make_hal_league(lhal_weight=0.5)
        late = train_recurrent_late(opening_model, timesteps=65536, league=league,
                                     hidden_size=64, use_shaping=True, shaping_preset="bridge",
                                     use_curriculum=True, p_opening=p_open)
        candidates.append({"tag": tag, "family": "R3", "late_model": late,
                           "ppo_steps": 65536, "hidden_size": 64})

    return candidates


# ── Lane B: Static safety gate ────────────────────────────────────────

def lane_b(candidates):
    print("\n" + "=" * 60)
    print("LANE B: Static Safety Gate")
    print("=" * 60)

    passing = []
    for cand in candidates:
        tag = cand["tag"]
        print(f"\n--- Static safety: {tag} ---")
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

    return passing


# ── Lane C: Stress test ───────────────────────────────────────────────

def stress_recurrent(cand, ppo_steps, seed=42):
    tag = cand["tag"]
    opening_model = get_opening_model()
    print(f"\n  Stress {ppo_steps} steps for {tag}...")
    league = make_hal_league(lhal_weight=0.5, seed=seed)
    base_env = OpeningAutoPlayEnv(
        opening_model=opening_model, opening_horizon=OPENING_HORIZON,
        opponent=league, agent_role="baku", seed=seed,
        use_shaping=True, shaping_preset="bridge",
    )
    env = ActionClampEnv(base_env)
    model = RecurrentPPO(
        "MlpLstmPolicy", env, learning_rate=3e-4, n_steps=512, batch_size=64,
        n_epochs=10, gamma=0.99, verbose=0, seed=seed,
        policy_kwargs=dict(lstm_hidden_size=cand.get("hidden_size", 64), n_lstm_layers=1),
    )
    model.learn(total_timesteps=ppo_steps)
    selector = build_recurrent_selector(model)
    res = eval_suite(selector, games=24, tag=f"{tag}_stress{ppo_steps}", lhal_games=24)
    return {"ppo_steps": ppo_steps, "results": res, "late_model": model}


def lane_c(candidates):
    print("\n" + "=" * 60)
    print("LANE C: Recurrent Adaptation Stress")
    print("=" * 60)

    passing = []
    for cand in candidates:
        tag = cand["tag"]
        cand["stress_results"] = {}
        for steps in [2048, 8192]:
            st = stress_recurrent(cand, ppo_steps=steps)
            cand["stress_results"][steps] = st
            res = st["results"]
            bp = res.get("bridge_pressure", {}).get("r7_rate", 0)
            ht = res.get("hal_death_trade", {}).get("r7_rate", 0)
            hp = res.get("hal_pressure", {}).get("r7_rate", 0)
            r9 = res.get("seeded_r9", {}).get("rate", 0)
            lh = res.get("learned_hal", {}).get("win_rate", 0)
            gate = bp >= 0.76 and ht >= 0.32 and hp >= 0.32 and r9 >= 1.0
            st["gate_pass"] = gate
            pct = lambda v: f"{v*100:.0f}%"
            status = "PASS" if gate else "FAIL"
            print(f"  >> {tag} @ {steps}: {status} (bp={pct(bp)} ht={pct(ht)} hp={pct(hp)} r9={pct(r9)} lhal={pct(lh)})")
        if cand["stress_results"].get(2048, {}).get("gate_pass"):
            passing.append(cand)
    return passing


# ── Lane D: Full validation ───────────────────────────────────────────

def lane_d(candidates):
    print("\n" + "=" * 60)
    print("LANE D: Full Validation")
    print("=" * 60)
    for cand in candidates:
        tag = cand["tag"]
        print(f"\n--- 50-game validation: {tag} ---")
        res = build_and_eval(cand["late_model"], f"{tag}_50g", games=50, lhal_games=50)
        cand["full_validation"] = res


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

        s8 = cand.get("stress_results", {}).get(8192, {}).get("results", {})
        lh_8 = s8.get("learned_hal", {}).get("win_rate", 0)
        s2 = cand.get("stress_results", {}).get(2048, {}).get("results", {})

        static_ok = bp_s >= 0.80 and ht_s >= 0.36 and hp_s >= 0.36 and r9_s >= 1.0
        stress8_ok = (
            s8.get("bridge_pressure", {}).get("r7_rate", 0) >= 0.76
            and s8.get("hal_death_trade", {}).get("r7_rate", 0) >= 0.32
            and s8.get("hal_pressure", {}).get("r7_rate", 0) >= 0.32
            and s8.get("seeded_r9", {}).get("rate", 0) >= 1.0
        )
        improved = lh_s >= 0.24 or lh_8 >= 0.24 or ht_s >= 0.40 or hp_s >= 0.40
        primary = static_ok and stress8_ok and improved

        stress2_ok = (
            s2.get("bridge_pressure", {}).get("r7_rate", 0) >= 0.76
            and s2.get("hal_death_trade", {}).get("r7_rate", 0) >= 0.32
            and s2.get("hal_pressure", {}).get("r7_rate", 0) >= 0.32
            and s2.get("seeded_r9", {}).get("rate", 0) >= 1.0
        )
        modest = lh_s >= 0.20 or ht_s >= 0.38 or hp_s >= 0.38
        secondary = static_ok and stress2_ok and modest

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
    print("SPRINT: Recurrent Late Head")
    print("=" * 60)

    lane0_results = lane_0()
    all_candidates = lane_a()
    static_passing = lane_b(all_candidates)

    if not static_passing:
        print("\n!! ALL CANDIDATES FAIL STATIC GATE !!")
        promotion = {"best_candidate": None, "promotion_bar": None,
                     "primary_met": False, "secondary_met": False}
    else:
        lane_c(static_passing)
        lane_d(static_passing)
        promotion = check_promotion(static_passing)

    out = {
        "sprint": "recurrent-late-head", "date": "2026-03-27",
        "lane0": lane0_results,
        "candidates": [{k: v for k, v in c.items() if k != "late_model"}
                       for c in all_candidates],
        "promotion": promotion,
    }
    out_dir = Path("docs/reports/json"); out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "sprint_recurrent_late_head_2026-03-27.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")

    elapsed = time.time() - start
    print(f"\nSprint completed in {elapsed:.0f}s")
    print(f"Promotion: {promotion.get('promotion_bar') or 'FAIL'}")
    if promotion.get("best_candidate"):
        print(f"Best candidate: {promotion['best_candidate']}")


if __name__ == "__main__":
    main()
