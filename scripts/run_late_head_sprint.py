#!/usr/bin/env python3
"""Sprint: Modular Late-Head Adaptation — full execution.

Tests whether stronger late-head training signal can produce real
improvement (learned-Hal win rate, scripted ht/hp) while keeping the
frozen opening head intact.
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
from training.curriculum import get_scenario
from training.modular_policy import (
    LateCurriculumEnv,
    OpeningAutoPlayEnv,
    build_modular_selector,
    train_opening_model_bc,
    verify_opening_accuracy,
)


# ── Paths ──────────────────────────────────────────────────────────────
def _resolve(stem: str) -> str:
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
LEARNED_HAL = _resolve("hal_vs_promoted_baku_prior_4096")

HAL_OVERRIDES = {3: 10}
HAL_OPPONENTS = ["hal_death_trade", "hal_pressure"]
ALL_OPPONENTS = ["bridge_pressure", "hal_death_trade", "hal_pressure"]
OPENING_HORIZON = 8


# ── Shared opening model (built once, reused) ─────────────────────────
_opening_model_cache = None

def get_opening_model():
    global _opening_model_cache
    if _opening_model_cache is None:
        model, _ = train_opening_model_bc(
            base_model_path=BASE_MODEL,
            opponent_names=HAL_OPPONENTS,
            action_overrides=HAL_OVERRIDES,
            opening_horizon=OPENING_HORIZON,
            epochs=80, lr=1e-3,
        )
        _opening_model_cache = model
    return _opening_model_cache


# ── Evaluation helpers ─────────────────────────────────────────────────

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


def evaluate_selector(selector, opponent_name: str, games: int = 24,
                       seed: int = 42, opponent_model_path: str | None = None) -> EvalResult:
    wins = 0; total_hr = 0; r7_count = 0; r7_eligible = 0; deaths = 0
    for gi in range(games):
        if opponent_model_path:
            opp = create_model_opponent(opponent_model_path, agent_role="hal")
        else:
            opp = create_scripted_opponent(opponent_name)
        env = DTHEnv(opponent=opp, agent_role="baku", seed=seed + gi)
        obs, _ = env.reset()
        selector.reset()
        reached_r7 = current_route_stage_flags(env.game).get("round7_pressure", False)
        started_before_r7 = stage_is_eligible_from_start(env.game, "round7_pressure")
        while True:
            mask = env.action_masks()
            action, _ = selector.predict(obs, action_masks=mask, deterministic=True)
            obs, reward, term, trunc, info = env.step(int(action))
            reached_r7 = reached_r7 or current_route_stage_flags(env.game).get("round7_pressure", False)
            if term or trunc:
                won = bool(term and env.game.winner is env.agent and reward > 0)
                wins += int(won)
                total_hr += len(env.game.history)
                if started_before_r7:
                    r7_eligible += 1
                    r7_count += int(reached_r7)
                deaths += env.agent.deaths
                break
    n = games
    return EvalResult(
        opponent=opponent_name, games=n, wins=wins,
        win_rate=wins / n, avg_half_rounds=total_hr / n,
        r7_count=r7_count, r7_eligible=r7_eligible,
        r7_rate=r7_count / r7_eligible if r7_eligible else 0,
        deaths_by_agent=deaths,
    )


def eval_r9_seeded(model: MaskablePPO, seed: int = 42) -> dict:
    opp = create_scripted_opponent("safe")
    env = DTHEnv(opponent=opp, agent_role="baku", seed=seed,
                 scenario_sampler=lambda _rng: get_scenario("round9_pre_leap"))
    obs, _ = env.reset()
    while True:
        mask = env.action_masks()
        action, _ = model.predict(obs, action_masks=mask, deterministic=True)
        obs, reward, term, trunc, _ = env.step(int(action))
        if term or trunc:
            won = bool(term and env.game.winner is env.agent and reward > 0)
            return {"wins": int(won), "games": 1, "rate": float(won)}
    return {"wins": 0, "games": 1, "rate": 0.0}


def eval_suite(selector, games: int = 24, seed: int = 42, tag: str = "",
               r9_model: MaskablePPO | None = None, lhal_games: int = 0) -> dict:
    results = {}
    pct = lambda v: f"{v*100:.0f}%"
    for opp in ALL_OPPONENTS:
        res = evaluate_selector(selector, opp, games=games, seed=seed)
        results[opp] = asdict(res)
        print(f"  {tag} vs {opp}: r7={pct(res.r7_rate)} wins={pct(res.win_rate)} avg_hr={res.avg_half_rounds:.1f}")

    if r9_model is not None:
        r9_res = eval_r9_seeded(r9_model, seed=seed)
        results["seeded_r9"] = r9_res
        print(f"  {tag} seeded r9: {r9_res['wins']}/{r9_res['games']} ({pct(r9_res['rate'])})")

    if lhal_games > 0:
        lhal_res = evaluate_selector(selector, "learned_hal", games=lhal_games, seed=seed,
                                      opponent_model_path=LEARNED_HAL)
        results["learned_hal"] = asdict(lhal_res)
        print(f"  {tag} vs learned_hal: wins={pct(lhal_res.win_rate)} deaths={lhal_res.deaths_by_agent} avg_hr={lhal_res.avg_half_rounds:.1f}")

    return results


# ── League builders ────────────────────────────────────────────────────

def make_hal_league(lhal_weight: float = 0.5, seed: int = 42) -> WeightedOpponentLeague:
    entries = [
        LeagueEntry(label="ht", weight=1.0, opponent=create_scripted_opponent("hal_death_trade")),
        LeagueEntry(label="hp", weight=1.0, opponent=create_scripted_opponent("hal_pressure")),
        LeagueEntry(label="lhal", weight=lhal_weight, opponent=create_model_opponent(LEARNED_HAL, agent_role="hal")),
    ]
    return WeightedOpponentLeague(entries, seed=seed)


def make_lhal_heavy_league(lhal_weight: float = 4.0, scripted_weight: float = 1.0,
                            seed: int = 42) -> WeightedOpponentLeague:
    entries = [
        LeagueEntry(label="ht", weight=scripted_weight, opponent=create_scripted_opponent("hal_death_trade")),
        LeagueEntry(label="hp", weight=scripted_weight, opponent=create_scripted_opponent("hal_pressure")),
        LeagueEntry(label="lhal", weight=lhal_weight, opponent=create_model_opponent(LEARNED_HAL, agent_role="hal")),
    ]
    return WeightedOpponentLeague(entries, seed=seed)


# ── Training helpers ───────────────────────────────────────────────────

def train_late_model_autoplay(opening_model, timesteps: int, league, seed: int = 42,
                               use_shaping: bool = False, shaping_preset: str = "light"):
    env = OpeningAutoPlayEnv(
        opening_model=opening_model,
        opening_horizon=OPENING_HORIZON,
        opponent=league,
        agent_role="baku",
        seed=seed,
        use_shaping=use_shaping,
        shaping_preset=shaping_preset,
    )
    model = MaskablePPO.load(BASE_MODEL, env=env)
    model.learn(total_timesteps=timesteps)
    return model


def train_late_model_curriculum(opening_model, timesteps: int, league,
                                 p_opening: float = 0.5,
                                 late_scenarios: list[str] | None = None,
                                 seed: int = 42,
                                 use_shaping: bool = True,
                                 shaping_preset: str = "bridge"):
    env = LateCurriculumEnv(
        opening_model=opening_model,
        opening_horizon=OPENING_HORIZON,
        p_opening=p_opening,
        late_scenarios=late_scenarios or ["round7_pressure", "round8_bridge"],
        opponent=league,
        agent_role="baku",
        seed=seed,
        use_shaping=use_shaping,
        shaping_preset=shaping_preset,
    )
    model = MaskablePPO.load(BASE_MODEL, env=env)
    model.learn(total_timesteps=timesteps)
    return model


def build_and_eval(late_model, tag: str, games: int = 24, lhal_games: int = 24):
    opening_model = get_opening_model()
    selector = build_modular_selector(
        bp_specialist_path=BP_SPECIALIST,
        base_model_path=BASE_MODEL,
        opening_model=opening_model,
        late_model=late_model,
        opening_horizon=OPENING_HORIZON,
    )
    return eval_suite(selector, games=games, tag=tag, r9_model=late_model,
                      lhal_games=lhal_games)


# ── Lane 0: Baseline reproduction ─────────────────────────────────────

def lane_0() -> dict:
    print("\n" + "=" * 60)
    print("LANE 0: Baseline Reproduction (A1_h8)")
    print("=" * 60)

    opening_model = get_opening_model()
    base_late = MaskablePPO.load(BASE_MODEL)

    print("\n--- Static baseline ---")
    static_res = build_and_eval(base_late, "baseline_static", games=24, lhal_games=24)

    return {"static": static_res}


# ── Lane A: Late-head signal families ─────────────────────────────────

def lane_a() -> list[dict]:
    print("\n" + "=" * 60)
    print("LANE A: Late-Head Signal Families")
    print("=" * 60)

    opening_model = get_opening_model()
    candidates = []

    # ── Family L1: Longer PPO on OpeningAutoPlayEnv ──
    for steps, tag_suffix in [(32768, "32k"), (65536, "64k")]:
        tag = f"L1_{tag_suffix}"
        print(f"\n--- {tag}: longer PPO, {steps} steps ---")
        league = make_hal_league(lhal_weight=0.5)
        late_model = train_late_model_autoplay(opening_model, timesteps=steps, league=league)
        candidates.append({
            "tag": tag, "family": "L1", "late_model": late_model,
            "ppo_steps": steps, "description": f"OpeningAutoPlayEnv, {steps} steps, ht+hp+lhal(0.5)",
        })

    # ── Family L2: Scenario-augmented late curriculum ──
    for p_open, tag_suffix in [(0.5, "50"), (0.3, "30")]:
        tag = f"L2_p{tag_suffix}"
        print(f"\n--- {tag}: scenario curriculum, p_opening={p_open}, 32k steps ---")
        league = make_hal_league(lhal_weight=0.5)
        late_model = train_late_model_curriculum(
            opening_model, timesteps=32768, league=league,
            p_opening=p_open, late_scenarios=["round7_pressure", "round8_bridge"],
            use_shaping=True, shaping_preset="bridge",
        )
        candidates.append({
            "tag": tag, "family": "L2", "late_model": late_model,
            "ppo_steps": 32768, "description": f"LateCurriculum p_opening={p_open}, shaping=bridge",
        })

    # ── Family L3: Learned-Hal-heavy curriculum ──
    for lhal_w, tag_suffix in [(4.0, "4x"), (8.0, "8x")]:
        tag = f"L3_lhal{tag_suffix}"
        print(f"\n--- {tag}: learned-Hal-heavy, lhal_weight={lhal_w}, 32k steps ---")
        league = make_lhal_heavy_league(lhal_weight=lhal_w, scripted_weight=1.0)
        late_model = train_late_model_autoplay(
            opening_model, timesteps=32768, league=league,
            use_shaping=True, shaping_preset="light",
        )
        candidates.append({
            "tag": tag, "family": "L3", "late_model": late_model,
            "ppo_steps": 32768, "description": f"OpeningAutoPlay, lhal_weight={lhal_w}, shaping=light",
        })

    return candidates


# ── Lane B: Static safety gate ────────────────────────────────────────

def lane_b(candidates: list[dict]) -> list[dict]:
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


# ── Lane C: Adaptation evaluation + stress ────────────────────────────

def stress_test(cand: dict, ppo_steps: int, seed: int = 42) -> dict:
    """PPO-stress the late model and evaluate."""
    tag = cand["tag"]
    opening_model = get_opening_model()

    print(f"\n  Stress {ppo_steps} steps for {tag}...")
    league = make_hal_league(lhal_weight=0.5, seed=seed)
    env = OpeningAutoPlayEnv(
        opening_model=opening_model, opening_horizon=OPENING_HORIZON,
        opponent=league, agent_role="baku", seed=seed,
    )
    late_model = MaskablePPO.load(BASE_MODEL, env=env)
    late_model.learn(total_timesteps=ppo_steps)

    selector = build_modular_selector(
        bp_specialist_path=BP_SPECIALIST, base_model_path=BASE_MODEL,
        opening_model=opening_model, late_model=late_model,
        opening_horizon=OPENING_HORIZON,
    )
    res = eval_suite(selector, games=24, tag=f"{tag}_stress{ppo_steps}",
                     r9_model=late_model, lhal_games=24)
    return {"ppo_steps": ppo_steps, "results": res, "late_model": late_model}


def lane_c(candidates: list[dict]) -> list[dict]:
    print("\n" + "=" * 60)
    print("LANE C: Adaptation Evaluation + Stress")
    print("=" * 60)

    passing = []
    for cand in candidates:
        tag = cand["tag"]
        cand["stress_results"] = {}

        for steps in [2048, 8192]:
            st = stress_test(cand, ppo_steps=steps)
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

def lane_d(candidates: list[dict]) -> None:
    print("\n" + "=" * 60)
    print("LANE D: Full Validation")
    print("=" * 60)

    for cand in candidates:
        tag = cand["tag"]
        print(f"\n--- 50-game validation: {tag} ---")
        res = build_and_eval(cand["late_model"], f"{tag}_50g", games=50, lhal_games=50)
        cand["full_validation"] = res


# ── Promotion check ───────────────────────────────────────────────────

def check_promotion(candidates: list[dict], baseline_lhal_wr: float = 0.16) -> dict:
    best = None
    best_bar = None

    for cand in candidates:
        static = cand.get("full_validation", cand.get("static_results", {}))
        bp_s = static.get("bridge_pressure", {}).get("r7_rate", 0)
        ht_s = static.get("hal_death_trade", {}).get("r7_rate", 0)
        hp_s = static.get("hal_pressure", {}).get("r7_rate", 0)
        r9_s = static.get("seeded_r9", {}).get("rate", 0)
        lh_s = static.get("learned_hal", {}).get("win_rate", 0)

        s8192 = cand.get("stress_results", {}).get(8192, {}).get("results", {})
        bp_8 = s8192.get("bridge_pressure", {}).get("r7_rate", 0)
        ht_8 = s8192.get("hal_death_trade", {}).get("r7_rate", 0)
        hp_8 = s8192.get("hal_pressure", {}).get("r7_rate", 0)
        r9_8 = s8192.get("seeded_r9", {}).get("rate", 0)
        lh_8 = s8192.get("learned_hal", {}).get("win_rate", 0)

        s2048 = cand.get("stress_results", {}).get(2048, {}).get("results", {})
        bp_2 = s2048.get("bridge_pressure", {}).get("r7_rate", 0)
        ht_2 = s2048.get("hal_death_trade", {}).get("r7_rate", 0)
        hp_2 = s2048.get("hal_pressure", {}).get("r7_rate", 0)
        r9_2 = s2048.get("seeded_r9", {}).get("rate", 0)
        lh_2 = s2048.get("learned_hal", {}).get("win_rate", 0)

        # Primary: static 80/36/36 + stress@8192 76/32/32 + improvement
        static_ok = bp_s >= 0.80 and ht_s >= 0.36 and hp_s >= 0.36 and r9_s >= 1.0
        stress8_ok = bp_8 >= 0.76 and ht_8 >= 0.32 and hp_8 >= 0.32 and r9_8 >= 1.0
        improved = lh_s >= 0.24 or lh_8 >= 0.24 or ht_s >= 0.40 or hp_s >= 0.40
        primary = static_ok and stress8_ok and improved

        # Secondary: static 80/36/36 + stress@2048 76/32/32 + modest improvement
        stress2_ok = bp_2 >= 0.76 and ht_2 >= 0.32 and hp_2 >= 0.32 and r9_2 >= 1.0
        modest = lh_s >= 0.20 or lh_2 >= 0.20 or ht_s >= 0.38 or hp_s >= 0.38
        secondary = static_ok and stress2_ok and modest

        if primary and best_bar != "primary":
            best = cand; best_bar = "primary"
        elif secondary and best is None:
            best = cand; best_bar = "secondary"

    return {
        "best_candidate": best["tag"] if best else None,
        "promotion_bar": best_bar,
        "primary_met": best_bar == "primary",
        "secondary_met": best_bar in ("primary", "secondary"),
    }


# ── Main ──────────────────────────────────────────────────────────────

def main():
    start = time.time()
    print("=" * 60)
    print("SPRINT: Modular Late-Head Adaptation")
    print("=" * 60)

    lane0_results = lane_0()
    all_candidates = lane_a()
    static_passing = lane_b(all_candidates)

    if not static_passing:
        print("\n!! ALL CANDIDATES FAIL STATIC GATE — SPRINT FAIL !!")
        promotion = {"best_candidate": None, "promotion_bar": None,
                     "primary_met": False, "secondary_met": False}
    else:
        stress_passing = lane_c(static_passing)
        lane_d(static_passing)  # validate all static-passing, not just stress-passing
        promotion = check_promotion(static_passing)

    # Save results
    out = {
        "sprint": "modular-late-head-adaptation",
        "date": "2026-03-27",
        "lane0": lane0_results,
        "candidates": [
            {k: v for k, v in c.items() if k != "late_model"}
            for c in all_candidates
        ],
        "promotion": promotion,
    }
    out_dir = Path("docs/reports/json")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "sprint_modular_late_head_adaptation_2026-03-27.json"
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
