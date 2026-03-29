#!/usr/bin/env python3
"""Pilot: Constrained Self-Play / Late-Head Adaptation at p=0.88.

Timeboxed pilot to determine whether any real adaptation signal exists
under the corrected p=0.88 regime. Reuses existing artifacts from the
full sprint and runs exactly 3 materially different adaptation families
at modest budgets (12k steps each).

Reused artifacts:
  - Opening BC model (trained fresh from BASELINE_OVERRIDES)
  - _fresh_hal_fresh_32k.zip (pre-trained Hal at p=0.88)
  - BASE_MODEL, BP_SPECIALIST checkpoints
  - All evaluation infrastructure from prior sprints

Skipped for time:
  - Fresh Hal generation (reusing existing _fresh_hal_fresh_32k)
  - Degeneracy check (already passed in full sprint)
  - Stress testing / PPO perturbation lanes
  - Safety gate → direct 50-game evaluation

3 Adaptation Families (materially different mechanisms):
  A1: MLP + OpeningAutoPlay + bridge shaping + balanced league (vanilla late PPO)
  A2: MLP + LateCurriculum + scenario augmentation + bridge shaping (diversity)
  A3: MLP + OpeningAutoPlay + lhal-heavy(4x) league + light shaping (targeted)
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

from src.Constants import PHYSICALITY_BAKU
from environment.dth_env import DTHEnv
from environment.opponents.factory import create_model_opponent, create_scripted_opponent
from environment.opponents.league import WeightedOpponentLeague, LeagueEntry
from environment.route_stages import current_route_stage_flags, stage_is_eligible_from_start
from training.curriculum import get_scenario
from training.modular_policy import (
    LateCurriculumEnv,
    OpeningAutoPlayEnv,
    build_modular_selector,
    collect_opening_samples,
    verify_opening_accuracy,
)
from training.behavior_clone import behavior_clone_policy

# ── Paths ──────────────────────────────────────────────────────────────
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

# Reuse existing learned Hals
LEARNED_HAL_PRIOR = _resolve("hal_vs_promoted_baku_prior_4096")
FRESH_HAL_32K = _resolve("_fresh_hal_fresh_32k")

BASELINE_ROUTE = [1, 1, 1, 10, 60, 1, 60, 1]
BASELINE_OVERRIDES = {t: BASELINE_ROUTE[t] for t in range(8)}
OPENING_HORIZON = 8
PILOT_STEPS = 12288  # 12k — modest budget per family
EVAL_GAMES = 50
HAL_OPPONENTS = ["hal_death_trade", "hal_pressure"]
ALL_OPPONENTS = ["bridge_pressure", "hal_death_trade", "hal_pressure"]


# ── Helpers ────────────────────────────────────────────────────────────

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
        opp = (
            create_model_opponent(opp_model_path, agent_role="hal")
            if opp_model_path
            else create_scripted_opponent(opp_name)
        )
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
                "round7_pressure", False
            )
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


def eval_suite(selector, games=50, seed=42, tag="", lhal_games=50, lhal_path=None):
    results = {}
    hal_path = lhal_path or FRESH_HAL_32K
    for opp in ALL_OPPONENTS:
        res = evaluate_selector(selector, opp, games=games, seed=seed)
        results[opp] = asdict(res)
        print(f"  {tag} vs {opp}: r7={pct(res.r7_rate)} wins={pct(res.win_rate)} avg_hr={res.avg_half_rounds:.1f}")

    r9 = eval_r9_seeded(selector, seed=seed)
    results["seeded_r9"] = r9
    print(f"  {tag} seeded r9: {r9['wins']}/{r9['games']} ({pct(r9['rate'])})")

    if lhal_games > 0:
        lh = evaluate_selector(
            selector, "learned_hal", games=lhal_games, seed=seed,
            opp_model_path=hal_path,
        )
        results["learned_hal"] = asdict(lh)
        print(f"  {tag} vs learned_hal: wins={pct(lh.win_rate)} deaths={lh.deaths_by_agent} avg_hr={lh.avg_half_rounds:.1f}")

    return results


# ── Opening model (built once) ────────────────────────────────────────

_opening_model_cache = None
_bc_stats_cache = None


def get_opening_model():
    global _opening_model_cache, _bc_stats_cache
    if _opening_model_cache is None:
        samples = collect_opening_samples(
            BASE_MODEL, HAL_OPPONENTS, BASELINE_OVERRIDES, OPENING_HORIZON,
            seeds=range(200),
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
        _bc_stats_cache = {
            "accuracy": accuracy["accuracy"],
            "misaligned": accuracy["total"] - accuracy["correct"],
            "correct": accuracy["correct"],
            "total": accuracy["total"],
        }
        _opening_model_cache = opening_model
        print(f"  Opening BC: {_bc_stats_cache['accuracy']:.1%} ({_bc_stats_cache['correct']}/{_bc_stats_cache['total']}), misaligned={_bc_stats_cache['misaligned']}")
    return _opening_model_cache


def get_bc_stats():
    get_opening_model()
    return _bc_stats_cache


def build_baseline_selector(late_model=None):
    opening_model = get_opening_model()
    if late_model is None:
        late_model = MaskablePPO.load(BASE_MODEL)
    return build_modular_selector(
        bp_specialist_path=BP_SPECIALIST, base_model_path=BASE_MODEL,
        opening_model=opening_model, late_model=late_model,
        opening_horizon=OPENING_HORIZON,
    )


# ── League builders ──────────────────────────────────────────────────

def make_balanced_league(lhal_weight=0.5, seed=42, lhal_path=None):
    hal_path = lhal_path or FRESH_HAL_32K
    return WeightedOpponentLeague([
        LeagueEntry(label="ht", weight=1.0,
                    opponent=create_scripted_opponent("hal_death_trade")),
        LeagueEntry(label="hp", weight=1.0,
                    opponent=create_scripted_opponent("hal_pressure")),
        LeagueEntry(label="lhal", weight=lhal_weight,
                    opponent=create_model_opponent(hal_path, agent_role="hal")),
    ], seed=seed)


def make_lhal_heavy_league(lhal_weight=4.0, scripted_weight=1.0, seed=42,
                            lhal_path=None):
    hal_path = lhal_path or FRESH_HAL_32K
    return WeightedOpponentLeague([
        LeagueEntry(label="ht", weight=scripted_weight,
                    opponent=create_scripted_opponent("hal_death_trade")),
        LeagueEntry(label="hp", weight=scripted_weight,
                    opponent=create_scripted_opponent("hal_pressure")),
        LeagueEntry(label="lhal", weight=lhal_weight,
                    opponent=create_model_opponent(hal_path, agent_role="hal")),
    ], seed=seed)


# ── Training helpers ─────────────────────────────────────────────────

def train_late_autoplay(timesteps, league, seed=42, use_shaping=True,
                         shaping_preset="bridge"):
    opening_model = get_opening_model()
    env = OpeningAutoPlayEnv(
        opening_model=opening_model, opening_horizon=OPENING_HORIZON,
        opponent=league, agent_role="baku", seed=seed,
        use_shaping=use_shaping, shaping_preset=shaping_preset,
    )
    model = MaskablePPO.load(BASE_MODEL, env=env)
    model.learn(total_timesteps=timesteps)
    return model


def train_late_curriculum(timesteps, league, p_opening=0.4,
                           late_scenarios=None, seed=42,
                           use_shaping=True, shaping_preset="bridge"):
    opening_model = get_opening_model()
    env = LateCurriculumEnv(
        opening_model=opening_model, opening_horizon=OPENING_HORIZON,
        p_opening=p_opening,
        late_scenarios=late_scenarios or ["round7_pressure", "round8_bridge"],
        opponent=league, agent_role="baku", seed=seed,
        use_shaping=use_shaping, shaping_preset=shaping_preset,
    )
    model = MaskablePPO.load(BASE_MODEL, env=env)
    model.learn(total_timesteps=timesteps)
    return model


def build_and_eval(late_model, tag, games=50, lhal_games=50, lhal_path=None):
    selector = build_baseline_selector(late_model)
    return eval_suite(selector, games=games, tag=tag, lhal_games=lhal_games,
                      lhal_path=lhal_path)


# ══════════════════════════════════════════════════════════════════════
# LANE 0: Baseline (reuse corrected p=0.88 selector)
# ══════════════════════════════════════════════════════════════════════

def lane_0_baseline():
    print("\n" + "=" * 60)
    print(f"LANE 0: p=0.88 Baseline (PHYSICALITY_BAKU={PHYSICALITY_BAKU})")
    print(f"Route: {BASELINE_ROUTE}")
    print(f"Learned Hal: fresh_hal_32k (reused)")
    print("=" * 60)

    bc_stats = get_bc_stats()
    print(f"  BC: {bc_stats['accuracy']:.1%} ({bc_stats['correct']}/{bc_stats['total']}), misaligned={bc_stats['misaligned']}")

    selector = build_baseline_selector()
    results = eval_suite(selector, games=EVAL_GAMES, tag="baseline",
                         lhal_games=EVAL_GAMES)
    results["bc_stats"] = bc_stats
    return results


# ══════════════════════════════════════════════════════════════════════
# LANE 1: 3 Adaptation Families (12k steps each)
# ══════════════════════════════════════════════════════════════════════

def lane_1_adaptation():
    print("\n" + "=" * 60)
    print(f"LANE 1: 3 Adaptation Families ({PILOT_STEPS} steps each)")
    print("=" * 60)

    candidates = []

    # ── A1: Vanilla late PPO (OpeningAutoPlay + bridge + balanced) ──
    tag = "A1_vanilla"
    print(f"\n--- {tag}: MLP AutoPlay, bridge shaping, balanced league, {PILOT_STEPS} steps ---")
    league = make_balanced_league(lhal_weight=0.5)
    late = train_late_autoplay(
        timesteps=PILOT_STEPS, league=league,
        use_shaping=True, shaping_preset="bridge",
    )
    candidates.append({
        "tag": tag, "family": "A1", "late_model": late,
        "mechanism": "vanilla_late_ppo",
        "ppo_steps": PILOT_STEPS,
        "description": "MLP OpeningAutoPlay, bridge shaping, balanced league (ht+hp+lhal@0.5)",
    })

    # ── A2: Scenario-augmented curriculum (LateCurriculum + bridge) ──
    tag = "A2_curriculum"
    print(f"\n--- {tag}: MLP Curriculum p=0.4, bridge shaping, balanced league, {PILOT_STEPS} steps ---")
    league = make_balanced_league(lhal_weight=0.5)
    late = train_late_curriculum(
        timesteps=PILOT_STEPS, league=league,
        p_opening=0.4, late_scenarios=["round7_pressure", "round8_bridge"],
        use_shaping=True, shaping_preset="bridge",
    )
    candidates.append({
        "tag": tag, "family": "A2", "late_model": late,
        "mechanism": "scenario_curriculum",
        "ppo_steps": PILOT_STEPS,
        "description": "MLP LateCurriculum p_opening=0.4, r7+r8 scenarios, bridge shaping, balanced league",
    })

    # ── A3: Learned-Hal-targeted (AutoPlay + lhal-heavy 2x + light) ──
    tag = "A3_lhal_target"
    print(f"\n--- {tag}: MLP AutoPlay, light shaping, lhal-heavy(2x), {PILOT_STEPS} steps ---")
    league = make_lhal_heavy_league(lhal_weight=2.0, scripted_weight=1.0)
    late = train_late_autoplay(
        timesteps=PILOT_STEPS, league=league,
        use_shaping=True, shaping_preset="light",
    )
    candidates.append({
        "tag": tag, "family": "A3", "late_model": late,
        "mechanism": "lhal_targeted",
        "ppo_steps": PILOT_STEPS,
        "description": "MLP OpeningAutoPlay, light shaping, lhal-heavy(2x) league",
    })

    return candidates


# ══════════════════════════════════════════════════════════════════════
# LANE 2: Direct 50-Game Evaluation (no separate gate)
# ══════════════════════════════════════════════════════════════════════

def lane_2_evaluation(candidates):
    print("\n" + "=" * 60)
    print(f"LANE 2: Direct {EVAL_GAMES}-Game Evaluation")
    print("=" * 60)

    for cand in candidates:
        tag = cand["tag"]
        print(f"\n--- Evaluating: {tag} ---")
        res = build_and_eval(cand["late_model"], tag, games=EVAL_GAMES,
                             lhal_games=EVAL_GAMES)
        res["bc_stats"] = get_bc_stats()
        cand["eval_results"] = res

    return candidates


# ══════════════════════════════════════════════════════════════════════
# Analysis & Verdicts
# ══════════════════════════════════════════════════════════════════════

def analyze_results(baseline, candidates):
    bl_bp = baseline.get("bridge_pressure", {}).get("r7_rate", 0)
    bl_ht = baseline.get("hal_death_trade", {}).get("r7_rate", 0)
    bl_hp = baseline.get("hal_pressure", {}).get("r7_rate", 0)
    bl_r9 = baseline.get("seeded_r9", {}).get("rate", 0)
    bl_lh = baseline.get("learned_hal", {}).get("win_rate", 0)

    best_candidate = None
    best_score = -1
    any_improvement = False

    for cand in candidates:
        v = cand.get("eval_results", {})
        c_bp = v.get("bridge_pressure", {}).get("r7_rate", 0)
        c_ht = v.get("hal_death_trade", {}).get("r7_rate", 0)
        c_hp = v.get("hal_pressure", {}).get("r7_rate", 0)
        c_r9 = v.get("seeded_r9", {}).get("rate", 0)
        c_lh = v.get("learned_hal", {}).get("win_rate", 0)

        # Safety: didn't regress badly on scripted opponents
        safe = c_bp >= bl_bp - 0.06 and c_ht >= bl_ht - 0.06 and c_hp >= bl_hp - 0.06

        # Improvement: any metric improved by >2pp
        improved_lh = c_lh > bl_lh + 0.02
        improved_ht = c_ht > bl_ht + 0.02
        improved_hp = c_hp > bl_hp + 0.02

        if improved_lh or improved_ht or improved_hp:
            any_improvement = True

        # Composite score: weighted sum of deltas (lhal_wr weighted 3x)
        score = (
            3.0 * (c_lh - bl_lh)
            + 1.0 * (c_ht - bl_ht)
            + 1.0 * (c_hp - bl_hp)
            + 0.5 * (c_bp - bl_bp)
        )

        cand["analysis"] = {
            "safe": safe,
            "improved_lh": improved_lh,
            "improved_ht": improved_ht,
            "improved_hp": improved_hp,
            "any_improvement": improved_lh or improved_ht or improved_hp,
            "composite_score": round(score, 4),
            "deltas": {
                "bp_r7": round(c_bp - bl_bp, 4),
                "ht_r7": round(c_ht - bl_ht, 4),
                "hp_r7": round(c_hp - bl_hp, 4),
                "lhal_wr": round(c_lh - bl_lh, 4),
            },
        }

        if safe and score > best_score:
            best_score = score
            best_candidate = cand

    calibration_verdict = (
        "SIGNAL_DETECTED" if any_improvement
        else "NO_SIGNAL_AT_12K"
    )

    return {
        "best_candidate": best_candidate["tag"] if best_candidate else None,
        "best_composite_score": round(best_score, 4) if best_candidate else None,
        "any_improvement": any_improvement,
        "calibration_verdict": calibration_verdict,
    }


# ══════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════

def main():
    start = time.time()
    print("=" * 60)
    print("PILOT: Self-Play / Late-Head Adaptation at p=0.88")
    print("Date: 2026-03-28")
    print(f"PHYSICALITY_BAKU: {PHYSICALITY_BAKU}")
    print(f"Budget per family: {PILOT_STEPS} steps")
    print(f"Eval games: {EVAL_GAMES}")
    print(f"Learned Hal: fresh_hal_32k (reused)")
    print("=" * 60)
    print("\nReused artifacts:")
    print(f"  - Opening BC: trained from {BASELINE_ROUTE}")
    print(f"  - Fresh Hal: _fresh_hal_fresh_32k (pre-trained at p=0.88)")
    print(f"  - BASE_MODEL: {Path(BASE_MODEL).name}")
    print(f"  - BP_SPECIALIST: {Path(BP_SPECIALIST).name}")
    print("\nSkipped for time:")
    print("  - Fresh Hal generation (reusing existing)")
    print("  - Degeneracy check (already passed)")
    print("  - Stress testing / PPO perturbation")
    print("  - Separate safety gate (direct eval instead)")

    # Lane 0: Baseline
    baseline = lane_0_baseline()
    bl_lh = baseline.get("learned_hal", {}).get("win_rate", 0)

    # Lane 1: 3 adaptation families
    candidates = lane_1_adaptation()

    # Lane 2: Direct evaluation
    candidates = lane_2_evaluation(candidates)

    # Analysis
    verdict = analyze_results(baseline, candidates)

    elapsed = time.time() - start

    # ── Build output ─────────────────────────────────────────────────
    bl_bp = baseline.get("bridge_pressure", {}).get("r7_rate", 0)
    bl_ht = baseline.get("hal_death_trade", {}).get("r7_rate", 0)
    bl_hp = baseline.get("hal_pressure", {}).get("r7_rate", 0)
    bl_r9 = baseline.get("seeded_r9", {}).get("rate", 0)
    bc_stats = baseline.get("bc_stats", get_bc_stats())

    out = {
        "sprint": "self-play-p088-pilot",
        "date": "2026-03-28",
        "physicality_baku": PHYSICALITY_BAKU,
        "pilot_budget_per_family": PILOT_STEPS,
        "eval_games": EVAL_GAMES,
        "baseline_route": BASELINE_ROUTE,
        "reused_artifacts": {
            "learned_hal": "fresh_hal_fresh_32k (pre-trained at p=0.88, 32768 steps)",
            "opening_bc": f"route {BASELINE_ROUTE}, 80 epochs, accuracy {bc_stats['accuracy']:.1%}",
            "base_model": Path(BASE_MODEL).name,
            "bp_specialist": Path(BP_SPECIALIST).name,
        },
        "skipped": [
            "Fresh Hal generation",
            "Degeneracy check",
            "Stress testing / PPO perturbation",
            "Separate safety gate lane",
        ],
        "baseline": {k: v for k, v in baseline.items() if k != "bc_stats"},
        "baseline_bc_stats": bc_stats,
        "candidates": [
            {k: v for k, v in c.items() if k != "late_model"}
            for c in candidates
        ],
        "verdict": verdict,
        "elapsed_seconds": round(elapsed),
    }

    # Write JSON
    out_dir = Path("docs/reports/json")
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "sprint_self_play_p088_pilot_2026-03-28.json"
    with open(json_path, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\nJSON written to {json_path}")

    # ── Terminal Summary ─────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("PILOT TERMINAL SUMMARY")
    print("=" * 60)
    print(f"  PHYSICALITY_BAKU:     {PHYSICALITY_BAKU}")
    print(f"  Budget per family:    {PILOT_STEPS} steps")
    print(f"  Eval games:           {EVAL_GAMES}")
    print(f"  BC misaligned:        {bc_stats.get('misaligned', '?')}")
    print(f"  ────────────────────────────────────────")
    print(f"  p=0.88 BASELINE ({EVAL_GAMES}-game):")
    print(f"    bp  r7:   {pct(bl_bp)}")
    print(f"    ht  r7:   {pct(bl_ht)}")
    print(f"    hp  r7:   {pct(bl_hp)}")
    print(f"    r9 seed:  {pct(bl_r9)}")
    print(f"    lhal wr:  {pct(bl_lh)}")
    print(f"  ────────────────────────────────────────")
    print(f"  CANDIDATES:")
    for cand in candidates:
        v = cand.get("eval_results", {})
        a = cand.get("analysis", {})
        d = a.get("deltas", {})
        c_bp = v.get("bridge_pressure", {}).get("r7_rate", 0)
        c_ht = v.get("hal_death_trade", {}).get("r7_rate", 0)
        c_hp = v.get("hal_pressure", {}).get("r7_rate", 0)
        c_r9 = v.get("seeded_r9", {}).get("rate", 0)
        c_lh = v.get("learned_hal", {}).get("win_rate", 0)
        delta_lh = d.get("lhal_wr", 0)
        safe_str = "SAFE" if a.get("safe") else "REGR"
        imp_str = "IMP" if a.get("any_improvement") else "---"
        print(
            f"    {cand['tag']:18s}: bp={pct(c_bp)} ht={pct(c_ht)} "
            f"hp={pct(c_hp)} r9={pct(c_r9)} "
            f"lhal={pct(c_lh)}({'+' if delta_lh >= 0 else ''}{pct(delta_lh)}) "
            f"[{safe_str}|{imp_str}] score={a.get('composite_score', '?')}"
        )
    print(f"  ────────────────────────────────────────")
    best = verdict.get("best_candidate") or "NONE"
    cal = verdict.get("calibration_verdict", "?")
    print(f"  Best candidate:       {best}")
    print(f"  Calibration verdict:  {cal}")
    print(f"  Any improvement:      {'YES' if verdict.get('any_improvement') else 'NO'}")
    print(f"  Sprint time:          {elapsed:.0f}s")
    print("=" * 60)

    return out


if __name__ == "__main__":
    main()
