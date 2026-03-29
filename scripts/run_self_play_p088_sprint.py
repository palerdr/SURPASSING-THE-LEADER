#!/usr/bin/env python3
"""Sprint: Constrained Self-Play / Modular Late-Head Adaptation at p=0.88.

Resolves whether old modular/self-play failures (2026-03-27, p=0.70) were
calibration artifacts or persist under the corrected p=0.88 regime.

Harness fixes vs prior late-head sprint:
  1. Full 8-turn BASELINE_OVERRIDES (was: partial {3: 10})
  2. Selector-based r9 eval (was: raw model r9)
  3. PHYSICALITY_BAKU=0.88 (was: 0.70)

Lanes:
  0: Reproduce p=0.88 baseline with corrected selector
  1: Learned Hal usefulness check (degeneracy + WR)
  2: Fresh Hal generation (if needed)
  3: Late-head adaptation families (F1-F4, 4 materially different)
  4: Safety gate (24-game)
  5: Full 50-game validation
  6: Promotion check + deliverables
"""

from __future__ import annotations

import sys
import os
import json
import time
import collections
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
from training.expert_selector import build_classifier

# Try importing RecurrentPPO; degrade gracefully if unavailable
try:
    from sb3_contrib import RecurrentPPO
    from training.modular_policy import ActionClampEnv, RecurrentModularBakuSelector
    HAS_RECURRENT = True
except ImportError:
    HAS_RECURRENT = False


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
LEARNED_HAL = _resolve("hal_vs_promoted_baku_prior_4096")

# ── HARNESS FIX #1: Full 8-turn baseline overrides ────────────────────
# Previous late-head sprint used partial {3: 10}. Corrected to full route.
BASELINE_ROUTE = [1, 1, 1, 10, 60, 1, 60, 1]
BASELINE_OVERRIDES = {t: BASELINE_ROUTE[t] for t in range(8)}
OPENING_HORIZON = 8

HAL_OPPONENTS = ["hal_death_trade", "hal_pressure"]
ALL_OPPONENTS = ["bridge_pressure", "hal_death_trade", "hal_pressure"]

# Promotion bars from task spec
FULL_PROMOTION = {
    "bp_r7": 0.84, "ht_r7": 0.48, "hp_r7": 0.48,
    "seeded_r9": 1.0, "misaligned": 0, "lhal_wr_delta": 0.08,
}
ACCEPTABLE_PASS = {
    "bp_r7": 0.80, "ht_r7": 0.46, "hp_r7": 0.46,
    "seeded_r9": 1.0, "misaligned": 0, "lhal_wr_delta": 0.04,
}
SAFETY_GATE = {"bp_r7": 0.76, "ht_r7": 0.36, "hp_r7": 0.36, "seeded_r9": 1.0}
BASELINE_LHAL_WR = 0.32  # current learned-Hal baseline from reaudit


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


def evaluate_selector(selector, opp_name, games=24, seed=42, opp_model_path=None):
    """Run games with the full selector, return structured result."""
    wins = 0
    total_hr = 0
    r7_count = 0
    r7_eligible = 0
    deaths = 0
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


# ── HARNESS FIX #2: Selector-based r9 eval ────────────────────────────
def eval_r9_seeded(selector, seed=42):
    """Evaluate seeded r9 through the full selector (not raw model)."""
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


def eval_suite(selector, games=50, seed=42, tag="", lhal_games=0, lhal_path=None):
    """Full evaluation suite: scripted opponents + seeded r9 + learned Hal."""
    results = {}
    hal_path = lhal_path or LEARNED_HAL
    for opp in ALL_OPPONENTS:
        res = evaluate_selector(selector, opp, games=games, seed=seed)
        results[opp] = asdict(res)
        print(
            f"  {tag} vs {opp}: r7={pct(res.r7_rate)} "
            f"wins={pct(res.win_rate)} avg_hr={res.avg_half_rounds:.1f}"
        )
    r9 = eval_r9_seeded(selector, seed=seed)
    results["seeded_r9"] = r9
    print(f"  {tag} seeded r9: {r9['wins']}/{r9['games']} ({pct(r9['rate'])})")
    if lhal_games > 0:
        lh = evaluate_selector(
            selector, "learned_hal", games=lhal_games, seed=seed,
            opp_model_path=hal_path,
        )
        results["learned_hal"] = asdict(lh)
        print(
            f"  {tag} vs learned_hal: wins={pct(lh.win_rate)} "
            f"deaths={lh.deaths_by_agent} avg_hr={lh.avg_half_rounds:.1f}"
        )
    return results


# ── Opening model builder (full 8-turn overrides) ────────────────────

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
        print(
            f"  Opening BC: {_bc_stats_cache['accuracy']:.1%} "
            f"({_bc_stats_cache['correct']}/{_bc_stats_cache['total']}), "
            f"misaligned={_bc_stats_cache['misaligned']}"
        )
    return _opening_model_cache


def get_bc_stats():
    get_opening_model()
    return _bc_stats_cache


def build_baseline_selector(late_model=None):
    """Build full modular selector with frozen opening and optional late model."""
    opening_model = get_opening_model()
    if late_model is None:
        late_model = MaskablePPO.load(BASE_MODEL)
    return build_modular_selector(
        bp_specialist_path=BP_SPECIALIST, base_model_path=BASE_MODEL,
        opening_model=opening_model, late_model=late_model,
        opening_horizon=OPENING_HORIZON,
    )


# ── League builders ───────────────────────────────────────────────────

def make_balanced_league(lhal_weight=0.5, seed=42, lhal_path=None):
    hal_path = lhal_path or LEARNED_HAL
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
    hal_path = lhal_path or LEARNED_HAL
    return WeightedOpponentLeague([
        LeagueEntry(label="ht", weight=scripted_weight,
                    opponent=create_scripted_opponent("hal_death_trade")),
        LeagueEntry(label="hp", weight=scripted_weight,
                    opponent=create_scripted_opponent("hal_pressure")),
        LeagueEntry(label="lhal", weight=lhal_weight,
                    opponent=create_model_opponent(hal_path, agent_role="hal")),
    ], seed=seed)


# ── Training helpers ──────────────────────────────────────────────────

def train_late_autoplay(timesteps, league, seed=42, use_shaping=True,
                         shaping_preset="bridge"):
    """Train MLP late model on OpeningAutoPlayEnv."""
    opening_model = get_opening_model()
    env = OpeningAutoPlayEnv(
        opening_model=opening_model, opening_horizon=OPENING_HORIZON,
        opponent=league, agent_role="baku", seed=seed,
        use_shaping=use_shaping, shaping_preset=shaping_preset,
    )
    model = MaskablePPO.load(BASE_MODEL, env=env)
    model.learn(total_timesteps=timesteps)
    return model


def train_late_curriculum(timesteps, league, p_opening=0.5,
                           late_scenarios=None, seed=42,
                           use_shaping=True, shaping_preset="bridge"):
    """Train MLP late model on LateCurriculumEnv."""
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


def train_recurrent_late(timesteps, league, hidden_size=64, seed=42,
                          use_shaping=True, shaping_preset="bridge"):
    """Train LSTM late model on OpeningAutoPlayEnv with ActionClampEnv."""
    if not HAS_RECURRENT:
        raise RuntimeError("RecurrentPPO not available")
    opening_model = get_opening_model()
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
        policy_kwargs=dict(lstm_hidden_size=hidden_size, n_lstm_layers=1),
    )
    model.learn(total_timesteps=timesteps)
    return model


def build_recurrent_selector(recurrent_late_model):
    """Build selector with recurrent late head."""
    opening_model = get_opening_model()
    bp_specialist = MaskablePPO.load(BP_SPECIALIST)
    clf = build_classifier(
        BASE_MODEL, ["bridge_pressure", "hal_death_trade"], classify_turn=2,
    )
    return RecurrentModularBakuSelector(
        bp_specialist=bp_specialist, opening_model=opening_model,
        late_model=recurrent_late_model, classifier=clf,
        opening_horizon=OPENING_HORIZON, classify_turn=2,
    )


def build_and_eval(late_model, tag, games=24, lhal_games=24,
                    is_recurrent=False, lhal_path=None):
    """Build selector from late model and run eval suite."""
    if is_recurrent:
        selector = build_recurrent_selector(late_model)
    else:
        selector = build_baseline_selector(late_model)
    return eval_suite(
        selector, games=games, tag=tag, lhal_games=lhal_games, lhal_path=lhal_path,
    )


# ── Degeneracy check for learned Hal ─────────────────────────────────

def check_hal_degeneracy(hal_path, n_games=20, seed=42):
    """Play n_games as Hal and check action diversity."""
    hal_model = MaskablePPO.load(hal_path)
    actions_by_role = collections.defaultdict(list)
    total_actions = 0

    for gi in range(n_games):
        opp = create_scripted_opponent("safe")
        env = DTHEnv(opponent=opp, agent_role="hal", seed=seed + gi)
        obs, _ = env.reset()
        while True:
            mask = env.action_masks()
            action, _ = hal_model.predict(obs, action_masks=mask, deterministic=True)
            action_second = int(action) + 1
            obs_1d = obs.flatten() if obs.ndim > 1 else obs
            role = "dropper" if float(obs_1d[7]) > 0.5 else "checker"
            actions_by_role[role].append(action_second)
            total_actions += 1
            obs, _, term, trunc, _ = env.step(int(action))
            if term or trunc:
                break

    dropper_actions = actions_by_role.get("dropper", [])
    checker_actions = actions_by_role.get("checker", [])
    dropper_unique = len(set(dropper_actions))
    checker_unique = len(set(checker_actions))
    dropper_counter = collections.Counter(dropper_actions)
    checker_counter = collections.Counter(checker_actions)
    all_actions = dropper_actions + checker_actions

    # Degenerate if total unique actions < 2 or identical distribution across roles
    degenerate = len(set(all_actions)) < 2
    if not degenerate:
        degenerate = (
            dropper_unique <= 1 and checker_unique <= 1
            and set(dropper_actions) == set(checker_actions)
        )

    return {
        "degenerate": degenerate,
        "total_actions": total_actions,
        "dropper_unique": dropper_unique,
        "checker_unique": checker_unique,
        "dropper_top3": dropper_counter.most_common(3),
        "checker_top3": checker_counter.most_common(3),
        "total_unique": len(set(all_actions)),
    }


# ── Fresh Hal generation ─────────────────────────────────────────────

def train_fresh_hal(timesteps, seed=42):
    """Train Hal against BASE_MODEL (Baku) at current p=0.88."""
    baku_opponent = create_model_opponent(BASE_MODEL, agent_role="hal")
    env = DTHEnv(opponent=baku_opponent, agent_role="hal", seed=seed)
    model = MaskablePPO(
        "MlpPolicy", env, learning_rate=3e-4, n_steps=2048,
        batch_size=64, n_epochs=10, gamma=0.99, verbose=1, seed=seed,
    )
    model.learn(total_timesteps=timesteps)
    return model


# ══════════════════════════════════════════════════════════════════════
# LANE 0: Baseline Reproduction
# ══════════════════════════════════════════════════════════════════════

def lane_0_baseline():
    print("\n" + "=" * 60)
    print(f"LANE 0: Baseline Reproduction (PHYSICALITY_BAKU={PHYSICALITY_BAKU})")
    print(f"Harness: Full 8-turn overrides, selector-based r9")
    print(f"Route: {BASELINE_ROUTE}")
    print("=" * 60)

    bc_stats = get_bc_stats()
    print(
        f"  BC: {bc_stats['accuracy']:.1%} "
        f"({bc_stats['correct']}/{bc_stats['total']}), "
        f"misaligned={bc_stats['misaligned']}"
    )

    selector = build_baseline_selector()
    results = eval_suite(selector, games=50, tag="baseline", lhal_games=50)
    results["bc_stats"] = bc_stats
    return results


# ══════════════════════════════════════════════════════════════════════
# LANE 1: Learned Hal Usefulness Check
# ══════════════════════════════════════════════════════════════════════

def lane_1_hal_check(baseline_lhal_wr):
    print("\n" + "=" * 60)
    print("LANE 1: Learned Hal Usefulness Check")
    print("=" * 60)

    print("\n--- Degeneracy check (20 games) ---")
    degen = check_hal_degeneracy(LEARNED_HAL, n_games=20)
    print(f"  Total actions: {degen['total_actions']}")
    print(f"  Unique actions: {degen['total_unique']}")
    print(f"  Dropper unique: {degen['dropper_unique']} top3: {degen['dropper_top3']}")
    print(f"  Checker unique: {degen['checker_unique']} top3: {degen['checker_top3']}")
    print(f"  Degenerate: {degen['degenerate']}")

    # Useful: nondegenerate AND provides real pressure (Baku WR < 50%)
    useful = not degen["degenerate"] and baseline_lhal_wr < 0.50
    print(f"\n  Baku WR vs prior_4096: {pct(baseline_lhal_wr)}")
    print(f"  Useful pressure opponent: {'YES' if useful else 'NO'}")

    return {
        "degeneracy": degen,
        "baseline_lhal_wr": baseline_lhal_wr,
        "useful": useful,
        "hal_path": LEARNED_HAL,
    }


# ══════════════════════════════════════════════════════════════════════
# LANE 2: Fresh Hal Generation
# ══════════════════════════════════════════════════════════════════════

def lane_2_fresh_hal(selector, existing_useful):
    print("\n" + "=" * 60)
    print("LANE 2: Fresh Hal Candidate Generation")
    print("=" * 60)

    if existing_useful:
        print("  Existing prior_4096 is useful. Generating fresh candidates as bonus.")
    else:
        print("  Existing prior_4096 NOT useful. Must generate fresh candidates.")

    fresh_candidates = []

    for steps, tag in [(32768, "fresh_32k"), (65536, "fresh_64k")]:
        print(f"\n--- {tag}: Training Hal, {steps} steps, p=0.88 ---")
        hal_model = train_fresh_hal(timesteps=steps, seed=42)

        save_path = f"models/checkpoints/_fresh_hal_{tag}"
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        hal_model.save(save_path)

        # Evaluate Baku vs this Hal
        print(f"  Evaluating Baku vs {tag} (50 games)...")
        res = evaluate_selector(
            selector, "hal_candidate", games=50, seed=42,
            opp_model_path=save_path,
        )
        baku_wr = res.win_rate

        useful = baku_wr < BASELINE_LHAL_WR
        print(
            f"  Baku WR vs {tag}: {pct(baku_wr)} "
            f"(baseline {pct(BASELINE_LHAL_WR)}, pushes_below={'YES' if useful else 'NO'})"
        )

        fresh_candidates.append({
            "tag": tag, "steps": steps, "save_path": save_path,
            "baku_wr": baku_wr, "useful": useful,
            "eval": asdict(res),
        })

    return fresh_candidates


# ══════════════════════════════════════════════════════════════════════
# LANE 3: Late-Head Adaptation Families
# ══════════════════════════════════════════════════════════════════════

def lane_3_adaptation(best_hal_path):
    print("\n" + "=" * 60)
    print("LANE 3: Late-Head Adaptation Families (p=0.88)")
    print(f"Using Hal opponent: {Path(best_hal_path).name}")
    print("=" * 60)

    candidates = []

    # ── F1: MLP + OpeningAutoPlay + bridge shaping + balanced league ──
    for steps, stag in [(32768, "32k"), (65536, "64k")]:
        tag = f"F1_{stag}"
        print(f"\n--- {tag}: MLP AutoPlay, bridge shaping, balanced league, {steps} ---")
        league = make_balanced_league(lhal_weight=0.5, lhal_path=best_hal_path)
        late = train_late_autoplay(
            timesteps=steps, league=league,
            use_shaping=True, shaping_preset="bridge",
        )
        candidates.append({
            "tag": tag, "family": "F1", "late_model": late, "is_recurrent": False,
            "ppo_steps": steps,
            "description": f"MLP OpeningAutoPlay bridge balanced {steps}",
        })

    # ── F2: MLP + LateCurriculum + bridge shaping ──
    for p_open, stag in [(0.3, "p30"), (0.5, "p50")]:
        tag = f"F2_{stag}"
        print(
            f"\n--- {tag}: MLP Curriculum p={p_open}, bridge shaping, "
            f"balanced league, 32k ---"
        )
        league = make_balanced_league(lhal_weight=0.5, lhal_path=best_hal_path)
        late = train_late_curriculum(
            timesteps=32768, league=league, p_opening=p_open,
            late_scenarios=["round7_pressure", "round8_bridge"],
            use_shaping=True, shaping_preset="bridge",
        )
        candidates.append({
            "tag": tag, "family": "F2", "late_model": late, "is_recurrent": False,
            "ppo_steps": 32768,
            "description": f"MLP Curriculum p_opening={p_open} bridge balanced 32k",
        })

    # ── F3: MLP + AutoPlay + learned-Hal-heavy(4x) league ──
    for steps, stag in [(32768, "32k"), (65536, "64k")]:
        tag = f"F3_{stag}"
        print(
            f"\n--- {tag}: MLP AutoPlay, light shaping, lhal-heavy(4x), {steps} ---"
        )
        league = make_lhal_heavy_league(
            lhal_weight=4.0, lhal_path=best_hal_path,
        )
        late = train_late_autoplay(
            timesteps=steps, league=league,
            use_shaping=True, shaping_preset="light",
        )
        candidates.append({
            "tag": tag, "family": "F3", "late_model": late, "is_recurrent": False,
            "ppo_steps": steps,
            "description": f"MLP AutoPlay light lhal_heavy(4x) {steps}",
        })

    # ── F4: LSTM h=64 + AutoPlay + bridge shaping (if available) ──
    if HAS_RECURRENT:
        tag = "F4_64k"
        print(f"\n--- {tag}: LSTM h=64 AutoPlay, bridge shaping, balanced, 65k ---")
        league = make_balanced_league(lhal_weight=0.5, lhal_path=best_hal_path)
        late = train_recurrent_late(
            timesteps=65536, league=league, hidden_size=64,
            use_shaping=True, shaping_preset="bridge",
        )
        candidates.append({
            "tag": tag, "family": "F4", "late_model": late, "is_recurrent": True,
            "ppo_steps": 65536,
            "description": "LSTM h=64 AutoPlay bridge balanced 65k",
        })
    else:
        print("\n--- F4: SKIPPED (RecurrentPPO not available) ---")

    return candidates


# ══════════════════════════════════════════════════════════════════════
# LANE 4: Safety Gate (24-game)
# ══════════════════════════════════════════════════════════════════════

def lane_4_safety_gate(candidates, best_hal_path):
    print("\n" + "=" * 60)
    print("LANE 4: Safety Gate (24-game)")
    print("=" * 60)

    passing = []
    for cand in candidates:
        tag = cand["tag"]
        print(f"\n--- Safety check: {tag} ---")
        res = build_and_eval(
            cand["late_model"], tag, games=24, lhal_games=24,
            is_recurrent=cand["is_recurrent"], lhal_path=best_hal_path,
        )
        cand["safety_results"] = res

        bp = res.get("bridge_pressure", {}).get("r7_rate", 0)
        ht = res.get("hal_death_trade", {}).get("r7_rate", 0)
        hp = res.get("hal_pressure", {}).get("r7_rate", 0)
        r9 = res.get("seeded_r9", {}).get("rate", 0)
        lh = res.get("learned_hal", {}).get("win_rate", 0)

        gate = (
            bp >= SAFETY_GATE["bp_r7"]
            and ht >= SAFETY_GATE["ht_r7"]
            and hp >= SAFETY_GATE["hp_r7"]
            and r9 >= SAFETY_GATE["seeded_r9"]
        )
        cand["safety_pass"] = gate

        status = "PASS" if gate else "FAIL"
        print(
            f"  >> {tag}: {status} "
            f"(bp={pct(bp)} ht={pct(ht)} hp={pct(hp)} r9={pct(r9)} lhal={pct(lh)})"
        )

        if gate:
            passing.append(cand)

    return passing


# ══════════════════════════════════════════════════════════════════════
# LANE 5: Full 50-Game Validation
# ══════════════════════════════════════════════════════════════════════

def lane_5_validation(candidates, best_hal_path):
    print("\n" + "=" * 60)
    print("LANE 5: Full 50-Game Validation")
    print("=" * 60)

    for cand in candidates:
        tag = cand["tag"]
        print(f"\n--- Full validation: {tag} ---")
        if cand["is_recurrent"]:
            selector = build_recurrent_selector(cand["late_model"])
        else:
            selector = build_baseline_selector(cand["late_model"])

        res = eval_suite(
            selector, games=50, tag=f"{tag}_50g", lhal_games=50,
            lhal_path=best_hal_path,
        )
        res["bc_stats"] = get_bc_stats()
        cand["full_validation"] = res


# ══════════════════════════════════════════════════════════════════════
# LANE 6: Promotion Check
# ══════════════════════════════════════════════════════════════════════

def check_promotion(candidates, baseline_lhal_wr):
    """Check candidates against promotion bars."""
    best = None
    best_level = None

    for cand in candidates:
        v = cand.get("full_validation", cand.get("safety_results", {}))
        bp = v.get("bridge_pressure", {}).get("r7_rate", 0)
        ht = v.get("hal_death_trade", {}).get("r7_rate", 0)
        hp = v.get("hal_pressure", {}).get("r7_rate", 0)
        r9 = v.get("seeded_r9", {}).get("rate", 0)
        bc = v.get("bc_stats", get_bc_stats())
        mis = bc.get("misaligned", 0)
        lh = v.get("learned_hal", {}).get("win_rate", 0)
        lh_delta = lh - baseline_lhal_wr

        # Full promotion
        full = (
            bp >= FULL_PROMOTION["bp_r7"]
            and ht >= FULL_PROMOTION["ht_r7"]
            and hp >= FULL_PROMOTION["hp_r7"]
            and r9 >= FULL_PROMOTION["seeded_r9"]
            and mis == FULL_PROMOTION["misaligned"]
            and lh_delta >= FULL_PROMOTION["lhal_wr_delta"]
        )

        # Acceptable pass
        acceptable = (
            bp >= ACCEPTABLE_PASS["bp_r7"]
            and ht >= ACCEPTABLE_PASS["ht_r7"]
            and hp >= ACCEPTABLE_PASS["hp_r7"]
            and r9 >= ACCEPTABLE_PASS["seeded_r9"]
            and mis == ACCEPTABLE_PASS["misaligned"]
            and lh_delta >= ACCEPTABLE_PASS["lhal_wr_delta"]
        )

        if full and best_level != "PROMOTED":
            best = cand
            best_level = "PROMOTED"
        elif acceptable and best is None:
            best = cand
            best_level = "ACCEPTABLE_PASS"

    return {
        "best_candidate": best["tag"] if best else None,
        "level": best_level,
        "promoted": best_level == "PROMOTED",
        "acceptable": best_level in ("PROMOTED", "ACCEPTABLE_PASS"),
    }


# ══════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════

def main():
    start = time.time()
    print("=" * 60)
    print("SPRINT: Constrained Self-Play / Modular Late-Head at p=0.88")
    print("Date: 2026-03-28")
    print(f"PHYSICALITY_BAKU: {PHYSICALITY_BAKU}")
    print("=" * 60)
    print("\nHarness fixes applied:")
    print("  1. Full 8-turn BASELINE_OVERRIDES (was: partial {3: 10})")
    print("  2. Selector-based r9 eval (was: raw model r9)")
    print(f"  3. PHYSICALITY_BAKU={PHYSICALITY_BAKU} (was: 0.70)")

    # Lane 0: Baseline
    baseline = lane_0_baseline()
    baseline_lhal_wr = baseline.get("learned_hal", {}).get("win_rate", 0)

    # Lane 1: Hal usefulness
    hal_check = lane_1_hal_check(baseline_lhal_wr)
    existing_useful = hal_check["useful"]

    # Lane 2: Fresh Hal generation
    selector = build_baseline_selector()
    fresh_hal = lane_2_fresh_hal(selector, existing_useful)

    # Pick best Hal: existing or fresh (lowest Baku WR = hardest Hal)
    best_hal_path = LEARNED_HAL
    best_baku_wr = baseline_lhal_wr
    for fc in fresh_hal:
        if fc["baku_wr"] < best_baku_wr:
            best_baku_wr = fc["baku_wr"]
            best_hal_path = fc["save_path"]

    any_hal_useful = existing_useful or any(fc["useful"] for fc in fresh_hal)
    if not any_hal_useful:
        print("\n!! NO USEFUL LEARNED HAL — using prior_4096 anyway !!")
        best_hal_path = LEARNED_HAL

    print(
        f"\nBest Hal for adaptation: {Path(best_hal_path).name} "
        f"(Baku WR={pct(best_baku_wr)})"
    )

    # Lane 3: Adaptation families
    all_candidates = lane_3_adaptation(best_hal_path)

    # Lane 4: Safety gate
    safe_candidates = lane_4_safety_gate(all_candidates, best_hal_path)

    # Lane 5: Full validation
    if safe_candidates:
        lane_5_validation(safe_candidates, best_hal_path)
    else:
        print("\n!! ALL CANDIDATES FAIL SAFETY GATE !!")

    # Lane 6: Promotion
    promotion = check_promotion(safe_candidates, baseline_lhal_wr)

    elapsed = time.time() - start

    # ── Build results ────────────────────────────────────────────────
    bp_r7 = baseline.get("bridge_pressure", {}).get("r7_rate", 0)
    ht_r7 = baseline.get("hal_death_trade", {}).get("r7_rate", 0)
    hp_r7 = baseline.get("hal_pressure", {}).get("r7_rate", 0)
    r9_rate = baseline.get("seeded_r9", {}).get("rate", 0)
    misaligned = baseline.get("bc_stats", {}).get("misaligned", -1)

    # Determine if old failures were calibration artifacts
    any_improvement = False
    for cand in all_candidates:
        v = cand.get("full_validation", cand.get("safety_results", {}))
        cand_lh = v.get("learned_hal", {}).get("win_rate", 0)
        cand_ht = v.get("hal_death_trade", {}).get("r7_rate", 0)
        cand_hp = v.get("hal_pressure", {}).get("r7_rate", 0)
        if (
            cand_lh > baseline_lhal_wr + 0.02
            or cand_ht > ht_r7 + 0.02
            or cand_hp > hp_r7 + 0.02
        ):
            any_improvement = True
            break

    calibration_verdict = (
        "CALIBRATION_ARTIFACT" if any_improvement else "STRUCTURAL_LIMITATION"
    )

    out = {
        "sprint": "self-play-late-head-p088",
        "date": "2026-03-28",
        "physicality_baku": PHYSICALITY_BAKU,
        "baseline_route": BASELINE_ROUTE,
        "harness_fixes": [
            "Full 8-turn BASELINE_OVERRIDES (was: partial {3: 10})",
            "Selector-based r9 eval (was: raw model r9)",
            f"PHYSICALITY_BAKU={PHYSICALITY_BAKU} (was: 0.70)",
        ],
        "baseline": baseline,
        "hal_usefulness": {
            k: v for k, v in hal_check.items()
            if k != "hal_path"
        },
        "hal_usefulness_path": str(hal_check["hal_path"]),
        "fresh_hal_candidates": fresh_hal,
        "best_hal": {"path": str(best_hal_path), "baku_wr": best_baku_wr},
        "candidates": [
            {k: v for k, v in c.items() if k != "late_model"}
            for c in all_candidates
        ],
        "safe_candidates": [c["tag"] for c in safe_candidates],
        "promotion": promotion,
        "calibration_verdict": calibration_verdict,
        "elapsed_seconds": round(elapsed),
    }

    # Write JSON
    out_dir = Path("docs/reports/json")
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "sprint_self_play_p088_2026-03-28.json"
    with open(json_path, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\nJSON written to {json_path}")

    # ── Terminal Summary ─────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("TERMINAL SUMMARY")
    print("=" * 60)
    print(f"  PHYSICALITY_BAKU:     {PHYSICALITY_BAKU}")
    print(f"  Baseline route:       {BASELINE_ROUTE}")
    print(f"  BC misaligned:        {misaligned}")
    print(f"  ────────────────────────────────────────")
    print(f"  p=0.88 Baseline (50-game):")
    print(
        f"    bp  r7:   {pct(bp_r7)}  "
        f"(full >={pct(FULL_PROMOTION['bp_r7'])} / accept >={pct(ACCEPTABLE_PASS['bp_r7'])})"
    )
    print(
        f"    ht  r7:   {pct(ht_r7)}  "
        f"(full >={pct(FULL_PROMOTION['ht_r7'])} / accept >={pct(ACCEPTABLE_PASS['ht_r7'])})"
    )
    print(
        f"    hp  r7:   {pct(hp_r7)}  "
        f"(full >={pct(FULL_PROMOTION['hp_r7'])} / accept >={pct(ACCEPTABLE_PASS['hp_r7'])})"
    )
    print(f"    r9 seed:  {pct(r9_rate)}")
    print(f"    lhal wr:  {pct(baseline_lhal_wr)}")
    print(f"  ────────────────────────────────────────")
    print(f"  Learned Hal:")
    print(f"    prior_4096 useful:  {'YES' if existing_useful else 'NO'}")
    print(f"    Fresh candidates:   {len(fresh_hal)}")
    print(
        f"    Best Hal:           {Path(best_hal_path).name} "
        f"(Baku WR={pct(best_baku_wr)})"
    )
    print(f"  ────────────────────────────────────────")
    print(f"  Adaptation Families:  {len(all_candidates)}")
    print(f"  Safety gate pass:     {len(safe_candidates)}/{len(all_candidates)}")

    if safe_candidates:
        print(f"  ────────────────────────────────────────")
        print(f"  Validated Candidates:")
        for cand in safe_candidates:
            v = cand.get("full_validation", cand.get("safety_results", {}))
            c_bp = v.get("bridge_pressure", {}).get("r7_rate", 0)
            c_ht = v.get("hal_death_trade", {}).get("r7_rate", 0)
            c_hp = v.get("hal_pressure", {}).get("r7_rate", 0)
            c_r9 = v.get("seeded_r9", {}).get("rate", 0)
            c_lh = v.get("learned_hal", {}).get("win_rate", 0)
            c_mis = v.get("bc_stats", get_bc_stats()).get("misaligned", 0)
            delta = c_lh - baseline_lhal_wr
            print(
                f"    {cand['tag']:20s}: bp={pct(c_bp)} ht={pct(c_ht)} "
                f"hp={pct(c_hp)} r9={pct(c_r9)} mis={c_mis} "
                f"lhal={pct(c_lh)} ({'+' if delta >= 0 else ''}{pct(delta)})"
            )

    print(f"  ────────────────────────────────────────")
    verdict = promotion.get("level") or "FAIL"
    print(f"  Verdict:              {verdict}")
    print(f"  Old-failure diagnosis: {calibration_verdict}")
    if promotion.get("best_candidate"):
        print(f"  Best candidate:       {promotion['best_candidate']}")
    print(f"  Sprint time:          {elapsed:.0f}s")
    print("=" * 60)

    return out


if __name__ == "__main__":
    main()
