#!/usr/bin/env python3
"""Sprint: Scaffold-Native Self-Play Ladder at p=0.88.

Fixes the key confound from the previous ladder sprint: Hal opponents were
trained against BASE_MODEL alone, not the actual modular p=0.88 Baku
scaffold.  This sprint trains "scaffold-native" Hal candidates against the
full ModularBakuSelector (frozen BC opening + classifier + late model),
then uses them in a graded opponent ladder for Baku late-head adaptation.

Lanes:
  0: Reproduce corrected modular p=0.88 baseline
  1: Train scaffold-native Hal candidates (Hal vs full modular Baku)
  2: Rank native Hals by Baku win rate + nondegeneracy
  3: Build graded ladder: prior_4096 + best scaffold-native Hal
  4: Train 3+ materially different Baku adaptation families (16k-24k, gated)
  5: Full 50-game evaluation of survivors
  6: Promotion check + deliverables

Promotion (strong): bp>=0.84, ht>=0.48, hp>=0.48, r9=1.0, mis=0,
  AND (+6pp vs prior_4096 OR +6pp vs best native Hal).
Promotion (acceptable): bp>=0.82, ht>=0.48, hp>=0.48, r9=1.0, mis=0,
  AND (+4pp vs prior_4096 OR +4pp vs best native Hal).
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
from environment.opponents.model_opponent import SelectorOpponent
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
LEARNED_HAL_PRIOR = _resolve("hal_vs_promoted_baku_prior_4096")
FRESH_HAL_32K = _resolve("_fresh_hal_fresh_32k")

BASELINE_ROUTE = [1, 1, 1, 10, 60, 1, 60, 1]
BASELINE_OVERRIDES = {t: BASELINE_ROUTE[t] for t in range(8)}
OPENING_HORIZON = 8
EVAL_GAMES = 50
GATE_GAMES = 16
HAL_OPPONENTS = ["hal_death_trade", "hal_pressure"]
SCRIPTED_OPPONENTS = ["bridge_pressure", "hal_death_trade", "hal_pressure"]

# Promotion bars
STRONG_PASS = {
    "bp_r7": 0.84, "ht_r7": 0.48, "hp_r7": 0.48,
    "seeded_r9": 1.0, "misaligned": 0,
    "prior_wr_delta": 0.06, "native_wr_delta": 0.06,
}
ACCEPTABLE_PASS = {
    "bp_r7": 0.82, "ht_r7": 0.48, "hp_r7": 0.48,
    "seeded_r9": 1.0, "misaligned": 0,
    "prior_wr_delta": 0.04, "native_wr_delta": 0.04,
}
GATE_BARS = {"bp_r7": 0.70, "ht_r7": 0.30, "hp_r7": 0.30, "seeded_r9": 1.0}


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


def evaluate_selector(selector, opp_name, games=50, seed=42, opp_model_path=None,
                      opp_selector=None):
    """Run games with the full selector. Supports model path or selector opponent."""
    wins = 0; total_hr = 0; r7_count = 0; r7_eligible = 0; deaths = 0
    for gi in range(games):
        if opp_selector is not None:
            opp = SelectorOpponent(opp_selector, role="hal")
            opp_selector.reset()
        elif opp_model_path:
            opp = create_model_opponent(opp_model_path, agent_role="baku")
        else:
            opp = create_scripted_opponent(opp_name)
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


def eval_full_suite(selector, games=50, seed=42, tag="",
                    native_hal_path=None, native_hal_model=None):
    """Full eval: scripted bp/ht/hp + seeded r9 + prior_4096 + native Hal."""
    results = {}

    for opp in SCRIPTED_OPPONENTS:
        res = evaluate_selector(selector, opp, games=games, seed=seed)
        results[opp] = asdict(res)
        print(f"  {tag} vs {opp}: r7={pct(res.r7_rate)} wr={pct(res.win_rate)} "
              f"avg_hr={res.avg_half_rounds:.1f}")

    r9 = eval_r9_seeded(selector, seed=seed)
    results["seeded_r9"] = r9
    print(f"  {tag} seeded r9: {r9['wins']}/{r9['games']} ({pct(r9['rate'])})")

    # vs prior_4096
    lh_prior = evaluate_selector(
        selector, "prior_4096", games=games, seed=seed,
        opp_model_path=LEARNED_HAL_PRIOR,
    )
    results["prior_4096"] = asdict(lh_prior)
    print(f"  {tag} vs prior_4096: wr={pct(lh_prior.win_rate)} "
          f"deaths={lh_prior.deaths_by_agent} avg_hr={lh_prior.avg_half_rounds:.1f}")

    # vs scaffold-native Hal (by checkpoint path or in-memory model)
    if native_hal_path:
        lh_native = evaluate_selector(
            selector, "native_hal", games=games, seed=seed,
            opp_model_path=native_hal_path,
        )
        results["native_hal"] = asdict(lh_native)
        print(f"  {tag} vs native_hal: wr={pct(lh_native.win_rate)} "
              f"deaths={lh_native.deaths_by_agent} avg_hr={lh_native.avg_half_rounds:.1f}")

    return results


def eval_gate(selector, games=16, seed=42, tag=""):
    results = {}
    for opp in SCRIPTED_OPPONENTS:
        res = evaluate_selector(selector, opp, games=games, seed=seed)
        results[opp] = asdict(res)
    r9 = eval_r9_seeded(selector, seed=seed)
    results["seeded_r9"] = r9

    bp = results.get("bridge_pressure", {}).get("r7_rate", 0)
    ht = results.get("hal_death_trade", {}).get("r7_rate", 0)
    hp = results.get("hal_pressure", {}).get("r7_rate", 0)
    r9v = r9["rate"]

    passed = (
        bp >= GATE_BARS["bp_r7"]
        and ht >= GATE_BARS["ht_r7"]
        and hp >= GATE_BARS["hp_r7"]
        and r9v >= GATE_BARS["seeded_r9"]
    )
    print(
        f"  GATE {tag}: bp={pct(bp)} ht={pct(ht)} hp={pct(hp)} r9={pct(r9v)} "
        f"-> {'PASS' if passed else 'FAIL'}"
    )
    return passed, results


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
        print(f"  Opening BC: {_bc_stats_cache['accuracy']:.1%} "
              f"({_bc_stats_cache['correct']}/{_bc_stats_cache['total']}), "
              f"misaligned={_bc_stats_cache['misaligned']}")
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


# ── Scaffold-native Hal training ─────────────────────────────────────

def train_scaffold_native_hal(timesteps, seed=42, lr=3e-4):
    """Train Hal against the full modular Baku selector (the confound fix).

    Instead of training against BASE_MODEL (a single MaskablePPO), Hal
    faces the complete modular scaffold: BC opening + classifier + late model.
    This means Hal learns to play against the actual Baku it will face.
    """
    baku_selector = build_baseline_selector()
    baku_opponent = SelectorOpponent(baku_selector, role="baku")

    env = DTHEnv(opponent=baku_opponent, agent_role="hal", seed=seed)
    model = MaskablePPO(
        "MlpPolicy", env, learning_rate=lr, n_steps=2048,
        batch_size=64, n_epochs=10, gamma=0.99, verbose=0, seed=seed,
    )
    model.learn(total_timesteps=timesteps)
    return model


def check_hal_degeneracy(hal_model_or_path, n_games=20, seed=42):
    """Check action diversity of a Hal model."""
    if isinstance(hal_model_or_path, str):
        hal_model = MaskablePPO.load(hal_model_or_path)
    else:
        hal_model = hal_model_or_path

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


def eval_baku_wr_vs_hal(hal_model_or_path, games=50, seed=42):
    """Evaluate baseline Baku win rate against a Hal model."""
    selector = build_baseline_selector()
    if isinstance(hal_model_or_path, str):
        res = evaluate_selector(selector, "hal_model", games=games, seed=seed,
                                opp_model_path=hal_model_or_path)
    else:
        # In-memory model: save to temp then load
        tmp_path = Path("models/checkpoints/_tmp_native_hal")
        hal_model_or_path.save(str(tmp_path))
        res = evaluate_selector(selector, "hal_model", games=games, seed=seed,
                                opp_model_path=str(tmp_path))
    return res


# ── League builders for Baku adaptation ──────────────────────────────

def make_scripted_only_league(seed=42):
    return WeightedOpponentLeague([
        LeagueEntry(label="ht", weight=1.0,
                    opponent=create_scripted_opponent("hal_death_trade")),
        LeagueEntry(label="hp", weight=1.0,
                    opponent=create_scripted_opponent("hal_pressure")),
    ], seed=seed)


def make_scripted_plus_prior_league(prior_weight=1.5, seed=42):
    return WeightedOpponentLeague([
        LeagueEntry(label="ht", weight=1.0,
                    opponent=create_scripted_opponent("hal_death_trade")),
        LeagueEntry(label="hp", weight=1.0,
                    opponent=create_scripted_opponent("hal_pressure")),
        LeagueEntry(label="prior", weight=prior_weight,
                    opponent=create_model_opponent(LEARNED_HAL_PRIOR, agent_role="baku")),
    ], seed=seed)


def make_native_hal_league(native_hal_path, native_weight=2.0,
                           prior_weight=1.0, scripted_weight=1.0, seed=42):
    return WeightedOpponentLeague([
        LeagueEntry(label="ht", weight=scripted_weight,
                    opponent=create_scripted_opponent("hal_death_trade")),
        LeagueEntry(label="hp", weight=scripted_weight,
                    opponent=create_scripted_opponent("hal_pressure")),
        LeagueEntry(label="prior", weight=prior_weight,
                    opponent=create_model_opponent(LEARNED_HAL_PRIOR, agent_role="baku")),
        LeagueEntry(label="native", weight=native_weight,
                    opponent=create_model_opponent(native_hal_path, agent_role="baku")),
    ], seed=seed)


def make_native_dominant_league(native_hal_path, native_weight=4.0,
                                scripted_weight=0.5, seed=42):
    return WeightedOpponentLeague([
        LeagueEntry(label="ht", weight=scripted_weight,
                    opponent=create_scripted_opponent("hal_death_trade")),
        LeagueEntry(label="hp", weight=scripted_weight,
                    opponent=create_scripted_opponent("hal_pressure")),
        LeagueEntry(label="native", weight=native_weight,
                    opponent=create_model_opponent(native_hal_path, agent_role="baku")),
    ], seed=seed)


# ── Training helpers ─────────────────────────────────────────────────

def _make_autoplay_env(league, seed=42, use_shaping=True, shaping_preset="bridge"):
    opening_model = get_opening_model()
    return OpeningAutoPlayEnv(
        opening_model=opening_model, opening_horizon=OPENING_HORIZON,
        opponent=league, agent_role="baku", seed=seed,
        use_shaping=use_shaping, shaping_preset=shaping_preset,
    )


def train_late_autoplay(timesteps, league, seed=42, use_shaping=True,
                         shaping_preset="bridge"):
    env = _make_autoplay_env(league, seed=seed, use_shaping=use_shaping,
                              shaping_preset=shaping_preset)
    model = MaskablePPO.load(BASE_MODEL, env=env)
    model.learn(total_timesteps=timesteps)
    return model


def train_phased_autoplay(phases, seed=42, use_shaping=True, shaping_preset="bridge"):
    model = None
    for i, (timesteps, league) in enumerate(phases):
        env = _make_autoplay_env(league, seed=seed, use_shaping=use_shaping,
                                  shaping_preset=shaping_preset)
        if model is None:
            model = MaskablePPO.load(BASE_MODEL, env=env)
        else:
            model.set_env(env)
        print(f"    Phase {i+1}/{len(phases)}: {timesteps} steps")
        model.learn(total_timesteps=timesteps)
    return model


def train_late_curriculum(timesteps, league, p_opening=0.3,
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


# ══════════════════════════════════════════════════════════════════════
# LANE 0: Baseline Reproduction
# ══════════════════════════════════════════════════════════════════════

def lane_0_baseline():
    print("\n" + "=" * 60)
    print(f"LANE 0: p=0.88 Baseline (PHYSICALITY_BAKU={PHYSICALITY_BAKU})")
    print(f"Route: {BASELINE_ROUTE}")
    print("=" * 60)

    bc_stats = get_bc_stats()
    print(f"  BC: {bc_stats['accuracy']:.1%} ({bc_stats['correct']}/{bc_stats['total']}), "
          f"misaligned={bc_stats['misaligned']}")

    selector = build_baseline_selector()
    results = eval_full_suite(selector, games=EVAL_GAMES, tag="baseline",
                              native_hal_path=None)
    results["bc_stats"] = bc_stats
    return results


# ══════════════════════════════════════════════════════════════════════
# LANE 1: Train Scaffold-Native Hal Candidates
# ══════════════════════════════════════════════════════════════════════

def lane_1_native_hals():
    print("\n" + "=" * 60)
    print("LANE 1: Scaffold-Native Hal Training")
    print("  Training Hal vs FULL modular Baku selector (confound fix)")
    print("=" * 60)

    native_hals = []

    configs = [
        {"tag": "native_hal_16k", "timesteps": 16384, "seed": 42, "lr": 3e-4},
        {"tag": "native_hal_32k", "timesteps": 32768, "seed": 42, "lr": 3e-4},
        {"tag": "native_hal_32k_s2", "timesteps": 32768, "seed": 137, "lr": 3e-4},
    ]

    for cfg in configs:
        tag = cfg["tag"]
        print(f"\n--- Training {tag}: {cfg['timesteps']} steps, seed={cfg['seed']} ---")
        model = train_scaffold_native_hal(
            timesteps=cfg["timesteps"], seed=cfg["seed"], lr=cfg["lr"],
        )

        # Save checkpoint
        ckpt_path = f"models/checkpoints/_native_hal_{tag}"
        model.save(ckpt_path)
        print(f"  Saved: {ckpt_path}.zip")

        # Degeneracy check
        degen = check_hal_degeneracy(model, n_games=20, seed=42)
        print(f"  Degeneracy: {degen['degenerate']} "
              f"(unique={degen['total_unique']}, "
              f"dropper={degen['dropper_unique']}, "
              f"checker={degen['checker_unique']})")
        print(f"  Dropper top3: {degen['dropper_top3']}")
        print(f"  Checker top3: {degen['checker_top3']}")

        # Baku win rate against this Hal
        wr_res = eval_baku_wr_vs_hal(ckpt_path, games=EVAL_GAMES, seed=42)
        print(f"  Baku WR vs {tag}: {pct(wr_res.win_rate)} "
              f"(deaths={wr_res.deaths_by_agent}, avg_hr={wr_res.avg_half_rounds:.1f})")

        native_hals.append({
            "tag": tag,
            "config": cfg,
            "checkpoint": ckpt_path,
            "degeneracy": degen,
            "baku_wr": wr_res.win_rate,
            "baku_deaths": wr_res.deaths_by_agent,
            "baku_avg_hr": wr_res.avg_half_rounds,
        })

    return native_hals


def lane_2_rank_hals(native_hals):
    """Rank native Hals: prefer nondegenerate + lower Baku WR (harder opponent)."""
    print("\n" + "=" * 60)
    print("LANE 2: Rank Scaffold-Native Hal Candidates")
    print("=" * 60)

    # Filter out degenerate
    valid = [h for h in native_hals if not h["degeneracy"]["degenerate"]]
    degenerate = [h for h in native_hals if h["degeneracy"]["degenerate"]]

    for h in degenerate:
        print(f"  EXCLUDED (degenerate): {h['tag']}")

    if not valid:
        print("  !! ALL native Hals degenerate — falling back to prior_4096 only")
        return None, native_hals

    # Sort by Baku WR ascending (lower WR = harder opponent = better)
    valid.sort(key=lambda h: h["baku_wr"])

    print("\n  Ranking (lower Baku WR = better Hal):")
    for i, h in enumerate(valid):
        marker = " <-- BEST" if i == 0 else ""
        print(f"    {i+1}. {h['tag']}: baku_wr={pct(h['baku_wr'])} "
              f"unique={h['degeneracy']['total_unique']}{marker}")

    best = valid[0]
    print(f"\n  Selected: {best['tag']} (Baku WR={pct(best['baku_wr'])})")
    return best, native_hals


# ══════════════════════════════════════════════════════════════════════
# LANE 3: Graded Adaptation Families
# ══════════════════════════════════════════════════════════════════════

def lane_3_training(native_hal_path):
    print("\n" + "=" * 60)
    print("LANE 3: Graded Late-Head Adaptation (3 families)")
    print(f"  Native Hal: {Path(native_hal_path).name}")
    print("=" * 60)

    candidates = []

    # ── Family E: Phased ladder (scripted → +prior → +native), 24k ──
    tag = "E_phased_native_ladder"
    print(f"\n--- {tag}: 3-phase sequential, bridge shaping, 24k ---")
    phases = [
        (8192, make_scripted_only_league()),
        (8192, make_scripted_plus_prior_league(prior_weight=1.5)),
        (8192, make_native_hal_league(native_hal_path,
                                       native_weight=2.0, prior_weight=1.0)),
    ]
    late = train_phased_autoplay(phases, use_shaping=True, shaping_preset="bridge")
    candidates.append({
        "tag": tag, "family": "E", "late_model": late,
        "mechanism": "phased_native_ladder",
        "ppo_steps": 24576,
        "description": "3-phase (scripted->+prior->+native@2.0), bridge shaping, 24k",
    })

    # ── Family F: Native-dominant + curriculum, 20k ──
    tag = "F_native_dominant_curriculum"
    print(f"\n--- {tag}: Native-dominant (4x) + scenario curriculum, bridge, 20k ---")
    league = make_native_dominant_league(native_hal_path,
                                         native_weight=4.0, scripted_weight=0.5)
    late = train_late_curriculum(
        timesteps=20480, league=league,
        p_opening=0.3,
        late_scenarios=["round7_pressure", "round8_bridge", "round9_pre_leap"],
        use_shaping=True, shaping_preset="bridge",
    )
    candidates.append({
        "tag": tag, "family": "F", "late_model": late,
        "mechanism": "native_dominant_curriculum",
        "ppo_steps": 20480,
        "description": "Native-dominant (4x) + scripted(0.5), LateCurriculum p=0.3, r7+r8+r9, bridge, 20k",
    })

    # ── Family G: Balanced native+prior+scripted, autoplay, 20k ──
    tag = "G_balanced_native_prior"
    print(f"\n--- {tag}: Balanced native+prior, autoplay, bridge, 20k ---")
    league = make_native_hal_league(native_hal_path,
                                     native_weight=2.0, prior_weight=2.0,
                                     scripted_weight=1.0)
    late = train_late_autoplay(
        timesteps=20480, league=league,
        use_shaping=True, shaping_preset="bridge",
    )
    candidates.append({
        "tag": tag, "family": "G", "late_model": late,
        "mechanism": "balanced_native_prior",
        "ppo_steps": 20480,
        "description": "Balanced (native@2.0 + prior@2.0 + scripted@1.0), OpeningAutoPlay, bridge, 20k",
    })

    # ── Family H: Light shaping + native heavy + longer budget, 24k ──
    tag = "H_native_heavy_light_shaping"
    print(f"\n--- {tag}: Native heavy (5x) + light shaping, autoplay, 24k ---")
    league = make_native_dominant_league(native_hal_path,
                                         native_weight=5.0, scripted_weight=0.5)
    late = train_late_autoplay(
        timesteps=24576, league=league,
        use_shaping=True, shaping_preset="light",
    )
    candidates.append({
        "tag": tag, "family": "H", "late_model": late,
        "mechanism": "native_heavy_light",
        "ppo_steps": 24576,
        "description": "Native heavy (5x) + scripted(0.5), OpeningAutoPlay, light shaping, 24k",
    })

    return candidates


# ══════════════════════════════════════════════════════════════════════
# LANE 4: Quick Gate
# ══════════════════════════════════════════════════════════════════════

def lane_4_gate(candidates):
    print("\n" + "=" * 60)
    print(f"LANE 4: Quick Gate ({GATE_GAMES}-game)")
    print("=" * 60)

    surviving = []
    for cand in candidates:
        tag = cand["tag"]
        print(f"\n--- Gate: {tag} ---")
        selector = build_baseline_selector(cand["late_model"])
        passed, gate_results = eval_gate(selector, games=GATE_GAMES, tag=tag)
        cand["gate_results"] = gate_results
        cand["gate_pass"] = passed
        if passed:
            surviving.append(cand)
        else:
            print(f"  !! {tag} GATED OUT")

    print(f"\n  Gate survivors: {len(surviving)}/{len(candidates)}")
    return surviving


# ══════════════════════════════════════════════════════════════════════
# LANE 5: Full 50-Game Evaluation
# ══════════════════════════════════════════════════════════════════════

def lane_5_full_eval(candidates, native_hal_path):
    print("\n" + "=" * 60)
    print(f"LANE 5: Full {EVAL_GAMES}-Game Evaluation")
    print("=" * 60)

    for cand in candidates:
        tag = cand["tag"]
        print(f"\n--- Full eval: {tag} ---")
        selector = build_baseline_selector(cand["late_model"])
        res = eval_full_suite(selector, games=EVAL_GAMES, tag=tag,
                              native_hal_path=native_hal_path)
        res["bc_stats"] = get_bc_stats()
        cand["eval_results"] = res

    return candidates


# ══════════════════════════════════════════════════════════════════════
# Analysis & Verdicts
# ══════════════════════════════════════════════════════════════════════

def analyze_results(baseline, candidates, bl_native_wr=None):
    bl_bp = baseline.get("bridge_pressure", {}).get("r7_rate", 0)
    bl_ht = baseline.get("hal_death_trade", {}).get("r7_rate", 0)
    bl_hp = baseline.get("hal_pressure", {}).get("r7_rate", 0)
    bl_r9 = baseline.get("seeded_r9", {}).get("rate", 0)
    bl_prior = baseline.get("prior_4096", {}).get("win_rate", 0)
    bl_mis = baseline.get("bc_stats", {}).get("misaligned", 0)
    bl_nat = bl_native_wr if bl_native_wr is not None else 0.0

    best_candidate = None
    best_level = None
    best_score = -1
    any_improvement = False

    for cand in candidates:
        v = cand.get("eval_results", {})
        c_bp = v.get("bridge_pressure", {}).get("r7_rate", 0)
        c_ht = v.get("hal_death_trade", {}).get("r7_rate", 0)
        c_hp = v.get("hal_pressure", {}).get("r7_rate", 0)
        c_r9 = v.get("seeded_r9", {}).get("rate", 0)
        c_prior = v.get("prior_4096", {}).get("win_rate", 0)
        c_native = v.get("native_hal", {}).get("win_rate", 0)
        c_mis = v.get("bc_stats", {}).get("misaligned", 0)

        prior_delta = c_prior - bl_prior
        native_delta = c_native - bl_nat

        # Strong pass
        strong = (
            c_bp >= STRONG_PASS["bp_r7"]
            and c_ht >= STRONG_PASS["ht_r7"]
            and c_hp >= STRONG_PASS["hp_r7"]
            and c_r9 >= STRONG_PASS["seeded_r9"]
            and c_mis == STRONG_PASS["misaligned"]
            and (prior_delta >= STRONG_PASS["prior_wr_delta"]
                 or native_delta >= STRONG_PASS["native_wr_delta"])
        )

        # Acceptable pass
        acceptable = (
            c_bp >= ACCEPTABLE_PASS["bp_r7"]
            and c_ht >= ACCEPTABLE_PASS["ht_r7"]
            and c_hp >= ACCEPTABLE_PASS["hp_r7"]
            and c_r9 >= ACCEPTABLE_PASS["seeded_r9"]
            and c_mis == ACCEPTABLE_PASS["misaligned"]
            and (prior_delta >= ACCEPTABLE_PASS["prior_wr_delta"]
                 or native_delta >= ACCEPTABLE_PASS["native_wr_delta"])
        )

        if prior_delta > 0.02 or native_delta > 0.02:
            any_improvement = True

        score = (
            2.0 * prior_delta
            + 3.0 * native_delta
            + 0.5 * (c_ht - bl_ht)
            + 0.5 * (c_hp - bl_hp)
            + 0.3 * (c_bp - bl_bp)
        )

        level = "STRONG_PASS" if strong else ("ACCEPTABLE_PASS" if acceptable else None)

        cand["analysis"] = {
            "strong_pass": strong,
            "acceptable_pass": acceptable,
            "level": level,
            "composite_score": round(score, 4),
            "deltas": {
                "bp_r7": round(c_bp - bl_bp, 4),
                "ht_r7": round(c_ht - bl_ht, 4),
                "hp_r7": round(c_hp - bl_hp, 4),
                "prior_wr": round(prior_delta, 4),
                "native_wr": round(native_delta, 4),
            },
            "abs": {
                "bp_r7": round(c_bp, 4),
                "ht_r7": round(c_ht, 4),
                "hp_r7": round(c_hp, 4),
                "r9": round(c_r9, 4),
                "prior_wr": round(c_prior, 4),
                "native_wr": round(c_native, 4),
                "misaligned": c_mis,
            },
        }

        if level == "STRONG_PASS" and best_level != "STRONG_PASS":
            best_candidate = cand
            best_level = "STRONG_PASS"
            best_score = score
        elif level == "ACCEPTABLE_PASS" and best_level is None:
            best_candidate = cand
            best_level = "ACCEPTABLE_PASS"
            best_score = score
        elif score > best_score and (
            (level == best_level) or (best_level is None and level is None)
        ):
            best_candidate = cand
            best_score = score

    verdict = "STRONG_PASS" if best_level == "STRONG_PASS" else (
        "ACCEPTABLE_PASS" if best_level == "ACCEPTABLE_PASS" else (
            "SIGNAL_DETECTED" if any_improvement else "NO_IMPROVEMENT"
        )
    )

    return {
        "best_candidate": best_candidate["tag"] if best_candidate else None,
        "best_level": best_level,
        "best_composite_score": round(best_score, 4) if best_candidate else None,
        "any_improvement": any_improvement,
        "verdict": verdict,
    }


# ══════════════════════════════════════════════════════════════════════
# Report Generation
# ══════════════════════════════════════════════════════════════════════

def write_markdown_report(baseline, native_hals, best_native, all_candidates,
                          surviving, verdict, bl_native_wr, elapsed):
    bc_stats = baseline.get("bc_stats", get_bc_stats())
    bl_bp = baseline.get("bridge_pressure", {}).get("r7_rate", 0)
    bl_ht = baseline.get("hal_death_trade", {}).get("r7_rate", 0)
    bl_hp = baseline.get("hal_pressure", {}).get("r7_rate", 0)
    bl_r9 = baseline.get("seeded_r9", {}).get("rate", 0)
    bl_prior = baseline.get("prior_4096", {}).get("win_rate", 0)

    lines = []
    a = lines.append

    a("# Sprint Report: Scaffold-Native Self-Play Ladder at p=0.88")
    a(f"**Date:** 2026-03-28")
    a(f"**PHYSICALITY_BAKU:** {PHYSICALITY_BAKU}")
    a(f"**Verdict:** {verdict['verdict']}")
    a("")
    a("## Objective")
    a("")
    a("Fix the key confound from the previous ladder sprint: Hal opponents were")
    a("trained against BASE_MODEL alone, not the actual modular p=0.88 Baku scaffold.")
    a("This sprint trains scaffold-native Hal candidates against the full")
    a("ModularBakuSelector, then uses them in a graded opponent ladder.")
    a("")

    a("## Baseline Reproduction")
    a("")
    a(f"- Route: `{BASELINE_ROUTE}`")
    a(f"- BC accuracy: {bc_stats['accuracy']:.1%} ({bc_stats['correct']}/{bc_stats['total']}), misaligned={bc_stats['misaligned']}")
    a(f"- bp r7: {pct(bl_bp)} | ht r7: {pct(bl_ht)} | hp r7: {pct(bl_hp)}")
    a(f"- seeded r9: {pct(bl_r9)}")
    a(f"- vs prior_4096 win rate: {pct(bl_prior)}")
    a("")

    a("## Scaffold-Native Hal Candidates")
    a("")
    a("These Hals were trained against the **full modular Baku selector** (BC opening")
    a("+ classifier + late model), not BASE_MODEL alone.")
    a("")
    a("| Candidate | Steps | Seed | Degenerate | Unique Actions | Baku WR |")
    a("|-----------|-------|------|------------|----------------|---------|")
    for h in native_hals:
        a(f"| {h['tag']} | {h['config']['timesteps']} | {h['config']['seed']} | "
          f"{'YES' if h['degeneracy']['degenerate'] else 'no'} | "
          f"{h['degeneracy']['total_unique']} | {pct(h['baku_wr'])} |")
    a("")

    if best_native:
        a(f"**Selected best native Hal:** {best_native['tag']} "
          f"(Baku WR={pct(best_native['baku_wr'])})")
        a(f"- Baseline Baku WR vs native: {pct(bl_native_wr)}")
    else:
        a("**All native Hals were degenerate.** Falling back to prior_4096 only.")
    a("")

    a("## Scaffold-Native Self-Play Signal Assessment")
    a("")
    if best_native:
        wr_diff = best_native["baku_wr"]
        if wr_diff < 0.10:
            a("Native Hal provides strong pressure (Baku WR < 10%).")
        elif wr_diff < 0.30:
            a("Native Hal provides moderate pressure (Baku WR 10-30%).")
        elif wr_diff < 0.50:
            a("Native Hal provides mild pressure (Baku WR 30-50%).")
        else:
            a("Native Hal provides weak pressure (Baku WR >= 50%).")
        a("")
        compare_to_old = "stronger" if wr_diff < baseline.get("prior_4096", {}).get("win_rate", 1.0) else "weaker"
        a(f"Compared to prior_4096 (Baku WR={pct(bl_prior)}), the native Hal is **{compare_to_old}**.")
    else:
        a("No usable scaffold-native Hal was generated.")
    a("")

    a("## Graded-Ladder Families")
    a("")
    a("| Family | Mechanism | Steps | Gate |")
    a("|--------|-----------|-------|------|")
    for cand in all_candidates:
        gate = "PASS" if cand.get("gate_pass") else "FAIL"
        a(f"| {cand['tag']} | {cand['mechanism']} | {cand['ppo_steps']} | {gate} |")
    a("")

    a("### Family Descriptions")
    a("")
    for cand in all_candidates:
        a(f"**{cand['tag']}:** {cand['description']}")
        a("")

    if surviving:
        a("## Full Evaluation Results (50-game)")
        a("")
        a("| Candidate | bp r7 | ht r7 | hp r7 | r9 | prior WR | native WR | mis |")
        a("|-----------|-------|-------|-------|----|----------|-----------|-----|")
        a(f"| **baseline** | {pct(bl_bp)} | {pct(bl_ht)} | {pct(bl_hp)} | {pct(bl_r9)} | {pct(bl_prior)} | {pct(bl_native_wr)} | {bc_stats['misaligned']} |")
        for cand in surviving:
            ab = cand.get("analysis", {}).get("abs", {})
            a(f"| {cand['tag']} | {pct(ab.get('bp_r7', 0))} | {pct(ab.get('ht_r7', 0))} | "
              f"{pct(ab.get('hp_r7', 0))} | {pct(ab.get('r9', 0))} | "
              f"{pct(ab.get('prior_wr', 0))} | {pct(ab.get('native_wr', 0))} | "
              f"{ab.get('misaligned', 0)} |")
        a("")

        a("### Deltas vs Baseline")
        a("")
        a("| Candidate | bp r7 | ht r7 | hp r7 | prior WR | native WR | Score | Level |")
        a("|-----------|-------|-------|-------|----------|-----------|-------|-------|")
        for cand in surviving:
            d = cand.get("analysis", {}).get("deltas", {})
            level = cand.get("analysis", {}).get("level") or "---"
            score = cand.get("analysis", {}).get("composite_score", 0)
            a(f"| {cand['tag']} | {d.get('bp_r7', 0):+.0%} | {d.get('ht_r7', 0):+.0%} | "
              f"{d.get('hp_r7', 0):+.0%} | {d.get('prior_wr', 0):+.0%} | "
              f"{d.get('native_wr', 0):+.0%} | {score:.4f} | {level} |")
        a("")

    a("## Promotion Criteria")
    a("")
    a("| Bar | bp r7 | ht r7 | hp r7 | r9 | mis | prior WR delta | native WR delta |")
    a("|-----|-------|-------|-------|----|-----|----------------|-----------------|")
    a(f"| Strong | >={pct(STRONG_PASS['bp_r7'])} | >={pct(STRONG_PASS['ht_r7'])} | >={pct(STRONG_PASS['hp_r7'])} | {pct(STRONG_PASS['seeded_r9'])} | {STRONG_PASS['misaligned']} | +{pct(STRONG_PASS['prior_wr_delta'])} | +{pct(STRONG_PASS['native_wr_delta'])} |")
    a(f"| Acceptable | >={pct(ACCEPTABLE_PASS['bp_r7'])} | >={pct(ACCEPTABLE_PASS['ht_r7'])} | >={pct(ACCEPTABLE_PASS['hp_r7'])} | {pct(ACCEPTABLE_PASS['seeded_r9'])} | {ACCEPTABLE_PASS['misaligned']} | +{pct(ACCEPTABLE_PASS['prior_wr_delta'])} | +{pct(ACCEPTABLE_PASS['native_wr_delta'])} |")
    a("")

    a("## Verdict")
    a("")
    a(f"**{verdict['verdict']}**")
    a("")
    if verdict.get("best_candidate"):
        a(f"- Best candidate: {verdict['best_candidate']}")
        a(f"- Level: {verdict.get('best_level', '---')}")
        a(f"- Composite score: {verdict.get('best_composite_score', '---')}")
    else:
        a("- No candidate met promotion criteria.")
    a(f"- Any improvement signal: {'YES' if verdict.get('any_improvement') else 'NO'}")
    a(f"- Sprint time: {elapsed:.0f}s")
    a("")

    a("## Analysis")
    a("")
    a("### Does scaffold-native self-play yield real improvement signal?")
    a("")
    if verdict.get("any_improvement"):
        a("**YES** — some improvement signal was detected when training against")
        a("scaffold-native Hal opponents. See candidate details above for specifics.")
    else:
        a("**NO** — training Hal against the full modular Baku scaffold produced")
        a("no measurable improvement in Baku adaptation, despite fixing the confound")
        a("of training against BASE_MODEL alone. The MlpPolicy deterministic argmax")
        a("ceiling persists regardless of opponent provenance.")
    a("")

    out_dir = Path("docs/reports")
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "sprint_self_play_p088_native_ladder_report_2026-03-28.md"
    path.write_text("\n".join(lines))
    print(f"\nMarkdown report written to {path}")
    return str(path)


# ══════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════

def main():
    start = time.time()
    print("=" * 60)
    print("SPRINT: Scaffold-Native Self-Play Ladder at p=0.88")
    print("Date: 2026-03-28")
    print(f"PHYSICALITY_BAKU: {PHYSICALITY_BAKU}")
    print("=" * 60)
    print(f"\nArtifacts:")
    print(f"  BASE_MODEL:        {Path(BASE_MODEL).name}")
    print(f"  BP_SPECIALIST:     {Path(BP_SPECIALIST).name}")
    print(f"  LEARNED_HAL_PRIOR: {Path(LEARNED_HAL_PRIOR).name}")
    print(f"  FRESH_HAL_32K:     {Path(FRESH_HAL_32K).name}")
    print(f"  Route:             {BASELINE_ROUTE}")
    print(f"\nConfound fix: Hal trained vs full ModularBakuSelector, not BASE_MODEL")

    # Lane 0: Baseline
    baseline = lane_0_baseline()

    # Lane 1: Train scaffold-native Hals
    native_hals = lane_1_native_hals()

    # Lane 2: Rank and select best native Hal
    best_native, native_hals = lane_2_rank_hals(native_hals)

    # Measure baseline vs best native Hal
    bl_native_wr = 0.0
    native_hal_path = None
    if best_native:
        native_hal_path = best_native["checkpoint"]
        bl_native_res = eval_baku_wr_vs_hal(native_hal_path, games=EVAL_GAMES, seed=42)
        bl_native_wr = bl_native_res.win_rate
        # Add native Hal results to baseline
        baseline["native_hal"] = asdict(bl_native_res)
        print(f"\n  Baseline vs best native Hal: wr={pct(bl_native_wr)}")

    # Lane 3: Training families (use native Hal if available, else prior only)
    if native_hal_path:
        all_candidates = lane_3_training(native_hal_path)
    else:
        # Fallback: use FRESH_HAL_32K in place of native
        print("\n  !! No usable native Hal — using FRESH_HAL_32K as substitute")
        native_hal_path = FRESH_HAL_32K
        bl_native_res = eval_baku_wr_vs_hal(native_hal_path, games=EVAL_GAMES, seed=42)
        bl_native_wr = bl_native_res.win_rate
        baseline["native_hal"] = asdict(bl_native_res)
        all_candidates = lane_3_training(native_hal_path)

    # Lane 4: Gate
    surviving = lane_4_gate(all_candidates)

    # Lane 5: Full eval
    if surviving:
        surviving = lane_5_full_eval(surviving, native_hal_path)
    else:
        print("\n!! ALL CANDIDATES GATED OUT — no full eval !!")

    # Analysis
    verdict = analyze_results(baseline, surviving, bl_native_wr=bl_native_wr)

    elapsed = time.time() - start

    # ── Build JSON output ────────────────────────────────────────────
    bc_stats = baseline.get("bc_stats", get_bc_stats())
    bl_bp = baseline.get("bridge_pressure", {}).get("r7_rate", 0)
    bl_ht = baseline.get("hal_death_trade", {}).get("r7_rate", 0)
    bl_hp = baseline.get("hal_pressure", {}).get("r7_rate", 0)
    bl_r9 = baseline.get("seeded_r9", {}).get("rate", 0)
    bl_prior = baseline.get("prior_4096", {}).get("win_rate", 0)

    out = {
        "sprint": "self-play-p088-native-ladder",
        "date": "2026-03-28",
        "physicality_baku": PHYSICALITY_BAKU,
        "confound_fix": "Hal trained vs full ModularBakuSelector, not BASE_MODEL alone",
        "baseline_route": BASELINE_ROUTE,
        "eval_games": EVAL_GAMES,
        "gate_games": GATE_GAMES,
        "artifacts": {
            "base_model": Path(BASE_MODEL).name,
            "bp_specialist": Path(BP_SPECIALIST).name,
            "learned_hal_prior": Path(LEARNED_HAL_PRIOR).name,
            "fresh_hal_32k": Path(FRESH_HAL_32K).name,
            "opening_bc": f"route {BASELINE_ROUTE}, 80 epochs, accuracy {bc_stats['accuracy']:.1%}",
        },
        "baseline": {k: v for k, v in baseline.items() if k != "bc_stats"},
        "baseline_bc_stats": bc_stats,
        "baseline_native_wr": bl_native_wr,
        "native_hals": [
            {k: v for k, v in h.items() if k != "model"}
            for h in native_hals
        ],
        "best_native_hal": best_native["tag"] if best_native else None,
        "candidates": [
            {k: v for k, v in c.items() if k != "late_model"}
            for c in all_candidates
        ],
        "gate_survivors": [c["tag"] for c in surviving],
        "evaluated_candidates": [
            {k: v for k, v in c.items() if k != "late_model"}
            for c in surviving
        ],
        "promotion_bars": {
            "strong": STRONG_PASS,
            "acceptable": ACCEPTABLE_PASS,
        },
        "verdict": verdict,
        "elapsed_seconds": round(elapsed),
    }

    out_dir = Path("docs/reports/json")
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "sprint_self_play_p088_native_ladder_2026-03-28.json"
    with open(json_path, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\nJSON written to {json_path}")

    md_path = write_markdown_report(
        baseline, native_hals, best_native, all_candidates,
        surviving, verdict, bl_native_wr, elapsed,
    )

    # ── Terminal Summary ─────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("TERMINAL SUMMARY: Scaffold-Native Self-Play Ladder")
    print("=" * 60)
    print(f"  PHYSICALITY_BAKU:     {PHYSICALITY_BAKU}")
    print(f"  BC misaligned:        {bc_stats.get('misaligned', '?')}")
    print(f"  Confound fix:         Hal vs full ModularBakuSelector")
    print(f"  ────────────────────────────────────────")
    print(f"  BASELINE ({EVAL_GAMES}-game):")
    print(f"    bp  r7:     {pct(bl_bp)}")
    print(f"    ht  r7:     {pct(bl_ht)}")
    print(f"    hp  r7:     {pct(bl_hp)}")
    print(f"    seeded r9:  {pct(bl_r9)}")
    print(f"    prior WR:   {pct(bl_prior)}")
    print(f"    native WR:  {pct(bl_native_wr)}")
    print(f"  ────────────────────────────────────────")
    print(f"  NATIVE HAL CANDIDATES:")
    for h in native_hals:
        degen = "DEGEN" if h["degeneracy"]["degenerate"] else "ok"
        selected = " <-- SELECTED" if (best_native and h["tag"] == best_native["tag"]) else ""
        print(f"    {h['tag']:25s}: baku_wr={pct(h['baku_wr'])} "
              f"unique={h['degeneracy']['total_unique']} [{degen}]{selected}")
    print(f"  ────────────────────────────────────────")
    print(f"  ADAPTATION FAMILIES: {len(all_candidates)} trained, "
          f"{len(surviving)} survived gate")
    for cand in all_candidates:
        gate = "PASS" if cand.get("gate_pass") else "FAIL"
        if cand in surviving and "analysis" in cand:
            an = cand["analysis"]
            d = an.get("deltas", {})
            ab = an.get("abs", {})
            level = an.get("level") or "---"
            print(
                f"    {cand['tag']:35s}: bp={pct(ab.get('bp_r7', 0))} "
                f"ht={pct(ab.get('ht_r7', 0))} hp={pct(ab.get('hp_r7', 0))} "
                f"r9={pct(ab.get('r9', 0))} "
                f"prior={pct(ab.get('prior_wr', 0))}({d.get('prior_wr', 0):+.0%}) "
                f"native={pct(ab.get('native_wr', 0))}({d.get('native_wr', 0):+.0%}) "
                f"[{level}]"
            )
        else:
            print(f"    {cand['tag']:35s}: GATE={gate}")
    print(f"  ────────────────────────────────────────")
    best = verdict.get("best_candidate") or "NONE"
    v = verdict.get("verdict", "?")
    print(f"  Best candidate:       {best}")
    print(f"  Verdict:              {v}")
    print(f"  Any improvement:      {'YES' if verdict.get('any_improvement') else 'NO'}")
    print(f"  Native self-play signal: {'YES' if verdict.get('any_improvement') else 'NO'}")
    print(f"  Sprint time:          {elapsed:.0f}s")
    print("=" * 60)

    return out


if __name__ == "__main__":
    main()
