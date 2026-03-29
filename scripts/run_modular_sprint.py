#!/usr/bin/env python3
"""Sprint: Modular Opening Head — full execution.

Runs all lanes (0 through D) of the modular opening head sprint.
Tests whether separate opening parameters prevent the opening-route
collapse under PPO perturbation.
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
from environment.opponents.factory import (
    build_opponent_league,
    create_model_opponent,
    create_scripted_opponent,
)
from environment.route_stages import current_route_stage_flags, stage_is_eligible_from_start
from src.Constants import LS_WINDOW_START
from training.expert_selector import build_classifier, build_controller_selector
from training.modular_policy import (
    ModularBakuSelector,
    OpeningAutoPlayEnv,
    OpeningTruncatedEnv,
    build_modular_selector,
    collect_opening_samples,
    train_opening_model_bc,
    verify_opening_accuracy,
)

# ── Paths ──────────────────────────────────────────────────────────────
def _resolve_checkpoint(stem: str) -> str:
    candidates = (
        Path("models/checkpoints") / stem,
        Path("models/checkpoints") / f"{stem}.zip",
        Path("STL/models/checkpoints") / stem,
        Path("STL/models/checkpoints") / f"{stem}.zip",
    )
    for candidate in candidates:
        if candidate.exists():
            return str(candidate.with_suffix("")) if candidate.suffix == ".zip" else str(candidate)
    raise FileNotFoundError(f"Could not resolve checkpoint '{stem}' in models/checkpoints or STL/models/checkpoints")


BASE_MODEL = _resolve_checkpoint("baku_vs_bridge_pressure_opening_to_round7_bc_only_bc_combined_0.3")
BP_SPECIALIST = _resolve_checkpoint("baku_vs_bridge_pressure_opening_to_round7_bc_only_bc_spec_e50_lr5.5")
LEARNED_HAL = _resolve_checkpoint("hal_vs_promoted_baku_prior_4096")

# The known T4=10 override: turn index 3 (0-indexed) → action_second 10
HAL_OVERRIDES = {3: 10}

# Opponents for data collection and evaluation
HAL_OPPONENTS = ["hal_death_trade", "hal_pressure"]
ALL_OPPONENTS = ["bridge_pressure", "hal_death_trade", "hal_pressure"]


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
    r9_count: int
    r9_eligible: int
    r9_rate: float
    deaths_by_agent: int


def evaluate_selector(selector, opponent_name: str, games: int = 24, seed: int = 42,
                       opponent_model_path: str | None = None) -> EvalResult:
    """Evaluate a selector-like object against an opponent."""
    wins = 0
    total_hr = 0
    r7_count = 0
    r7_eligible = 0
    r9_count = 0
    r9_eligible = 0
    deaths = 0

    for gi in range(games):
        if opponent_model_path:
            opp = create_model_opponent(opponent_model_path, agent_role="hal")
        else:
            opp = create_scripted_opponent(opponent_name)

        env = DTHEnv(opponent=opp, agent_role="baku", seed=seed + gi)
        obs, info = env.reset()
        selector.reset()

        reached_r7 = current_route_stage_flags(env.game).get("round7_pressure", False)
        reached_r9 = current_route_stage_flags(env.game).get("round9_pre_leap", False)
        started_before_r7 = stage_is_eligible_from_start(env.game, "round7_pressure")
        started_before_r9 = stage_is_eligible_from_start(env.game, "round9_pre_leap")

        while True:
            mask = env.action_masks()
            action, _ = selector.predict(obs, action_masks=mask, deterministic=True)
            obs, reward, term, trunc, info = env.step(int(action))

            flags = current_route_stage_flags(env.game)
            reached_r7 = reached_r7 or flags.get("round7_pressure", False)
            reached_r9 = reached_r9 or flags.get("round9_pre_leap", False)

            if term or trunc:
                won = bool(term and env.game.winner is env.agent and reward > 0)
                wins += int(won)
                total_hr += len(env.game.history)
                if started_before_r7:
                    r7_eligible += 1
                    r7_count += int(reached_r7)
                if started_before_r9:
                    r9_eligible += 1
                    r9_count += int(reached_r9)
                deaths += env.agent.deaths
                break

    n = games
    return EvalResult(
        opponent=opponent_name,
        games=n,
        wins=wins,
        win_rate=wins / n if n else 0,
        avg_half_rounds=total_hr / n if n else 0,
        r7_count=r7_count,
        r7_eligible=r7_eligible,
        r7_rate=r7_count / r7_eligible if r7_eligible else 0,
        r9_count=r9_count,
        r9_eligible=r9_eligible,
        r9_rate=r9_count / r9_eligible if r9_eligible else 0,
        deaths_by_agent=deaths,
    )


def eval_r9_seeded(model: MaskablePPO, seed: int = 42) -> dict:
    """Evaluate a model on the single seeded round9_pre_leap scenario.

    Uses a single deterministic game at seed=42, matching the prior
    sprint baseline evaluation method. This is a pass/fail test:
    the agent must win this specific deterministic game.
    """
    from training.curriculum import get_scenario
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
               r9_model: MaskablePPO | None = None) -> dict:
    """Run full evaluation suite against all scripted opponents + seeded r9.

    r9_model: the model to use for seeded r9 eval (the late/base model,
    not the selector). Falls back to selector-based eval if None.
    """
    results = {}
    for opp in ALL_OPPONENTS:
        res = evaluate_selector(selector, opp, games=games, seed=seed)
        results[opp] = asdict(res)
        pct = lambda v: f"{v*100:.0f}%"
        print(f"  {tag} vs {opp}: r7={pct(res.r7_rate)} wins={pct(res.win_rate)} avg_hr={res.avg_half_rounds:.1f}")

    # Seeded r9: use the late/base model directly
    if r9_model is not None:
        r9_res = eval_r9_seeded(r9_model, seed=seed)
    else:
        # Fallback: use selector (works if selector routes correctly for late game)
        from training.curriculum import get_scenario
        r9_wins = 0
        r9_games = 10
        for gi in range(r9_games):
            opp = create_scripted_opponent("safe")
            env = DTHEnv(opponent=opp, agent_role="baku", seed=seed + gi,
                         scenario_sampler=lambda _rng: get_scenario("round9_pre_leap"))
            obs, _ = env.reset()
            selector.reset()
            while True:
                mask = env.action_masks()
                action, _ = selector.predict(obs, action_masks=mask, deterministic=True)
                obs, reward, term, trunc, _ = env.step(int(action))
                if term or trunc:
                    if term and env.game.winner is env.agent and reward > 0:
                        r9_wins += 1
                    break
        r9_res = {"wins": r9_wins, "games": r9_games, "rate": r9_wins / r9_games}

    results["seeded_r9"] = r9_res
    print(f"  {tag} seeded r9: {r9_res['wins']}/{r9_res['games']} ({r9_res['rate']*100:.0f}%)")
    return results


def ppo_perturb_model(model_path: str, timesteps: int, seed: int = 42) -> MaskablePPO:
    """PPO-train a model against a mixed league (bp+ht+hp+learned_hal).

    Matches the prior sprint training setup to reproduce the known collapse.
    """
    from environment.opponents.league import WeightedOpponentLeague, LeagueEntry
    entries = [
        LeagueEntry(label="bp", weight=1.0, opponent=create_scripted_opponent("bridge_pressure")),
        LeagueEntry(label="ht", weight=1.0, opponent=create_scripted_opponent("hal_death_trade")),
        LeagueEntry(label="hp", weight=1.0, opponent=create_scripted_opponent("hal_pressure")),
        LeagueEntry(label="lhal", weight=0.5, opponent=create_model_opponent(LEARNED_HAL, agent_role="hal")),
    ]
    opponent = WeightedOpponentLeague(entries, seed=seed)
    env = DTHEnv(opponent=opponent, agent_role="baku", seed=seed)
    model = MaskablePPO.load(model_path, env=env)
    model.learn(total_timesteps=timesteps)
    return model


# ── Lane 0: Baseline collapse reproduction ─────────────────────────────

def lane_0():
    """Reproduce the old architecture collapse under PPO perturbation."""
    print("\n" + "=" * 60)
    print("LANE 0: Baseline Collapse Reproduction")
    print("=" * 60)

    # Build old-architecture ControllerSelector
    old_selector = build_controller_selector(
        bp_specialist_path=BP_SPECIALIST,
        base_model_path=BASE_MODEL,
        hal_overrides=HAL_OVERRIDES,
        controller_turns=(2, 4),
        classify_turn=2,
        controller_type="lookup",
    )

    base_model = MaskablePPO.load(BASE_MODEL)

    # Static eval
    print("\n--- Static eval (old architecture) ---")
    static_results = eval_suite(old_selector, games=24, tag="old_static", r9_model=base_model)

    # PPO perturbation at 2048, 4096, 8192 steps (1, 2, 4 policy updates)
    perturbed_results = {}
    for steps in [2048, 4096, 8192]:
        print(f"\n--- PPO perturbation {steps} steps (old architecture) ---")
        perturbed_model = ppo_perturb_model(BASE_MODEL, timesteps=steps)
        # Rebuild old selector with perturbed base
        perturbed_selector = build_controller_selector(
            bp_specialist_path=BP_SPECIALIST,
            base_model_path=BASE_MODEL,  # classifier uses original base
            hal_overrides=HAL_OVERRIDES,
            controller_turns=(2, 4),
            classify_turn=2,
            controller_type="lookup",
        )
        # Replace the base_model in the selector with the perturbed one
        perturbed_selector.base_model = perturbed_model
        res = eval_suite(perturbed_selector, games=24, tag=f"old_ppo{steps}", r9_model=perturbed_model)
        perturbed_results[steps] = res

    return {
        "static": static_results,
        "perturbed": perturbed_results,
    }


# ── Lane A: Build modular candidates ──────────────────────────────────

def build_family_a(opening_horizon: int, tag: str) -> dict:
    """Family A: Frozen opening (BC-only) + trainable late."""
    print(f"\n--- Family A [{tag}]: frozen opening, horizon={opening_horizon} ---")
    opening_model, samples = train_opening_model_bc(
        base_model_path=BASE_MODEL,
        opponent_names=HAL_OPPONENTS,
        action_overrides=HAL_OVERRIDES,
        opening_horizon=opening_horizon,
        epochs=80,
        lr=1e-3,
    )
    stats = verify_opening_accuracy(opening_model, samples)
    late_model = MaskablePPO.load(BASE_MODEL)

    return {
        "tag": tag,
        "family": "A",
        "opening_horizon": opening_horizon,
        "bc_accuracy": stats["accuracy"],
        "bc_samples": stats["total"],
        "opening_model": opening_model,
        "late_model": late_model,
        "opening_trainable": False,
    }


def build_family_b(opening_horizon: int, opening_ppo_steps: int, tag: str) -> dict:
    """Family B: Trainable opening (BC + opening PPO) + trainable late."""
    print(f"\n--- Family B [{tag}]: trainable opening, horizon={opening_horizon}, opening_ppo={opening_ppo_steps} ---")

    # Step 1: BC-initialize the opening model
    opening_model, samples = train_opening_model_bc(
        base_model_path=BASE_MODEL,
        opponent_names=HAL_OPPONENTS,
        action_overrides=HAL_OVERRIDES,
        opening_horizon=opening_horizon,
        epochs=80,
        lr=1e-3,
    )
    bc_stats = verify_opening_accuracy(opening_model, samples)

    # Step 2: PPO-train the opening model on truncated env
    print(f"  Opening PPO: {opening_ppo_steps} steps on truncated env...")
    opp = create_scripted_opponent("hal_death_trade")
    trunc_env = OpeningTruncatedEnv(
        opening_horizon=opening_horizon,
        opponent=opp,
        agent_role="baku",
        seed=42,
    )
    # Transfer the BC-trained model to the truncated env
    opening_model.set_env(trunc_env)
    opening_model.learn(total_timesteps=opening_ppo_steps)

    # Verify accuracy after PPO
    ppo_stats = verify_opening_accuracy(opening_model, samples)
    print(f"  Opening accuracy after PPO: {ppo_stats['accuracy']:.1%} (was {bc_stats['accuracy']:.1%})")

    late_model = MaskablePPO.load(BASE_MODEL)

    return {
        "tag": tag,
        "family": "B",
        "opening_horizon": opening_horizon,
        "opening_ppo_steps": opening_ppo_steps,
        "bc_accuracy": bc_stats["accuracy"],
        "ppo_accuracy": ppo_stats["accuracy"],
        "bc_samples": bc_stats["total"],
        "opening_model": opening_model,
        "late_model": late_model,
        "opening_trainable": True,
    }


def lane_a() -> list[dict]:
    """Build all modular architecture candidates."""
    print("\n" + "=" * 60)
    print("LANE A: Modular Architecture Families")
    print("=" * 60)

    candidates = []

    # Family A variants: frozen opening with different horizons
    candidates.append(build_family_a(opening_horizon=8, tag="A1_h8"))
    candidates.append(build_family_a(opening_horizon=12, tag="A2_h12"))

    # Family B variants: trainable opening with PPO
    candidates.append(build_family_b(opening_horizon=8, opening_ppo_steps=2048, tag="B1_h8_ppo2k"))
    candidates.append(build_family_b(opening_horizon=8, opening_ppo_steps=4096, tag="B2_h8_ppo4k"))

    return candidates


# ── Lane B: Static parity gate ────────────────────────────────────────

def lane_b(candidates: list[dict]) -> list[dict]:
    """Static parity gate: 24-game pilot."""
    print("\n" + "=" * 60)
    print("LANE B: Static Parity Gate")
    print("=" * 60)

    passing = []
    for cand in candidates:
        tag = cand["tag"]
        print(f"\n--- Static parity: {tag} ---")

        selector = build_modular_selector(
            bp_specialist_path=BP_SPECIALIST,
            base_model_path=BASE_MODEL,
            opening_model=cand["opening_model"],
            late_model=cand["late_model"],
            opening_horizon=cand["opening_horizon"],
        )

        results = eval_suite(selector, games=24, tag=tag, r9_model=cand["late_model"])

        bp_r7 = results.get("bridge_pressure", {}).get("r7_rate", 0)
        ht_r7 = results.get("hal_death_trade", {}).get("r7_rate", 0)
        hp_r7 = results.get("hal_pressure", {}).get("r7_rate", 0)
        r9_rate = results.get("seeded_r9", {}).get("rate", 0)

        gate_pass = bp_r7 >= 0.76 and ht_r7 >= 0.32 and hp_r7 >= 0.32 and r9_rate >= 1.0
        cand["static_results"] = results
        cand["static_pass"] = gate_pass

        if gate_pass:
            print(f"  >> {tag}: PASS static gate")
            passing.append(cand)
        else:
            print(f"  >> {tag}: FAIL static gate (bp={bp_r7:.0%} ht={ht_r7:.0%} hp={hp_r7:.0%} r9={r9_rate:.0%})")

    # 50-game validation for passing candidates
    for cand in passing:
        tag = cand["tag"]
        print(f"\n--- 50-game validation: {tag} ---")
        selector = build_modular_selector(
            bp_specialist_path=BP_SPECIALIST,
            base_model_path=BASE_MODEL,
            opening_model=cand["opening_model"],
            late_model=cand["late_model"],
            opening_horizon=cand["opening_horizon"],
        )
        results_50 = eval_suite(selector, games=50, tag=f"{tag}_50g", r9_model=cand["late_model"])
        cand["static_results_50"] = results_50

    return passing


# ── Lane C: PPO Stress Test ───────────────────────────────────────────

def stress_test_candidate(cand: dict, ppo_steps: int, seed: int = 42) -> dict:
    """PPO-train the late model and evaluate the full modular selector."""
    tag = cand["tag"]
    horizon = cand["opening_horizon"]
    opening_model = cand["opening_model"]

    print(f"\n  PPO stress {ppo_steps} steps for {tag}...")

    # Build training env: auto-play opening, late model trains on T_{horizon}+
    from environment.opponents.league import WeightedOpponentLeague, LeagueEntry
    entries = [
        LeagueEntry(label="ht", weight=1.0, opponent=create_scripted_opponent("hal_death_trade")),
        LeagueEntry(label="hp", weight=1.0, opponent=create_scripted_opponent("hal_pressure")),
        LeagueEntry(label="lhal", weight=0.5, opponent=create_model_opponent(LEARNED_HAL, agent_role="hal")),
    ]
    opponent = WeightedOpponentLeague(entries, seed=seed)
    train_env = OpeningAutoPlayEnv(
        opening_model=opening_model,
        opening_horizon=horizon,
        opponent=opponent,
        agent_role="baku",
        seed=seed,
    )

    # Train the late model
    late_model = MaskablePPO.load(BASE_MODEL, env=train_env)
    late_model.learn(total_timesteps=ppo_steps)

    # Build full selector with the PPO-perturbed late model
    selector = build_modular_selector(
        bp_specialist_path=BP_SPECIALIST,
        base_model_path=BASE_MODEL,
        opening_model=opening_model,
        late_model=late_model,
        opening_horizon=horizon,
    )

    results = eval_suite(selector, games=24, tag=f"{tag}_ppo{ppo_steps}", r9_model=late_model)
    return {
        "ppo_steps": ppo_steps,
        "results": results,
        "late_model": late_model,
    }


def lane_c(candidates: list[dict]) -> list[dict]:
    """PPO stress test for all passing candidates."""
    print("\n" + "=" * 60)
    print("LANE C: PPO Stress Test")
    print("=" * 60)

    stress_passing = []
    for cand in candidates:
        tag = cand["tag"]
        cand["stress_results"] = {}

        for steps in [2048, 4096, 8192]:
            stress = stress_test_candidate(cand, ppo_steps=steps)
            cand["stress_results"][steps] = stress

            res = stress["results"]
            bp_r7 = res.get("bridge_pressure", {}).get("r7_rate", 0)
            ht_r7 = res.get("hal_death_trade", {}).get("r7_rate", 0)
            hp_r7 = res.get("hal_pressure", {}).get("r7_rate", 0)
            r9_rate = res.get("seeded_r9", {}).get("rate", 0)

            gate = bp_r7 >= 0.76 and ht_r7 >= 0.32 and hp_r7 >= 0.32 and r9_rate >= 1.0
            stress["gate_pass"] = gate

            pct = lambda v: f"{v*100:.0f}%"
            status = "PASS" if gate else "FAIL"
            print(f"  >> {tag} @ {steps}: {status} (bp={pct(bp_r7)} ht={pct(ht_r7)} hp={pct(hp_r7)} r9={pct(r9_rate)})")

        # Check if at least 2048 gate passed
        if cand["stress_results"][2048].get("gate_pass"):
            stress_passing.append(cand)

    return stress_passing


# ── Lane D: Learned-Hal Robustness ────────────────────────────────────

def lane_d(candidates: list[dict]) -> None:
    """Evaluate passing candidates against the learned Hal opponent."""
    print("\n" + "=" * 60)
    print("LANE D: Learned-Hal Robustness")
    print("=" * 60)

    for cand in candidates:
        tag = cand["tag"]
        # Use the stress-tested late model (from strongest passing PPO level)
        stress_8192 = cand.get("stress_results", {}).get(8192)
        stress_4096 = cand.get("stress_results", {}).get(4096)
        if stress_8192 and stress_8192.get("gate_pass"):
            late_model = stress_8192["late_model"]
            ppo_label = "8192"
        elif stress_4096 and stress_4096.get("gate_pass"):
            late_model = stress_4096["late_model"]
            ppo_label = "4096"
        else:
            stress_2048 = cand.get("stress_results", {}).get(2048)
            if stress_2048:
                late_model = stress_2048["late_model"]
                ppo_label = "2048"
            else:
                late_model = cand["late_model"]
                ppo_label = "static"

        selector = build_modular_selector(
            bp_specialist_path=BP_SPECIALIST,
            base_model_path=BASE_MODEL,
            opening_model=cand["opening_model"],
            late_model=late_model,
            opening_horizon=cand["opening_horizon"],
        )

        print(f"\n--- Learned-Hal eval: {tag} (late model from PPO {ppo_label}) ---")
        res = evaluate_selector(
            selector, "learned_hal", games=50, seed=42,
            opponent_model_path=LEARNED_HAL,
        )
        cand["learned_hal_result"] = asdict(res)
        print(f"  >> {tag} vs learned Hal: wins={res.win_rate:.0%} deaths={res.deaths_by_agent} avg_hr={res.avg_half_rounds:.1f}")


# ── Promotion check ───────────────────────────────────────────────────

def check_promotion(candidates: list[dict]) -> dict:
    """Check promotion bars for all candidates."""
    best = None
    best_bar = None

    for cand in candidates:
        tag = cand["tag"]
        static_50 = cand.get("static_results_50", cand.get("static_results", {}))

        bp_s = static_50.get("bridge_pressure", {}).get("r7_rate", 0)
        ht_s = static_50.get("hal_death_trade", {}).get("r7_rate", 0)
        hp_s = static_50.get("hal_pressure", {}).get("r7_rate", 0)
        r9_s = static_50.get("seeded_r9", {}).get("rate", 0)

        # Check stress results — use 4096 for primary, 2048 for secondary
        stress_4096 = cand.get("stress_results", {}).get(4096, {})
        stress_2048 = cand.get("stress_results", {}).get(2048, {})

        s4096 = stress_4096.get("results", {})
        bp_4096 = s4096.get("bridge_pressure", {}).get("r7_rate", 0)
        ht_4096 = s4096.get("hal_death_trade", {}).get("r7_rate", 0)
        hp_4096 = s4096.get("hal_pressure", {}).get("r7_rate", 0)
        r9_4096 = s4096.get("seeded_r9", {}).get("rate", 0)

        s2048 = stress_2048.get("results", {})
        bp_2048 = s2048.get("bridge_pressure", {}).get("r7_rate", 0)
        ht_2048 = s2048.get("hal_death_trade", {}).get("r7_rate", 0)
        hp_2048 = s2048.get("hal_pressure", {}).get("r7_rate", 0)
        r9_2048 = s2048.get("seeded_r9", {}).get("rate", 0)

        learned_hal = cand.get("learned_hal_result", {})
        lh_wr = learned_hal.get("win_rate", 0)

        # Primary bar (stress at 4096 PPO steps = 2 policy updates)
        primary = (
            bp_s >= 0.80 and ht_s >= 0.36 and hp_s >= 0.36 and r9_s >= 1.0
            and bp_4096 >= 0.80 and ht_4096 >= 0.32 and hp_4096 >= 0.32 and r9_4096 >= 1.0
            and (lh_wr >= 0.24 or ht_4096 >= 0.40 or hp_4096 >= 0.40)
        )

        # Secondary bar (stress at 2048 PPO steps = 1 policy update)
        secondary = (
            bp_s >= 0.80 and ht_s >= 0.36 and hp_s >= 0.36 and r9_s >= 1.0
            and bp_2048 >= 0.76 and ht_2048 >= 0.32 and hp_2048 >= 0.32 and r9_2048 >= 1.0
        )

        if primary and (best_bar != "primary"):
            best = cand
            best_bar = "primary"
        elif secondary and best is None:
            best = cand
            best_bar = "secondary"

    return {
        "best_candidate": best["tag"] if best else None,
        "promotion_bar": best_bar,
        "primary_met": best_bar == "primary",
        "secondary_met": best_bar in ("primary", "secondary"),
    }


# ── Report generation ─────────────────────────────────────────────────

def serialize_results(candidates: list[dict], lane0: dict, promotion: dict) -> dict:
    """Build JSON-serializable results dict."""
    serialized_candidates = []
    for c in candidates:
        sc = {
            "tag": c["tag"],
            "family": c["family"],
            "opening_horizon": c["opening_horizon"],
            "bc_accuracy": c.get("bc_accuracy"),
            "opening_trainable": c.get("opening_trainable"),
            "static_pass": c.get("static_pass"),
            "static_results": c.get("static_results"),
            "static_results_50": c.get("static_results_50"),
            "stress_results": {},
            "learned_hal_result": c.get("learned_hal_result"),
        }
        for steps, sr in c.get("stress_results", {}).items():
            sc["stress_results"][steps] = {
                "ppo_steps": sr.get("ppo_steps"),
                "results": sr.get("results"),
                "gate_pass": sr.get("gate_pass"),
            }
        serialized_candidates.append(sc)

    return {
        "sprint": "modular-opening-head",
        "date": "2026-03-27",
        "lane0": lane0,
        "candidates": serialized_candidates,
        "promotion": promotion,
    }


# ── Main ──────────────────────────────────────────────────────────────

def main():
    start = time.time()

    print("=" * 60)
    print("SPRINT: Modular Opening Head")
    print("=" * 60)

    # Lane 0
    lane0_results = lane_0()

    # Lane A
    all_candidates = lane_a()

    # Lane B
    static_passing = lane_b(all_candidates)

    if not static_passing:
        print("\n!! ALL CANDIDATES FAIL STATIC PARITY — SPRINT FAIL !!")
        promotion = {"best_candidate": None, "promotion_bar": None, "primary_met": False, "secondary_met": False}
        results = serialize_results(all_candidates, lane0_results, promotion)
        save_results(results)
        return

    # Lane C
    stress_passing = lane_c(static_passing)

    # Lane D (run on all static-passing candidates, not just stress-passing)
    lane_d(static_passing)

    # Promotion check
    promotion = check_promotion(static_passing)

    # Save results
    results = serialize_results(all_candidates, lane0_results, promotion)
    save_results(results)

    elapsed = time.time() - start
    print(f"\nSprint completed in {elapsed:.0f}s")
    print(f"Promotion: {promotion['promotion_bar'] or 'FAIL'}")
    if promotion["best_candidate"]:
        print(f"Best candidate: {promotion['best_candidate']}")


def save_results(results: dict):
    """Save JSON results."""
    out_dir = Path("docs/reports/json")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "sprint_modular_opening_head_2026-03-27.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
