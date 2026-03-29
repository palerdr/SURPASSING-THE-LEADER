#!/usr/bin/env python3
"""Sprint: Opening Re-Audit under PHYSICALITY_BAKU = 0.88.

Reproduces baseline, runs counterfactual audit T0-T7, evaluates candidate
routes, and writes the required deliverables.
"""

from __future__ import annotations
import sys, os, json, time, itertools
from pathlib import Path
from dataclasses import dataclass, asdict
from copy import deepcopy

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
    OpeningAutoPlayEnv, build_modular_selector, collect_opening_samples,
    verify_opening_accuracy,
)
from training.behavior_clone import TraceSample, behavior_clone_policy
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
HAL_OPPONENTS = ["hal_death_trade", "hal_pressure"]
ALL_OPPONENTS = ["bridge_pressure", "hal_death_trade", "hal_pressure"]
OPENING_HORIZON = 8

BASELINE_ROUTE = [1, 1, 1, 10, 60, 1, 60, 1]
BASELINE_OVERRIDES = {t: BASELINE_ROUTE[t] for t in range(8)}

# Hard promotion bars from task spec
HARD_BAR = {"bp_r7": 0.80, "ht_r7": 0.40, "hp_r7": 0.40, "seeded_r9": 1.0, "misaligned": 0}


# ── Evaluation helpers ─────────────────────────────────────────────────

def pct(v):
    return f"{v*100:.0f}%"


@dataclass
class EvalResult:
    opponent: str; games: int; wins: int; win_rate: float
    avg_half_rounds: float; r7_count: int; r7_eligible: int; r7_rate: float
    deaths_by_agent: int


def eval_route_r7(overrides: dict[int, int], opponents: list[str],
                   n_seeds: int = 50, base_seed: int = 42) -> dict:
    """Evaluate an opening route measuring r7 rate across opponents."""
    model = MaskablePPO.load(BASE_MODEL)
    r7_count = 0; r7_eligible = 0; wins = 0; total = 0

    for opp_name in opponents:
        for si in range(n_seeds):
            seed = base_seed + si
            env = DTHEnv(opponent=create_scripted_opponent(opp_name), agent_role="baku", seed=seed)
            obs, _ = env.reset()
            alive = True
            reached_r7 = current_route_stage_flags(env.game).get("round7_pressure", False)

            for t in range(OPENING_HORIZON):
                mask = env.action_masks()
                action_idx = overrides[t] - 1 if t in overrides else int(model.predict(obs, action_masks=mask, deterministic=True)[0])
                obs, _, term, trunc, _ = env.step(action_idx)
                reached_r7 = reached_r7 or current_route_stage_flags(env.game).get("round7_pressure", False)
                if term or trunc:
                    alive = False; break

            if not alive:
                total += 1; r7_eligible += 1; r7_count += int(reached_r7); continue

            while True:
                mask = env.action_masks()
                action_idx = int(model.predict(obs, action_masks=mask, deterministic=True)[0])
                obs, reward, term, trunc, _ = env.step(action_idx)
                reached_r7 = reached_r7 or current_route_stage_flags(env.game).get("round7_pressure", False)
                if term or trunc: break

            total += 1; r7_eligible += 1; r7_count += int(reached_r7)
            won = bool(term and env.game.winner is env.agent and reward > 0)
            wins += int(won)

    return {"r7_count": r7_count, "r7_eligible": r7_eligible,
            "r7_rate": r7_count / r7_eligible if r7_eligible else 0,
            "wins": wins, "total": total, "win_rate": wins / total if total else 0}


def evaluate_selector(selector, opp_name, games=24, seed=42, opp_model_path=None):
    wins = 0; total_hr = 0; r7_count = 0; r7_eligible = 0; deaths = 0
    for gi in range(games):
        opp = (create_model_opponent(opp_model_path, agent_role="hal")
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
            reached_r7 = reached_r7 or current_route_stage_flags(env.game).get("round7_pressure", False)
            if term or trunc:
                won = bool(term and env.game.winner is env.agent and reward > 0)
                wins += int(won); total_hr += len(env.game.history)
                if eligible: r7_eligible += 1; r7_count += int(reached_r7)
                deaths += env.agent.deaths; break
    n = games
    return EvalResult(opp_name, n, wins, wins/n, total_hr/n,
                      r7_count, r7_eligible, r7_count/r7_eligible if r7_eligible else 0, deaths)


def eval_r9_seeded(selector, seed=42):
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


def eval_suite(selector, games=50, seed=42, tag="", lhal_games=0):
    results = {}
    for opp in ALL_OPPONENTS:
        res = evaluate_selector(selector, opp, games=games, seed=seed)
        results[opp] = asdict(res)
        print(f"  {tag} vs {opp}: r7={pct(res.r7_rate)} wins={pct(res.win_rate)} avg_hr={res.avg_half_rounds:.1f}")
    r9 = eval_r9_seeded(selector, seed=seed)
    results["seeded_r9"] = r9
    print(f"  {tag} seeded r9: {r9['wins']}/{r9['games']} ({pct(r9['rate'])})")
    if lhal_games > 0:
        lh = evaluate_selector(selector, "learned_hal", games=lhal_games, seed=seed, opp_model_path=LEARNED_HAL)
        results["learned_hal"] = asdict(lh)
        print(f"  {tag} vs learned_hal: wins={pct(lh.win_rate)} deaths={lh.deaths_by_agent} avg_hr={lh.avg_half_rounds:.1f}")
    return results


# ── Opening model builder ─────────────────────────────────────────────

def build_opening_model_from_overrides(overrides: dict[int, int]):
    samples = collect_opening_samples(
        BASE_MODEL, HAL_OPPONENTS, overrides, OPENING_HORIZON, seeds=range(200),
    )
    env = DTHEnv(opponent=create_scripted_opponent("hal_death_trade"), agent_role="baku", seed=0)
    opening_model = MaskablePPO("MlpPolicy", env, verbose=0, seed=42)
    behavior_clone_policy(opening_model, samples, epochs=80, batch_size=64, learning_rate=1e-3)
    accuracy = verify_opening_accuracy(opening_model, samples)
    misaligned = accuracy["total"] - accuracy["correct"]
    return opening_model, samples, {"accuracy": accuracy["accuracy"], "misaligned": misaligned,
                                     "correct": accuracy["correct"], "total": accuracy["total"]}


def build_selector_for_route(overrides: dict[int, int]):
    opening_model, samples, bc_stats = build_opening_model_from_overrides(overrides)
    late_model = MaskablePPO.load(BASE_MODEL)
    sel = build_modular_selector(
        bp_specialist_path=BP_SPECIALIST, base_model_path=BASE_MODEL,
        opening_model=opening_model, late_model=late_model,
        opening_horizon=OPENING_HORIZON,
    )
    return sel, opening_model, bc_stats


# ── Lane 0: Baseline Reproduction ────────────────────────────────────

def lane_0_baseline():
    print("\n" + "=" * 60)
    print(f"LANE 0: Baseline Reproduction (PHYSICALITY_BAKU={PHYSICALITY_BAKU})")
    print(f"Route: {BASELINE_ROUTE}")
    print("=" * 60)

    sel, _opening, bc_stats = build_selector_for_route(BASELINE_OVERRIDES)
    print(f"  BC accuracy: {bc_stats['accuracy']:.1%} ({bc_stats['correct']}/{bc_stats['total']}), misaligned={bc_stats['misaligned']}")

    # 50-game eval on all opponents + learned hal
    results = eval_suite(sel, games=50, tag="baseline", lhal_games=50)
    results["bc_stats"] = bc_stats
    return results


# ── Lane A: Counterfactual Audit T0-T7 ──────────────────────────────

def lane_a_counterfactual():
    print("\n" + "=" * 60)
    print("LANE A: Counterfactual Audit (T0-T7)")
    print("=" * 60)

    # Roles: T0=C, T1=D, T2=C, T3=D, T4=C, T5=D, T6=C, T7=D
    test_actions = [1, 2, 5, 10, 15, 20, 30, 40, 50, 55, 60]
    audit = {}

    print(f"\nBaseline route: {BASELINE_ROUTE}")
    baseline_hal = eval_route_r7(BASELINE_OVERRIDES, HAL_OPPONENTS, n_seeds=50)
    print(f"Baseline hal-family r7: {pct(baseline_hal['r7_rate'])} ({baseline_hal['r7_count']}/{baseline_hal['r7_eligible']})")
    baseline_bp = eval_route_r7(BASELINE_OVERRIDES, ["bridge_pressure"], n_seeds=50)
    print(f"Baseline bp r7: {pct(baseline_bp['r7_rate'])} ({baseline_bp['r7_count']}/{baseline_bp['r7_eligible']})")
    audit["baseline_hal"] = baseline_hal
    audit["baseline_bp"] = baseline_bp

    improvements_found = []

    for turn in range(OPENING_HORIZON):
        baseline_action = BASELINE_ROUTE[turn]
        turn_results = {}
        best_action = baseline_action
        best_r7 = baseline_hal["r7_rate"]
        role = "C" if turn % 2 == 0 else "D"

        print(f"\n  T{turn} ({role}, baseline={baseline_action}):")

        for action_second in test_actions:
            if action_second == baseline_action:
                turn_results[action_second] = {"hal": baseline_hal, "note": "baseline"}
                continue

            overrides = dict(BASELINE_OVERRIDES)
            overrides[turn] = action_second
            res = eval_route_r7(overrides, HAL_OPPONENTS, n_seeds=50)
            turn_results[action_second] = {"hal": res}

            marker = ""
            delta = res["r7_rate"] - baseline_hal["r7_rate"]
            if delta > 0:
                marker = f" +{pct(delta)}"
                if res["r7_rate"] > best_r7:
                    best_r7 = res["r7_rate"]
                    best_action = action_second
            elif delta < 0:
                marker = f" {pct(delta)}"

            print(f"    a={action_second:2d}: hal_r7={pct(res['r7_rate'])} ({res['r7_count']}/{res['r7_eligible']}){marker}")

        improvement = best_r7 - baseline_hal["r7_rate"]
        audit[f"T{turn}"] = {
            "role": role, "baseline_action": baseline_action,
            "best_action": best_action, "best_r7": best_r7,
            "improvement": improvement,
            "results": {str(k): v for k, v in turn_results.items()},
        }

        if best_action != baseline_action:
            print(f"    ** T{turn} best: a={best_action} r7={pct(best_r7)} (+{pct(improvement)})")
            improvements_found.append((turn, best_action, improvement))
        else:
            print(f"    ** T{turn}: baseline a={baseline_action} is optimal")

    audit["improvements_found"] = [(t, a, round(imp, 4)) for t, a, imp in improvements_found]
    return audit


# ── Lane B: Candidate Route Construction ────────────────────────────

def lane_b_candidates(audit):
    print("\n" + "=" * 60)
    print("LANE B: Candidate Route Construction")
    print("=" * 60)

    candidates = []
    baseline_r7 = audit["baseline_hal"]["r7_rate"]
    improvements = audit.get("improvements_found", [])

    if not improvements:
        print("  No single-turn improvements found. Baseline is locally optimal.")
        return candidates

    # S1: Apply ALL single-turn improvements simultaneously
    improved_ov = dict(BASELINE_OVERRIDES)
    diffs = []
    for turn, action, imp in improvements:
        improved_ov[turn] = action
        diffs.append(f"T{turn}={action}")
    tag = "S1_all_improvements"
    print(f"\n--- {tag}: {', '.join(diffs)} ---")
    res = eval_route_r7(improved_ov, HAL_OPPONENTS, n_seeds=50)
    bp_res = eval_route_r7(improved_ov, ["bridge_pressure"], n_seeds=50)
    print(f"  hal r7: {pct(res['r7_rate'])} (baseline {pct(baseline_r7)}), bp r7: {pct(bp_res['r7_rate'])}")
    candidates.append({"tag": tag, "overrides": improved_ov, "hal_r7": res, "bp_r7": bp_res, "diffs": diffs})

    # S2: Individual top improvements
    improvements_sorted = sorted(improvements, key=lambda x: -x[2])
    for i, (turn, action, imp) in enumerate(improvements_sorted[:3]):
        tag = f"S2_top{i+1}_T{turn}eq{action}"
        ov = dict(BASELINE_OVERRIDES)
        ov[turn] = action
        print(f"\n--- {tag}: T{turn}={action} (single, +{pct(imp)}) ---")
        res = eval_route_r7(ov, HAL_OPPONENTS, n_seeds=50)
        bp_res = eval_route_r7(ov, ["bridge_pressure"], n_seeds=50)
        print(f"  hal r7: {pct(res['r7_rate'])}, bp r7: {pct(bp_res['r7_rate'])}")
        candidates.append({"tag": tag, "overrides": ov, "hal_r7": res, "bp_r7": bp_res,
                          "diffs": [f"T{turn}={action}"]})

    # S3: Pairwise combinations
    if len(improvements_sorted) >= 2:
        for (t1, a1, _), (t2, a2, _) in itertools.combinations(improvements_sorted[:3], 2):
            tag = f"S3_T{t1}eq{a1}_T{t2}eq{a2}"
            ov = dict(BASELINE_OVERRIDES)
            ov[t1] = a1; ov[t2] = a2
            print(f"\n--- {tag} ---")
            res = eval_route_r7(ov, HAL_OPPONENTS, n_seeds=50)
            bp_res = eval_route_r7(ov, ["bridge_pressure"], n_seeds=50)
            print(f"  hal r7: {pct(res['r7_rate'])}, bp r7: {pct(bp_res['r7_rate'])}")
            candidates.append({"tag": tag, "overrides": ov, "hal_r7": res, "bp_r7": bp_res,
                              "diffs": [f"T{t1}={a1}", f"T{t2}={a2}"]})

    return candidates


# ── Lane C: Full Validation of Candidates ────────────────────────────

def lane_c_validate(candidates):
    print("\n" + "=" * 60)
    print("LANE C: Full Validation (50-game + misalignment check)")
    print("=" * 60)

    validated = []
    for cand in candidates:
        tag = cand["tag"]
        print(f"\n--- {tag} ---")
        sel, opening_model, bc_stats = build_selector_for_route(cand["overrides"])
        cand["bc_stats"] = bc_stats
        print(f"  BC accuracy: {bc_stats['accuracy']:.1%}, misaligned={bc_stats['misaligned']}")

        res = eval_suite(sel, games=50, tag=tag, lhal_games=50)
        cand["full_validation"] = res

        route = [cand["overrides"].get(t, BASELINE_ROUTE[t]) for t in range(OPENING_HORIZON)]
        route_diffs = [(t, BASELINE_ROUTE[t], route[t]) for t in range(8) if BASELINE_ROUTE[t] != route[t]]
        cand["route"] = route
        cand["route_diffs"] = route_diffs

        bp = res.get("bridge_pressure", {}).get("r7_rate", 0)
        ht = res.get("hal_death_trade", {}).get("r7_rate", 0)
        hp = res.get("hal_pressure", {}).get("r7_rate", 0)
        r9 = res.get("seeded_r9", {}).get("rate", 0)
        misaligned = bc_stats["misaligned"]
        has_diff = len(route_diffs) > 0

        meets_bar = (bp >= HARD_BAR["bp_r7"] and ht >= HARD_BAR["ht_r7"]
                     and hp >= HARD_BAR["hp_r7"] and r9 >= HARD_BAR["seeded_r9"]
                     and misaligned == HARD_BAR["misaligned"] and has_diff)
        cand["meets_hard_bar"] = meets_bar

        status = "PASS" if meets_bar else "FAIL"
        print(f"  >> {tag}: {status} (bp={pct(bp)} ht={pct(ht)} hp={pct(hp)} r9={pct(r9)} mis={misaligned} diffs={len(route_diffs)})")

        if meets_bar:
            validated.append(cand)

    return validated


# ── Promotion Check ──────────────────────────────────────────────────

def check_promotion(validated):
    if not validated:
        return {"best_candidate": None, "promotion_bar": None, "met": False}

    best = None
    for cand in validated:
        v = cand.get("full_validation", {})
        ht = v.get("hal_death_trade", {}).get("r7_rate", 0)
        hp = v.get("hal_pressure", {}).get("r7_rate", 0)
        score = ht + hp  # rank by hal-family performance
        if best is None or score > best[1]:
            best = (cand, score)

    return {
        "best_candidate": best[0]["tag"] if best else None,
        "route": best[0].get("route") if best else None,
        "met": True,
    }


# ── Main ──────────────────────────────────────────────────────────────

def main():
    start = time.time()
    print("=" * 60)
    print(f"SPRINT: Opening Re-Audit under PHYSICALITY_BAKU={PHYSICALITY_BAKU}")
    print(f"Date: 2026-03-28")
    print("=" * 60)

    # Lane 0: Baseline
    baseline = lane_0_baseline()

    # Lane A: Counterfactual
    audit = lane_a_counterfactual()

    # Lane B: Candidates
    candidates = lane_b_candidates(audit)

    # Lane C: Validation (only if candidates exist)
    validated = []
    promotion = {"best_candidate": None, "promotion_bar": None, "met": False}
    if candidates:
        validated = lane_c_validate(candidates)
        promotion = check_promotion(validated)
    else:
        print("\n!! NO CANDIDATE ROUTES — BASELINE IS LOCALLY OPTIMAL !!")

    elapsed = time.time() - start

    # ── Build output ──────────────────────────────────────────────────
    bp_r7 = baseline.get("bridge_pressure", {}).get("r7_rate", 0)
    ht_r7 = baseline.get("hal_death_trade", {}).get("r7_rate", 0)
    hp_r7 = baseline.get("hal_pressure", {}).get("r7_rate", 0)
    r9_rate = baseline.get("seeded_r9", {}).get("rate", 0)
    lhal_wr = baseline.get("learned_hal", {}).get("win_rate", 0)
    misaligned = baseline.get("bc_stats", {}).get("misaligned", -1)

    # Compare to old regime (from sprint_opening_route_search_2026-03-27.json)
    old_regime = {"bp_r7": 0.792, "ht_r7": 0.333, "hp_r7": 0.333, "hal_combined_r7": 0.36}

    out = {
        "sprint": "opening-reaudit-p088",
        "date": "2026-03-28",
        "physicality_baku": PHYSICALITY_BAKU,
        "baseline_route": BASELINE_ROUTE,
        "baseline": baseline,
        "old_regime_comparison": old_regime,
        "audit_summary": {
            "improvements_found": audit.get("improvements_found", []),
            "baseline_hal_r7": audit.get("baseline_hal", {}).get("r7_rate", 0),
            "baseline_bp_r7": audit.get("baseline_bp", {}).get("r7_rate", 0),
        },
        "candidates": [{k: v for k, v in c.items() if k not in {"opening_model"}}
                       for c in candidates],
        "validated": [{k: v for k, v in c.items() if k not in {"opening_model"}}
                      for c in validated],
        "promotion": promotion,
        "elapsed_seconds": round(elapsed),
    }

    # Write JSON
    out_dir = Path("docs/reports/json"); out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "sprint_opening_reaudit_p088_2026-03-28.json"
    with open(json_path, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\nJSON written to {json_path}")

    # ── Terminal Summary ─────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("TERMINAL SUMMARY")
    print("=" * 60)
    print(f"  PHYSICALITY_BAKU:  {PHYSICALITY_BAKU}")
    print(f"  Baseline route:    {BASELINE_ROUTE}")
    print(f"  BC misaligned:     {misaligned}")
    print(f"  ────────────────────────────────────────")
    print(f"  Baseline (50-game, p=0.88):")
    print(f"    bp  r7:   {pct(bp_r7)}  (bar ≥80%)")
    print(f"    ht  r7:   {pct(ht_r7)}  (bar ≥40%)")
    print(f"    hp  r7:   {pct(hp_r7)}  (bar ≥40%)")
    print(f"    r9 seed:  {pct(r9_rate)}  (bar =100%)")
    print(f"    lhal wr:  {pct(lhal_wr)}")
    print(f"  ────────────────────────────────────────")
    print(f"  Old regime (p=0.70, 24-game):")
    print(f"    bp  r7:   {pct(old_regime['bp_r7'])}")
    print(f"    ht  r7:   {pct(old_regime['ht_r7'])}")
    print(f"    hp  r7:   {pct(old_regime['hp_r7'])}")
    print(f"    hal comb: {pct(old_regime['hal_combined_r7'])}")
    print(f"  ────────────────────────────────────────")
    ceiling_moved = (ht_r7 > old_regime["ht_r7"] + 0.05 or hp_r7 > old_regime["hp_r7"] + 0.05)
    print(f"  Ceiling moved:     {'YES' if ceiling_moved else 'NO'}")
    print(f"  Improvements:      {len(audit.get('improvements_found', []))}")
    print(f"  Candidates built:  {len(candidates)}")
    print(f"  Candidates pass:   {len(validated)}")
    verdict = "PROMOTED" if promotion.get("met") else "FAIL"
    print(f"  Verdict:           {verdict}")
    if promotion.get("best_candidate"):
        print(f"  Best candidate:    {promotion['best_candidate']}")
        print(f"  Promoted route:    {promotion['route']}")
    print(f"  Sprint time:       {elapsed:.0f}s")
    print("=" * 60)

    return out


if __name__ == "__main__":
    main()
