#!/usr/bin/env python3
"""Sprint: Opening Route Search — full execution.

Searches for better T0-T7 opening routes to break the 36% hal-family ceiling.
"""

from __future__ import annotations
import sys, os, json, time, itertools
from pathlib import Path
from dataclasses import dataclass, asdict
from copy import deepcopy

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from sb3_contrib import MaskablePPO

from environment.dth_env import DTHEnv
from environment.opponents.factory import create_model_opponent, create_scripted_opponent
from environment.opponents.league import WeightedOpponentLeague, LeagueEntry
from environment.route_stages import current_route_stage_flags, stage_is_eligible_from_start
from training.curriculum import get_scenario
from training.modular_policy import (
    OpeningAutoPlayEnv, build_modular_selector, train_opening_model_bc,
    collect_opening_samples,
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

# Full baseline opening route (action_second, 1-indexed).
# ALL turns must be specified because the base model's defaults (check@1 everywhere)
# are wrong for the hal-family route.
BASELINE_ROUTE = [1, 1, 1, 10, 60, 1, 60, 1]
BASELINE_OVERRIDES = {t: BASELINE_ROUTE[t] for t in range(8)}


# ── Evaluation helpers ─────────────────────────────────────────────────

def eval_route_r7(overrides: dict[int, int], opponents: list[str],
                   n_seeds: int = 50, base_seed: int = 42) -> dict:
    """Evaluate an opening route (defined by overrides) measuring r7 rate.

    r7 is tracked with OR across all steps (not just at game end) because
    the route_stage flag is only True at specific (round, half) moments.
    """
    model = MaskablePPO.load(BASE_MODEL)
    r7_count = 0
    r7_eligible = 0
    wins = 0
    total = 0

    for opp_name in opponents:
        for si in range(n_seeds):
            seed = base_seed + si
            env = DTHEnv(opponent=create_scripted_opponent(opp_name), agent_role="baku", seed=seed)
            obs, _ = env.reset()
            alive = True
            reached_r7 = current_route_stage_flags(env.game).get("round7_pressure", False)

            # Play opening with overrides
            for t in range(OPENING_HORIZON):
                mask = env.action_masks()
                if t in overrides:
                    action_idx = overrides[t] - 1
                else:
                    action_idx = int(model.predict(obs, action_masks=mask, deterministic=True)[0])
                obs, _, term, trunc, _ = env.step(action_idx)
                reached_r7 = reached_r7 or current_route_stage_flags(env.game).get("round7_pressure", False)
                if term or trunc:
                    alive = False
                    break

            if not alive:
                total += 1
                r7_eligible += 1
                r7_count += int(reached_r7)
                continue

            # Play remaining turns with base model
            while True:
                mask = env.action_masks()
                action_idx = int(model.predict(obs, action_masks=mask, deterministic=True)[0])
                obs, reward, term, trunc, _ = env.step(action_idx)
                reached_r7 = reached_r7 or current_route_stage_flags(env.game).get("round7_pressure", False)
                if term or trunc:
                    break

            total += 1
            r7_eligible += 1
            r7_count += int(reached_r7)
            won = bool(term and env.game.winner is env.agent and reward > 0)
            wins += int(won)

    return {
        "r7_count": r7_count, "r7_eligible": r7_eligible,
        "r7_rate": r7_count / r7_eligible if r7_eligible else 0,
        "wins": wins, "total": total, "win_rate": wins / total if total else 0,
    }


@dataclass
class EvalResult:
    opponent: str; games: int; wins: int; win_rate: float
    avg_half_rounds: float; r7_count: int; r7_eligible: int; r7_rate: float
    deaths_by_agent: int


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
                deaths += env.agent.deaths
                break
    n = games
    return EvalResult(opp_name, n, wins, wins/n, total_hr/n,
                      r7_count, r7_eligible, r7_count/r7_eligible if r7_eligible else 0, deaths)


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
        lh = evaluate_selector(selector, "learned_hal", games=lhal_games, seed=seed, opp_model_path=LEARNED_HAL)
        results["learned_hal"] = asdict(lh)
        print(f"  {tag} vs learned_hal: wins={pct(lh.win_rate)} deaths={lh.deaths_by_agent} avg_hr={lh.avg_half_rounds:.1f}")
    return results


# ── Opening model builder from overrides ──────────────────────────────

def build_opening_model_from_overrides(overrides: dict[int, int]):
    """BC-train a frozen opening model using the given action overrides."""
    samples = collect_opening_samples(
        BASE_MODEL, HAL_OPPONENTS, overrides, OPENING_HORIZON, seeds=range(200),
    )
    env = DTHEnv(opponent=create_scripted_opponent("hal_death_trade"), agent_role="baku", seed=0)
    opening_model = MaskablePPO("MlpPolicy", env, verbose=0, seed=42)
    behavior_clone_policy(opening_model, samples, epochs=80, batch_size=64, learning_rate=1e-3)
    return opening_model


def build_selector_for_route(overrides: dict[int, int]):
    """Build a full ModularBakuSelector with a custom opening route."""
    opening_model = build_opening_model_from_overrides(overrides)
    late_model = MaskablePPO.load(BASE_MODEL)
    return build_modular_selector(
        bp_specialist_path=BP_SPECIALIST, base_model_path=BASE_MODEL,
        opening_model=opening_model, late_model=late_model,
        opening_horizon=OPENING_HORIZON,
    ), opening_model


# ── Lane 0: Baseline ──────────────────────────────────────────────────

def lane_0():
    print("\n" + "=" * 60)
    print("LANE 0: Baseline Reproduction")
    print("=" * 60)
    sel, _ = build_selector_for_route(BASELINE_OVERRIDES)
    return eval_suite(sel, games=24, tag="baseline", lhal_games=24)


# ── Lane A: Counterfactual Audit ──────────────────────────────────────

def lane_a():
    print("\n" + "=" * 60)
    print("LANE A: Counterfactual Audit (T0-T7)")
    print("=" * 60)

    # Roles: T0=C, T1=D, T2=C, T3=D, T4=C, T5=D, T6=C, T7=D
    test_actions = [1, 5, 10, 20, 30, 40, 50, 60]
    audit = {}

    print(f"\nBaseline route: {BASELINE_ROUTE}")
    baseline_hal = eval_route_r7(BASELINE_OVERRIDES, HAL_OPPONENTS, n_seeds=50)
    print(f"Baseline hal r7: {baseline_hal['r7_rate']:.0%} ({baseline_hal['r7_count']}/{baseline_hal['r7_eligible']})")
    audit["baseline"] = baseline_hal

    # Single-turn deviations
    for turn in range(OPENING_HORIZON):
        baseline_action = BASELINE_ROUTE[turn]
        turn_results = {}
        best_action = baseline_action
        best_r7 = baseline_hal["r7_rate"]

        for action_second in test_actions:
            if action_second == baseline_action:
                turn_results[action_second] = baseline_hal
                continue

            overrides = dict(BASELINE_OVERRIDES)
            overrides[turn] = action_second
            res = eval_route_r7(overrides, HAL_OPPONENTS, n_seeds=50)
            turn_results[action_second] = res

            marker = ""
            if res["r7_rate"] > best_r7:
                best_r7 = res["r7_rate"]
                best_action = action_second
                marker = " <<<< IMPROVEMENT"

            print(f"  T{turn} action={action_second:2d}: hal_r7={res['r7_rate']:.0%} ({res['r7_count']}/{res['r7_eligible']}){marker}")

        audit[f"T{turn}"] = {
            "results": {k: v for k, v in turn_results.items()},
            "baseline_action": baseline_action,
            "best_action": best_action,
            "best_r7": best_r7,
            "improvement": best_r7 - baseline_hal["r7_rate"],
        }

        if best_action != baseline_action:
            print(f"  ** T{turn} best: action={best_action} r7={best_r7:.0%} (+{best_r7 - baseline_hal['r7_rate']:.0%})")

    return audit


# ── Lane B: Opening Search ────────────────────────────────────────────

def lane_b(audit: dict):
    """Build candidate routes from audit findings."""
    print("\n" + "=" * 60)
    print("LANE B: Opening Search Families")
    print("=" * 60)

    candidates = []
    baseline_r7 = audit["baseline"]["r7_rate"]

    # S1: Apply all single-turn improvements simultaneously
    improved_overrides = dict(BASELINE_OVERRIDES)
    diffs = []
    for turn in range(OPENING_HORIZON):
        turn_data = audit.get(f"T{turn}", {})
        best_action = turn_data.get("best_action", BASELINE_ROUTE[turn])
        if best_action != BASELINE_ROUTE[turn]:
            improved_overrides[turn] = best_action
            diffs.append(f"T{turn}={best_action}")

    if diffs:
        tag = "S1_all_improvements"
        print(f"\n--- {tag}: {', '.join(diffs)} ---")
        res = eval_route_r7(improved_overrides, HAL_OPPONENTS, n_seeds=50)
        print(f"  hal r7: {res['r7_rate']:.0%} (baseline {baseline_r7:.0%})")
        candidates.append({"tag": tag, "overrides": improved_overrides, "route_hal_r7": res})

    # S2: Try top-2 individual improvements
    turn_improvements = []
    for turn in range(OPENING_HORIZON):
        td = audit.get(f"T{turn}", {})
        imp = td.get("improvement", 0)
        if imp > 0:
            turn_improvements.append((turn, td["best_action"], imp))
    turn_improvements.sort(key=lambda x: -x[2])

    for i, (turn, action, imp) in enumerate(turn_improvements[:3]):
        tag = f"S2_top{i+1}_T{turn}eq{action}"
        print(f"\n--- {tag}: T{turn}={action} (single turn, +{imp:.0%}) ---")
        ov = dict(BASELINE_OVERRIDES)
        ov[turn] = action
        res = eval_route_r7(ov, HAL_OPPONENTS, n_seeds=50)
        print(f"  hal r7: {res['r7_rate']:.0%}")
        candidates.append({"tag": tag, "overrides": ov, "route_hal_r7": res})

    # S3: Pairwise combinations of top improvements
    if len(turn_improvements) >= 2:
        for (t1, a1, _), (t2, a2, _) in itertools.combinations(turn_improvements[:3], 2):
            tag = f"S3_T{t1}eq{a1}_T{t2}eq{a2}"
            print(f"\n--- {tag} ---")
            ov = dict(BASELINE_OVERRIDES)
            ov[t1] = a1
            ov[t2] = a2
            res = eval_route_r7(ov, HAL_OPPONENTS, n_seeds=50)
            print(f"  hal r7: {res['r7_rate']:.0%}")
            candidates.append({"tag": tag, "overrides": ov, "route_hal_r7": res})

    # S4: Broader search on checker turns with top dropper actions locked
    best_dropper = dict(BASELINE_OVERRIDES)
    for t, a, _ in turn_improvements:
        if BASELINE_ROUTE[t] != a:  # Only keep improvements
            best_dropper[t] = a

    # Try a few more checker values with all dropper improvements locked
    for check_val in [30, 35, 40, 45, 50, 55]:
        for check_turn in [0, 2]:  # T0 and T2 are the risky checker turns
            ov = dict(best_dropper)
            ov[check_turn] = check_val
            tag = f"S4_T{check_turn}eq{check_val}"
            res = eval_route_r7(ov, HAL_OPPONENTS, n_seeds=50)
            if res["r7_rate"] > baseline_r7 + 0.02:
                print(f"  {tag}: hal r7={res['r7_rate']:.0%} <<<")
                candidates.append({"tag": tag, "overrides": ov, "route_hal_r7": res})

    return candidates


# ── Lane C: Static Safety Gate ────────────────────────────────────────

def lane_c(candidates):
    print("\n" + "=" * 60)
    print("LANE C: Static Safety Gate")
    print("=" * 60)

    passing = []
    for cand in candidates:
        tag = cand["tag"]
        print(f"\n--- {tag} ---")
        sel, opening_model = build_selector_for_route(cand["overrides"])
        cand["opening_model"] = opening_model
        res = eval_suite(sel, games=24, tag=tag, lhal_games=24)
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


# ── Lane D: Full Validation ───────────────────────────────────────────

def lane_d(passing):
    print("\n" + "=" * 60)
    print("LANE D: Full Validation (50-game)")
    print("=" * 60)
    for cand in passing:
        tag = cand["tag"]
        print(f"\n--- {tag} ---")
        sel, _ = build_selector_for_route(cand["overrides"])
        res = eval_suite(sel, games=50, tag=f"{tag}_50g", lhal_games=50)
        cand["full_validation"] = res

        # Route difference
        route = [cand["overrides"].get(t, BASELINE_ROUTE[t]) for t in range(OPENING_HORIZON)]
        diffs = [(t, BASELINE_ROUTE[t], route[t]) for t in range(8) if BASELINE_ROUTE[t] != route[t]]
        cand["route"] = route
        cand["route_diffs"] = diffs
        print(f"  Route: {route}")
        print(f"  Diffs from baseline: {diffs}")


# ── Lane E: Stability Check ──────────────────────────────────────────

def lane_e(passing):
    print("\n" + "=" * 60)
    print("LANE E: Late-Head Stability Check")
    print("=" * 60)
    for cand in passing:
        tag = cand["tag"]
        opening_model = cand["opening_model"]
        print(f"\n--- {tag}: PPO stress 2048 ---")
        league = WeightedOpponentLeague([
            LeagueEntry(label="ht", weight=1.0, opponent=create_scripted_opponent("hal_death_trade")),
            LeagueEntry(label="hp", weight=1.0, opponent=create_scripted_opponent("hal_pressure")),
            LeagueEntry(label="lhal", weight=0.5, opponent=create_model_opponent(LEARNED_HAL, agent_role="hal")),
        ], seed=42)
        train_env = OpeningAutoPlayEnv(
            opening_model=opening_model, opening_horizon=OPENING_HORIZON,
            opponent=league, agent_role="baku", seed=42,
            use_shaping=True, shaping_preset="bridge",
        )
        late_model = MaskablePPO.load(BASE_MODEL, env=train_env)
        late_model.learn(total_timesteps=2048)
        sel = build_modular_selector(
            bp_specialist_path=BP_SPECIALIST, base_model_path=BASE_MODEL,
            opening_model=opening_model, late_model=late_model,
            opening_horizon=OPENING_HORIZON,
        )
        res = eval_suite(sel, games=24, tag=f"{tag}_stress2048", lhal_games=24)
        cand["stability"] = res
        bp = res.get("bridge_pressure", {}).get("r7_rate", 0)
        ht = res.get("hal_death_trade", {}).get("r7_rate", 0)
        hp = res.get("hal_pressure", {}).get("r7_rate", 0)
        r9 = res.get("seeded_r9", {}).get("rate", 0)
        gate = bp >= 0.76 and ht >= 0.32 and hp >= 0.32 and r9 >= 1.0
        cand["stability_pass"] = gate
        pct = lambda v: f"{v*100:.0f}%"
        print(f"  >> {tag} stability: {'PASS' if gate else 'FAIL'} (bp={pct(bp)} ht={pct(ht)} hp={pct(hp)} r9={pct(r9)})")


# ── Promotion ─────────────────────────────────────────────────────────

def check_promotion(candidates):
    best = None; best_bar = None
    for cand in candidates:
        v = cand.get("full_validation", {})
        bp = v.get("bridge_pressure", {}).get("r7_rate", 0)
        ht = v.get("hal_death_trade", {}).get("r7_rate", 0)
        hp = v.get("hal_pressure", {}).get("r7_rate", 0)
        r9 = v.get("seeded_r9", {}).get("rate", 0)
        lh = v.get("learned_hal", {}).get("win_rate", 0)
        has_diff = len(cand.get("route_diffs", [])) > 0
        stable = cand.get("stability_pass", False)

        primary = (bp >= 0.80 and ht >= 0.44 and hp >= 0.44 and r9 >= 1.0
                   and lh >= 0.20 and has_diff and stable)
        secondary = (bp >= 0.80 and ht >= 0.40 and hp >= 0.40 and r9 >= 1.0
                     and has_diff and stable)

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
    print("SPRINT: Opening Route Search")
    print("=" * 60)

    lane0 = lane_0()
    audit = lane_a()
    candidates = lane_b(audit)

    if not candidates:
        print("\n!! NO IMPROVED ROUTES FOUND !!")
        promotion = {"best_candidate": None, "promotion_bar": None, "primary_met": False, "secondary_met": False}
    else:
        passing = lane_c(candidates)
        if not passing:
            print("\n!! ALL CANDIDATES FAIL STATIC GATE !!")
            promotion = {"best_candidate": None, "promotion_bar": None, "primary_met": False, "secondary_met": False}
        else:
            lane_d(passing)
            lane_e(passing)
            promotion = check_promotion(passing)

    out = {
        "sprint": "opening-route-search", "date": "2026-03-27",
        "lane0": lane0, "audit": {k: v for k, v in audit.items() if k == "baseline"},
        "candidates": [{k: v for k, v in c.items() if k not in {"opening_model", "late_model"}}
                       for c in candidates],
        "promotion": promotion,
    }
    out_dir = Path("docs/reports/json"); out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "sprint_opening_route_search_2026-03-27.json", "w") as f:
        json.dump(out, f, indent=2, default=str)

    elapsed = time.time() - start
    print(f"\nSprint completed in {elapsed:.0f}s")
    print(f"Promotion: {promotion.get('promotion_bar') or 'FAIL'}")
    if promotion.get("best_candidate"):
        print(f"Best candidate: {promotion['best_candidate']}")


if __name__ == "__main__":
    main()
