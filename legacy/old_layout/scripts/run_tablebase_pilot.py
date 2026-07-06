#!/usr/bin/env python3
"""Tablebase pilot (plan ticket 14): retire the cost-model risk, go/no-go.

Stages:
  1. SWEEP    — engine-equivalence of the analytic transition map over
                randomized states (the load-bearing verification).
  2. TIER0    — exact solve of the all-deaths-fatal LIMIT GAME over the
                post-leap quotient: V[dropper, cylH, cylB], 2x300x300 =
                180,000 states, one LP each, single core. This is the
                deep-tail anchor AND the per-state cost measurement.
  3. AUDIT    — Tier-0 values vs solve_exact_finite_horizon on the band
                where the limit game IS the real game (both cylinders
                >= 295: every death is overflow-fatal), gap <= 1e-9.
  4. EPOCH    — timing probe of a real-chance epoch sweep (interval form:
                out-of-epoch survive branches bracketed [-1, +1], two LPs
                per state) on a subsample; extrapolates Tier A/B cost.
  5. VERDICT  — GO if Tier-0 per-state cost <= 10 ms (plan section 7.6).

Artifacts: checkpoints/tablebase_pilot/ (limit-game V table + sha256 +
report JSON). Resumable: stages skip when their artifact exists.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from environment.cfr.backward import (
    MAX_ST,
    assemble_payoff_matrix,
    verify_stage_outcomes_against_engine,
)
from environment.cfr.exact import ExactSearchConfig, solve_exact_finite_horizon, solve_minimax
from src.Constants import CYLINDER_MAX, PHYSICALITY_BAKU, PHYSICALITY_HAL
from src.Game import Game
from src.Player import Player
from src.Referee import Referee

OUT_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "checkpoints",
    "tablebase_pilot",
)
CYL = int(CYLINDER_MAX)  # 300

# Quotient convention: V[bit, cyl_hal, cyl_baku], bit 0 = Hal drops (Baku
# checks), bit 1 = Baku drops (Hal checks). Hal-perspective values. The
# dropper alternates every half-round (engine roles), and post-leap the
# clock is dynamics-irrelevant, so (bit, cylinders) is the whole state.


def make_game(*, clock, half, hal_cyl=0.0, baku_cyl=0.0, hal_ttd=0.0, baku_ttd=0.0,
              hal_deaths=0, baku_deaths=0):
    hal = Player(name="Hal", physicality=PHYSICALITY_HAL)
    baku = Player(name="Baku", physicality=PHYSICALITY_BAKU)
    game = Game(player1=hal, player2=baku, referee=Referee(), first_dropper=hal)
    game.seed(0)
    game.game_clock = float(clock)
    game.current_half = half
    hal.cylinder, baku.cylinder = float(hal_cyl), float(baku_cyl)
    hal.ttd, baku.ttd = float(hal_ttd), float(baku_ttd)
    hal.deaths, baku.deaths = hal_deaths, baku_deaths
    game.referee.cprs_performed = hal_deaths + baku_deaths
    return game


# ── Stage 1: equivalence sweep ────────────────────────────────────────────


def stage_sweep(n_states: int, seed: int) -> dict:
    from tests.test_backward_map import random_reachable_game  # reuse the generator

    rng = np.random.default_rng(seed)
    start = time.perf_counter()
    for i in range(n_states):
        verify_stage_outcomes_against_engine(random_reachable_game(rng))
    elapsed = time.perf_counter() - start
    print(f"[sweep] {n_states} random states engine-equivalent (exact) in {elapsed:.1f}s")
    return {"states": n_states, "elapsed_s": round(elapsed, 2), "result": "exact-equal"}


# ── Stage 2: Tier-0 limit game ────────────────────────────────────────────


def solve_limit_game(progress_every: int = 60) -> tuple[np.ndarray, dict]:
    """Backward induction over the post-leap quotient with all deaths fatal."""
    V = np.zeros((2, CYL, CYL), dtype=np.float64)
    lp_time = 0.0
    states = 0
    start = time.perf_counter()

    for cyl_sum in range(2 * (CYL - 1), -1, -1):
        ch_lo = max(0, cyl_sum - (CYL - 1))
        ch_hi = min(CYL - 1, cyl_sum)
        for ch in range(ch_lo, ch_hi + 1):
            cb = cyl_sum - ch
            for bit in (0, 1):
                if bit == 0:
                    # Baku checks: his cylinder grows; Baku death => +1.
                    terminal = 1.0
                    room = CYL - 1 - cb  # largest ST keeping cb+st <= 299
                    succ = np.full(MAX_ST, terminal)
                    if room > 0:
                        take = min(MAX_ST, room)
                        succ[:take] = V[1, ch, cb + 1 : cb + 1 + take]
                else:
                    # Hal checks: Hal death => -1.
                    terminal = -1.0
                    room = CYL - 1 - ch
                    succ = np.full(MAX_ST, terminal)
                    if room > 0:
                        take = min(MAX_ST, room)
                        succ[:take] = V[0, ch + 1 : ch + 1 + take, cb]

                matrix = assemble_payoff_matrix(60, 60, succ, terminal)
                t0 = time.perf_counter()
                if bit == 0:
                    _, value = solve_minimax(matrix)          # Hal (row) maximizes
                else:
                    _, neg_value = solve_minimax(-matrix)     # Baku (row) minimizes Hal value
                    value = -neg_value
                lp_time += time.perf_counter() - t0
                V[bit, ch, cb] = value
                states += 1
        if cyl_sum % progress_every == 0:
            done = time.perf_counter() - start
            print(f"[tier0] cyl_sum={cyl_sum:3d} states={states:6d} "
                  f"elapsed={done:6.1f}s ({1000*done/max(states,1):.2f} ms/state)")

    elapsed = time.perf_counter() - start
    stats = {
        "states": states,
        "elapsed_s": round(elapsed, 1),
        "ms_per_state": round(1000 * elapsed / states, 3),
        "lp_share": round(lp_time / elapsed, 3),
    }
    return V, stats


# ── Stage 3: audit vs the exact forward solver ────────────────────────────


def stage_audit(V: np.ndarray, n_samples: int, seed: int) -> dict:
    """Compare Tier-0 values to solve_exact_finite_horizon where the limit
    game equals the real game (both cylinders >= 295 => all deaths fatal)."""
    rng = np.random.default_rng(seed)
    config = ExactSearchConfig()
    gaps = []
    checked = 0
    skipped_unresolved = 0
    for _ in range(n_samples):
        ch = int(rng.integers(295, CYL))
        cb = int(rng.integers(295, CYL))
        bit = int(rng.integers(0, 2))
        # Post-leap clock; pick half so the right player drops (first_dropper=Hal).
        half = 1 if bit == 0 else 2
        game = make_game(clock=3661.0, half=half, hal_cyl=ch, baku_cyl=cb)
        result = solve_exact_finite_horizon(game, half_round_horizon=3, config=config)
        if result.unresolved_probability > 0.0:
            skipped_unresolved += 1
            continue
        gap = abs(result.value_for_hal - V[bit, ch, cb])
        gaps.append(gap)
        checked += 1

    max_gap = float(max(gaps)) if gaps else None
    print(f"[audit] checked={checked} skipped(unresolved)={skipped_unresolved} "
          f"max_gap={max_gap}")
    return {
        "checked": checked,
        "skipped_unresolved": skipped_unresolved,
        "max_gap": max_gap,
        "pass": bool(max_gap is not None and max_gap <= 1e-9),
    }


# ── Stage 4: real-chance epoch timing probe ───────────────────────────────


def stage_epoch_probe(n_states: int, seed: int) -> dict:
    """Interval sweep timing on epoch (ttdH=60, ttdB=0, cprs=1): two LPs
    per state, fail-survive branches bracketed [-1, +1]. Measures the Tier
    A/B per-state cost including survival-probability lookups."""
    # Engine-derived survival probabilities per (checker, cylinder).
    p_table = np.zeros((2, CYL))
    probe_hal = Player(name="Hal", physicality=PHYSICALITY_HAL)
    probe_hal.ttd = 60.0
    probe_baku = Player(name="Baku", physicality=PHYSICALITY_BAKU)
    referee = Referee()
    referee.cprs_performed = 1
    for cyl in range(CYL):
        dd = min(cyl + 60, CYL)
        p_table[1, cyl] = referee.compute_survival_probability(probe_hal, death_duration=dd)
        p_table[0, cyl] = referee.compute_survival_probability(probe_baku, death_duration=dd)

    rng = np.random.default_rng(seed)
    V_lo = rng.uniform(-1, 1, size=(2, CYL, CYL))  # stand-in tables: timing only
    V_hi = np.minimum(V_lo + rng.uniform(0, 0.2, size=V_lo.shape), 1.0)

    start = time.perf_counter()
    for _ in range(n_states):
        ch = int(rng.integers(0, CYL))
        cb = int(rng.integers(0, CYL))
        bit = int(rng.integers(0, 2))
        terminal = 1.0 if bit == 0 else -1.0
        checker_cyl = cb if bit == 0 else ch
        p = p_table[bit, checker_cyl]
        fail_lo = p * (-1.0) + (1 - p) * terminal
        fail_hi = p * (+1.0) + (1 - p) * terminal
        for table, fail in ((V_lo, fail_lo), (V_hi, fail_hi)):
            room = CYL - 1 - checker_cyl
            succ = np.full(MAX_ST, terminal)
            if room > 0:
                take = min(MAX_ST, room)
                if bit == 0:
                    succ[:take] = table[1, ch, cb + 1 : cb + 1 + take]
                else:
                    succ[:take] = table[0, ch + 1 : ch + 1 + take, cb]
            matrix = assemble_payoff_matrix(60, 60, succ, fail)
            if bit == 0:
                solve_minimax(matrix)
            else:
                solve_minimax(-matrix)
    elapsed = time.perf_counter() - start
    ms = 1000 * elapsed / n_states
    print(f"[epoch] {n_states} interval states in {elapsed:.1f}s ({ms:.2f} ms/state, 2 LPs each)")
    return {"states": n_states, "elapsed_s": round(elapsed, 1), "ms_per_state": round(ms, 3)}


# ── Main ──────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sweep-states", type=int, default=10_000)
    parser.add_argument("--epoch-states", type=int, default=5_000)
    parser.add_argument("--audit-samples", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--skip-sweep", action="store_true")
    args = parser.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    report: dict = {"seed": args.seed}

    if not args.skip_sweep:
        report["sweep"] = stage_sweep(args.sweep_states, args.seed)

    v_path = os.path.join(OUT_DIR, "limit_game_v.npy")
    if os.path.exists(v_path):
        V = np.load(v_path)
        print(f"[tier0] reusing {v_path}")
        report["tier0"] = {"reused": True}
    else:
        V, stats = solve_limit_game()
        np.save(v_path, V)
        report["tier0"] = stats

    sha = hashlib.sha256(V.tobytes()).hexdigest()
    report["tier0_sha256"] = sha
    print(f"[tier0] sha256={sha[:16]}…  V[0,0,0]={V[0,0,0]:+.4f} V[1,0,0]={V[1,0,0]:+.4f}")

    report["audit"] = stage_audit(V, args.audit_samples, args.seed)
    report["epoch"] = stage_epoch_probe(args.epoch_states, args.seed)

    tier0_ms = report["tier0"].get("ms_per_state")
    verdict = "GO" if (tier0_ms is not None and tier0_ms <= 10.0 and report["audit"]["pass"]) else (
        "GO (cached tier0)" if report["tier0"].get("reused") and report["audit"]["pass"] else "NO-GO"
    )
    report["verdict"] = verdict

    with open(os.path.join(OUT_DIR, "pilot_report.json"), "w") as fh:
        json.dump(report, fh, indent=1)
    print(f"\nVERDICT: {verdict}")
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
