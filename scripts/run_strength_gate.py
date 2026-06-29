#!/usr/bin/env python3
"""SPRT match gate: run the agent through the frozen opponent ladder.

Plan tickets 9+10. Plays the SolverAgent (Hal seat) against each named
scripted opponent via ``training.strength.run_ladder``, prints per-opponent
W/D/L + Wilson CI + simplified-GSPRT verdict (H1: elo>=ELO1 vs H0:
elo<=ELO0 against the 50% null), and exits non-zero if any gate check
fails.

Examples:
    python scripts/run_strength_gate.py --agent-iterations 50 --games 20 \
        --seed 0 --opponents random,safe,baku_lsr_engineering,pattern_reader
    python scripts/run_strength_gate.py --skip-agent --games 20   # cheap smoke
    python scripts/run_strength_gate.py --quick                   # capped run

Runtime budget: SolverAgent is ~0.3-0.7 s/move at 50 iterations and games
run ~10-30 half-rounds, so 20 games/opponent is minutes-per-rung. The
per-state search cache makes repeat states cheap across games.

Note ``play_match`` already bounds every game at 200 half-rounds (longer
games are scored as draws, cause "unfinished"), so the ladder cannot hang
on a single runaway game.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training.strength import gate_report, reset_per_game, run_ladder

# ── Gate thresholds ──────────────────────────────────────────────────
# PROVISIONAL until baselines accumulate: these are first-measurement
# floors, not calibrated targets. Tighten once a few gate runs exist.
MIN_WINRATE_VS_RANDOM = 0.70   # provisional — any competent policy should clear
MIN_WINRATE_VS_SAFE = 0.55     # provisional — must beat the 60/1 safe baseline
# Exploitation check: a deterministic agent collapses against the
# pattern reader while staying fine vs "safe"; bound the gap.
MAX_PATTERN_READER_GAP = 0.15  # provisional — |wr(safe) - wr(pattern_reader)| cap

# SPRT hypotheses: H0 "no stronger than even" vs H1 "at least +50 Elo".
ELO0 = 0.0
ELO1 = 50.0

# --quick caps (smoke-test scale).
QUICK_MAX_GAMES = 8
QUICK_MAX_ITERATIONS = 30


def build_hal_choose_action(args):
    """Construct the Hal-seat callable: SolverAgent or CanonicalHal."""
    if args.skip_agent:
        from hal.hal_opponent import CanonicalHal

        # CanonicalHal carries belief/memory state across calls, so wrap
        # it for per-game resets just like the ladder's Baku opponents.
        return reset_per_game(CanonicalHal(seed=args.seed)), "CanonicalHal"

    from hal.agent import DEFAULT_CHECKPOINT, SolverAgent, make_choose_action

    checkpoint = args.checkpoint or DEFAULT_CHECKPOINT
    agent = SolverAgent(
        checkpoint,
        player_name="Hal",
        iterations=args.agent_iterations,
        seed=args.seed,
        use_tier_a=args.use_tier_a,
        tier_a_width=args.tier_a_width,
    )
    label = (
        f"SolverAgent(iterations={args.agent_iterations}, seed={args.seed}, "
        f"checkpoint={checkpoint}, tier_a={args.use_tier_a})"
    )
    # No per-game reset wrapper: the agent's policy is a pure function of
    # the public state (state-seeded, cached search), so resets only cost
    # time. The action-sampling RNG intentionally persists.
    return make_choose_action(agent), label


def run_gate(args) -> int:
    opponents = [name.strip() for name in args.opponents.split(",") if name.strip()]
    games = args.games
    if args.quick:
        games = min(games, QUICK_MAX_GAMES)
        args.agent_iterations = min(args.agent_iterations, QUICK_MAX_ITERATIONS)

    hal_choose_action, hal_label = build_hal_choose_action(args)

    print(f"Hal seat : {hal_label}")
    print(f"Ladder   : {', '.join(opponents)}")
    print(f"Games    : {games}/opponent   seed={args.seed}")
    print(f"SPRT     : H0 elo<={ELO0}  H1 elo>={ELO1}  alpha=beta=0.05")
    print()

    t0 = time.time()
    results = run_ladder(
        hal_choose_action,
        opponents,
        n_games=games,
        seed=args.seed,
    )
    elapsed = time.time() - t0

    report = gate_report(results, elo0=ELO0, elo1=ELO1)

    header = (
        f"{'opponent':<24}{'W':>4}{'D':>4}{'L':>4}{'win%':>8}"
        f"{'wilson 95% CI':>20}{'LLR':>9}{'SPRT':>10}{'len':>7}"
    )
    print(header)
    print("-" * len(header))
    for name in opponents:
        stats = report["opponents"][name]
        ci = f"[{stats['wilson_lo']:.3f}, {stats['wilson_hi']:.3f}]"
        llr = stats["llr"]
        llr_str = f"{llr:+.2f}" if abs(llr) != float("inf") else ("+inf" if llr > 0 else "-inf")
        print(
            f"{name:<24}{stats['wins']:>4}{stats['draws']:>4}{stats['losses']:>4}"
            f"{stats['win_rate']:>8.3f}{ci:>20}{llr_str:>9}{stats['sprt']:>10}"
            f"{stats['avg_game_length_half_rounds']:>7.1f}"
        )
    overall = report["overall"]
    print("-" * len(header))
    print(
        f"{'OVERALL':<24}{overall['wins']:>4}{overall['draws']:>4}"
        f"{overall['losses']:>4}{overall['win_rate']:>8.3f}"
        f"{'':>20}{'':>9}{'':>10}{'':>7}"
    )
    print()
    for name in opponents:
        causes = report["opponents"][name]["cause_of_termination"]
        print(f"  {name}: causes {causes}")
    print(f"\nElapsed: {elapsed:.1f}s")

    # ── Gate checks (only for rungs actually played) ─────────────────
    failures: list[str] = []

    def win_rate(name: str) -> float | None:
        stats = report["opponents"].get(name)
        return stats["win_rate"] if stats else None

    wr_random = win_rate("random")
    if wr_random is not None and wr_random < MIN_WINRATE_VS_RANDOM:
        failures.append(
            f"win-rate vs random {wr_random:.3f} < {MIN_WINRATE_VS_RANDOM}"
        )

    wr_safe = win_rate("safe")
    if wr_safe is not None and wr_safe < MIN_WINRATE_VS_SAFE:
        failures.append(f"win-rate vs safe {wr_safe:.3f} < {MIN_WINRATE_VS_SAFE}")

    wr_reader = win_rate("pattern_reader")
    if wr_safe is not None and wr_reader is not None:
        gap = wr_safe - wr_reader
        if gap > MAX_PATTERN_READER_GAP:
            failures.append(
                f"pattern_reader exploitation: win-rate drop vs safe "
                f"{gap:.3f} > {MAX_PATTERN_READER_GAP} "
                f"(safe={wr_safe:.3f}, pattern_reader={wr_reader:.3f}) — "
                "policy looks too deterministic"
            )

    print()
    if failures:
        print("GATE: FAIL")
        for failure in failures:
            print(f"  - {failure}")
        return 1
    print("GATE: PASS (all thresholds provisional)")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument(
        "--agent-iterations", type=int, default=50,
        help="MCTS iterations per SolverAgent move (default 50)",
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="ValueNet checkpoint for SolverAgent; default is hal.agent.DEFAULT_CHECKPOINT",
    )
    parser.add_argument(
        "--use-tier-a",
        action="store_true",
        help="Wrap the SolverAgent leaf evaluator with Tier A tablebase lookup.",
    )
    parser.add_argument(
        "--tier-a-width",
        type=float,
        default=0.0,
        help="Maximum Tier A interval width used directly by the runtime evaluator.",
    )
    parser.add_argument(
        "--games", type=int, default=20,
        help="games per ladder opponent (default 20)",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--opponents",
        default="random,safe,baku_lsr_engineering,pattern_reader",
        help="comma-separated factory opponent names",
    )
    parser.add_argument(
        "--quick", action="store_true",
        help=f"cap games at {QUICK_MAX_GAMES} and iterations at "
             f"{QUICK_MAX_ITERATIONS} for a fast pass",
    )
    parser.add_argument(
        "--skip-agent", action="store_true",
        help="run CanonicalHal instead of the SolverAgent (cheap smoke)",
    )
    args = parser.parse_args()
    return run_gate(args)


if __name__ == "__main__":
    raise SystemExit(main())
