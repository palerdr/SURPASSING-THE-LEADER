#!/usr/bin/env python3
"""Tier A decision event: integration diagnostics + next long-run recommendation."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stl.solver.evaluator import TablebaseEvaluator, ValueNetEvaluator
from stl.solver.exact import ExactSearchConfig
from stl.solver.selective import selective_solve
from stl.play.agent import DEFAULT_CHECKPOINT, SolverAgent, make_choose_action
from stl.learning.model import extract_features
from stl.engine.game import PHYSICALITY_BAKU, PHYSICALITY_HAL
from stl.engine.game import Game
from stl.engine.game import Player
from stl.engine.game import Referee
from stl.learning.strength import agent_policy, best_response_interval, gate_report, run_ladder
from stl.solver.tier_a import TierAEvaluator, TierALookup, frontier_interval_fn
from stl.learning.train import load_checkpoint, make_predict_fn
from stl.learning.targets import ValueTarget, save_targets, source_breakdown
from stl.commands.compare_ladder import _json_safe


SOURCE_TIER_A_MIDPOINT = "tier_a_midpoint"
SOURCE_TIER_A_SELECTIVE = "tier_a_selective_h1"


def make_game(
    *,
    clock: float = 3661.0,
    half: int = 1,
    hal_cyl: float = 0.0,
    baku_cyl: float = 0.0,
    hal_deaths: int = 0,
    baku_deaths: int = 0,
    hal_ttd: float = 0.0,
    baku_ttd: float = 0.0,
) -> Game:
    hal = Player(name="Hal", physicality=PHYSICALITY_HAL)
    baku = Player(name="Baku", physicality=PHYSICALITY_BAKU)
    game = Game(player1=hal, player2=baku, referee=Referee(), first_dropper=hal)
    game.seed(0)
    game.game_clock = float(clock)
    game.current_half = int(half)
    hal.cylinder = float(hal_cyl)
    baku.cylinder = float(baku_cyl)
    hal.deaths = int(hal_deaths)
    baku.deaths = int(baku_deaths)
    hal.ttd = float(hal_ttd)
    baku.ttd = float(baku_ttd)
    game.referee.cprs_performed = int(hal_deaths + baku_deaths)
    return game


def policy_vectors(game: Game, result) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    drop = np.zeros(61, dtype=np.float32)
    check = np.zeros(61, dtype=np.float32)
    drop_mask = np.zeros(61, dtype=np.float32)
    check_mask = np.zeros(61, dtype=np.float32)
    for second, prob in zip(result.drop_seconds, result.dropper_strategy):
        drop[second - 1] = float(prob)
        drop_mask[second - 1] = 1.0
    for second, prob in zip(result.check_seconds, result.checker_strategy):
        check[second - 1] = float(prob)
        check_mask[second - 1] = 1.0
    return drop, check, drop_mask, check_mask


def build_evaluators(checkpoint: str, lookup: TierALookup, runtime_width: float):
    net = load_checkpoint(checkpoint)
    predict = make_predict_fn(net)
    base = TablebaseEvaluator(fallback=ValueNetEvaluator(predict))
    tier_a = TierAEvaluator(fallback=base, lookup=lookup, max_width=runtime_width)
    return base, tier_a


def diagnostic_states() -> dict[str, Game]:
    return {
        "postleap_fresh_h1": make_game(half=1),
        "postleap_fresh_h2": make_game(half=2),
        "postleap_mid_120": make_game(half=1, hal_cyl=120, baku_cyl=120),
        "postleap_near_230": make_game(half=1, hal_cyl=230, baku_cyl=230),
        "postleap_boundary_239": make_game(half=1, hal_cyl=239, baku_cyl=239),
        "postleap_fatal_240": make_game(half=1, hal_cyl=240, baku_cyl=240),
        "postleap_fatal_280": make_game(half=2, hal_cyl=280, baku_cyl=280),
        "postleap_terminal_299": make_game(half=1, hal_cyl=299, baku_cyl=299),
        "d1_hal_120_near": make_game(half=2, hal_cyl=230, baku_cyl=230, hal_deaths=1, hal_ttd=120),
        "d1_baku_120_near": make_game(half=1, hal_cyl=230, baku_cyl=230, baku_deaths=1, baku_ttd=120),
        "preleap_miss": make_game(clock=3300.0, half=1, hal_cyl=240, baku_cyl=240),
    }


def lookup_diagnostics(lookup: TierALookup, *, label_width: float) -> dict:
    rows = {}
    hits = low_width = 0
    miss_reasons: dict[str, int] = {}
    for name, game in diagnostic_states().items():
        result = lookup.lookup(game)
        if result.interval is None:
            reason = result.miss_reason or "unknown"
            miss_reasons[reason] = miss_reasons.get(reason, 0) + 1
            rows[name] = {"hit": False, "reason": reason}
            continue
        interval = result.interval
        hits += 1
        if interval.width <= label_width:
            low_width += 1
        rows[name] = {
            "hit": True,
            "source": interval.source,
            "lo": interval.lo,
            "hi": interval.hi,
            "width": interval.width,
            "midpoint": interval.midpoint,
        }
    return {
        "states": rows,
        "hits": hits,
        "low_width_hits": low_width,
        "miss_reasons": miss_reasons,
    }


def run_exploitability(args, base_eval, tier_eval, lookup: TierALookup) -> dict:
    scenarios = {
        "postleap_fresh": make_game(half=1),
        "postleap_230": make_game(half=1, hal_cyl=230, baku_cyl=230),
        "postleap_240": make_game(half=1, hal_cyl=240, baku_cyl=240),
        "d1_hal_120_230": make_game(half=2, hal_cyl=230, baku_cyl=230, hal_deaths=1, hal_ttd=120),
    }
    out = {}
    for name, game in scenarios.items():
        baseline_agent = SolverAgent(
            args.checkpoint,
            player_name="Hal",
            iterations=args.agent_iterations,
            seed=args.seed,
            evaluator=base_eval,
        )
        tier_agent = SolverAgent(
            args.checkpoint,
            player_name="Hal",
            iterations=args.agent_iterations,
            seed=args.seed,
            evaluator=tier_eval,
        )
        baseline = best_response_interval(
            game,
            agent_policy(baseline_agent),
            depth=args.br_depth,
            frozen_name="Hal",
            support_mass=args.support_mass,
            max_states=args.max_states,
        )
        tier = best_response_interval(
            game,
            agent_policy(tier_agent),
            depth=args.br_depth,
            frozen_name="Hal",
            support_mass=args.support_mass,
            max_states=args.max_states,
            frontier_fn=frontier_interval_fn(lookup),
        )
        out[name] = {
            "baseline": {
                "lo": baseline.lo,
                "hi": baseline.hi,
                "width": baseline.width,
                "states": baseline.states_solved,
                "frontier_hits": baseline.frontier_hits,
                "tablebase_frontier_hits": baseline.tablebase_frontier_hits,
            },
            "tier_a": {
                "lo": tier.lo,
                "hi": tier.hi,
                "width": tier.width,
                "states": tier.states_solved,
                "frontier_hits": tier.frontier_hits,
                "tablebase_frontier_hits": tier.tablebase_frontier_hits,
            },
            "width_reduction": baseline.width - tier.width,
        }
    return out


def run_ladder_pair(args, base_eval, tier_eval) -> dict:
    opponents = [name.strip() for name in args.opponents.split(",") if name.strip()]
    base_agent = SolverAgent(
        args.checkpoint,
        player_name="Hal",
        iterations=args.agent_iterations,
        seed=args.seed,
        evaluator=base_eval,
    )
    tier_agent = SolverAgent(
        args.checkpoint,
        player_name="Hal",
        iterations=args.agent_iterations,
        seed=args.seed,
        evaluator=tier_eval,
    )
    base_results = run_ladder(make_choose_action(base_agent), opponents, n_games=args.games, seed=args.seed)
    tier_results = run_ladder(make_choose_action(tier_agent), opponents, n_games=args.games, seed=args.seed)
    return {
        "opponents": opponents,
        "baseline": gate_report(base_results),
        "tier_a": gate_report(tier_results),
    }


def generate_target_pilot(args, tier_eval, lookup: TierALookup, out_dir: Path) -> dict:
    candidates = [
        game
        for game in diagnostic_states().values()
        if lookup.lookup(game).interval is not None
    ]
    targets: list[ValueTarget] = []
    exact_config = ExactSearchConfig()
    for game in candidates[: args.target_limit]:
        hit = lookup.lookup(game).interval
        if hit is None or hit.width > args.label_width:
            continue
        result = selective_solve(game, 1, exact_config, evaluator=tier_eval)
        drop, check, drop_mask, check_mask = policy_vectors(game, result)
        source = SOURCE_TIER_A_SELECTIVE if drop.sum() > 0 and check.sum() > 0 else SOURCE_TIER_A_MIDPOINT
        targets.append(
            ValueTarget(
                features=extract_features(game),
                value=float(hit.midpoint),
                source=source,
                horizon=0,
                dropper_dist=drop,
                checker_dist=check,
                dropper_legal_mask=drop_mask,
                checker_legal_mask=check_mask,
                unresolved_probability=float(hit.width),
            )
        )

    path = out_dir / "tier_a_target_pilot.npz"
    save_targets(targets, path)
    policy_targets = sum(1 for t in targets if t.dropper_dist.sum() > 0 and t.checker_dist.sum() > 0)
    return {
        "path": str(path),
        "records": len(targets),
        "policy_targets": policy_targets,
        "source_breakdown": source_breakdown(targets),
    }


def recommend(report: dict, *, label_width: float) -> str:
    lookup = report["lookup"]
    exploit = report.get("exploitability", {})
    target = report.get("target_pilot", {})
    reductions = [row["width_reduction"] for row in exploit.values()]
    frontier_hits = [
        row["tier_a"]["tablebase_frontier_hits"]
        for row in exploit.values()
    ]
    if lookup["low_width_hits"] < 3 or max(frontier_hits, default=0) == 0:
        return "Tier B tablebase generation: Tier A is valid but the decision frontier still needs deeper death epochs."
    if target.get("records", 0) >= 3 and max(reductions, default=0.0) > 0.25 * label_width:
        return "Policy/value generation + training: Tier A gives usable labels and improves search enough to mint a larger corpus."
    return "More strength evaluation: integration works, but run larger SPRT/exploitability samples before committing heavy generation."


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--tier-a-dir", default=str(Path("checkpoints") / "tablebase" / "tier_a"))
    parser.add_argument("--out-dir", default=str(Path("checkpoints") / "tier_a_decision"))
    parser.add_argument("--runtime-width", type=float, default=0.0)
    parser.add_argument("--label-width", type=float, default=0.05)
    parser.add_argument("--agent-iterations", type=int, default=40)
    parser.add_argument("--br-depth", type=int, default=2)
    parser.add_argument("--games", type=int, default=4)
    parser.add_argument("--opponents", default="random,safe,pattern_reader")
    parser.add_argument("--target-limit", type=int, default=12)
    parser.add_argument("--support-mass", type=float, default=1.0)
    parser.add_argument("--max-states", type=int, default=200_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--skip-ladder", action="store_true")
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()

    if args.quick:
        args.agent_iterations = min(args.agent_iterations, 10)
        args.br_depth = min(args.br_depth, 1)
        args.games = min(args.games, 1)
        args.target_limit = min(args.target_limit, 6)
        args.opponents = "safe"

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    lookup = TierALookup(args.tier_a_dir, verify=True)
    base_eval, tier_eval = build_evaluators(args.checkpoint, lookup, args.runtime_width)

    start = time.time()
    report = {
        "config": vars(args),
        "manifest_entries": len(lookup.verify_manifest()),
        "lookup": lookup_diagnostics(lookup, label_width=args.label_width),
    }
    report["exploitability"] = run_exploitability(args, base_eval, tier_eval, lookup)
    if not args.skip_ladder:
        report["ladder"] = run_ladder_pair(args, base_eval, tier_eval)
    report["target_pilot"] = generate_target_pilot(args, tier_eval, lookup, out_dir)
    report["recommendation"] = recommend(report, label_width=args.label_width)
    report["elapsed_seconds"] = round(time.time() - start, 2)

    report_path = out_dir / "report.json"
    with report_path.open("w", encoding="utf-8") as fh:
        json.dump(_json_safe(report), fh, indent=2, allow_nan=False)

    print(f"Tier A decision event report: {report_path}")
    print(f"Lookup hits: {report['lookup']['hits']} low-width: {report['lookup']['low_width_hits']}")
    print(f"Target pilot: {report['target_pilot']}")
    print(f"Recommendation: {report['recommendation']}")
    print(f"Elapsed: {report['elapsed_seconds']}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
