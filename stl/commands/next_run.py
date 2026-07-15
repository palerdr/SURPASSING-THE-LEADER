#!/usr/bin/env python3
"""Summarize solver promotion evidence and emit the next long-run plan.

This is the post-promotion companion to the Tier A frontier event. The frontier
event answered whether Tier A was useful enough to try a longer auxiliary run;
this event consumes the resulting promotion/rejection/runtime reports and
answers what the next expensive job should be.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stl.commands.compare_ladder import _json_safe


DEFAULT_ACCEPTED_REPORT = (
    Path("checkpoints")
    / "checkpoint_promotion_event"
    / "tier_a_50k_lowimpact_accept_report.json"
)
DEFAULT_RUNTIME_REPORT = (
    Path("checkpoints")
    / "tier_a_runtime_compare"
    / "current_default_exact_seeded_seeds012_g20_report.json"
)
DEFAULT_BASE_TARGETS = Path("checkpoints") / "gen_tier_a_aux_50k_w001_targets.npz"
DEFAULT_HOLDOUT = Path("checkpoints") / "ceiling_holdout_clean.npz"


def load_json(path: str | Path) -> dict:
    with Path(path).open(encoding="utf-8") as fh:
        return json.load(fh)


def _width_tag(width: float) -> str:
    text = f"{width:.4f}".rstrip("0").rstrip(".")
    return "w" + text.replace(".", "")


def _decision(report: dict) -> dict:
    return report.get("decision", {})


def accepted_summary(report: dict) -> dict:
    decision = _decision(report)
    metrics = decision.get("metrics", {})
    checkpoints = metrics.get("checkpoints", {})
    return {
        "status": decision.get("status"),
        "checkpoint": report.get("candidate_checkpoint")
        or checkpoints.get("calibration_candidate"),
        "held_out_overall_mse": metrics.get("held_out_overall_mse"),
        "ladder_wins_delta": metrics.get("ladder_wins_delta"),
        "opponent_wins_delta": metrics.get("opponent_wins_delta", {}),
        "exploitability_median_midpoint_delta": metrics.get(
            "exploitability_median_midpoint_delta"
        ),
        "exploitability_verdicts": metrics.get("exploitability_verdicts", {}),
        "policy_drift": metrics.get("policy_drift"),
    }


def rejected_summary(report: dict) -> dict:
    decision = _decision(report)
    metrics = decision.get("metrics", {})
    return {
        "status": decision.get("status"),
        "checkpoint": report.get("candidate_checkpoint"),
        "reasons": list(decision.get("reasons", [])),
        "held_out_overall_mse": metrics.get("held_out_overall_mse"),
        "ladder_wins_delta": metrics.get("ladder_wins_delta"),
        "opponent_wins_delta": metrics.get("opponent_wins_delta", {}),
        "exploitability_verdicts": metrics.get("exploitability_verdicts", {}),
        "policy_drift": metrics.get("policy_drift"),
    }


def runtime_summary(report: dict | None) -> dict | None:
    if report is None:
        return None
    aggregate = report.get("aggregate", {})
    delta = aggregate.get("delta", report.get("delta", {}).get("overall", {}))
    return {
        "checkpoint": report.get("config", {}).get("checkpoint"),
        "tier_a_width": report.get("config", {}).get("tier_a_width"),
        "wins_delta": delta.get("wins_delta"),
        "losses_delta": delta.get("losses_delta"),
        "evaluator_stats": aggregate.get("tier_a_evaluator_stats"),
    }


def _has_live_strength_failure(row: dict) -> bool:
    if row.get("ladder_wins_delta", 0) < 0:
        return True
    verdicts = row.get("exploitability_verdicts", {})
    return int(verdicts.get("candidate_certified_worse", 0)) > 0


def _d1_is_unsafe(rejected: list[dict]) -> bool:
    for row in rejected:
        checkpoint = str(row.get("checkpoint", "")).lower()
        reasons = " ".join(str(item).lower() for item in row.get("reasons", []))
        if "d1hal" in checkpoint or "d1_hal" in checkpoint or "d1-hal" in reasons:
            if _has_live_strength_failure(row):
                return True
    return False


def _d0_append_is_unsafe(rejected: list[dict]) -> bool:
    for row in rejected:
        checkpoint = str(row.get("checkpoint", "")).lower()
        reasons = " ".join(str(item).lower() for item in row.get("reasons", []))
        if ("d0" in checkpoint or "d0" in reasons) and _has_live_strength_failure(row):
            return True
    return False


def _mcts_refresh_is_unsafe(rejected: list[dict]) -> bool:
    for row in rejected:
        checkpoint = str(row.get("checkpoint", "")).lower()
        if "mcts" in checkpoint and _has_live_strength_failure(row):
            return True
    return False


def _pattern_reader_regressed(rejected: list[dict]) -> bool:
    for row in rejected:
        deltas = row.get("opponent_wins_delta", {})
        if int(deltas.get("pattern_reader", 0)) < 0:
            return True
    return False


def _runtime_should_remain_diagnostic(runtime: dict | None) -> bool:
    if runtime is None:
        return True
    delta = runtime.get("wins_delta")
    return delta is None or int(delta) <= 0


def _post_training_commands(*, checkpoint: str, candidate: Path, out_name: str) -> list[str]:
    drift_report = (
        Path("checkpoints")
        / "checkpoint_policy_drift"
        / f"{out_name}_vs_current_iter200_seeds012_report.json"
    )
    ladder_report = (
        Path("checkpoints")
        / "checkpoint_ladder_compare"
        / f"{out_name}_vs_current_seeds012_g20_report.json"
    )
    exploit_report = (
        Path("checkpoints")
        / "checkpoint_exploitability_compare"
        / f"{out_name}_vs_current_seeds012_report.json"
    )
    promotion_report = (
        Path("checkpoints")
        / "checkpoint_promotion_event"
        / f"{out_name}_promotion_report.json"
    )
    return [
        (
            f"python scripts/compare_checkpoints_policy_drift.py --champion-checkpoint {checkpoint} "
            f"--candidate-checkpoint {candidate} --iterations 200 --seeds 0,1,2 --out {drift_report}"
        ),
        (
            f"python scripts/compare_checkpoints_ladder.py --champion-checkpoint {checkpoint} "
            f"--candidate-checkpoint {candidate} --agent-iterations 30 --games 20 "
            f"--seeds 0,1,2 --out {ladder_report}"
        ),
        (
            f"python scripts/compare_checkpoints_exploitability.py --champion-checkpoint {checkpoint} "
            f"--candidate-checkpoint {candidate} --iterations 30 --depth 1 --seeds 0,1,2 "
            f"--out {exploit_report}"
        ),
        (
            f"python scripts/run_checkpoint_promotion_event.py --calibration-report {candidate.parent / 'gate_report.json'} "
            f"--ladder-report {ladder_report} --exploitability-report {exploit_report} "
            f"--policy-drift-report {drift_report} --out {promotion_report}"
        ),
    ]


def build_next_run(args, accepted: dict, rejected: list[dict], runtime: dict | None) -> dict:
    reasons: list[str] = []
    if accepted.get("status") != "accepted":
        return {
            "status": "blocked_until_promoted_checkpoint",
            "rationale": ["no accepted promotion report was supplied"],
            "target_generation_command": None,
            "training_command": None,
            "post_training_commands": [],
            "acceptance_rule": None,
        }

    checkpoint = accepted["checkpoint"]
    prev_mse = accepted["held_out_overall_mse"]
    d1_unsafe = _d1_is_unsafe(rejected)
    d0_append_unsafe = _d0_append_is_unsafe(rejected)
    mcts_refresh_unsafe = _mcts_refresh_is_unsafe(rejected)
    pattern_reader_regressed = _pattern_reader_regressed(rejected)
    runtime_diagnostic = _runtime_should_remain_diagnostic(runtime)
    if mcts_refresh_unsafe and pattern_reader_regressed:
        out_dir = (
            Path("checkpoints")
            / f"gen_current_mcts300_resolve_tiera50k_lowimpact_s{args.next_seed}"
        )
        out_targets = Path("checkpoints") / f"{out_dir.name}_targets.npz"
        reasons = [
            "static Tier A appends and the first MCTS-refresh probe improved calibration but failed live strength",
            "the recurring live failure is pattern_reader, including at the deployment-like 200-iteration budget",
            "the next generation should harden high-stakes search labels with critical-state subgame resolve before another promotion attempt",
        ]
        command = (
            f"python scripts/run_gen_iteration.py --in-checkpoint {checkpoint} "
            f"--init-checkpoint {checkpoint} --out-dir {out_dir} --out-targets {out_targets} "
            f"--anchor-targets checkpoints/ceiling_corpus.npz "
            f"--extra-targets checkpoints/tier_a_targets_50k_w001.npz "
            f"--held-out-targets {args.held_out_targets} --iterations 300 --epochs {args.epochs} "
            f"--learning-rate {args.learning_rate:g} --hidden-dim 192 --split-interior "
            f"--weight-decay 1e-4 --tablebase-weight 15 --interior-weight 100 "
            f"--tier-a-weight {args.tier_a_weight:g} --tier-a-policy-weight 0.0 "
            f"--tier-a-replicate 1 --subgame-resolve-at-critical "
            f"--subgame-resolve-horizon 1 --subgame-resolve-cfr-iters 2000 "
            f"--bootstrap-critical-only --bootstrap-max-states {args.critical_bootstrap_max_states} "
            f"--prev-gen-holdout-mse {prev_mse}"
        )
        candidate = out_dir / "best.pt"
        pattern_report = (
            Path("checkpoints")
            / "checkpoint_ladder_compare"
            / f"{out_dir.name}_vs_current_pattern_iter200_seeds012_g10_report.json"
        )
        return {
            "status": "needs_pattern_reader_hardening_generation",
            "rationale": reasons,
            "target_generation_command": None,
            "training_command": command,
            "post_training_commands": [
                (
                    f"python scripts/compare_checkpoints_ladder.py --champion-checkpoint {checkpoint} "
                    f"--candidate-checkpoint {candidate} --agent-iterations 200 --games 10 "
                    f"--seeds 0,1,2 --opponents pattern_reader --out {pattern_report}"
                ),
                *_post_training_commands(
                    checkpoint=checkpoint,
                    candidate=candidate,
                    out_name=out_dir.name,
                ),
            ],
            "acceptance_rule": (
                "Do not run full promotion unless the 200-iteration pattern_reader diagnostic is non-regressing. "
                "Then require held-out, deterministic ladder, certified exploitability, policy-drift, "
                "and trace-readability gates."
            ),
        }

    if d0_append_unsafe:
        out_dir = (
            Path("checkpoints")
            / f"gen_current_mcts300_tiera50k_lowimpact_s{args.next_seed}"
        )
        out_targets = Path("checkpoints") / f"{out_dir.name}_targets.npz"
        reasons = [
            "current promoted checkpoint passed calibration, deterministic ladder, and certified exploitability gates",
            "both d1-Hal and d0-only saved-corpus Tier A append probes improved calibration but failed live-strength promotion checks",
            "the next attempt should refresh MCTS bootstrap labels from the current policy instead of adding more static Tier A rows",
        ]
        if runtime_diagnostic:
            reasons.append(
                "Tier A runtime leaf replacement remains diagnostic, so the generation uses Tier A only as low-weight auxiliary labels"
            )
        train_command = (
            f"python scripts/run_gen_iteration.py --in-checkpoint {checkpoint} "
            f"--init-checkpoint {checkpoint} --out-dir {out_dir} --out-targets {out_targets} "
            f"--anchor-targets checkpoints/ceiling_corpus.npz "
            f"--extra-targets checkpoints/tier_a_targets_50k_w001.npz "
            f"--held-out-targets {args.held_out_targets} --iterations 300 --epochs {args.epochs} "
            f"--learning-rate {args.learning_rate:g} "
            f"--hidden-dim 192 --split-interior --weight-decay 1e-4 "
            f"--tablebase-weight 15 --interior-weight 100 "
            f"--tier-a-weight {args.tier_a_weight:g} --tier-a-policy-weight 0.0 "
            f"--tier-a-replicate 1 --prev-gen-holdout-mse {prev_mse}"
        )
        return {
            "status": "ready_for_mcts_bootstrap_refresh",
            "rationale": reasons,
            "target_generation_command": None,
            "training_command": train_command,
            "post_training_commands": _post_training_commands(
                checkpoint=checkpoint,
                candidate=out_dir / "best.pt",
                out_name=out_dir.name,
            ),
            "acceptance_rule": (
                "Promote only if the fresh MCTS-bootstrap generation beats the current held-out MSE, "
                "the deterministic checkpoint ladder is non-regressing overall and by opponent, "
                "and certified exploitability has zero candidate-certified-worse cases."
            ),
        }

    death_filter = "d0" if d1_unsafe else "all"
    include_d1_flag = "--no-include-d1" if d1_unsafe else "--include-d1"
    width_tag = _width_tag(args.next_target_max_width)
    target_path = (
        Path("checkpoints")
        / f"tier_a_targets_{death_filter}_{args.next_target_count // 1000}k_{width_tag}_s{args.next_seed}.npz"
    )
    out_dir = (
        Path("checkpoints")
        / (
            f"gen_tier_a_aux_plus_{death_filter}_{args.next_target_count // 1000}k_"
            f"{width_tag}_ft_lr{args.learning_rate:g}_tw{args.tier_a_weight:g}_pw0_s{args.next_seed}"
        )
    )

    reasons.append(
        "current promoted checkpoint passed calibration, deterministic ladder, and certified exploitability gates"
    )
    if d1_unsafe:
        reasons.append(
            "d1-Hal follow-ups improved calibration but failed live-strength promotion checks, so the next scale-up excludes d1 states"
        )
    if runtime_diagnostic:
        reasons.append(
            "Tier A runtime leaf replacement is neutral or unproven, so the next run uses Tier A as training data/frontier evidence only"
        )

    target_command = (
        f"python scripts/run_tier_a_targets.py --out {target_path} "
        f"--source tier_a --limit {args.next_target_count} "
        f"--max-width {args.next_target_max_width} --runtime-width 0.0 "
        f"--policy-horizon 1 --death-filter {death_filter} {include_d1_flag} "
        f"--verify-manifest --seed {args.next_seed}"
    )
    train_command = (
        f"python scripts/train_saved_corpus_gate.py --targets {args.base_targets} "
        f"--extra-targets {target_path} --out-dir {out_dir} "
        f"--held-out-targets {args.held_out_targets} --epochs {args.epochs} "
        f"--learning-rate {args.learning_rate:g} --weight-decay 1e-4 "
        f"--hidden-dim 192 --init-checkpoint {checkpoint} "
        f"--terminal-weight 30 --horizon-weight 10 --tablebase-weight 15 "
        f"--interior-weight 100 --tier-a-weight {args.tier_a_weight:g} "
        f"--tier-a-policy-weight 0.0 --early-stopping-patience 10 "
        f"--prev-gen-holdout-mse {prev_mse}"
    )
    candidate = out_dir / "best.pt"
    post_commands = _post_training_commands(
        checkpoint=checkpoint,
        candidate=candidate,
        out_name=out_dir.name,
    )
    return {
        "status": "ready_for_d0_tier_a_scaleup" if d1_unsafe else "ready_for_tier_a_scaleup",
        "rationale": reasons,
        "target_generation_command": target_command,
        "training_command": train_command,
        "post_training_commands": post_commands,
        "acceptance_rule": (
            "Promote only if the saved-corpus gate beats the current held-out MSE, "
            "the deterministic checkpoint ladder is non-regressing overall and by opponent, "
            "and certified exploitability has zero candidate-certified-worse cases."
        ),
    }


def build_report(args, accepted_report: dict, rejected_reports: list[dict], runtime_report: dict | None) -> dict:
    accepted = accepted_summary(accepted_report)
    rejected = [rejected_summary(report) for report in rejected_reports]
    runtime = runtime_summary(runtime_report)
    return {
        "config": vars(args),
        "current": accepted,
        "rejected_candidates": rejected,
        "runtime": runtime,
        "next_run": build_next_run(args, accepted, rejected, runtime),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--accepted-promotion-report", default=str(DEFAULT_ACCEPTED_REPORT))
    parser.add_argument(
        "--rejected-promotion-report",
        action="append",
        default=[],
        help="Rejected promotion report to mine for unsafe distributions. Repeatable.",
    )
    parser.add_argument("--runtime-report", default=str(DEFAULT_RUNTIME_REPORT))
    parser.add_argument("--base-targets", default=str(DEFAULT_BASE_TARGETS))
    parser.add_argument("--held-out-targets", default=str(DEFAULT_HOLDOUT))
    parser.add_argument("--next-target-count", type=int, default=150_000)
    parser.add_argument("--next-target-max-width", type=float, default=0.01)
    parser.add_argument("--next-seed", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=3e-6)
    parser.add_argument("--tier-a-weight", type=float, default=5e-4)
    parser.add_argument("--critical-bootstrap-max-states", type=int, default=24)
    parser.add_argument(
        "--out",
        default=str(Path("checkpoints") / "solver_next_event" / "report.json"),
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    accepted = load_json(args.accepted_promotion_report)
    rejected = [load_json(path) for path in args.rejected_promotion_report]
    runtime = load_json(args.runtime_report) if args.runtime_report else None
    report = build_report(args, accepted, rejected, runtime)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fh:
        json.dump(_json_safe(report), fh, indent=2, allow_nan=False)

    next_run = report["next_run"]
    print(f"Solver next event report: {out_path}")
    print(f"Current checkpoint: {report['current']['checkpoint']}")
    print(f"Next status: {next_run['status']}")
    for reason in next_run["rationale"]:
        print(f"  - {reason}")
    print("target generation:", next_run["target_generation_command"])
    print("training:", next_run["training_command"])
    print("post-training:")
    for command in next_run["post_training_commands"]:
        print(f"  {command}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
