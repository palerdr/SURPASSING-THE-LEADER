#!/usr/bin/env python3
"""Decide whether a trained checkpoint is promotable from saved gate reports.

The promotion rule intentionally combines three independent pieces of evidence:

1. Calibration gate passed and, by default, improved held-out MSE.
2. Deterministic checkpoint ladder is non-regressing overall.
3. Certified exploitability comparison is non-regressing.

This keeps a calibration-only fine-tune from silently becoming the default when
it loses live strength, which happened in the rejected d1-Hal append runs.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace


def _norm_path(value) -> str | None:
    if value in (None, ""):
        return None
    return Path(str(value)).resolve(strict=False).as_posix().lower()


def load_json(path: str | Path) -> dict:
    with Path(path).open(encoding="utf-8") as fh:
        return json.load(fh)


def _opponent_deltas(ladder_report: dict) -> dict[str, int]:
    opponents = ladder_report["aggregate"].get("opponents", {})
    return {
        name: int(row["delta"]["wins_delta"])
        for name, row in opponents.items()
    }


def _report_checkpoints(
    calibration_report: dict,
    ladder_report: dict,
    exploitability_report: dict,
    policy_drift_report: dict | None = None,
) -> dict:
    ladder_cfg = ladder_report.get("config", {})
    exploit_cfg = exploitability_report.get("config", {})
    drift_cfg = (policy_drift_report or {}).get("config", {})
    checkpoints = {
        "calibration_candidate": calibration_report.get("checkpoint"),
        "ladder_candidate": ladder_cfg.get("candidate_checkpoint"),
        "exploitability_candidate": exploit_cfg.get("candidate_checkpoint"),
        "ladder_champion": ladder_cfg.get("champion_checkpoint"),
        "exploitability_champion": exploit_cfg.get("champion_checkpoint"),
    }
    if policy_drift_report is not None:
        checkpoints["policy_drift_candidate"] = drift_cfg.get("candidate_checkpoint")
        checkpoints["policy_drift_champion"] = drift_cfg.get("champion_checkpoint")
    return checkpoints


def _checkpoint_consistency_reasons(checkpoints: dict) -> list[str]:
    reasons: list[str] = []
    candidate_keys = (
        "calibration_candidate",
        "ladder_candidate",
        "exploitability_candidate",
    )
    if "policy_drift_candidate" in checkpoints:
        candidate_keys = (*candidate_keys, "policy_drift_candidate")
    candidate_values = {key: _norm_path(checkpoints.get(key)) for key in candidate_keys}
    missing = [key for key, value in candidate_values.items() if value is None]
    if missing:
        reasons.append("missing candidate checkpoint in report(s): " + ", ".join(missing))
    present = {key: value for key, value in candidate_values.items() if value is not None}
    if len(set(present.values())) > 1:
        reasons.append(
            "candidate checkpoint mismatch across reports: "
            + ", ".join(f"{key}={checkpoints.get(key)}" for key in candidate_keys)
        )

    ladder_champion = _norm_path(checkpoints.get("ladder_champion"))
    exploit_champion = _norm_path(checkpoints.get("exploitability_champion"))
    drift_champion = (
        _norm_path(checkpoints.get("policy_drift_champion"))
        if "policy_drift_champion" in checkpoints
        else None
    )
    if ladder_champion is None or exploit_champion is None:
        reasons.append("missing champion checkpoint in ladder or exploitability report")
    elif ladder_champion != exploit_champion:
        reasons.append(
            "champion checkpoint mismatch across reports: "
            f"ladder={checkpoints.get('ladder_champion')} "
            f"exploitability={checkpoints.get('exploitability_champion')}"
        )
    if "policy_drift_champion" in checkpoints:
        if drift_champion is None:
            reasons.append("missing champion checkpoint in policy drift report")
        elif ladder_champion is not None and drift_champion != ladder_champion:
            reasons.append(
                "champion checkpoint mismatch across reports: "
                f"ladder={checkpoints.get('ladder_champion')} "
                f"policy_drift={checkpoints.get('policy_drift_champion')}"
            )
    return reasons


def decide_promotion(
    calibration_report: dict,
    ladder_report: dict,
    exploitability_report: dict,
    args,
    policy_drift_report: dict | None = None,
) -> dict:
    reasons: list[str] = []
    warnings: list[str] = []
    checkpoints = _report_checkpoints(
        calibration_report,
        ladder_report,
        exploitability_report,
        policy_drift_report,
    )
    reasons.extend(_checkpoint_consistency_reasons(checkpoints))

    gate_passed = bool(calibration_report.get("gate_passed"))
    if not gate_passed:
        reasons.append(f"calibration gate failed: {calibration_report.get('gate_error')}")

    if args.require_calibration_improvement and calibration_report.get("beats_prev_gen") is not True:
        reasons.append("held-out calibration did not strictly beat previous generation")

    ladder_delta = int(ladder_report["aggregate"]["delta"]["wins_delta"])
    if ladder_delta < args.min_ladder_wins_delta:
        reasons.append(
            f"ladder wins_delta {ladder_delta} < required {args.min_ladder_wins_delta}"
        )

    opponent_deltas = _opponent_deltas(ladder_report)
    bad_opponents = {
        name: delta
        for name, delta in opponent_deltas.items()
        if delta < args.min_opponent_wins_delta
    }
    if bad_opponents:
        reasons.append(
            "opponent regression(s): "
            + ", ".join(f"{name} {delta:+d}" for name, delta in sorted(bad_opponents.items()))
        )

    exploit_summary = exploitability_report["summary"]
    median_delta = float(exploit_summary["median_midpoint_delta"])
    if median_delta < args.min_exploitability_median_delta:
        reasons.append(
            f"exploitability median midpoint delta {median_delta:.6f} < required "
            f"{args.min_exploitability_median_delta:.6f}"
        )

    certified_worse = int(exploit_summary.get("verdicts", {}).get("candidate_certified_worse", 0))
    if certified_worse > args.max_certified_worse:
        reasons.append(
            f"candidate has {certified_worse} certified-worse exploitability case(s); "
            f"allowed {args.max_certified_worse}"
        )

    overlap = int(exploit_summary.get("verdicts", {}).get("intervals_overlap", 0))
    if overlap:
        warnings.append(f"{overlap} exploitability case(s) overlap; not proof of improvement")

    policy_drift_summary = None
    if policy_drift_report is not None:
        policy_drift_summary = policy_drift_report.get("summary", {})

    status = "accepted" if not reasons else "rejected"
    return {
        "status": status,
        "reasons": reasons,
        "warnings": warnings,
        "metrics": {
            "checkpoints": checkpoints,
            "held_out_overall_mse": calibration_report.get("held_out_overall_mse"),
            "beats_prev_gen": calibration_report.get("beats_prev_gen"),
            "ladder_wins_delta": ladder_delta,
            "opponent_wins_delta": opponent_deltas,
            "exploitability_median_midpoint_delta": median_delta,
            "exploitability_verdicts": exploit_summary.get("verdicts", {}),
            "policy_drift": policy_drift_summary,
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration-report", required=True)
    parser.add_argument("--ladder-report", required=True)
    parser.add_argument("--exploitability-report", required=True)
    parser.add_argument("--policy-drift-report", default=None)
    parser.add_argument(
        "--out",
        default=str(Path("checkpoints") / "checkpoint_promotion_event" / "report.json"),
    )
    parser.add_argument("--min-ladder-wins-delta", type=int, default=1)
    parser.add_argument("--min-opponent-wins-delta", type=int, default=0)
    parser.add_argument("--min-exploitability-median-delta", type=float, default=0.0)
    parser.add_argument("--max-certified-worse", type=int, default=0)
    parser.add_argument(
        "--require-calibration-improvement",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    calibration = load_json(args.calibration_report)
    ladder = load_json(args.ladder_report)
    exploitability = load_json(args.exploitability_report)
    policy_drift = load_json(args.policy_drift_report) if args.policy_drift_report else None
    decision = decide_promotion(calibration, ladder, exploitability, args, policy_drift)
    report = {
        "config": vars(args),
        "candidate_checkpoint": calibration.get("checkpoint"),
        "decision": decision,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)

    print(f"Promotion event: {decision['status'].upper()}")
    for reason in decision["reasons"]:
        print(f"  reject: {reason}")
    for warning in decision["warnings"]:
        print(f"  warn: {warning}")
    print(f"Metrics: {decision['metrics']}")
    print(f"Report: {out_path}")
    return 0 if decision["status"] == "accepted" else 1


def _args_for_tests(**overrides):
    defaults = dict(
        min_ladder_wins_delta=1,
        min_opponent_wins_delta=0,
        min_exploitability_median_delta=0.0,
        max_certified_worse=0,
        require_calibration_improvement=True,
    )
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


if __name__ == "__main__":
    raise SystemExit(main())
