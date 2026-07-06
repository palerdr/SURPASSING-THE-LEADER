#!/usr/bin/env python3
"""Decide whether a trained checkpoint is promotable from saved gate reports.

The promotion rule intentionally combines independent pieces of evidence:

1. Calibration gate passed and, by default, improved held-out MSE.
2. Deterministic checkpoint ladder is non-regressing overall.
3. Certified exploitability comparison is non-regressing.
4. Policy drift/readability diagnostics stay within configured thresholds.

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


def _required_ladder_opponents(args) -> tuple[str, ...]:
    raw = getattr(args, "required_ladder_opponents", None)
    if isinstance(raw, str):
        raw = [raw]
    names = ["pattern_reader", *(raw or [])]
    required: list[str] = []
    for name in names:
        text = str(name)
        if text and text not in required:
            required.append(text)
    return tuple(required)


def _report_checkpoints(
    calibration_report: dict,
    ladder_report: dict,
    exploitability_report: dict,
    policy_drift_report: dict | None = None,
    trace_policy_gate_report: dict | None = None,
) -> dict:
    ladder_cfg = ladder_report.get("config", {})
    exploit_cfg = exploitability_report.get("config", {})
    drift_cfg = (policy_drift_report or {}).get("config", {})
    trace_cfg = (trace_policy_gate_report or {}).get("config", {})
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
    if trace_policy_gate_report is not None:
        checkpoints["trace_policy_gate_candidate"] = trace_cfg.get("candidate_checkpoint")
        checkpoints["trace_policy_gate_champion"] = trace_cfg.get("champion_checkpoint")
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
    if "trace_policy_gate_candidate" in checkpoints:
        candidate_keys = (*candidate_keys, "trace_policy_gate_candidate")
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
    if "trace_policy_gate_champion" in checkpoints:
        trace_champion = _norm_path(checkpoints.get("trace_policy_gate_champion"))
        if trace_champion is None:
            reasons.append("missing champion checkpoint in trace policy gate report")
        elif ladder_champion is not None and trace_champion != ladder_champion:
            reasons.append(
                "champion checkpoint mismatch across reports: "
                f"ladder={checkpoints.get('ladder_champion')} "
                f"trace_policy_gate={checkpoints.get('trace_policy_gate_champion')}"
            )
    return reasons


def _policy_settings_from_config(config: dict) -> dict:
    return {
        "policy_ensemble_size": int(config.get("policy_ensemble_size", 1)),
        "policy_uniform_mix": float(config.get("policy_uniform_mix", 0.0)),
        "resolve_at_critical": bool(config.get("resolve_at_critical", False)),
        "resolve_horizon": int(config.get("resolve_horizon", 3)),
    }


def _policy_setting_consistency_reasons(
    ladder_report: dict,
    exploitability_report: dict,
    policy_drift_report: dict | None = None,
    trace_policy_gate_report: dict | None = None,
) -> tuple[list[str], dict]:
    settings = {
        "ladder": _policy_settings_from_config(ladder_report.get("config", {})),
        "exploitability": _policy_settings_from_config(
            exploitability_report.get("config", {})
        ),
    }
    if policy_drift_report is not None:
        settings["policy_drift"] = _policy_settings_from_config(
            policy_drift_report.get("config", {})
        )
    if trace_policy_gate_report is not None:
        settings["trace_policy_gate"] = _policy_settings_from_config(
            trace_policy_gate_report.get("config", {})
        )

    unique = {
        (
            value["policy_ensemble_size"],
            value["policy_uniform_mix"],
            value["resolve_at_critical"],
            value["resolve_horizon"],
        )
        for value in settings.values()
    }
    if len(unique) <= 1:
        return [], settings
    return [
        "policy setting mismatch across reports: "
        + ", ".join(f"{name}={value}" for name, value in settings.items())
    ], settings


def _policy_budget_from_config(config: dict, *, ladder: bool = False) -> int | None:
    key = "agent_iterations" if ladder else "iterations"
    value = config.get(key)
    return None if value is None else int(value)


def _policy_budget_consistency_reasons(
    ladder_report: dict,
    exploitability_report: dict,
    policy_drift_report: dict | None = None,
    trace_policy_gate_report: dict | None = None,
) -> tuple[list[str], dict]:
    budgets = {
        "ladder": _policy_budget_from_config(
            ladder_report.get("config", {}),
            ladder=True,
        ),
        "exploitability": _policy_budget_from_config(
            exploitability_report.get("config", {})
        ),
    }
    if policy_drift_report is not None:
        budgets["policy_drift"] = _policy_budget_from_config(
            policy_drift_report.get("config", {})
        )
    if trace_policy_gate_report is not None:
        budgets["trace_policy_gate"] = _policy_budget_from_config(
            trace_policy_gate_report.get("config", {})
        )

    present = {name: value for name, value in budgets.items() if value is not None}
    if len(set(present.values())) <= 1:
        return [], budgets
    return [
        "policy budget mismatch across reports: "
        + ", ".join(f"{name}={value}" for name, value in budgets.items())
    ], budgets


def decide_promotion(
    calibration_report: dict,
    ladder_report: dict,
    exploitability_report: dict,
    args,
    policy_drift_report: dict | None = None,
    trace_policy_gate_report: dict | None = None,
) -> dict:
    reasons: list[str] = []
    warnings: list[str] = []
    checkpoints = _report_checkpoints(
        calibration_report,
        ladder_report,
        exploitability_report,
        policy_drift_report,
        trace_policy_gate_report,
    )
    reasons.extend(_checkpoint_consistency_reasons(checkpoints))
    policy_setting_reasons, policy_settings = _policy_setting_consistency_reasons(
        ladder_report,
        exploitability_report,
        policy_drift_report,
        trace_policy_gate_report,
    )
    reasons.extend(policy_setting_reasons)
    policy_budget_reasons, policy_budgets = _policy_budget_consistency_reasons(
        ladder_report,
        exploitability_report,
        policy_drift_report,
        trace_policy_gate_report,
    )
    reasons.extend(policy_budget_reasons)

    gate_passed = bool(calibration_report.get("gate_passed"))
    if not gate_passed:
        reasons.append(f"calibration gate failed: {calibration_report.get('gate_error')}")

    if args.require_calibration_improvement and calibration_report.get("beats_prev_gen") is not True:
        reasons.append("held-out calibration did not strictly beat previous generation")

    if args.require_policy_drift_report and policy_drift_report is None:
        reasons.append("policy drift report is required")

    if args.require_trace_policy_gate_report and trace_policy_gate_report is None:
        reasons.append("trace policy gate report is required")

    ladder_delta = int(ladder_report["aggregate"]["delta"]["wins_delta"])
    if ladder_delta < args.min_ladder_wins_delta:
        reasons.append(
            f"ladder wins_delta {ladder_delta} < required {args.min_ladder_wins_delta}"
        )

    opponent_deltas = _opponent_deltas(ladder_report)
    required_opponents = _required_ladder_opponents(args)
    missing_opponents = [
        opponent for opponent in required_opponents if opponent not in opponent_deltas
    ]
    if missing_opponents:
        reasons.append(
            "required ladder opponent(s) missing: " + ", ".join(missing_opponents)
        )
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
        max_tv = policy_drift_summary.get("max_tv")
        if max_tv is not None and float(max_tv) > args.max_policy_drift_tv:
            reasons.append(
                f"policy drift max_tv {float(max_tv):.6f} > required "
                f"{args.max_policy_drift_tv:.6f}"
            )
        mean_tv = policy_drift_summary.get("mean_tv")
        if mean_tv is not None and float(mean_tv) > args.max_policy_drift_mean_tv:
            reasons.append(
                f"policy drift mean_tv {float(mean_tv):.6f} > required "
                f"{args.max_policy_drift_mean_tv:.6f}"
            )
        max_entropy_drop = policy_drift_summary.get("max_entropy_drop")
        if (
            max_entropy_drop is not None
            and float(max_entropy_drop) > args.max_policy_entropy_drop
        ):
            reasons.append(
                f"policy entropy drop {float(max_entropy_drop):.6f} > required "
                f"{args.max_policy_entropy_drop:.6f}"
            )
        min_candidate_entropy = policy_drift_summary.get("min_candidate_entropy")
        min_champion_entropy = policy_drift_summary.get("min_champion_entropy")
        if (
            min_candidate_entropy is not None
            and float(min_candidate_entropy) < args.min_policy_candidate_entropy
        ):
            candidate_entropy = float(min_candidate_entropy)
            champion_entropy = (
                None if min_champion_entropy is None else float(min_champion_entropy)
            )
            if champion_entropy is None or candidate_entropy < champion_entropy - 1.0e-9:
                suffix = (
                    ""
                    if champion_entropy is None
                    else f" and below champion floor {champion_entropy:.6f}"
                )
                reasons.append(
                    f"policy candidate entropy {candidate_entropy:.6f} < required "
                    f"{args.min_policy_candidate_entropy:.6f}{suffix}"
                )

    trace_policy_gate_summary = None
    if trace_policy_gate_report is not None:
        trace_policy_gate_summary = trace_policy_gate_report.get("decision", {})
        if trace_policy_gate_summary.get("status") != "passed":
            gate_reasons = trace_policy_gate_summary.get("reasons", [])
            if gate_reasons:
                reasons.append(
                    "trace policy gate failed: " + "; ".join(str(item) for item in gate_reasons)
                )
            else:
                reasons.append("trace policy gate failed")

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
            "trace_policy_gate": trace_policy_gate_summary,
            "policy_settings": policy_settings,
            "policy_budgets": policy_budgets,
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration-report", required=True)
    parser.add_argument("--ladder-report", required=True)
    parser.add_argument("--exploitability-report", required=True)
    parser.add_argument("--policy-drift-report", default=None)
    parser.add_argument("--trace-policy-gate-report", default=None)
    parser.add_argument(
        "--out",
        default=str(Path("checkpoints") / "checkpoint_promotion_event" / "report.json"),
    )
    parser.add_argument("--min-ladder-wins-delta", type=int, default=1)
    parser.add_argument("--min-opponent-wins-delta", type=int, default=0)
    parser.add_argument(
        "--required-ladder-opponent",
        dest="required_ladder_opponents",
        action="append",
        default=None,
        help=(
            "Opponent rung that must be present in the ladder report. Repeatable. "
            "Defaults to pattern_reader so policy-readability canaries cannot be "
            "omitted from promotion evidence."
        ),
    )
    parser.add_argument("--min-exploitability-median-delta", type=float, default=0.0)
    parser.add_argument("--max-certified-worse", type=int, default=0)
    parser.add_argument("--max-policy-drift-tv", type=float, default=0.30)
    parser.add_argument("--max-policy-drift-mean-tv", type=float, default=0.10)
    parser.add_argument("--max-policy-entropy-drop", type=float, default=0.20)
    parser.add_argument("--min-policy-candidate-entropy", type=float, default=0.05)
    parser.add_argument(
        "--require-policy-drift-report",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Require a policy-drift/readability report before promotion.",
    )
    parser.add_argument(
        "--require-trace-policy-gate-report",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Require a trace policy gate report before promotion.",
    )
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
    trace_policy_gate = (
        load_json(args.trace_policy_gate_report)
        if args.trace_policy_gate_report
        else None
    )
    decision = decide_promotion(
        calibration,
        ladder,
        exploitability,
        args,
        policy_drift,
        trace_policy_gate,
    )
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
        required_ladder_opponents=("pattern_reader",),
        min_exploitability_median_delta=0.0,
        max_certified_worse=0,
        max_policy_drift_tv=0.30,
        max_policy_drift_mean_tv=0.10,
        max_policy_entropy_drop=0.20,
        min_policy_candidate_entropy=0.05,
        require_policy_drift_report=True,
        require_trace_policy_gate_report=False,
        require_calibration_improvement=True,
    )
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


if __name__ == "__main__":
    raise SystemExit(main())
