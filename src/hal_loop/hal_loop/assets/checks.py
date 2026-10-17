"""Asset checks on `offline_evaluation`: the likelihood gate and the exploitation gate.

The exploitation gate plays real games against the complete DTH tablebase.
It runs only when its op config sets with_simulation to true.
Otherwise it yields no result, so Dagster records the check as skipped.
"""

import json
from pathlib import Path

from dagster import (
    AssetCheckExecutionContext,
    AssetCheckResult,
    AssetCheckSpec,
    Field,
    asset_check,
    multi_asset_check,
)

from hal_loop import gates
from hal_loop.prior import HistogramEmulator
from hal_loop.resources import DEFAULT_DTH_ARTIFACT_DIR, PathsResource
from hal_loop.warehouse import load_histograms

EXPLOITATION_DESCRIPTION = (
    "Skipped unless the op config sets with_simulation: true. When enabled, it plays a "
    "paired-seat series of the adaptive provider with the candidate prior and of the "
    "equilibrium provider against a histogram emulator fitted on the test split. It passes "
    "when the adaptive mean score is at most 0.01 below the equilibrium mean score."
)


@asset_check(
    asset="offline_evaluation",
    description=(
        f"Passes when the card verdict is {gates.REQUIRED_VERDICT}, the test split holds at least "
        f"{gates.MINIMUM_SCORED_PLAYERS} scored players and {gates.MINIMUM_SCORED_MOVES} scored moves, "
        "and the candidate beats production per move with a 95% bootstrap interval above zero."
    ),
)
def likelihood_gate(offline_evaluation: dict) -> AssetCheckResult:
    passed, description = gates.likelihood_gate(offline_evaluation)
    pooled = offline_evaluation["pooled"]
    return AssetCheckResult(
        passed=passed,
        description=description,
        metadata={
            "difference": pooled["difference"],
            "interval": str(pooled["interval"]),
            "version": offline_evaluation["version"],
            "verdict": offline_evaluation["verdict"],
            "scored_split": offline_evaluation["split"],
            "scored_players": offline_evaluation["players"],
            "scored_moves": pooled["moves"],
        },
    )


@multi_asset_check(
    specs=[AssetCheckSpec("exploitation_gate", asset="offline_evaluation", description=EXPLOITATION_DESCRIPTION)],
    can_subset=True,
    config_schema={
        "with_simulation": Field(bool, default_value=False),
        "games": Field(int, default_value=200),
        "dth_artifact_dir": Field(str, default_value=str(DEFAULT_DTH_ARTIFACT_DIR)),
    },
)
def exploitation_gate(context: AssetCheckExecutionContext, offline_evaluation: dict, paths: PathsResource):
    config = context.op_execution_context.op_config
    if not config["with_simulation"]:
        context.log.info("exploitation_gate skipped: set with_simulation: true in its op config to run it")
        return
    from hal_loop.simulation import build_provider, paired_series

    artifact = Path(config["dth_artifact_dir"])
    prior_path = Path(offline_evaluation["directory"]) / "prior.json"
    emulator_counts = load_histograms(paths.warehouse).population("test")
    results = {}
    for label, kind, prior in (("adaptive", "adaptive-dth", prior_path), ("equilibrium", "dth", None)):
        provider = build_provider(kind, artifact, prior_path=prior)
        results[label] = paired_series(provider, lambda: HistogramEmulator(emulator_counts), games=config["games"])
    passed, description = gates.exploitation_gate(results["adaptive"]["mean_score"], results["equilibrium"]["mean_score"])
    yield AssetCheckResult(
        check_name="exploitation_gate",
        passed=passed,
        description=description,
        metadata={"series": json.dumps(results, sort_keys=True)},
    )
