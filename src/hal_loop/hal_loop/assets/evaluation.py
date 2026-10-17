"""The `offline_evaluation` asset: score the candidate prior on the test split.

The validation split chose the prior's strength, so the score reads the test split.

Production is src/arena/config/adaptive_prior_current.json when that file exists.
Without it, the uniform prior stands in for production.
"""

import json
from pathlib import Path

from dagster import AssetExecutionContext, Output, asset

from hal_loop.evaluate import evaluate_candidate
from hal_loop.resources import PathsResource
from hal_loop.warehouse import load_histograms

CURRENT_PRIOR = "adaptive_prior_current.json"


def load_production(config_dir: Path) -> dict | None:
    path = config_dir / CURRENT_PRIOR
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


@asset(group_name="model", compute_kind="python")
def offline_evaluation(context: AssetExecutionContext, adaptive_prior: dict, paths: PathsResource) -> Output[dict]:
    """Write data/models/<version>/evaluation.json and return it."""
    directory = Path(adaptive_prior["directory"])
    candidate = json.loads((directory / "prior.json").read_text(encoding="utf-8"))
    production = load_production(Path(paths.arena_config_dir))
    evaluation = evaluate_candidate(load_histograms(paths.warehouse), candidate, production)
    evaluation["version"] = adaptive_prior["version"]
    evaluation["verdict"] = adaptive_prior["verdict"]
    evaluation["directory"] = str(directory)
    (directory / "evaluation.json").write_text(json.dumps(evaluation, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    pooled = evaluation["pooled"]
    context.log.info(f"pooled difference {pooled['difference']} with interval {pooled['interval']}")
    return Output(
        evaluation,
        metadata={
            "version": adaptive_prior["version"],
            "production_source": evaluation["production_source"],
            "pooled_difference": pooled["difference"],
            "pooled_interval": str(pooled["interval"]),
            "verdict": evaluation["verdict"],
            "scored_split": evaluation["split"],
            "scored_players": evaluation["players"],
            "scored_moves": pooled["moves"],
        },
    )
