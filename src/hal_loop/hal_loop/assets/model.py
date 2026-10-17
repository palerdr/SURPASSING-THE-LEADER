"""The `adaptive_prior` asset: fit the role prior on the training split.

The version is the first 12 hex digits of a SHA-256 over the protocol constants and
the sorted (player_id, human_role, split, second, count) rows of role_histograms in the
train and validation splits. Those rows are the whole input of the fit, so a rewritten
game or a dbt change that moves a count also moves the version, and the same counts
give the same version.
"""

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from dagster import AssetExecutionContext, AssetKey, Output, asset

from hal_loop.prior import (
    DEFAULT_PSEUDOCOUNT,
    ROLES,
    STRENGTH_GRID,
    fit_role_prior,
    select_strength,
    validate_prior,
)
from hal_loop.resources import PathsResource
from hal_loop.tracking import log_prior_run
from hal_loop.evaluate import HISTORY_SPLITS, SCORED_SPLIT
from hal_loop.warehouse import Histograms, histogram_rows, load_histograms, training_games

FIT_SPLITS = ("train", "validation")
PROTOCOL = {
    "protocol": "adaptive-prior-v1",
    "pseudocount": DEFAULT_PSEUDOCOUNT,
    "strength_grid": list(STRENGTH_GRID),
    "fit_split": "train",
    "selection_split": "validation",
    "evaluation_split": SCORED_SPLIT,
    "evaluation_history_splits": list(HISTORY_SPLITS),
}
MINIMUM_TRAINING_GAMES = 50


def prior_version(rows: list[tuple]) -> str:
    """Hash the sorted train and validation histogram rows with the protocol constants."""
    payload = json.dumps({"histograms": sorted(list(row) for row in rows), "protocol": PROTOCOL}, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]


def fit_from_histograms(histograms: Histograms) -> tuple[dict, dict]:
    """Return the fitted prior and each role's strength-selection summary."""
    totals = histograms.population("train")
    players = {role: [counts for _, counts in histograms.player_counts(role)] for role in ROLES}
    prior = fit_role_prior(totals, DEFAULT_PSEUDOCOUNT, players_by_role=players)
    selection = {}
    for role in ROLES:
        strength, scores = select_strength(totals[role], players[role], DEFAULT_PSEUDOCOUNT)
        held_out = float(sum(p.held_out.sum() for p in players[role]))
        selection[role] = {
            "strength": strength,
            "training_moves": float(totals[role].sum()),
            "held_out_moves": held_out,
            "players_scored": len(players[role]),
            "grid_log_likelihood": {str(s): v for s, v in scores.items()},
            "selected_log_likelihood_per_move": scores[strength] / held_out if scores else None,
        }
    return prior, selection


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


@asset(
    deps=[AssetKey("role_histograms"), AssetKey("move_context")],
    group_name="model",
    compute_kind="python",
)
def adaptive_prior(context: AssetExecutionContext, paths: PathsResource) -> Output[dict]:
    """Fit, version, register, and write data/models/<version>/prior.json and card.json.

    A version that MLflow already registered keeps its model version, and the run logs nothing new.

    The fit uses the moves of finished, stopped, and abandoned training games. The verdict
    counts finished training games alone: it is "fitted" at 50 or more and
    "insufficient data" below 50.
    """
    histograms = load_histograms(paths.warehouse)
    games = training_games(paths.warehouse)
    prior, selection = fit_from_histograms(histograms)
    validate_prior(prior)
    version = prior_version(histogram_rows(paths.warehouse, FIT_SPLITS))
    verdict = "fitted" if games.finished >= MINIMUM_TRAINING_GAMES else "insufficient data"
    directory = paths.models_dir / version
    directory.mkdir(parents=True, exist_ok=True)
    card = {
        "version": version,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "protocol": PROTOCOL,
        "training_games": len(games.game_keys),
        "finished_training_games": games.finished,
        "training_rows": games.moves,
        "players": games.players,
        "minimum_training_games": MINIMUM_TRAINING_GAMES,
        "verdict_rule": "fitted when finished_training_games >= minimum_training_games",
        "verdict": verdict,
        "selection": selection,
    }
    _write_json(directory / "prior.json", prior)
    _write_json(directory / "card.json", card)
    params = {
        "pseudocount": DEFAULT_PSEUDOCOUNT,
        "training_rows": games.moves,
        "training_games": len(games.game_keys),
        "finished_training_games": games.finished,
        "players": games.players,
        "version": version,
        "verdict": verdict,
        **{f"strength_{role}": prior[role]["strength"] for role in ROLES},
    }
    metrics = {f"training_moves_{role}": selection[role]["training_moves"] for role in ROLES}
    for role in ROLES:
        if selection[role]["selected_log_likelihood_per_move"] is not None:
            metrics[f"selection_log_likelihood_{role}"] = selection[role]["selected_log_likelihood_per_move"]
    run_id, model_version, registered = log_prior_run(
        paths.mlruns, version=version, params=params, metrics=metrics,
        artifacts=[directory / "prior.json", directory / "card.json"],
    )
    card.update(mlflow_run_id=run_id, registered_model_version=model_version)
    _write_json(directory / "card.json", card)
    action = "registered model version" if registered else "kept the existing model version"
    context.log.info(f"prior {version}: {verdict} on {games.finished} finished training games; {action} {model_version}")
    return Output(
        {"version": version, "directory": str(directory), "verdict": verdict, "mlflow_run_id": run_id},
        metadata={"version": version, "mlflow_run_id": run_id, "verdict": verdict,
                  "finished_training_games": games.finished, "registered_model_version": model_version,
                  "newly_registered": registered},
    )
