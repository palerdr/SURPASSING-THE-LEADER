"""Log a fitted prior to the local MLflow store and register it as a model version.

The backend store is the SQLite database data/mlruns/mlflow.db.
Run artifacts go under data/mlruns/artifacts.
Each prior version registers once: a model version tagged with the same prior_version
is returned as it is, and no new run is logged.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

EXPERIMENT = "adaptive_prior"
REGISTERED_MODEL = "adaptive_prior"
DATABASE = "mlflow.db"
ARTIFACTS = "artifacts"
VERSION_TAG = "prior_version"


def tracking_uri(mlruns: Path) -> str:
    """Return the SQLite tracking URI of the store under `mlruns`."""
    return f"sqlite:///{(mlruns / DATABASE).resolve()}"


def find_model_version(client, version: str):
    """Return the registered model version tagged with this prior version, or None."""
    found = client.search_model_versions(f"name = '{REGISTERED_MODEL}' and tags.{VERSION_TAG} = '{version}'")
    return min(found, key=lambda entry: int(entry.version)) if found else None


def log_prior_run(
    mlruns: Path,
    *,
    version: str,
    params: Mapping[str, object],
    metrics: Mapping[str, float],
    artifacts: Sequence[Path],
) -> tuple[str, str, bool]:
    """Log one fit and return its MLflow run id, its registered model version, and whether this call registered it."""
    from mlflow import MlflowClient
    from mlflow.entities import Metric, Param
    from mlflow.exceptions import MlflowException
    from mlflow.utils.time import get_current_time_millis

    mlruns.mkdir(parents=True, exist_ok=True)
    uri = tracking_uri(mlruns)
    client = MlflowClient(tracking_uri=uri, registry_uri=uri)
    existing = find_model_version(client, version)
    if existing is not None:
        return existing.run_id, str(existing.version), False
    experiment = client.get_experiment_by_name(EXPERIMENT)
    experiment_id = (
        experiment.experiment_id
        if experiment
        else client.create_experiment(EXPERIMENT, artifact_location=(mlruns / ARTIFACTS).resolve().as_uri())
    )
    run = client.create_run(experiment_id, run_name=version)
    run_id = run.info.run_id
    now = get_current_time_millis()
    client.log_batch(
        run_id,
        metrics=[Metric(key, float(value), now, 0) for key, value in metrics.items()],
        params=[Param(key, str(value)) for key, value in params.items()],
    )
    for artifact in artifacts:
        client.log_artifact(run_id, str(artifact))
    try:
        client.create_registered_model(REGISTERED_MODEL)
    except MlflowException as error:
        if error.error_code != "RESOURCE_ALREADY_EXISTS":
            raise
    model_version = client.create_model_version(
        REGISTERED_MODEL, source=run.info.artifact_uri, run_id=run_id, tags={VERSION_TAG: version}
    )
    client.set_terminated(run_id)
    return run_id, str(model_version.version), True
