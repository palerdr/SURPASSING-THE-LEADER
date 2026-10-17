"""The `propose_prior` job and the sensor that starts it.

The op writes a passing prior as the next numbered config and as adaptive_prior_current.json
under the model's proposal folder, and it writes the model card as markdown beside the model.
With dry_run false, it also copies the two files into a temporary git worktree of origin/main,
commits them on a new branch there, pushes the branch, and opens a pull request.
The live checkout keeps its branch and its working tree.
"""

import json
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

from dagster import (
    AssetCheckKey,
    AssetKey,
    Config,
    Failure,
    OpExecutionContext,
    RunConfig,
    RunRequest,
    SensorEvaluationContext,
    SkipReason,
    job,
    op,
    sensor,
)
from dagster._core.storage.asset_check_execution_record import COMPLETED_ASSET_CHECK_EXECUTION_RECORD_STATUSES

from hal_loop import gates
from hal_loop.jobs.card import render_card
from hal_loop.resources import DEFAULT_ARENA_CONFIG_DIR, REPOSITORY_ROOT, PathsResource

CURRENT_PRIOR = "adaptive_prior_current.json"
VERSIONED = re.compile(r"^adaptive_prior_v(\d+)\.json$")
LIKELIHOOD_CHECK = AssetCheckKey(AssetKey("offline_evaluation"), "likelihood_gate")
PROPOSAL_DIR = "proposal"
BASE_REF = "origin/main"


class ProposeConfig(Config):
    version: str
    dry_run: bool = True
    config_dir: str = str(DEFAULT_ARENA_CONFIG_DIR)
    repo_dir: str = str(REPOSITORY_ROOT)


def next_config_number(config_dir: Path) -> int:
    """Return one more than the highest n of adaptive_prior_v<n>.json."""
    numbers = [int(m.group(1)) for p in config_dir.glob("adaptive_prior_v*.json") if (m := VERSIONED.match(p.name))]
    return max(numbers, default=0) + 1


def has_proposal(models_dir: Path, version: str) -> bool:
    """Return True when a proposal for this model version already exists."""
    return any((models_dir / version / PROPOSAL_DIR).glob("adaptive_prior_v*.json"))


def _run(command: list[str], cwd: Path) -> str:
    """Run one command and return its stdout. A failure raises with the command's stderr."""
    result = subprocess.run(command, cwd=cwd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"{' '.join(command)} failed in {cwd} with exit code {result.returncode}: "
            f"{result.stderr.strip() or result.stdout.strip()}"
        )
    return result.stdout.strip()


def open_pull_request(repo: Path, files: list[Path], config_subdir: Path, number: int, card_path: Path, version: str) -> str:
    """Commit the files on hal-loop/prior-v<n> in a temporary worktree of origin/main and open a pull request.

    `config_subdir` is the config folder relative to the repository root.
    The live checkout at `repo` keeps its branch, index, and working tree.
    """
    branch = f"hal-loop/prior-v{number}"
    parent = Path(tempfile.mkdtemp(prefix="hal-loop-proposal-"))
    worktree = parent / "worktree"
    _run(["git", "fetch", "origin", "main"], repo)
    _run(["git", "worktree", "add", str(worktree), "-b", branch, BASE_REF], repo)
    try:
        target = worktree / config_subdir
        target.mkdir(parents=True, exist_ok=True)
        relative = []
        for path in files:
            shutil.copyfile(path, target / path.name)
            relative.append(str(config_subdir / path.name))
        _run(["git", "add", "--", *relative], worktree)
        _run(["git", "commit", "-m", f"Propose adaptive prior v{number} ({version})", "--", *relative], worktree)
        _run(["git", "push", "-u", "origin", branch], worktree)
        return _run(
            ["gh", "pr", "create", "--base", "main", "--head", branch,
             "--title", f"Adaptive prior v{number} ({version})", "--body-file", str(card_path)],
            worktree,
        )
    finally:
        _run(["git", "worktree", "remove", "--force", str(worktree)], repo)
        shutil.rmtree(parent, ignore_errors=True)


@op
def propose_prior_op(context: OpExecutionContext, config: ProposeConfig, paths: PathsResource) -> dict:
    """Write the next prior config and its card, then open a pull request unless dry_run."""
    directory = paths.models_dir / config.version
    card = json.loads((directory / "card.json").read_text(encoding="utf-8"))
    problem = gates.verdict_problem(card.get("verdict"))
    if problem:
        raise Failure(f"prior {config.version} cannot be proposed: {problem}")
    evaluation = json.loads((directory / "evaluation.json").read_text(encoding="utf-8"))
    passed, description = gates.likelihood_gate(evaluation)
    if not passed:
        raise Failure(f"prior {config.version} did not pass the likelihood gate: {description}")
    config_dir = Path(config.config_dir)
    number = next_config_number(config_dir)
    # The proposal lands beside the model. Only a pull request reaches the repository.
    target_dir = directory / PROPOSAL_DIR
    target_dir.mkdir(parents=True, exist_ok=True)
    versioned = target_dir / f"adaptive_prior_v{number}.json"
    current = target_dir / CURRENT_PRIOR
    shutil.copyfile(directory / "prior.json", versioned)
    shutil.copyfile(versioned, current)
    card_path = directory / "card.md"
    card_path.write_text(render_card(card, evaluation, config_name=versioned.name), encoding="utf-8")
    result = {"number": number, "versioned": str(versioned), "current": str(current), "card": str(card_path), "pull_request": None}
    if config.dry_run:
        context.log.info(f"dry run: wrote {versioned.name} and {CURRENT_PRIOR} under {target_dir}; no branch or pull request")
    else:
        repo = Path(config.repo_dir).resolve()
        subdir = config_dir.resolve().relative_to(repo)
        result["pull_request"] = open_pull_request(repo, [versioned, current], subdir, number, card_path, config.version)
    return result


@job(description="Write a passing prior as the next versioned config and propose it in a pull request.")
def propose_prior():
    propose_prior_op()


@sensor(job=propose_prior, minimum_interval_seconds=60)
def evaluation_passed_sensor(context: SensorEvaluationContext, paths: PathsResource):
    """Request a dry-run propose_prior once for each model version whose likelihood_gate passed.

    The sensor reads finished check records alone. A PLANNED record belongs to a run in
    progress or a failed run, so the cursor stays on the last finished record.
    """
    storage = context.instance.event_log_storage
    latest = storage.get_asset_check_execution_history(LIKELIHOOD_CHECK, limit=1)
    finished = storage.get_asset_check_execution_history(
        LIKELIHOOD_CHECK, limit=1, status=COMPLETED_ASSET_CHECK_EXECUTION_RECORD_STATUSES
    )
    cursor = int(context.cursor or 0)
    if not finished or finished[0].id <= cursor:
        if latest and latest[0].id > cursor:
            return SkipReason(f"the latest likelihood_gate record is {latest[0].status.value}; waiting for a finished one")
        return SkipReason("no new likelihood_gate evaluation")
    record = finished[0]
    context.update_cursor(str(record.id))
    if record.event is None or record.event.dagster_event is None:
        return SkipReason("the latest finished likelihood_gate record has no event")
    evaluation = record.event.dagster_event.asset_check_evaluation_data
    if not evaluation.passed:
        return SkipReason("the latest likelihood_gate evaluation failed")
    version = evaluation.metadata["version"].value
    if has_proposal(paths.models_dir, version):
        return SkipReason(f"a proposal for prior {version} already exists")
    return RunRequest(
        run_key=f"propose:{version}",
        run_config=RunConfig(ops={"propose_prior_op": ProposeConfig(version=version)}),
    )
