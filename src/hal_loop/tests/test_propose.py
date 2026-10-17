"""The propose_prior op in dry-run mode, the pull-request command sequence, and the sensor that requests the op."""

import json
import shutil
import sys
from pathlib import Path

import pytest
from dagster import DagsterInstance, RunConfig, RunRequest, SkipReason, asset, build_sensor_context, materialize

from hal_loop.assets.checks import likelihood_gate
from hal_loop.jobs import propose
from hal_loop.jobs.propose import ProposeConfig, evaluation_passed_sensor, open_pull_request, propose_prior
from hal_loop.prior import ROLES, uniform_prior
from hal_loop.resources import DEFAULT_ARENA_CONFIG_DIR, PathsResource

VERSION = "0123456789ab"


def evaluation(difference: float, interval: list[float], verdict: str = "fitted") -> dict:
    summary = {
        "moves": 120.0,
        "difference": difference,
        "interval": interval,
        "log_likelihood_per_move": {"candidate": -2.0 + difference, "uniform": -2.0, "production": -2.0},
    }
    return {
        "version": VERSION,
        "verdict": verdict,
        "split": "test",
        "players": 5,
        "production_source": "uniform",
        "bootstrap": {"resamples": 2000, "seed": 20261017},
        "roles": {role: dict(summary) for role in ROLES},
        "pooled": summary,
    }


def model_dir(data_dir: Path, result: dict, verdict: str = "fitted") -> Path:
    directory = data_dir / "models" / VERSION
    directory.mkdir(parents=True)
    prior = uniform_prior(strength=5.0)
    (directory / "prior.json").write_text(json.dumps(prior))
    card = {
        "version": VERSION,
        "training_games": 60,
        "finished_training_games": 55,
        "minimum_training_games": 50,
        "players": 5,
        "verdict": verdict,
        "mlflow_run_id": "run-1",
        "selection": {role: {"strength": 5.0, "training_moves": 100.0} for role in ROLES},
    }
    (directory / "card.json").write_text(json.dumps(card))
    (directory / "evaluation.json").write_text(json.dumps(result))
    return directory


def run_propose(data_dir: Path, config_dir: Path):
    return propose_prior.execute_in_process(
        run_config=RunConfig(ops={"propose_prior_op": ProposeConfig(version=VERSION, config_dir=str(config_dir))}),
        resources={"paths": PathsResource(data_dir=str(data_dir))},
        raise_on_error=False,
    )


def failure_message(result) -> str:
    return " ".join(str(event.event_specific_data.error.message) for event in result.all_events if event.is_step_failure)


def test_dry_run_writes_next_versioned_config_and_card(tmp_path):
    config_dir = tmp_path / "config"
    shutil.copytree(DEFAULT_ARENA_CONFIG_DIR, config_dir)
    (config_dir / "adaptive_prior_v2.json").write_text("{}")
    directory = model_dir(tmp_path / "data", evaluation(0.1, [0.02, 0.2]))
    before = sorted(path.name for path in DEFAULT_ARENA_CONFIG_DIR.iterdir())

    result = run_propose(tmp_path / "data", config_dir)
    assert result.success
    output = result.output_for_node("propose_prior_op")
    assert output["number"] == 3 and output["pull_request"] is None
    # The dry run leaves the config folder alone and writes the proposal beside the model.
    versioned = directory / "proposal" / "adaptive_prior_v3.json"
    current = directory / "proposal" / "adaptive_prior_current.json"
    assert output["versioned"] == str(versioned)
    assert not (config_dir / "adaptive_prior_v3.json").exists()
    assert versioned.read_bytes() == (directory / "prior.json").read_bytes()
    assert current.read_bytes() == versioned.read_bytes()
    card = (directory / "card.md").read_text()
    assert f"# Adaptive prior {VERSION}" in card and "adaptive_prior_v3.json" in card
    assert "The test split holds 5 scored players" in card

    # The number comes from the config folder, which the dry run did not change.
    assert run_propose(tmp_path / "data", config_dir).output_for_node("propose_prior_op")["number"] == 3
    assert sorted(path.name for path in DEFAULT_ARENA_CONFIG_DIR.iterdir()) == before


def test_failing_evaluation_writes_nothing(tmp_path):
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    directory = model_dir(tmp_path / "data", evaluation(0.1, [-0.02, 0.2]))
    assert not run_propose(tmp_path / "data", config_dir).success
    assert list(config_dir.iterdir()) == []
    assert not (directory / "proposal").exists()


def test_insufficient_data_card_refuses_the_proposal(tmp_path):
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    # The evaluation passes on its own numbers; the card's verdict alone refuses it.
    directory = model_dir(tmp_path / "data", evaluation(0.1, [0.02, 0.2]), verdict="insufficient data")
    result = run_propose(tmp_path / "data", config_dir)
    assert not result.success
    assert "card verdict is 'insufficient data'" in failure_message(result)
    assert list(config_dir.iterdir()) == []
    assert not (directory / "proposal").exists() and not (directory / "card.md").exists()


class Recorder:
    """Stand in for propose._run and record each command with its working directory."""

    def __init__(self, fail_on: str | None = None):
        self.calls: list[tuple[list[str], Path]] = []
        self.fail_on = fail_on

    def __call__(self, command, cwd):
        self.calls.append((list(command), Path(cwd)))
        if self.fail_on and self.fail_on in command:
            raise RuntimeError(f"{' '.join(command)} failed: remote rejected")
        return "https://github.com/example/pull/7" if command[:3] == ["gh", "pr", "create"] else ""


def pull_request_inputs(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    files = [tmp_path / "adaptive_prior_v3.json", tmp_path / "adaptive_prior_current.json"]
    for path in files:
        path.write_text("{}")
    card = tmp_path / "card.md"
    card.write_text("# card")
    return repo, files, card


def test_pull_request_works_in_a_worktree_and_never_switches_the_checkout(tmp_path, monkeypatch):
    recorder = Recorder()
    monkeypatch.setattr(propose, "_run", recorder)
    repo, files, card = pull_request_inputs(tmp_path)
    url = open_pull_request(repo, files, Path("src/arena/config"), 3, card, VERSION)
    assert url == "https://github.com/example/pull/7"

    in_repo = [command for command, cwd in recorder.calls if cwd == repo]
    assert not any(command[:2] in (["git", "switch"], ["git", "checkout"]) for command in in_repo)
    assert not any(command[1] in ("switch", "checkout") for command, _ in recorder.calls if command[0] == "git")
    assert in_repo[0] == ["git", "fetch", "origin", "main"]
    worktree = Path(in_repo[1][3])
    assert in_repo[1] == ["git", "worktree", "add", str(worktree), "-b", "hal-loop/prior-v3", "origin/main"]
    assert in_repo[-1] == ["git", "worktree", "remove", "--force", str(worktree)]
    in_worktree = [command[:3] for command, cwd in recorder.calls if cwd == worktree]
    assert in_worktree == [["git", "add", "--"], ["git", "commit", "-m"], ["git", "push", "-u"], ["gh", "pr", "create"]]
    assert not worktree.parent.exists()


def test_pull_request_removes_the_worktree_after_a_failure(tmp_path, monkeypatch):
    recorder = Recorder(fail_on="push")
    monkeypatch.setattr(propose, "_run", recorder)
    repo, files, card = pull_request_inputs(tmp_path)
    with pytest.raises(RuntimeError, match="remote rejected"):
        open_pull_request(repo, files, Path("src/arena/config"), 3, card, VERSION)
    assert recorder.calls[-1][0][:3] == ["git", "worktree", "remove"]


def test_run_reports_stderr(tmp_path):
    command = [sys.executable, "-c", "import sys; sys.stderr.write('fatal: branch exists'); sys.exit(3)"]
    with pytest.raises(RuntimeError, match="exit code 3: fatal: branch exists"):
        propose._run(command, tmp_path)


def check_result(instance, result: dict | None):
    """Materialize offline_evaluation and likelihood_gate; a None result fails the asset."""

    @asset(name="offline_evaluation")
    def synthetic_evaluation() -> dict:
        if result is None:
            raise RuntimeError("the evaluation failed")
        return result

    materialize([synthetic_evaluation, likelihood_gate], instance=instance, raise_on_error=False)


def sense(instance, data_dir: Path, cursor: str | None = None):
    context = build_sensor_context(instance=instance, cursor=cursor, resources={"paths": PathsResource(data_dir=str(data_dir))})
    return evaluation_passed_sensor(context), context.cursor


def test_sensor_requests_the_job_after_a_passing_gate(tmp_path):
    with DagsterInstance.ephemeral() as instance:
        assert isinstance(sense(instance, tmp_path)[0], SkipReason)
        check_result(instance, evaluation(0.1, [0.02, 0.2]))
        request, cursor = sense(instance, tmp_path)
        assert isinstance(request, RunRequest)
        assert request.run_key == f"propose:{VERSION}"
        assert request.run_config["ops"]["propose_prior_op"]["config"]["version"] == VERSION
        assert cursor is not None
        assert isinstance(sense(instance, tmp_path, cursor)[0], SkipReason)


def test_sensor_skips_a_planned_record_and_keeps_its_cursor(tmp_path):
    with DagsterInstance.ephemeral() as instance:
        check_result(instance, evaluation(0.1, [0.02, 0.2]))
        _, cursor = sense(instance, tmp_path)
        # A failed run leaves the check's PLANNED record as the newest one.
        check_result(instance, None)
        latest = instance.event_log_storage.get_asset_check_execution_history(propose.LIKELIHOOD_CHECK, limit=1)
        assert latest[0].status.value == "PLANNED"
        result, after = sense(instance, tmp_path, cursor)
        assert isinstance(result, SkipReason) and "PLANNED" in result.skip_message
        assert after == cursor
        # The next finished record moves the cursor past the PLANNED one.
        check_result(instance, evaluation(0.1, [0.02, 0.2]))
        result, after = sense(instance, tmp_path, cursor)
        assert isinstance(result, RunRequest) and int(after) > int(cursor)


def test_sensor_keys_runs_on_the_version_and_skips_a_proposed_version(tmp_path):
    with DagsterInstance.ephemeral() as instance:
        check_result(instance, evaluation(0.1, [0.02, 0.2]))
        first, cursor = sense(instance, tmp_path)
        # The same version passing again gives the same run key, which Dagster runs once.
        check_result(instance, evaluation(0.1, [0.02, 0.2]))
        second, cursor = sense(instance, tmp_path, cursor)
        assert first.run_key == second.run_key == f"propose:{VERSION}"
        # Once a proposal exists for the version, the sensor requests nothing.
        proposal = tmp_path / "models" / VERSION / "proposal"
        proposal.mkdir(parents=True)
        (proposal / "adaptive_prior_v1.json").write_text("{}")
        check_result(instance, evaluation(0.1, [0.02, 0.2]))
        third, _ = sense(instance, tmp_path, cursor)
        assert isinstance(third, SkipReason) and "already exists" in third.skip_message

