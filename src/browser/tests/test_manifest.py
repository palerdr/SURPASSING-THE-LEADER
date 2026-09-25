"""The runtime manifest covers the hosted browser runtime.

``browser.deploy.manifest.RUNTIME_FILES`` is the one list of files that
``prepare_vercel`` copies into the Vercel bundle and that ``production``
hashes into the hosted code version. A module that the production entry
imports but the list omits is missing from the bundle, and the deployed
function fails at cold start.

The closure check runs the production entry in a fresh interpreter under each
``STL_HAL_POLICY`` value, because this test process may already hold modules
from other tests. The run builds the hosted app with an in-memory store and a
stand-in agent, then plays one half-round through a restart, so a module that
a request imports counts too.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys

import pytest

from browser.deploy.manifest import POLICIES, REPOSITORY_ROOT, RUNTIME_FILES, version_entries

TRAINING_MODULES = ("torch", "gymnasium", "stable_baselines3", "sb3_contrib")
# The terminal app and the lab never enter the hosted runtime.
FORBIDDEN_PROJECTS = ("terminal", "hal_lab")

_RUN_PRODUCTION = """
import json, sys, types
from pathlib import Path

root = Path(sys.argv[1])
sys.path.insert(0, str(root / "src"))

import numpy as np
from fastapi.testclient import TestClient

from browser.deploy import production
from dth.agent import CertifiedStageGame


class Agent:
    tablebase = types.SimpleNamespace(metadata={"code_config_digest": "closure"})

    def decide(self, state):
        uniform = np.full(60, 1.0 / 60)
        return types.SimpleNamespace(drop_policy=uniform, check_policy=uniform)

    def stage_game(self, state):
        uniform = np.full(60, 1.0 / 60)
        return CertifiedStageGame(
            state=tuple(int(value) for value in state), value=1.0 / 60, matrix=np.eye(60),
            drop_policy=uniform, check_policy=uniform.copy(), saddle_gap=0.0,
        )


class Store:
    def __init__(self):
        self.rows = {}

    async def get(self, key):
        return self.rows.get(key)

    async def compare_set(self, key, old, new):
        if self.rows.get(key) != old:
            return False
        self.rows[key] = new
        return True


production.CompleteDTHAgent = lambda *args, **kwargs: Agent()
production.RedisSessionStore = lambda *args: Store()
app = production.create_production_app(Path("unused"))
with TestClient(app, base_url="https://testserver") as client:
    def post(path, state, **values):
        response = client.post(path, json={"sequence": state["sequence"], **values})
        assert response.status_code == 200, response.text
        return response.json()

    assert client.get("/api/health").status_code == 200
    assert client.get("/api/rules").status_code == 200
    state = post("/api/session/begin", client.get("/api/session").json())
    state = post("/api/session/action", state, second=state["legal_seconds"][0])
    state = post("/api/session/ack", state)
    post("/api/session/restart", state)
    assert client.get("/api/transcript").status_code == 200

source = (root / "src").resolve()
files = set()
for module in list(sys.modules.values()):
    location = getattr(module, "__file__", None)
    if location and Path(location).resolve().is_relative_to(source):
        files.add(Path(location).resolve().relative_to(root.resolve()).as_posix())
print(json.dumps({"files": sorted(files), "modules": sorted(sys.modules)}))
"""


@pytest.fixture(scope="module", params=POLICIES)
def production_closure(request) -> dict[str, list[str]]:
    environment = {**os.environ, "STL_HAL_POLICY": request.param}
    for name in ("SUPABASE_URL", "SUPABASE_SECRET_KEY", "SUPABASE_SERVICE_ROLE_KEY"):
        environment.pop(name, None)
    result = subprocess.run(
        [sys.executable, "-I", "-c", _RUN_PRODUCTION, str(REPOSITORY_ROOT)],
        capture_output=True,
        text=True,
        env=environment,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout.strip().splitlines()[-1])


def test_the_manifest_covers_the_first_party_closure_of_production(production_closure) -> None:
    listed = {entry.path for entry in RUNTIME_FILES}
    assert sorted(set(production_closure["files"]) - listed) == []


def test_the_production_closure_imports_no_training_stack(production_closure) -> None:
    loaded = {name.split(".", 1)[0] for name in production_closure["modules"]}
    assert sorted(loaded & set(TRAINING_MODULES)) == []


def test_the_production_closure_imports_no_terminal_or_lab_module(production_closure) -> None:
    loaded = {name.split(".", 1)[0] for name in production_closure["modules"]}
    assert sorted(loaded & set(FORBIDDEN_PROJECTS)) == []


def test_each_manifest_file_exists_once() -> None:
    paths = [entry.path for entry in RUNTIME_FILES]
    assert len(paths) == len(set(paths))
    assert [path for path in paths if not (REPOSITORY_ROOT / path).is_file()] == []


def test_the_version_holds_production_and_leaves_out_the_name_filter() -> None:
    for policy in POLICIES:
        versioned = {entry.path for entry in version_entries(policy)}
        assert "src/browser/deploy/production.py" in versioned
        assert "src/browser/names.py" not in versioned
