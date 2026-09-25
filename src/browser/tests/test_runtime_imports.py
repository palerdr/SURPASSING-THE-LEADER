"""The hosted browser runtime imports without the training stack or the terminal.

Each check runs in a fresh interpreter, because this test process may already
hold torch or the terminal modules from another test module.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

SOURCE_ROOT = Path(__file__).resolve().parents[2]

# The hosted providers, the translated opponent memory, and the Vercel entry.
RUNTIME_MODULES = (
    "arena.policies.perfect_hal",
    "arena.policies.translated_hal",
    "arena.translated_hal_adapter",
    "browser.opponent_memory",
    "browser.deploy.production",
)
TRAINING_MODULES = ("torch", "gymnasium", "stable_baselines3", "sb3_contrib")
TERMINAL_MODULES = ("terminal", "terminal.cli", "terminal.tui")

# The rules and art routes imported the terminal renderer inside the request
# handler, so an import-time check alone would not see that edge.
_SERVE_RULES_AND_ART = """
from fastapi.testclient import TestClient
from arena.presentation.scene_art import SceneArt
from browser.app import create_app
client = TestClient(create_app(hal_factory=object, art_loader=SceneArt, webclient_dist=None))
assert client.get("/api/rules").status_code == 200
assert client.get("/art/hal/idle/0.png").status_code == 404
client.get("/art/panel/stl_rules")
"""


def _loaded_modules(candidates: tuple[str, ...], exercise: str = "") -> list[str]:
    code = (
        "import importlib, json, sys\n"
        "sys.path.insert(0, sys.argv[1])\n"
        "for name in json.loads(sys.argv[2]):\n"
        "    importlib.import_module(name)\n"
        f"{exercise}\n"
        "print(json.dumps(sorted(set(json.loads(sys.argv[3])) & set(sys.modules))))\n"
    )
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            code,
            str(SOURCE_ROOT),
            json.dumps(RUNTIME_MODULES),
            json.dumps(candidates),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout.strip().splitlines()[-1])


def test_hosted_runtime_imports_without_training_dependencies() -> None:
    assert _loaded_modules(TRAINING_MODULES) == []


def test_hosted_runtime_does_not_import_the_terminal_surface() -> None:
    assert _loaded_modules(TERMINAL_MODULES, _SERVE_RULES_AND_ART) == []
