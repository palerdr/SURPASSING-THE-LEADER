"""The one list of files that the hosted browser runtime needs.

``prepare_vercel`` copies each file to ``runtime/<path>`` in the Vercel
bundle, so the bundle keeps the repository layout below ``runtime/``. The DTH
``code_config_digest`` labels its inputs ``src/dth/...`` and ``uv.lock``, and
``CompleteDTHAgent`` checks that digest at each cold start, so the layout must
match.

``production`` hashes the files whose ``in_version`` flag is set into the
hosted code version, together with the artifact's ``code_config_digest``. A
file enters the version only under the ``STL_HAL_POLICY`` values in its
``policies``. A new version ends every live hosted session, so a file stays
out of the version when a change to it must not end games: ``names.py`` holds
the leaderboard word list (see ``DEPLOYMENT.md``).

``tests/test_manifest.py`` checks that the list holds each first-party module
that ``browser.deploy.production`` imports under both policies.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from dth.agent import runtime_source_files

# The repository root, or the ``runtime/`` directory inside the bundle.
REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
POLICIES = ("exact", "translated-v1")
_TRANSLATED = ("translated-v1",)


@dataclass(frozen=True)
class RuntimeFile:
    """One runtime file, by its path from the repository root."""

    path: str
    in_version: bool = False
    # The STL_HAL_POLICY values that load the file.
    policies: tuple[str, ...] = POLICIES


RUNTIME_FILES: tuple[RuntimeFile, ...] = (
    # The game session and the referee.
    RuntimeFile("src/arena/__init__.py"),
    RuntimeFile("src/arena/agent.py", in_version=True),
    RuntimeFile("src/arena/contracts.py", in_version=True),
    RuntimeFile("src/arena/dth_adapter.py", in_version=True),
    RuntimeFile("src/arena/session.py", in_version=True),
    RuntimeFile("src/arena/variants.py"),
    RuntimeFile("src/arena/presentation/__init__.py"),
    RuntimeFile("src/arena/presentation/rules_text.py"),
    RuntimeFile("src/arena/presentation/scene_art.py"),
    RuntimeFile("src/arena/presentation/sprites.py"),
    RuntimeFile("src/stl/__init__.py"),
    RuntimeFile("src/stl/engine/__init__.py"),
    RuntimeFile("src/stl/engine/actions.py", in_version=True),
    RuntimeFile("src/stl/engine/game.py", in_version=True),
    # The browser server, its hosted wrapper, and this deploy entry.
    RuntimeFile("src/browser/__init__.py"),
    RuntimeFile("src/browser/app.py", in_version=True),
    RuntimeFile("src/browser/hosted.py", in_version=True),
    RuntimeFile("src/browser/ledger.py", in_version=True),
    RuntimeFile("src/browser/names.py"),
    RuntimeFile("src/browser/schema.py", in_version=True),
    RuntimeFile("src/browser/deploy/__init__.py"),
    RuntimeFile("src/browser/deploy/manifest.py", in_version=True),
    RuntimeFile("src/browser/deploy/production.py", in_version=True),
    # The translated Hal candidate and its opponent memory.
    RuntimeFile("src/arena/translated_hal_adapter.py", in_version=True, policies=_TRANSLATED),
    RuntimeFile("src/arena/policies/__init__.py", policies=_TRANSLATED),
    RuntimeFile("src/arena/policies/perfect_hal.py", in_version=True, policies=_TRANSLATED),
    RuntimeFile("src/arena/policies/translated_hal.py", in_version=True, policies=_TRANSLATED),
    RuntimeFile(
        "src/arena/config/translated_hal_v1_selection.json", in_version=True, policies=_TRANSLATED
    ),
    RuntimeFile("src/browser/opponent_memory.py", in_version=True, policies=_TRANSLATED),
    # The certified DTH agent. code_config_digest stands for these files in
    # the version.
    *(RuntimeFile(path) for path in runtime_source_files()),
)


def version_entries(policy: str) -> tuple[RuntimeFile, ...]:
    """Return the files that enter the hosted code version under ``policy``."""

    if policy not in POLICIES:
        raise ValueError(f"STL_HAL_POLICY must be one of {', '.join(POLICIES)}")
    return tuple(entry for entry in RUNTIME_FILES if entry.in_version and policy in entry.policies)
