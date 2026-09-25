"""The repository root holds governance files and a fixed set of directories.

The check reads the entries a clone contains: the files git tracks, plus
untracked files that git does not ignore. A clone holds no ignored tool state,
such as `.venv/` or a stale `build/`, so the check skips it.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]

GOVERNANCE = frozenset(
    {
        ".gitattributes",
        ".github",
        ".gitignore",
        "AGENTS.md",
        "CLAUDE.md",
        "Cargo.lock",
        "Cargo.toml",
        "README.md",
        "pyproject.toml",
        "uv.lock",
    }
)
# docs/ holds the canonical contracts, paper/ the TeX papers, src/ the
# projects, and tests/ the repository meta-tests (tests/meta) and the
# terminal-versus-browser parity tests (tests/parity). outputs/ is the ignored
# legacy store of the Hal research runs; frozen evidence keys name its paths.
DIRECTORIES = frozenset({"docs", "paper", "src", "tests", "outputs"})
# Tracked root entries that a later restructure stage moves. Remove a name in
# the change that moves it; the second test fails while a listed name is gone.
LEGACY = frozenset()


def _root_entries() -> set[str]:
    if not (ROOT / ".git").exists():
        pytest.skip("the root layout check reads a git checkout")
    listing = subprocess.run(
        ["git", "-C", str(ROOT), "ls-files", "-z", "--cached", "--others", "--exclude-standard"],
        capture_output=True,
        check=True,
    ).stdout
    return {entry.decode("utf-8").split("/", 1)[0] for entry in listing.split(b"\0") if entry}


def test_every_root_entry_is_listed():
    unexpected = _root_entries() - GOVERNANCE - DIRECTORIES - LEGACY
    assert not unexpected, f"new root entries {sorted(unexpected)}; move them under src/, docs/, or tests/"


def test_legacy_entries_still_exist():
    gone = LEGACY - _root_entries()
    assert not gone, f"{sorted(gone)} left the root; remove them from LEGACY"
