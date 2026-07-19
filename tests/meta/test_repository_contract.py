from __future__ import annotations

import ast
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
PROJECTS = ("stl", "dth", "toy")
IGNORED_PARTS = {
    ".git",
    ".venv",
    ".pytest_cache",
    ".ruff_cache",
    "graphify-out",
    "outputs",
    "checkpoints",
    "artifacts",
    "target",
    "build",
    "__pycache__",
}


def _source_files(suffix: str):
    for path in ROOT.rglob(f"*{suffix}"):
        if not IGNORED_PARTS.intersection(path.relative_to(ROOT).parts):
            yield path


def test_only_root_readme_remains():
    assert [path.relative_to(ROOT) for path in _source_files("README.md")] == [Path("README.md")]


def test_markdown_is_owned_by_an_approved_context_root():
    approved_roots = {"docs", "papers", "stl", "dth", "toy", "crates", ".codex"}
    for path in _source_files(".md"):
        relative = path.relative_to(ROOT)
        if len(relative.parts) == 1:
            assert relative.name in {"README.md", "AGENTS.md"}
        else:
            assert relative.parts[0] in approved_roots, relative


def test_evidence_line_links_resolve_and_include_their_marker():
    pattern = re.compile(r"\((?P<path>[^)]+\.md)#L(?P<start>\d+)-L(?P<end>\d+)\)")
    links = []
    for document in (ROOT / "docs").glob("*.md"):
        for match in pattern.finditer(document.read_text(encoding="utf-8")):
            target = (document.parent / match.group("path")).resolve()
            start, end = int(match.group("start")), int(match.group("end"))
            lines = target.read_text(encoding="utf-8").splitlines()
            assert 1 <= start <= end <= len(lines)
            assert "evidence:" in "\n".join(lines[start - 1 : end])
            links.append((document, target, start, end))
    assert len(links) == 5


def test_solver_projects_do_not_import_each_other():
    for project in PROJECTS:
        forbidden = set(PROJECTS) - {project}
        for path in (ROOT / project).rglob("*.py"):
            if IGNORED_PARTS.intersection(path.relative_to(ROOT).parts):
                continue
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
                module = None
                if isinstance(node, ast.ImportFrom):
                    module = node.module
                elif isinstance(node, ast.Import):
                    for alias in node.names:
                        assert alias.name.split(".", 1)[0] not in forbidden, path
                if module is not None:
                    assert module.split(".", 1)[0] not in forbidden, path


def test_removed_python_namespaces_do_not_reappear():
    for path in _source_files(".py"):
        if path.resolve() == Path(__file__).resolve():
            continue
        text = path.read_text(encoding="utf-8")
        assert "stl.toy" not in text, path
        assert not re.search(r"(?:from|import)\s+pure(?:\.|\s)", text), path


def test_solver_configs_own_their_default_artifact_paths():
    for path in (ROOT / "stl" / "config").rglob("*.yaml"):
        text = path.read_text(encoding="utf-8")
        assert not re.search(r"(?<!stl/)outputs/", text), path
        assert not re.search(r"(?<!stl/)checkpoints/", text), path
    for project in ("dth", "toy"):
        config_root = ROOT / project / "config"
        if not config_root.exists():
            continue
        for path in config_root.rglob("*.yaml"):
            text = path.read_text(encoding="utf-8")
            assert "outputs/regen2rl" not in text, path
            assert re.search(rf"\b(?:{project}/|\.)", text) or "output" not in text, path
