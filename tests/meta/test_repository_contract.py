from __future__ import annotations

import ast
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = ROOT / "src"
PROJECTS = ("stl", "dth", "abstract")
IMPLEMENTATIONS = (*PROJECTS, "crates", "dth_ocaml", "arena")
IGNORED_PARTS = {
    ".git",
    ".venv",
    ".tools",
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


# A subtree's README is also its instruction file, so exactly these are the
# ones CLAUDE.md imports. A README deeper inside a subtree is ordinary
# documentation and is deliberately not an instruction file.
INSTRUCTION_READMES = frozenset(
    [f"src/{name}/README.md" for name in IMPLEMENTATIONS] + ["docs/papers/README.md"]
)


def test_readmes_are_owned_by_root_or_a_subtree():
    for path in _source_files("README.md"):
        relative = path.relative_to(ROOT)
        assert (
            relative == Path("README.md")
            or relative.as_posix() in INSTRUCTION_READMES
            or (relative.parts[0] == "src" and relative.parts[1] in IMPLEMENTATIONS)
        ), relative


def test_markdown_is_owned_by_an_approved_context_root():
    approved_roots = {"docs", "src", ".codex"}
    for path in _source_files(".md"):
        relative = path.relative_to(ROOT)
        if len(relative.parts) == 1:
            assert relative.name in {"README.md", "AGENTS.md", "CLAUDE.md"}
        else:
            assert relative.parts[0] in approved_roots, relative
            if relative.parts[0] == "src":
                assert len(relative.parts) >= 3
                assert relative.parts[1] in IMPLEMENTATIONS, relative


EVIDENCE = ROOT / "docs" / "game-sources" / "EVIDENCE.md"


def _declared_evidence_ids() -> set[str]:
    text = EVIDENCE.read_text(encoding="utf-8")
    return set(re.findall(r"<!-- evidence:(E-[A-Z-]+) -->", text))


def test_every_evidence_id_has_a_matching_anchor():
    text = EVIDENCE.read_text(encoding="utf-8")
    anchors = set(re.findall(r'<a id="([a-z-]+)"></a>', text))
    declared = _declared_evidence_ids()
    assert declared, "the evidence ledger declares no evidence IDs"
    assert {identifier.lower() for identifier in declared} == anchors


def test_evidence_links_resolve_to_a_declared_id():
    """Docs must cite evidence by stable ID anchor, never by line number.

    Line-number links rot on every edit to the ledger; the previous contract
    enforced them and they had already broken.
    """
    declared = {identifier.lower() for identifier in _declared_evidence_ids()}
    anchor_link = re.compile(r"\((?P<path>[^()\[\]\s]*EVIDENCE\.md)#(?P<anchor>[a-z-]+)\)")
    line_link = re.compile(r"\([^()\[\]\s]*\.md#L\d+-L\d+\)")
    links = []
    for document in (ROOT / "docs").rglob("*.md"):
        text = document.read_text(encoding="utf-8")
        assert not line_link.search(text), f"{document} uses a forbidden line-number link"
        for match in anchor_link.finditer(text):
            target = (document.parent / match.group("path")).resolve()
            assert target == EVIDENCE.resolve(), f"{document} -> {target}"
            assert match.group("anchor") in declared, f"{document} -> #{match.group('anchor')}"
            links.append((document, match.group("anchor")))
    assert links, "no document cites the evidence ledger"


def test_claude_md_imports_every_subtree_readme():
    """Every subtree README is an instruction file and must be imported.

    The root AGENTS.md is imported separately as the global rules; subtree
    guidance lives in that subtree's README so one file serves a reader and an
    agent. READMEs nested deeper inside a subtree are documentation, not
    instructions, and are excluded by INSTRUCTION_READMES.
    """

    claude = (ROOT / "CLAUDE.md").read_text(encoding="utf-8")
    imported = set(re.findall(r"^@(\S*README\.md)\s*$", claude, flags=re.MULTILINE))
    on_disk = {
        path.relative_to(ROOT).as_posix()
        for path in _source_files("README.md")
        if path.relative_to(ROOT).as_posix() in INSTRUCTION_READMES
    }
    assert on_disk == INSTRUCTION_READMES, f"instruction README missing on disk: {INSTRUCTION_READMES - on_disk}"
    assert imported == on_disk, f"missing {on_disk - imported}, stale {imported - on_disk}"


def test_solver_projects_do_not_import_each_other():
    for project in PROJECTS:
        forbidden = set(PROJECTS) - {project}
        for path in (SOURCE_ROOT / project).rglob("*.py"):
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
        assert "stl.abstract" not in text, path
        assert not re.search(r"(?:from|import)\s+pure(?:\.|\s)", text), path


def test_solver_configs_own_their_default_artifact_paths():
    for path in (SOURCE_ROOT / "stl" / "config").rglob("*.yaml"):
        text = path.read_text(encoding="utf-8")
        assert not re.search(r"(?<!src/stl/)outputs/", text), path
        assert not re.search(r"(?<!src/stl/)checkpoints/", text), path
    for project in ("dth", "abstract"):
        config_root = SOURCE_ROOT / project / "config"
        if not config_root.exists():
            continue
        for path in config_root.rglob("*.yaml"):
            text = path.read_text(encoding="utf-8")
            assert "src/stl/outputs/" not in text, path
            assert re.search(rf"\b(?:{project}/|\.)", text) or "output" not in text, path
