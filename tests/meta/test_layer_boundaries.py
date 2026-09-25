"""Every Python import under src/ follows the rules in docs/PROJECTS.toml.

The registry gives each project three kinds of import rule:

- ``may_import``: the modules of other projects that the project can import.
  Each entry must sit under a ``public_interfaces`` entry of its owner. An
  import inside the same project passes.
- ``forbid_imports`` and ``forbid_imports_exempt``: the static torch firewall.
  No file of the project imports a listed package, except the listed files.
- ``core_modules`` and ``core_may_import``: a sub-boundary inside a project.

The check covers every ``src/<id>/**/*.py``, tests included. It reads each
import statement in the file, including imports inside functions, because a
lazy import is still a dependency. The torch firewall skips imports under
``if TYPE_CHECKING:``, because those never run.

A ``[[consumer]]`` entry, such as ``paper``, names a directory outside src/
whose scripts import projects. The check holds each of its scripts to the
entry's ``may_import`` and to the owners' ``public_interfaces``.

``FORBIDDEN_EDGES`` names imports that no ``may_import`` entry can allow: the
library imports no app and not the lab, and the terminal app imports neither
the browser app nor the lab. The rule matches the top-level package name, so
it holds before the target project exists.
"""

from __future__ import annotations

import ast
import os
import tomllib
from dataclasses import dataclass
from functools import cache
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = ROOT / "src"
REGISTRY = ROOT / "docs" / "PROJECTS.toml"
IGNORED_PARTS = frozenset(
    {
        ".git",
        ".lake",
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
        "_build",
        "_opam",
        "__pycache__",
        "node_modules",
        "dist",
    }
)
# The package marker of arena.policies must stay empty: a runtime that loads
# one provider must not load torch through a sibling module.
IMPORT_FREE = ("src/arena/policies/__init__.py",)
# Imports that no registry entry can allow, by importer and then by the
# top-level package of the target.
FORBIDDEN_EDGES = {
    "arena": ("terminal", "browser", "hal_lab"),
    "terminal": ("browser", "hal_lab"),
}

DOCUMENT = tomllib.loads(REGISTRY.read_text(encoding="utf-8"))
PROJECTS = {str(entry["id"]): entry for entry in DOCUMENT["project"]}
CONSUMERS = {str(entry["id"]): entry for entry in DOCUMENT.get("consumer", [])}


@dataclass(frozen=True)
class Import:
    line: int
    # The statement passes a rule when any candidate passes it. For
    # ``from a.b import c`` the candidates are ``a.b`` and ``a.b.c``, because
    # ``c`` can be a submodule or a name defined in ``a.b``.
    candidates: tuple[str, ...]
    type_checking: bool


def _under(module: str, prefix: str) -> bool:
    return module == prefix or module.startswith(prefix + ".")


def _interface(name: str) -> str:
    return name.removesuffix(".py")


def _is_python(entry: dict) -> bool:
    return "python" in entry["languages"]


def _import_root(entry: dict) -> Path:
    """The directory that must be on sys.path to import the project's modules."""

    base = ROOT / str(entry["path"])
    return base.parent if (base / "__init__.py").is_file() else base


@cache
def module_owners() -> dict[str, str]:
    """Map each first-party top-level module name to the project that owns it.

    A package project owns the package named after it. A flat project, such as
    dth_compact, owns its top-level modules. A package directory under src/
    that the registry omits owns its own name, so an import of it never passes
    as a third-party import.
    """

    owners: dict[str, str] = {}
    for project_id, entry in PROJECTS.items():
        if not _is_python(entry):
            continue
        base = ROOT / str(entry["path"])
        if (base / "__init__.py").is_file():
            owners[project_id] = project_id
        else:
            for path in base.glob("*.py"):
                owners[path.stem] = project_id
    for path in SOURCE_ROOT.iterdir():
        if (path / "__init__.py").is_file():
            owners.setdefault(path.name, path.name)
    return owners


def owner_of(module: str) -> str | None:
    return module_owners().get(module.split(".", 1)[0])


def module_name(entry: dict, path: Path) -> str:
    parts = list(path.relative_to(_import_root(entry)).with_suffix("").parts)
    if parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts)


def _is_type_checking(test: ast.expr) -> bool:
    return (isinstance(test, ast.Name) and test.id == "TYPE_CHECKING") or (
        isinstance(test, ast.Attribute) and test.attr == "TYPE_CHECKING"
    )


def imports_of(source: str, module: str, is_package: bool = False) -> list[Import]:
    """List the import statements of one module's source."""

    found: list[Import] = []
    package = module.split(".") if is_package else module.split(".")[:-1]

    def record(node: ast.AST, type_checking: bool) -> None:
        if isinstance(node, ast.Import):
            for alias in node.names:
                found.append(Import(node.lineno, (alias.name,), type_checking))
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                base = package[: len(package) - (node.level - 1)]
                target = ".".join(base + ([node.module] if node.module else []))
            else:
                target = node.module or ""
            for alias in node.names:
                names = (target,) if alias.name == "*" else (target, f"{target}.{alias.name}".lstrip("."))
                candidates = tuple(name for name in names if name)
                if candidates:
                    found.append(Import(node.lineno, candidates, type_checking))
        visit(node, type_checking)

    def visit(node: ast.AST, type_checking: bool) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.If) and _is_type_checking(child.test):
                for statement in child.body:
                    record(statement, True)
                for statement in child.orelse:
                    record(statement, type_checking)
            else:
                record(child, type_checking)

    visit(ast.parse(source), False)
    return found


def _walk(base: Path):
    for directory, names, files in os.walk(base):
        names[:] = sorted(name for name in names if name not in IGNORED_PARTS)
        for name in sorted(files):
            if name.endswith(".py"):
                yield Path(directory) / name


@cache
def python_files() -> tuple[tuple[str, str, str, list[Import]], ...]:
    """List (project id, module name, repository path, imports) for src/ files."""

    found = []
    for project_id, entry in PROJECTS.items():
        for path in _walk(ROOT / str(entry["path"])):
            module = module_name(entry, path)
            source = path.read_text(encoding="utf-8")
            imports = imports_of(source, module, path.name == "__init__.py")
            found.append((project_id, module, path.relative_to(ROOT).as_posix(), imports))
    return tuple(found)


@cache
def consumer_files() -> tuple[tuple[str, str, list[Import]], ...]:
    """List (consumer id, repository path, imports) for the consumers' scripts."""

    found = []
    for consumer_id, entry in CONSUMERS.items():
        base = ROOT / str(entry["path"])
        for path in _walk(base):
            module = ".".join(path.relative_to(base).with_suffix("").parts)
            imports = imports_of(path.read_text(encoding="utf-8"), module)
            found.append((consumer_id, path.relative_to(ROOT).as_posix(), imports))
    return tuple(found)


def layer_violations(project_id: str, imports: list[Import]) -> list[str]:
    entry = PROJECTS[project_id]
    allowed = [str(prefix) for prefix in entry.get("may_import", [])]
    return _violations(project_id, allowed, imports)


def consumer_violations(consumer_id: str, imports: list[Import]) -> list[str]:
    allowed = [_interface(str(prefix)) for prefix in CONSUMERS[consumer_id].get("may_import", [])]
    return _violations(None, allowed, imports)


def _violations(own: str | None, allowed: list[str], imports: list[Import]) -> list[str]:
    problems = []
    for statement in imports:
        owner = owner_of(statement.candidates[0])
        if owner is None or owner == own:
            continue
        interfaces = [_interface(str(name)) for name in PROJECTS.get(owner, {}).get("public_interfaces", [])]
        if any(
            any(_under(name, prefix) for prefix in allowed) and any(_under(name, face) for face in interfaces)
            for name in statement.candidates
        ):
            continue
        problems.append(f"line {statement.line} imports {statement.candidates[-1]} ({owner}); may_import = {allowed}")
    return problems


def core_violations(project_id: str, module: str, imports: list[Import]) -> list[str]:
    entry = PROJECTS[project_id]
    core = [str(name) for name in entry.get("core_modules", [])]
    if module not in core:
        return []
    allowed = core + [str(prefix) for prefix in entry.get("core_may_import", [])]
    problems = []
    for statement in imports:
        if owner_of(statement.candidates[0]) is None:
            continue
        if any(_under(name, prefix) for name in statement.candidates for prefix in allowed):
            continue
        problems.append(f"line {statement.line} imports {statement.candidates[-1]}; a core module may import {allowed}")
    return problems


def forbidden_edge_hits(project_id: str, imports: list[Import]) -> list[str]:
    forbidden = FORBIDDEN_EDGES.get(project_id, ())
    return [
        f"line {statement.line} imports {statement.candidates[-1]}; {project_id} never imports {list(forbidden)}"
        for statement in imports
        if statement.candidates[0].split(".", 1)[0] in forbidden
    ]


def firewall_hits(imports: list[Import], forbidden: list[str]) -> list[str]:
    return [
        f"line {statement.line} imports {statement.candidates[0]}"
        for statement in imports
        if not statement.type_checking and statement.candidates[0].split(".", 1)[0] in forbidden
    ]


def _report(problems: dict[str, list[str]]) -> str:
    return "\n".join(f"{path}: {problem}" for path, items in sorted(problems.items()) for problem in items)


# Registry shape


def test_every_project_declares_may_import_under_public_interfaces():
    problems = []
    for project_id, entry in PROJECTS.items():
        allowed = entry.get("may_import")
        if not isinstance(allowed, list):
            problems.append(f"{project_id}: may_import must be a list")
            continue
        for prefix in allowed:
            owner = owner_of(str(prefix))
            interfaces = [_interface(str(name)) for name in PROJECTS.get(owner, {}).get("public_interfaces", [])]
            if owner is None or owner == project_id:
                problems.append(f"{project_id}: may_import {prefix!r} names no other Python project")
            elif not any(_under(str(prefix), face) for face in interfaces):
                problems.append(f"{project_id}: may_import {prefix!r} is outside {owner}'s public interfaces")
    assert not problems, "\n".join(problems)


def test_every_consumer_lies_outside_src_and_imports_public_interfaces():
    problems = []
    for consumer_id, entry in CONSUMERS.items():
        base = ROOT / str(entry["path"])
        if consumer_id in PROJECTS:
            problems.append(f"{consumer_id}: a consumer id must differ from every project id")
        if not base.is_dir() or base.resolve().is_relative_to(SOURCE_ROOT.resolve()):
            problems.append(f"{consumer_id}: path {entry['path']} must be a directory outside src/")
        allowed = entry.get("may_import")
        if not isinstance(allowed, list):
            problems.append(f"{consumer_id}: may_import must be a list")
            continue
        for prefix in allowed:
            name = _interface(str(prefix))
            owner = owner_of(name)
            interfaces = [_interface(str(face)) for face in PROJECTS.get(owner, {}).get("public_interfaces", [])]
            if owner is None:
                problems.append(f"{consumer_id}: may_import {prefix!r} names no Python project")
            elif not any(_under(name, face) for face in interfaces):
                problems.append(f"{consumer_id}: may_import {prefix!r} is outside {owner}'s public interfaces")
    assert not problems, "\n".join(problems)


def test_no_may_import_entry_opens_a_forbidden_edge():
    problems = []
    for project_id, forbidden in FORBIDDEN_EDGES.items():
        assert project_id in PROJECTS, f"FORBIDDEN_EDGES names the unregistered project {project_id}"
        for prefix in PROJECTS[project_id].get("may_import", []):
            if str(prefix).split(".", 1)[0] in forbidden:
                problems.append(f"{project_id}: may_import {prefix!r} opens a forbidden edge")
    assert not problems, "\n".join(problems)


def test_exempt_files_exist_and_still_import_a_forbidden_package():
    imports = {path: found for _, _, path, found in python_files()}
    problems = []
    for project_id, entry in PROJECTS.items():
        exempt = entry.get("forbid_imports_exempt", [])
        forbidden = entry.get("forbid_imports", [])
        if exempt and not forbidden:
            problems.append(f"{project_id}: forbid_imports_exempt needs forbid_imports")
        for path in exempt:
            if not str(path).startswith(f"{entry['path']}/"):
                problems.append(f"{project_id}: exempt file {path} lies outside {entry['path']}")
            elif path not in imports:
                problems.append(f"{project_id}: exempt file {path} does not exist; remove it from the list")
            elif not firewall_hits(imports[path], forbidden):
                problems.append(f"{project_id}: exempt file {path} imports none of {forbidden}; remove it")
    assert not problems, "\n".join(problems)


def test_core_modules_exist():
    for project_id, entry in PROJECTS.items():
        modules = {module for owner, module, _, _ in python_files() if owner == project_id}
        for name in entry.get("core_modules", []):
            assert name in modules, f"{project_id}: core module {name} does not exist"


def test_every_python_directory_under_src_is_a_registered_project():
    for path in sorted(SOURCE_ROOT.iterdir()):
        if path.name in IGNORED_PARTS:
            continue
        if path.is_file():
            assert path.suffix != ".py", f"{path.relative_to(ROOT)} belongs to no project"
        elif next(_walk(path), None) is not None:
            assert path.name in PROJECTS, f"src/{path.name} holds Python files but is not in {REGISTRY.name}"


# Import rules on the tree


def test_cross_project_imports_follow_may_import():
    problems = {path: layer_violations(owner, found) for owner, _, path, found in python_files()}
    assert not any(problems.values()), _report(problems)


def test_consumer_scripts_follow_may_import():
    problems = {path: consumer_violations(consumer, found) for consumer, path, found in consumer_files()}
    assert not any(problems.values()), _report(problems)


def test_core_modules_import_their_allowed_modules_alone():
    problems = {path: core_violations(owner, module, found) for owner, module, path, found in python_files()}
    assert not any(problems.values()), _report(problems)


def test_forbidden_edges_stay_closed():
    problems = {path: forbidden_edge_hits(owner, found) for owner, _, path, found in python_files()}
    assert not any(problems.values()), _report(problems)


def test_torch_firewall():
    problems = {}
    for owner, _, path, found in python_files():
        entry = PROJECTS[owner]
        if path in entry.get("forbid_imports_exempt", []):
            continue
        problems[path] = firewall_hits(found, entry.get("forbid_imports", []))
    assert not any(problems.values()), _report(problems)


def test_import_free_package_markers_import_nothing():
    for relative in IMPORT_FREE:
        assert not imports_of((ROOT / relative).read_text(encoding="utf-8"), "", True), relative


# The checker itself


def test_checker_flags_a_peer_that_imports_arena():
    found = imports_of("import arena\n", "dth.solver")
    assert layer_violations("dth", found) == ["line 1 imports arena (arena); may_import = []"]


def test_checker_flags_forbidden_edges_before_the_target_exists():
    found = imports_of("import browser.app\nfrom hal_lab import cli\nfrom arena import session\n", "terminal.cli")
    assert forbidden_edge_hits("terminal", found) == [
        "line 1 imports browser.app; terminal never imports ['browser', 'hal_lab']",
        "line 2 imports hal_lab.cli; terminal never imports ['browser', 'hal_lab']",
    ]
    found = imports_of("from terminal.cli import main\nimport hal_lab.cli\nimport stl.engine.game\n", "arena.x")
    assert forbidden_edge_hits("arena", found) == [
        "line 1 imports terminal.cli.main; arena never imports ['terminal', 'browser', 'hal_lab']",
        "line 2 imports hal_lab.cli; arena never imports ['terminal', 'browser', 'hal_lab']",
    ]


def test_checker_accepts_public_interfaces_and_rejects_internals():
    public = imports_of("from dth import agent\nfrom dth.agent import CompleteDTHAgent\nimport stl.engine.game\n", "arena.x")
    assert layer_violations("arena", public) == []
    private = imports_of("from dth import complete_tablebase\n", "arena.x")
    assert len(layer_violations("arena", private)) == 1
    relative = imports_of("from ..solver import leap_build\n", "stl.engine.game")
    assert relative[0].candidates == ("stl.solver", "stl.solver.leap_build")
    assert layer_violations("stl", relative) == []


def test_checker_holds_the_paper_to_stl_reader_and_the_compact_solver():
    public = imports_of(
        "from stl.reader import open_leap\nfrom stl import reader\nimport main as solver\nimport numpy\n",
        "make_stl_figures",
    )
    assert consumer_violations("paper", public) == []
    internal = imports_of(
        "from stl.solver.leap_audit import PackedReader\nimport stl.engine.game\nimport arena.session\n",
        "make_stl_figures",
    )
    assert consumer_violations("paper", internal) == [
        "line 1 imports stl.solver.leap_audit.PackedReader (stl); may_import = ['stl.reader', 'main']",
        "line 2 imports stl.engine.game (stl); may_import = ['stl.reader', 'main']",
        "line 3 imports arena.session (arena); may_import = ['stl.reader', 'main']",
    ]


def test_checker_holds_core_modules_to_stl_engine():
    allowed = imports_of("from arena.contracts import X\nfrom stl.engine.game import Game\n", "arena.match")
    assert core_violations("arena", "arena.match", allowed) == []
    adapter = imports_of("from arena.dth_adapter import X\nimport numpy\n", "arena.match")
    assert core_violations("arena", "arena.match", adapter) == [
        "line 1 imports arena.dth_adapter.X; a core module may import "
        "['arena.contracts', 'arena.agent', 'arena.session', 'arena.match', 'arena.variants', 'stl.engine']"
    ]


def test_checker_sees_lazy_imports_and_skips_type_checking_blocks():
    source = (
        "from typing import TYPE_CHECKING\n"
        "if TYPE_CHECKING:\n"
        "    import torch\n"
        "def load():\n"
        "    import gymnasium\n"
        "    from stable_baselines3 import PPO\n"
    )
    hits = firewall_hits(imports_of(source, "arena.x"), ["torch", "gymnasium", "stable_baselines3"])
    assert hits == ["line 5 imports gymnasium", "line 6 imports stable_baselines3"]
