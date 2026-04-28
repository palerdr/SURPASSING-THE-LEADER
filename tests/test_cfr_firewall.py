"""Static firewall for rigorous CFR modules.

Two complementary checks live here. The first catches *cross-module shaping
leaks* — any import inside ``environment/cfr/`` of a module known to carry
reward shaping, stage labels, or RL-agent observations. The second catches
*internal abstraction leaks* — any identifier defined or referenced inside
``environment/cfr/`` whose name signals bucketing, discretization, or
representative-value substitution. Together they make "rigorous CFR core"
a property the build can check, not just a convention in conversation.
"""

from __future__ import annotations

import ast
import pathlib


FORBIDDEN_IMPORTS = {
    "environment.awareness",
    "environment.observation",
    "environment.reward",
    "environment.route_math",
    "environment.route_stages",
}


FORBIDDEN_IDENTIFIER_SUBSTRINGS = (
    "bucket",
    "discretize",
    "quantize",
    "AbstractState",
    "representative_",
    "_BUCKET_SIZE",
    "NUM_BUCKETS",
)


def _imported_modules(path: pathlib.Path) -> set[str]:
    tree = ast.parse(path.read_text(), filename=str(path))
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                modules.add(alias.name)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            modules.add(node.module)
    return modules


def _identifiers(path: pathlib.Path) -> set[str]:
    tree = ast.parse(path.read_text(), filename=str(path))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, ast.Name):
            names.add(node.id)
        elif isinstance(node, ast.Attribute):
            names.add(node.attr)
        elif isinstance(node, ast.alias):
            names.add(node.name.split(".")[-1])
            if node.asname:
                names.add(node.asname)
    return names


def test_environment_cfr_has_no_shaping_or_observation_imports():
    root = pathlib.Path(__file__).resolve().parents[1]
    cfr_files = sorted((root / "environment" / "cfr").glob("**/*.py"))
    assert cfr_files

    violations: list[tuple[str, str]] = []
    for path in cfr_files:
        for module in _imported_modules(path):
            for forbidden in FORBIDDEN_IMPORTS:
                if module == forbidden or module.startswith(f"{forbidden}."):
                    violations.append((str(path.relative_to(root)), module))

    assert violations == []


def test_environment_cfr_has_no_bucketing_identifiers():
    root = pathlib.Path(__file__).resolve().parents[1]
    cfr_files = sorted((root / "environment" / "cfr").glob("**/*.py"))
    assert cfr_files

    violations: list[tuple[str, str, str]] = []
    for path in cfr_files:
        for name in _identifiers(path):
            for token in FORBIDDEN_IDENTIFIER_SUBSTRINGS:
                if token.lower() in name.lower():
                    violations.append((str(path.relative_to(root)), name, token))

    assert violations == []

